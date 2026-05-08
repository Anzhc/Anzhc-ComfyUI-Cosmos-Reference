import math
import types

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.conds
from comfy.ldm.modules.attention import optimized_attention


COSMOS_REF_ATTENTION_KEY = "cosmos_ref_attention"
COSMOS_REF_STRENGTH_KEY = "cosmos_reference_strength"
COSMOS_REF_METHOD_KEY = "cosmos_reference_method"
COSMOS_REF_METHOD_TEMPORAL_MASK = "temporal_mask"
COSMOS_REF_METHOD_KV_GATING = "kv_gating"
COSMOS_REF_METHODS = (COSMOS_REF_METHOD_TEMPORAL_MASK, COSMOS_REF_METHOD_KV_GATING)


def ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def scalar_strength(value, default=1.0):
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.flatten()[0].detach().cpu())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return float(default)
        return scalar_strength(value[0], default=default)
    return float(value)


def normalize_ref_latents(ref_latents):
    if ref_latents is None:
        return []
    if isinstance(ref_latents, (list, tuple)):
        return list(ref_latents)
    return [ref_latents]


def normalize_method(value, default=COSMOS_REF_METHOD_KV_GATING):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return normalize_method(value[0], default=default)

    method = str(value)
    if method not in COSMOS_REF_METHODS:
        return default
    return method


def select_reference_indices(n_ref, strength, device):
    if strength <= 0.0 or n_ref <= 0:
        return None

    if strength < 1.0:
        keep = max(1, int(round(n_ref * strength)))
        return torch.linspace(0, n_ref - 1, keep, device=device).round().long().unique()

    return torch.arange(n_ref, device=device)


def ref_kv_gated_attention_op(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D, transformer_options=None):
    if transformer_options is None:
        transformer_options = {}

    info = transformer_options.get(COSMOS_REF_ATTENTION_KEY, None)
    if info is not None:
        target_tokens = int(info["target_tokens"])
        total_tokens = int(info["total_tokens"])
        strength = float(info["strength"])

        is_video_self_attn = (
            q_B_S_H_D.shape[1] == total_tokens
            and k_B_S_H_D.shape[1] == total_tokens
            and v_B_S_H_D.shape[1] == total_tokens
            and 0 < target_tokens < total_tokens
        )

        if is_video_self_attn and strength != 1.0:
            k_target = k_B_S_H_D[:, :target_tokens]
            v_target = v_B_S_H_D[:, :target_tokens]
            k_ref = k_B_S_H_D[:, target_tokens:]
            v_ref = v_B_S_H_D[:, target_tokens:]
            n_ref = k_ref.shape[1]

            if strength <= 0.0 or n_ref == 0:
                k_B_S_H_D = k_target
                v_B_S_H_D = v_target
            elif strength < 1.0:
                idx = select_reference_indices(n_ref, strength, k_ref.device)
                k_B_S_H_D = torch.cat([k_target, k_ref[:, idx]], dim=1)
                v_B_S_H_D = torch.cat([v_target, v_ref[:, idx]], dim=1)
            else:
                whole = int(math.floor(strength))
                frac = strength - whole

                k_refs = [k_ref] * whole
                v_refs = [v_ref] * whole

                if frac > 0.0:
                    idx = select_reference_indices(n_ref, frac, k_ref.device)
                    if idx is not None:
                        k_refs.append(k_ref[:, idx])
                        v_refs.append(v_ref[:, idx])

                k_B_S_H_D = torch.cat([k_target] + k_refs, dim=1)
                v_B_S_H_D = torch.cat([v_target] + v_refs, dim=1)

    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape

    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(
        in_q_shape[0],
        in_q_shape[-2],
        -1,
        in_q_shape[-1],
    )
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(
        in_k_shape[0],
        in_k_shape[-2],
        -1,
        in_k_shape[-1],
    )
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(
        in_k_shape[0],
        in_k_shape[-2],
        -1,
        in_k_shape[-1],
    )

    return optimized_attention(
        q_B_H_S_D,
        k_B_H_S_D,
        v_B_H_S_D,
        in_q_shape[-2],
        skip_reshape=True,
        transformer_options=transformer_options,
    )


def match_batch(ref, target_batch):
    ref_batch = ref.shape[0]
    if ref_batch == target_batch:
        return ref
    if ref_batch == 1:
        return ref.expand(target_batch, -1, -1, -1, -1)
    if target_batch % ref_batch == 0:
        return ref.repeat(target_batch // ref_batch, 1, 1, 1, 1)
    if ref_batch > target_batch:
        return ref[:target_batch]
    repeat_count = ceil_div(target_batch, ref_batch)
    return ref.repeat(repeat_count, 1, 1, 1, 1)[:target_batch]


def match_first_dim(tensor, target_batch):
    tensor_batch = tensor.shape[0]
    if tensor_batch == target_batch:
        return tensor

    repeat_shape = (1,) * (tensor.ndim - 1)
    if tensor_batch == 1:
        return tensor.expand(target_batch, *[-1] * (tensor.ndim - 1))
    if target_batch % tensor_batch == 0:
        return tensor.repeat((target_batch // tensor_batch,) + repeat_shape)
    if tensor_batch > target_batch:
        return tensor[:target_batch]

    repeat_count = ceil_div(target_batch, tensor_batch)
    return tensor.repeat((repeat_count,) + repeat_shape)[:target_batch]


def as_5d_ref(ref):
    if ref.ndim == 4:
        ref = ref.unsqueeze(2)
    if ref.ndim != 5:
        raise RuntimeError(f"Cosmos reference latent must be 4D or 5D, got rank {ref.ndim}.")
    return ref


def align_temporal_boundary(orig_x, patch_temporal):
    orig_t = orig_x.shape[2]
    if patch_temporal <= 1:
        return orig_x, orig_t, 0

    aligned_t = ceil_div(orig_t, patch_temporal) * patch_temporal
    pad_t = aligned_t - orig_t
    if pad_t == 0:
        return orig_x, orig_t, 0

    temporal_pad = orig_x.new_zeros(
        orig_x.shape[0],
        orig_x.shape[1],
        pad_t,
        orig_x.shape[3],
        orig_x.shape[4],
    )
    return torch.cat([orig_x, temporal_pad], dim=2), orig_t, pad_t


def make_temporal_padding_channel(padding_mask, x_B_C_T_H_W):
    b, _, t, h, w = x_B_C_T_H_W.shape
    dtype = x_B_C_T_H_W.dtype
    device = x_B_C_T_H_W.device

    if padding_mask is None:
        return torch.zeros(b, 1, t, h, w, dtype=dtype, device=device)

    mask = padding_mask.to(device=device, dtype=dtype)
    mask = match_first_dim(mask, b)

    if mask.ndim == 5:
        if mask.shape[1] != 1:
            mask = mask[:, :1]

        if mask.shape[2] < t:
            mask = F.pad(mask, (0, 0, 0, 0, 0, t - mask.shape[2]), value=1.0)
        elif mask.shape[2] > t:
            mask = mask[:, :, :t]

        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask.float(), size=(t, h, w), mode="nearest").to(dtype=dtype)
        return mask

    if mask.ndim == 4 and mask.shape[1] == t and mask.shape[1] != 1:
        mask = mask.unsqueeze(1)
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask.float(), size=(t, h, w), mode="nearest").to(dtype=dtype)
        return mask

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    if mask.ndim == 4:
        if mask.shape[1] != 1:
            mask = mask[:, :1]

        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask.float(), size=(h, w), mode="nearest").to(dtype=dtype)
        return mask.unsqueeze(2).repeat(1, 1, t, 1, 1)

    raise RuntimeError(
        f"Unsupported Cosmos padding_mask shape: {tuple(mask.shape)}. "
        "Expected [B, 1, T, H, W], [B, T, H, W], [B, 1, H, W], or [B, H, W]."
    )


def prepare_refs_for_concat(ref_latents, new_x):
    refs = []
    for ref in normalize_ref_latents(ref_latents):
        if not torch.is_tensor(ref):
            continue

        ref = as_5d_ref(ref)
        ref = ref.to(dtype=new_x.dtype, device=new_x.device)
        ref = match_batch(ref, new_x.shape[0])

        if ref.shape[1] != new_x.shape[1] or ref.shape[3:] != new_x.shape[3:]:
            raise RuntimeError(
                "Cosmos reference latent shape mismatch: "
                f"target={tuple(new_x.shape)}, reference={tuple(ref.shape)}."
            )

        refs.append(ref)
    return refs


def patched_prepare_embedded_sequence(
    self,
    x_B_C_T_H_W,
    fps=None,
    padding_mask=None,
):
    if self.concat_padding_mask:
        padding_channel = make_temporal_padding_channel(padding_mask, x_B_C_T_H_W)
        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_channel], dim=1)

    x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

    if self.extra_per_block_abs_pos_emb:
        extra_pos_emb = self.extra_pos_embedder(
            x_B_T_H_W_D,
            fps=fps,
            device=x_B_C_T_H_W.device,
            dtype=x_B_C_T_H_W.dtype,
        )
    else:
        extra_pos_emb = None

    if "rope" in self.pos_emb_cls.lower():
        return (
            x_B_T_H_W_D,
            self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device),
            extra_pos_emb,
        )

    x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, device=x_B_C_T_H_W.device)
    return x_B_T_H_W_D, None, extra_pos_emb


def call_model_apply(prev_wrapper, model_apply, model_kwargs):
    if prev_wrapper is not None:
        return prev_wrapper(model_apply, model_kwargs)

    mk = model_kwargs.copy()
    x_val = mk.pop("input")
    t_val = mk.pop("timestep")
    cond_dict = mk.pop("c", {})
    return model_apply(x_val, t_val, **cond_dict, **mk)


def conditioning_with_reference(conditioning, reference_samples, strength, control_method):
    out = []
    for cond in conditioning:
        metadata = cond[1].copy()
        refs = normalize_ref_latents(metadata.get("reference_latents", None))
        refs.append(reference_samples)
        metadata["reference_latents"] = refs
        metadata[COSMOS_REF_STRENGTH_KEY] = float(strength)
        metadata[COSMOS_REF_METHOD_KEY] = normalize_method(
            control_method,
            default=COSMOS_REF_METHOD_TEMPORAL_MASK,
        )
        out.append([cond[0], metadata])
    return out


class CosmosReferenceConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "reference_latent": ("LATENT",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                "control_method": (
                    [COSMOS_REF_METHOD_TEMPORAL_MASK, COSMOS_REF_METHOD_KV_GATING],
                    {"default": COSMOS_REF_METHOD_TEMPORAL_MASK},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "Cosmos/Reference"

    def apply(self, positive, negative, reference_latent, strength, control_method=COSMOS_REF_METHOD_TEMPORAL_MASK):
        reference_samples = reference_latent.get("samples", None)
        if reference_samples is None:
            raise RuntimeError("Cosmos Reference Conditioning requires a LATENT with a samples tensor.")

        return (
            conditioning_with_reference(positive, reference_samples, strength, control_method),
            conditioning_with_reference(negative, reference_samples, strength, control_method),
        )


class ApplyCosmosReferenceModelPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Cosmos/Reference"

    def patch(self, model):
        m = model.clone()

        original_extra_conds = m.model.extra_conds

        def custom_extra_conds(self, **kwargs):
            out = original_extra_conds(**kwargs)

            strength = scalar_strength(kwargs.get(COSMOS_REF_STRENGTH_KEY, None), default=1.0)
            method = normalize_method(
                kwargs.get(COSMOS_REF_METHOD_KEY, None),
                default=COSMOS_REF_METHOD_KV_GATING,
            )
            if COSMOS_REF_STRENGTH_KEY in kwargs:
                out[COSMOS_REF_STRENGTH_KEY] = comfy.conds.CONDConstant(strength)
            if COSMOS_REF_METHOD_KEY in kwargs:
                out[COSMOS_REF_METHOD_KEY] = comfy.conds.CONDConstant(method)

            ref_latents = normalize_ref_latents(kwargs.get("reference_latents", None))
            if ref_latents and (method == COSMOS_REF_METHOD_TEMPORAL_MASK or strength > 0.0):
                latents = []
                for lat in ref_latents:
                    if torch.is_tensor(lat):
                        latents.append(self.process_latent_in(lat))
                if latents:
                    out["ref_latents"] = comfy.conds.CONDList(latents)
            return out

        bound_extra_conds = types.MethodType(custom_extra_conds, m.model)
        m.add_object_patch("extra_conds", bound_extra_conds)

        diffusion_model = getattr(m.model, "diffusion_model", None)
        if hasattr(diffusion_model, "prepare_embedded_sequence"):
            bound_prepare_embedded_sequence = types.MethodType(
                patched_prepare_embedded_sequence,
                diffusion_model,
            )
            m.add_object_patch(
                "diffusion_model.prepare_embedded_sequence",
                bound_prepare_embedded_sequence,
            )

        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is not None:
            for i, block in enumerate(blocks):
                self_attn = getattr(block, "self_attn", None)
                if hasattr(self_attn, "attn_op"):
                    m.add_object_patch(
                        f"diffusion_model.blocks.{i}.self_attn.attn_op",
                        ref_kv_gated_attention_op,
                    )

        prev_wrapper = m.model_options.get("model_function_wrapper", None)

        def ref_latent_unet_wrapper(model_apply, model_kwargs):
            c_dict = model_kwargs.get("c", {})
            ref_latents = c_dict.get("ref_latents", None)
            strength = scalar_strength(c_dict.get(COSMOS_REF_STRENGTH_KEY, None), default=1.0)
            method = normalize_method(
                c_dict.get(COSMOS_REF_METHOD_KEY, None),
                default=COSMOS_REF_METHOD_KV_GATING,
            )

            if ref_latents is None:
                return call_model_apply(prev_wrapper, model_apply, model_kwargs)

            if method == COSMOS_REF_METHOD_KV_GATING and strength <= 0.0:
                return call_model_apply(prev_wrapper, model_apply, model_kwargs)

            orig_x = model_kwargs.get("input")
            if not torch.is_tensor(orig_x) or orig_x.ndim != 5:
                return call_model_apply(prev_wrapper, model_apply, model_kwargs)

            if method == COSMOS_REF_METHOD_TEMPORAL_MASK and not getattr(diffusion_model, "concat_padding_mask", False):
                raise RuntimeError("Cosmos temporal-mask reference control requires a model with concat_padding_mask=True.")

            patch_temporal = max(1, int(getattr(diffusion_model, "patch_temporal", 1)))
            patch_spatial = max(1, int(getattr(diffusion_model, "patch_spatial", 1)))

            new_x, orig_t, pad_t = align_temporal_boundary(orig_x, patch_temporal)
            target_t_aligned = new_x.shape[2]
            refs = prepare_refs_for_concat(ref_latents, new_x)
            if not refs:
                return call_model_apply(prev_wrapper, model_apply, model_kwargs)

            patched_c = c_dict.copy()
            transformer_options = dict(c_dict.get("transformer_options", {}))

            if method == COSMOS_REF_METHOD_TEMPORAL_MASK:
                target_mask = make_temporal_padding_channel(c_dict.get("padding_mask", None), orig_x)
                mask_chunks = [target_mask]

                if pad_t > 0:
                    mask_chunks.append(
                        orig_x.new_ones(
                            orig_x.shape[0],
                            1,
                            pad_t,
                            orig_x.shape[3],
                            orig_x.shape[4],
                        )
                    )

                for ref in refs:
                    ref_t = ref.shape[2]
                    ref_mask = new_x.new_full(
                        (
                            new_x.shape[0],
                            1,
                            ref_t,
                            new_x.shape[3],
                            new_x.shape[4],
                        ),
                        1.0 - strength,
                    )
                    new_x = torch.cat([new_x, ref], dim=2)
                    mask_chunks.append(ref_mask)

                patched_c["padding_mask"] = torch.cat(mask_chunks, dim=2)
                transformer_options.pop(COSMOS_REF_ATTENTION_KEY, None)

            else:
                for ref in refs:
                    new_x = torch.cat([new_x, ref], dim=2)

                total_t_aligned = ceil_div(new_x.shape[2], patch_temporal) * patch_temporal
                h_tokens = ceil_div(new_x.shape[3], patch_spatial)
                w_tokens = ceil_div(new_x.shape[4], patch_spatial)
                target_tokens = (target_t_aligned // patch_temporal) * h_tokens * w_tokens
                total_tokens = (total_t_aligned // patch_temporal) * h_tokens * w_tokens

                transformer_options[COSMOS_REF_ATTENTION_KEY] = {
                    "target_tokens": target_tokens,
                    "total_tokens": total_tokens,
                    "strength": strength,
                }

            patched_c["transformer_options"] = transformer_options
            patched_kwargs = model_kwargs.copy()
            patched_kwargs["input"] = new_x
            patched_kwargs["c"] = patched_c

            out = call_model_apply(prev_wrapper, model_apply, patched_kwargs)
            return out[:, :, :orig_t, :, :]

        m.set_model_unet_function_wrapper(ref_latent_unet_wrapper)

        return (m,)


NODE_CLASS_MAPPINGS = {
    "CosmosReferenceConditioning": CosmosReferenceConditioning,
    "ApplyCosmosReferenceModelPatch": ApplyCosmosReferenceModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CosmosReferenceConditioning": "Cosmos Reference Conditioning",
    "ApplyCosmosReferenceModelPatch": "Apply Cosmos Reference Model Patch",
}

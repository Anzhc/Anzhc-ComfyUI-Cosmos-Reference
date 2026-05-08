# Anzhc-Cosmos-Reference

ComfyUI custom nodes for Cosmos reference latent conditioning.

## Nodes

- **Cosmos Reference Conditioning**: adds a reference latent to positive and negative conditioning with strength and control method settings.
- **Apply Cosmos Reference Model Patch**: patches a Cosmos model so it can use reference latents during sampling.

## Conditioning methods

- **Temporal mask**: concatenates the reference latent as extra temporal content and uses the model padding mask as the control signal. Strength scales the reference mask, where lower values suppress the reference and higher values expose it more strongly.
- **KV gating**: makes reference tokens available through attention keys and values. Strength controls how much reference attention is used; values below `1.0` use a reduced subset, `1.0` uses the reference once, and values above `1.0` repeat reference tokens to increase influence.

Both methods require specialized training to work properly. Without that training, **KV gating** will produce visible artifacts, while **temporal mask** conditioning will have little to no effect.

## Acknowledgements

[Mirumo0u0/ComfyUI-Cosmos-Reference](https://github.com/Mirumo0u0/ComfyUI-Cosmos-Reference) for the initial Cosmos reference implementation.

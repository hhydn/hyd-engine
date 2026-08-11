import torch

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


_vae: AutoencoderKL


def on_vae_loaded(vae: AutoencoderKL) -> None:
    global _vae

    _vae = vae


def decode(latents: torch.Tensor) -> torch.Tensor:
    latents = latents / _vae.config.scaling_factor
    image: torch.Tensor = _vae.decode(latents, return_dict=False)[0]
    return (image / 2 + 0.5).clamp(0, 1)
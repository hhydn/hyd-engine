import torch

from collections.abc import Callable


_denoiser: torch.nn.Module


def on_denoiser_loaded(denoiser: torch.nn.Module) -> None:
    global _denoiser

    _denoiser = denoiser


def get_predictor(prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    return lambda latents, timestep: predict(latents, timestep, prompt_embeds, pooled_prompt_embeds)


def predict(latents: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor) -> torch.Tensor:
    height: int = latents.shape[-2] * 8
    width: int = latents.shape[-1] * 8

    time_ids: torch.Tensor = torch.tensor([[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=pooled_prompt_embeds.device).repeat(pooled_prompt_embeds.shape[0], 1)

    model_output: torch.Tensor = _denoiser(
        latents,
        timestep,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": time_ids},
        return_dict=False,
    )[0]

    return model_output
import torch

from PIL import Image
from pathlib import Path
from collections.abc import Callable

from . import text_encoder, denoiser, scheduler, sampler, vae


_latents: torch.Tensor
_output_dir: Path


def init(settings: dict[str, bool | int | str]) -> None:
    global _output_dir

    _output_dir = Path(str(settings["output_path"]))


def on_latent_changed(batch: int, width: int, height: int) -> None:
    global _latents

    _latents = torch.randn(
        batch,
        4,
        height // 8,
        width // 8,
        device="cpu",
        dtype=torch.float16,
    )


def on_generate_clicked() -> None:
    prompt_embeds, pooled_prompt_embeds = text_encoder.get_conditioning(_latents.shape[0])

    predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = denoiser.get_predictor(prompt_embeds, pooled_prompt_embeds)
    schedule: Callable[[float, float, torch.device], torch.Tensor] = scheduler.get_schedule()
    x = sampler.sample(_latents, schedule, predict)

    image = vae.decode(x)

    image = (image[0].permute(1, 2, 0) * 255).byte().cpu().numpy()
    Image.fromarray(image).save(_output_dir / "output.png")
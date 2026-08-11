import sys
import torch
import safetensors.torch

from typing import cast
from pathlib import Path
from collections.abc import Callable

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers.models.clip import CLIPTextModel, CLIPTextModelWithProjection, CLIPTextConfig, CLIPTokenizer


class Signals:
    on_denoiser_loaded: list[Callable[..., object | None]] = []
    on_vae_loaded: list[Callable[..., object | None]] = []
    on_text_encoder_loaded: list[Callable[..., object | None]] = []
    on_text_encoder_2_loaded: list[Callable[..., object | None]] = []


_CONFIG_PATHS: tuple[Path, ...] = (
    Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "denoiser.json",
    Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "vae.json",
    Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "text_encoder.json",
    Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "text_encoder_2.json",
    Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "tokenizer",
    Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "tokenizer_2",
)

_paths_by_type: dict[type[object], list[Path]] = {}


def on_files_parsed(paths_by_type: dict[type[object], list[Path]]) -> None:
    _paths_by_type.clear()
    _paths_by_type.update(paths_by_type)


def on_model_changed(path: Path) -> None:
    model_type: type[object] = next(model_type for model_type, paths in _paths_by_type.items() if path in paths)

    if model_type is UNet2DConditionModel:
        model = _load_diffusers_model(model_type, path)
        [receiver(model) for receiver in Signals.on_denoiser_loaded]

    elif model_type is AutoencoderKL:
        model = _load_diffusers_model(model_type, path)
        [receiver(model) for receiver in Signals.on_vae_loaded]

    elif model_type is CLIPTextModel:
        model, tokenizer = _load_transformers_model(model_type, path)
        [receiver(model, tokenizer) for receiver in Signals.on_text_encoder_loaded]

    elif model_type is CLIPTextModelWithProjection:
        model, tokenizer = _load_transformers_model(model_type, path)
        [receiver(model, tokenizer) for receiver in Signals.on_text_encoder_2_loaded]

    print(f"{model_type.__name__} loaded")


def _load_diffusers_model(type: type[object], path: Path) -> torch.nn.Module:
    if type is UNet2DConditionModel:
        denoiser: UNet2DConditionModel = UNet2DConditionModel.from_single_file(str(path), config=str(_CONFIG_PATHS[0]), local_files_only=True, device="cpu")
        return denoiser

    elif type is AutoencoderKL:
        vae: AutoencoderKL = AutoencoderKL.from_single_file(str(path), config=str(_CONFIG_PATHS[1]), local_files_only=True, device="cpu")
        return vae

    raise TypeError(f"Unsupported model type: {type.__name__}")


def _load_transformers_model(type: type[object], path: Path) -> tuple[torch.nn.Module, object]:
    if type is CLIPTextModel:
        config: CLIPTextConfig = CLIPTextConfig.from_json_file(str(_CONFIG_PATHS[2]))
        text_encoder: CLIPTextModel = CLIPTextModel(config)

        state_dict: dict[str, torch.Tensor] = safetensors.torch.load_file(path, device="cpu")

        state_dict = {key.removeprefix("text_model."): value for key, value in state_dict.items()}
        text_encoder.load_state_dict(state_dict)

        tokenizer: CLIPTokenizer = cast(CLIPTokenizer, CLIPTokenizer.from_pretrained(_CONFIG_PATHS[4], local_files_only=True))

        return text_encoder, tokenizer

    elif type is CLIPTextModelWithProjection:
        config: CLIPTextConfig = CLIPTextConfig.from_json_file(str(_CONFIG_PATHS[3]))
        text_encoder_2: CLIPTextModelWithProjection = CLIPTextModelWithProjection(config)

        safetensors.torch.load_model(text_encoder_2, path, device="cpu")

        tokenizer_2: CLIPTokenizer = cast(CLIPTokenizer, CLIPTokenizer.from_pretrained(_CONFIG_PATHS[5], local_files_only=True))

        return text_encoder_2, tokenizer_2

    raise TypeError(f"Unsupported model type: {type.__name__}")
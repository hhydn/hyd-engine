import sys
import torch
import safetensors
import safetensors.torch

from pathlib import Path
from collections.abc import Callable

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers.models.clip import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


SIGNALS: tuple[str, ...] = (
    "on_files_parsed",
    "on_denoiser_loaded",
    "on_vae_loaded",
    "on_text_encoder_loaded",
)
emit: Callable[..., object]

_CONFIGS: dict[str, dict[str, Path]] = {directory.name: {path.stem: path for path in directory.glob("*.json")} | {path.name: path / "tokenizer.json" for path in directory.glob("tokenizer*")} for directory in (Path(sys.argv[0]).resolve().parent / "resources").iterdir() if directory.is_dir()}

_paths_by_type: dict[type[object], list[Path]] = {}


def init(settings: dict[str, bool | int | str]) -> None:
    model_dir = Path(str(settings["model_path"]))
    model_paths: list[Path] = list(model_dir.rglob("*.safetensors"))

    _paths_by_type.clear()
    _paths_by_type.update(_get_types(model_paths))
    emit("on_files_parsed", _paths_by_type)


def on_model_changed(model_path: Path) -> None:
    model_type, model = _load_model(model_path)

    if model_type is UNet2DConditionModel:
        emit("on_denoiser_loaded", model)
    elif model_type is AutoencoderKL:
        emit("on_vae_loaded", model)
    elif model_type is CLIPTextModel:
        emit("on_text_encoder_loaded", model)
    elif model_type is CLIPTextModelWithProjection:
        emit("on_text_encoder_loaded", model)

    print(f"{model_type.__name__} loaded")


def _get_types(model_paths: list[Path]) -> dict[type[object], list[Path]]:
    paths_by_type: dict[type[object], list[Path]] = {}

    for path in model_paths:
        file: safetensors.safe_open = safetensors.safe_open(path, framework="pt")
        with file:
            keys: list[str] = list(file.keys())

            denoiser_type: type[object] | None = _get_denoiser_type(keys)
            if denoiser_type is not None:
                paths_by_type.setdefault(denoiser_type, []).append(path)

            if "encoder.conv_in.weight" in keys and "decoder.conv_in.weight" in keys:
                paths_by_type.setdefault(AutoencoderKL, []).append(path)

            text_encoder_type: type[object] | None = _get_text_encoder_type(file, keys)
            if text_encoder_type is not None:
                paths_by_type.setdefault(text_encoder_type, []).append(path)

    return paths_by_type


def _get_denoiser_type(keys: list[str]) -> type[object] | None:
    if "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias" in keys and any(key.startswith("model.diffusion_model.") for key in keys):
        return UNet2DConditionModel

    if "conv_in.weight" in keys and "conv_out.weight" in keys and "time_embedding.linear_1.weight" in keys:
        return UNet2DConditionModel

    return None


def _get_text_encoder_type(file: safetensors.safe_open, keys: list[str]) -> type[object] | None:
    token_embedding_key: str = "text_model.embeddings.token_embedding.weight"

    if token_embedding_key in keys:
        embedding_shape: list[int] = list(file.get_slice(token_embedding_key).get_shape())
        hidden_size: int = embedding_shape[1]

        if hidden_size == 768:
            return CLIPTextModel
        elif hidden_size == 1280:
            return CLIPTextModelWithProjection

    return None


def _load_model(model_path: Path) -> tuple[type[object], torch.nn.Module]:
    model_type: type[object] = next(model_type for model_type, paths in _paths_by_type.items() if model_path in paths)

    if model_type is UNet2DConditionModel:
        config_path: Path = _CONFIGS["sdxl"]["denoiser"]
        denoiser: UNet2DConditionModel = UNet2DConditionModel.from_single_file(
            str(model_path),
            config=str(config_path),
            local_files_only=True,
            device="cuda",
        )
        return UNet2DConditionModel, denoiser

    elif model_type is AutoencoderKL:
        config_path: Path = _CONFIGS["sdxl"]["vae"]
        vae: AutoencoderKL = AutoencoderKL.from_single_file(
            str(model_path),
            config=str(config_path),
            local_files_only=True,
            device="cuda",
        )
        return AutoencoderKL, vae

    elif model_type is CLIPTextModel:
        config_path: Path = _CONFIGS["sdxl"]["text_encoder"]
        config: CLIPTextConfig = CLIPTextConfig.from_json_file(str(config_path))

        with torch.device("cuda"):
            text_encoder: CLIPTextModel = CLIPTextModel(config)

        safetensors.torch.load_model(text_encoder, model_path, device="cuda")

        return CLIPTextModel, text_encoder
        

    elif model_type is CLIPTextModelWithProjection:
        config_path: Path = _CONFIGS["sdxl"]["text_encoder_2"]
        config: CLIPTextConfig = CLIPTextConfig.from_json_file(str(config_path))

        with torch.device("cuda"):
            text_encoder_2: CLIPTextModelWithProjection = CLIPTextModelWithProjection(config)
            safetensors.torch.load_model(text_encoder_2, model_path, device="cuda")

        return CLIPTextModelWithProjection, text_encoder_2

    raise TypeError(f"Unsupported model type: {model_type.__name__}")
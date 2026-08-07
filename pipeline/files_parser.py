import transformers
from safetensors import safe_open

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from pathlib import Path
from collections.abc import Callable


emit: Callable[..., object]

_sdxl_file_dict: dict[
    type[UNet2DConditionModel]
    | type[AutoencoderKL]
    | type[transformers.CLIPTextModel]
    | type[transformers.CLIPTextModelWithProjection],
    list[Path],
] = {}


def set_emit(function: Callable[..., None]) -> None:
    global emit
    emit = function


def init(settings: dict[str, bool | int | str]) -> None:
    model_path = Path(str(settings["model_path"]))
    safetensors_paths: list[Path] = list(model_path.rglob("*.safetensors"))
    _parse_files(safetensors_paths)


def _parse_files(safetensors_paths: list[Path]) -> None:
    for path in safetensors_paths:
        with safe_open(path, framework="pt") as file:
            keys: list[str] = list(file.keys())

        if "conv_in.weight" in keys and "time_embedding.linear_1.weight" in keys:
            _sdxl_file_dict.setdefault(UNet2DConditionModel, []).append(path)

        elif "encoder.conv_in.weight" in keys and "decoder.conv_in.weight" in keys:
            _sdxl_file_dict.setdefault(AutoencoderKL, []).append(path)

        elif "text_projection.weight" in keys:
            _sdxl_file_dict.setdefault(transformers.CLIPTextModelWithProjection, []).append(path)

        elif "text_model.embeddings.token_embedding.weight" in keys:
            _sdxl_file_dict.setdefault(transformers.CLIPTextModel, []).append(path)

    emit("files_parsed", _sdxl_file_dict)

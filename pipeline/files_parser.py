from pathlib import Path
from collections.abc import Callable
from safetensors import safe_open

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection


emit: Callable[..., object]

_sdxl_file_dict: dict[
    type[UNet2DConditionModel] | type[AutoencoderKL] | type[CLIPTextModel] | type[CLIPTextModelWithProjection],
    list[Path],
] = {
    UNet2DConditionModel: [],
    AutoencoderKL: [],
    CLIPTextModel: [],
    CLIPTextModelWithProjection: [],
}


def init(settings: dict[str, bool | int | str]) -> None:
    model_path = Path(str(settings["model_path"]))
    safetensors_paths: list[Path] = list(model_path.rglob("*.safetensors"))
    _parse_files(safetensors_paths)


def _parse_files(safetensors_paths: list[Path]) -> None:
    for paths in _sdxl_file_dict.values():
        paths.clear()

    for path in safetensors_paths:
        with safe_open(path, framework="pt") as file:
            keys: list[str] = list(file.keys())

        is_sdxl_checkpoint: bool = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias" in keys and any(key.startswith("model.diffusion_model.") for key in keys)

        if is_sdxl_checkpoint:
            _sdxl_file_dict[UNet2DConditionModel].append(path)

        if "conv_in.weight" in keys and "conv_out.weight" in keys and "time_embedding.linear_1.weight" in keys:
            _sdxl_file_dict[UNet2DConditionModel].append(path)

        if "encoder.conv_in.weight" in keys and "decoder.conv_in.weight" in keys:
            _sdxl_file_dict[AutoencoderKL].append(path)

        if "text_projection.weight" in keys:
            _sdxl_file_dict[CLIPTextModelWithProjection].append(path)

        if "text_model.embeddings.token_embedding.weight" in keys:
            _sdxl_file_dict[CLIPTextModel].append(path)

    emit("files_parsed", _sdxl_file_dict)

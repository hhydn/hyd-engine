import torch
import safetensors

from pathlib import Path
from collections.abc import Callable

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers.models.clip import CLIPTextModel, CLIPTextModelWithProjection, CLIPTextConfig


class Signals:
    on_models_parsed: list[Callable[..., object | None]] = []


def init(settings: dict[str, bool | int | str]) -> None:
    parsed_models: dict[type[object], tuple[list[Path], list[torch.nn.Module]]] = {}

    model_dir: Path = Path(str(settings["model_path"]))
    model_paths: list[Path] = list(model_dir.rglob("*.safetensors"))
    paths_by_type: dict[type[object], list[Path]] = _get_types(model_paths)

    for model_type, paths in paths_by_type.items():
        meta_models: list[torch.nn.Module] = [Meta(model_type, path).model for path in paths]
        parsed_models[model_type] = (paths, meta_models)

    [receiver(parsed_models) for receiver in Signals.on_models_parsed]


def _get_types(model_paths: list[Path]) -> dict[type[object], list[Path]]:
    paths_by_type: dict[type[object], list[Path]] = {}

    for path in model_paths:
        file: safetensors.safe_open = safetensors.safe_open(path, framework="pt")
        with file:
            keys: list[str] = list(file.keys())

            diffusers_type: type[object] | None = _get_diffusers_type(keys)
            if diffusers_type is not None:
                paths_by_type.setdefault(diffusers_type, []).append(path)

            transformers_type: type[object] | None = _get_transformers_type(file, keys)
            if transformers_type is not None:
                paths_by_type.setdefault(transformers_type, []).append(path)

    return paths_by_type


def _get_diffusers_type(keys: list[str]) -> type[object] | None:
    if "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias" in keys and any(key.startswith("model.diffusion_model.") for key in keys):
        return UNet2DConditionModel

    if "conv_in.weight" in keys and "conv_out.weight" in keys and "time_embedding.linear_1.weight" in keys:
        return UNet2DConditionModel

    if "encoder.conv_in.weight" in keys and "decoder.conv_in.weight" in keys:
        return AutoencoderKL

    return None


def _get_transformers_type(file: safetensors.safe_open, keys: list[str]) -> type[object] | None:
    token_embedding_key: str = "text_model.embeddings.token_embedding.weight"

    if token_embedding_key in keys:
        embedding_shape: list[int] = list(file.get_slice(token_embedding_key).get_shape())
        hidden_size: int = embedding_shape[1]

        if hidden_size == 768:
            return CLIPTextModel
        elif hidden_size == 1280:
            return CLIPTextModelWithProjection

    return None


class Meta:
    import sys
    import json
    import safetensors.torch


    model: torch.nn.Module

    _CONFIG_PATHS: tuple[Path, ...] = (
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "denoiser.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "vae.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "text_encoder.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "text_encoder_2.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "tokenizer",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "tokenizer_2",
    )


    def __init__(self, type: type[object], path: Path) -> None:
        with torch.device("meta"):
            if type is UNet2DConditionModel:
                diffusers_config: dict[str, object] = Meta.json.loads(Meta._CONFIG_PATHS[0].read_text(encoding="utf-8"))
                self.model = UNet2DConditionModel.from_config(diffusers_config)
            elif type is AutoencoderKL:
                diffusers_config: dict[str, object] = Meta.json.loads(Meta._CONFIG_PATHS[1].read_text(encoding="utf-8"))
                self.model = AutoencoderKL.from_config(diffusers_config)
            elif type is CLIPTextModel:
                transformers_config: CLIPTextConfig = CLIPTextConfig.from_json_file(str(Meta._CONFIG_PATHS[2]))
                self.model = CLIPTextModel(transformers_config)
            elif type is CLIPTextModelWithProjection:
                transformers_config: CLIPTextConfig = CLIPTextConfig.from_json_file(str(Meta._CONFIG_PATHS[3]))
                self.model = CLIPTextModelWithProjection(transformers_config)
            else:
                raise TypeError(f"Unsupported model type: {type.__name__}")

        parameter_dtypes, buffer_dtypes = self.get_dtypes(path)
        for name, dtype in parameter_dtypes.items():
            parameter: torch.nn.Parameter = self.model.get_parameter(name)
            parameter.data = parameter.data.to(dtype=dtype)

        for name, dtype in buffer_dtypes.items():
            buffer: torch.Tensor = self.model.get_buffer(name)
            buffer.data = buffer.data.to(dtype=dtype)


    def get_dtypes(self, path: Path) -> tuple[dict[str, torch.dtype], dict[str, torch.dtype]]:
        parameter_names: set[str] = {name for name, _ in self.model.named_parameters()}
        buffer_names: set[str] = {name for name, _ in self.model.named_buffers()}

        parameter_dtypes: dict[str, torch.dtype] = {}
        buffer_dtypes: dict[str, torch.dtype] = {}

        file: safetensors.safe_open = safetensors.safe_open(path, framework="pt")
        with file:
            keys: list[str] = list(file.keys())
            for name in keys:
                dtype: torch.dtype = Meta.safetensors.torch._getdtype(file.get_slice(name).get_dtype())
                if name in parameter_names:
                    parameter_dtypes[name] = dtype
                elif name in buffer_names:
                    buffer_dtypes[name] = dtype

        return parameter_dtypes, buffer_dtypes
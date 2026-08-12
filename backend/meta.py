import torch
import safetensors

from pathlib import Path
from collections.abc import Callable

from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from transformers.modeling_utils import PreTrainedModel
from transformers.models.clip import CLIPTextModel, CLIPTextModelWithProjection, CLIPTextConfig


class Signals:
    on_meta_parsed: list[Callable[..., object | None]] = []


def init(settings: dict[str, bool | int | str]) -> None:
    parsed_models: dict[type[ModelMixin] | type[PreTrainedModel], tuple[list[Path], list[torch.nn.Module]]] = {}

    model_dir: Path = Path(str(settings["model_path"]))
    model_paths: list[Path] = list(model_dir.rglob("*.safetensors"))
    paths_by_type: dict[type[ModelMixin] | type[PreTrainedModel], list[Path]] = _get_types(model_paths)

    for model_type, paths in paths_by_type.items():
        meta_models: list[torch.nn.Module] = [MetaModel(settings, model_type, path).model for path in paths]
        parsed_models[model_type] = (paths, meta_models)

    [receiver(parsed_models) for receiver in Signals.on_meta_parsed]


def _get_types(model_paths: list[Path]) -> dict[type[ModelMixin] | type[PreTrainedModel], list[Path]]:
    paths_by_type: dict[type[ModelMixin] | type[PreTrainedModel], list[Path]] = {}

    for path in model_paths:
        file: safetensors.safe_open = safetensors.safe_open(path, framework="pt")
        with file:
            keys: list[str] = list(file.keys())
            model_type: type[ModelMixin] | type[PreTrainedModel] | None = _get_model_type(file, keys)

            if model_type is not None:
                paths_by_type.setdefault(model_type, []).append(path)

    return paths_by_type


def _get_model_type(file: safetensors.safe_open, keys: list[str]) -> type[ModelMixin] | type[PreTrainedModel] | None:
    if "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias" in keys and any(key.startswith("model.diffusion_model.") for key in keys):
        return UNet2DConditionModel

    if "conv_in.weight" in keys and "conv_out.weight" in keys and "time_embedding.linear_1.weight" in keys:
        return UNet2DConditionModel

    if "encoder.conv_in.weight" in keys and "decoder.conv_in.weight" in keys:
        return AutoencoderKL

    token_embedding_key: str = "text_model.embeddings.token_embedding.weight"
    if token_embedding_key in keys:
        embedding_shape: list[int] = list(file.get_slice(token_embedding_key).get_shape())
        hidden_size: int = embedding_shape[1]

        if hidden_size == 768:
            return CLIPTextModel
        elif hidden_size == 1280:
            return CLIPTextModelWithProjection

    return None


class MetaDtype:
    import safetensors.torch

    model: torch.nn.Module


    def __init__(self, settings: dict[str, bool | int | str], model_type: type[ModelMixin] | type[PreTrainedModel], path: Path) -> None:
        dtypes_by_model_type: dict[type[ModelMixin] | type[PreTrainedModel], torch.dtype | None] = {
            UNet2DConditionModel: getattr(torch, str(settings["denoiser_dtype"]), None),
            AutoencoderKL: getattr(torch, str(settings["vae_dtype"]), None),
            CLIPTextModel: getattr(torch, str(settings["text_encoder_dtype"]), None),
            CLIPTextModelWithProjection: getattr(torch, str(settings["text_encoder_dtype"]), None),
        }

        dtype: torch.dtype | None = dtypes_by_model_type.get(model_type)
        if dtype is not None:
            self._set_dtype_override(dtype)
        else:
            self._set_trained_dtype(path)


    def _set_trained_dtype(self, path: Path) -> None:
        file: safetensors.safe_open = safetensors.safe_open(path, framework="pt")
        with file:
            keys: list[str] = list(file.keys())

            for name, parameter in self.model.named_parameters():
                if name in keys:
                    dtype: torch.dtype = MetaDtype.safetensors.torch._getdtype(file.get_slice(name).get_dtype())
                    parameter.data = parameter.data.to(dtype=dtype)

            for name, buffer in self.model.named_buffers():
                if name in keys:
                    dtype: torch.dtype = MetaDtype.safetensors.torch._getdtype(file.get_slice(name).get_dtype())
                    buffer.data = buffer.data.to(dtype=dtype)


    def _set_dtype_override(self, dtype: torch.dtype) -> None:
        keep_in_fp32_modules: list[str] = list(getattr(self.model, "_keep_in_fp32_modules", []) or [])

        for name, parameter in self.model.named_parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(torch.float32 if any(module in name.split(".") for module in keep_in_fp32_modules) else dtype)

        for name, buffer in self.model.named_buffers():
            if buffer.is_floating_point():
                buffer.data = buffer.data.to(torch.float32 if any(module in name.split(".") for module in keep_in_fp32_modules) else dtype)


class MetaModel(MetaDtype):
    import sys
    import json


    _CONFIG_PATHS: tuple[Path, ...] = (
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "denoiser.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "vae.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "text_encoder.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "text_encoder_2.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "tokenizer",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "tokenizer_2",
    )


    def __init__(self, settings: dict[str, bool | int | str], model_type: type[ModelMixin] | type[PreTrainedModel], path: Path) -> None:
        with torch.device("meta"):
            if model_type is UNet2DConditionModel:
                diffusers_config: dict[str, object] = MetaModel.json.loads(MetaModel._CONFIG_PATHS[0].read_text(encoding="utf-8"))
                self.model = UNet2DConditionModel.from_config(diffusers_config)

            elif model_type is AutoencoderKL:
                diffusers_config: dict[str, object] = MetaModel.json.loads(MetaModel._CONFIG_PATHS[1].read_text(encoding="utf-8"))
                self.model = AutoencoderKL.from_config(diffusers_config)

            elif model_type is CLIPTextModel:
                transformers_config: CLIPTextConfig = CLIPTextConfig.from_json_file(str(MetaModel._CONFIG_PATHS[2]))
                self.model = CLIPTextModel(transformers_config)

            elif model_type is CLIPTextModelWithProjection:
                transformers_config: CLIPTextConfig = CLIPTextConfig.from_json_file(str(MetaModel._CONFIG_PATHS[3]))
                self.model = CLIPTextModelWithProjection(transformers_config)
    
            else:
                raise TypeError(f"Unsupported model type: {model_type.__name__}")

        super().__init__(settings, model_type, path)
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
    model_dir: Path = Path(str(settings["model_path"]))
    paths: list[Path] = list(model_dir.rglob("*.safetensors"))

    model_paths: list[Path] = []
    meta_models: list[ModelMixin | PreTrainedModel] = []

    for path in paths:
        model_type: type[ModelMixin] | type[PreTrainedModel] | None = _get_model_type(path)
        if model_type is not None:
            model_paths.append(path)
            meta_models.append(MetaModel(settings, model_type, path).model)

    models: tuple[list[Path], list[ModelMixin | PreTrainedModel]] = (model_paths, meta_models)
    [receiver(models) for receiver in Signals.on_meta_parsed]


def _get_model_type(path: Path) -> type[ModelMixin] | type[PreTrainedModel] | None:
    file: safetensors.safe_open = safetensors.safe_open(path, framework="pt")
    with file:
        keys: list[str] = list(file.keys())

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
    import torch
    import safetensors.torch


    model: ModelMixin | PreTrainedModel

    _DTYPE_SETTINGS: dict[type[ModelMixin] | type[PreTrainedModel], str] = {
        UNet2DConditionModel: "denoiser_dtype",
        AutoencoderKL: "vae_dtype",
        CLIPTextModel: "text_encoder_dtype",
        CLIPTextModelWithProjection: "text_encoder_dtype",
    }


    def __init__(self, settings: dict[str, bool | int | str], model_type: type[ModelMixin] | type[PreTrainedModel], path: Path) -> None:
        dtype: MetaDtype.torch.dtype | None = getattr(self.torch, str(settings[self._DTYPE_SETTINGS[model_type]]), None)

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
                    dtype: MetaDtype.torch.dtype = self.safetensors.torch._getdtype(file.get_slice(name).get_dtype())
                    parameter.data = parameter.data.to(dtype=dtype)

            for name, buffer in self.model.named_buffers():
                if name in keys:
                    dtype: MetaDtype.torch.dtype = self.safetensors.torch._getdtype(file.get_slice(name).get_dtype())
                    buffer.data = buffer.data.to(dtype=dtype)


    def _set_dtype_override(self, dtype: torch.dtype) -> None:
        keep_in_fp32_modules: list[str] = list(getattr(self.model, "_keep_in_fp32_modules", []) or [])

        for name, parameter in self.model.named_parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(self.torch.float32 if any(module in name.split(".") for module in keep_in_fp32_modules) else dtype)

        for name, buffer in self.model.named_buffers():
            if buffer.is_floating_point():
                buffer.data = buffer.data.to(self.torch.float32 if any(module in name.split(".") for module in keep_in_fp32_modules) else dtype)


class MetaModel(MetaDtype):
    import sys
    import json


    _CONFIG_PATHS: tuple[Path, ...] = (
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "denoiser.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "vae.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "text_encoder.json",
        Path(sys.argv[0]).resolve().parent / "resources" / "sdxl" / "text_encoder_2.json",
    )


    def __init__(self, settings: dict[str, bool | int | str], model_type: type[ModelMixin] | type[PreTrainedModel], path: Path) -> None:
        with self.torch.device("meta"):
            if model_type is UNet2DConditionModel:
                diffusers_config: dict[str, object] = self.json.loads(self._CONFIG_PATHS[0].read_text(encoding="utf-8"))
                self.model = UNet2DConditionModel.from_config(diffusers_config)

            elif model_type is AutoencoderKL:
                diffusers_config: dict[str, object] = self.json.loads(self._CONFIG_PATHS[1].read_text(encoding="utf-8"))
                self.model = AutoencoderKL.from_config(diffusers_config)

            elif model_type is CLIPTextModel:
                transformers_config: CLIPTextConfig = CLIPTextConfig.from_json_file(str(self._CONFIG_PATHS[2]))
                self.model = CLIPTextModel(transformers_config)

            elif model_type is CLIPTextModelWithProjection:
                transformers_config: CLIPTextConfig = CLIPTextConfig.from_json_file(str(self._CONFIG_PATHS[3]))
                self.model = CLIPTextModelWithProjection(transformers_config)
    
            else:
                raise TypeError(f"Unsupported model type: {model_type.__name__}")

        super().__init__(settings, model_type, path)
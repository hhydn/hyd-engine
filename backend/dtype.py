import torch

from enum import Enum


_fp16_accumulation: bool | None = None


class ModelType(Enum):
    UNET = 1
    VAE = 2
    TEXT_ENCODER = 3


_registered_models: dict[ModelType, torch.nn.Module] = {}
_dtype_overrides: dict[ModelType, torch.dtype] = {}


def init(settings: dict[str, bool | int | str]) -> None:
    global _fp16_accumulation

    _fp16_accumulation = settings["fp16_accumulation"] is True


def register_unet(model: torch.nn.Module, dtype_override: torch.dtype | None = None) -> None:
    _registered_models[ModelType.UNET] = model

    if dtype_override is not None:
        _dtype_overrides[ModelType.UNET] = dtype_override
        _set_dtype(ModelType.UNET, dtype_override)

    _set_fp16_accumulation()


def register_vae(model: torch.nn.Module, dtype_override: torch.dtype | None = None) -> None:
    _registered_models[ModelType.VAE] = model

    if dtype_override is not None:
        _dtype_overrides[ModelType.VAE] = dtype_override
        _set_dtype(ModelType.VAE, dtype_override)

    _set_fp16_accumulation()


def register_text_encoder(model: torch.nn.Module, dtype_override: torch.dtype | None = None) -> None:
    _registered_models[ModelType.TEXT_ENCODER] = model

    if dtype_override is not None:
        _dtype_overrides[ModelType.TEXT_ENCODER] = dtype_override
        _set_dtype(ModelType.TEXT_ENCODER, dtype_override)

    _set_fp16_accumulation()


def get_module_dtypes(module: torch.nn.Module) -> tuple[dict[str, torch.dtype], dict[str, torch.dtype]]:
    param_dtypes: dict[str, torch.dtype] = {name: parameter.dtype for name, parameter in module.named_parameters()}
    buffer_dtypes: dict[str, torch.dtype] = {name: buffer.dtype for name, buffer in module.named_buffers()}
    return param_dtypes, buffer_dtypes


def _set_dtype(model_type: ModelType, dtype_override: torch.dtype) -> None:
    for model in _registered_models.values():
        if _registered_models[model] == model_type and model.torch.dtype != _dtype_overrides[model_type]:
            model: torch.nn.Module = _registered_models[model_type]
            model.to(dtype=dtype_override)


def _set_fp16_accumulation() -> None:
    if _fp16_accumulation and not torch.backends.cuda.matmul.allow_fp16_accumulation and torch.cuda.get_device_capability()[0] >= 7:
        contains_fp16_gemm: bool = any(isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.float16 for model in _registered_models.values() for module in model.modules())

        if contains_fp16_gemm:
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            print("FP16 Accumulation: Enabled")
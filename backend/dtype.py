import torch


_denoiser_dtype: torch.dtype | None = None
_vae_dtype: torch.dtype | None = None
_text_encoder_dtype: torch.dtype | None = None
_fp16_accumulation: bool | None = None


def init(settings: dict[str, bool | int | str]) -> None:
    global _denoiser_dtype, _vae_dtype, _text_encoder_dtype, _fp16_accumulation

    _denoiser_dtype = getattr(torch, str(settings["denoiser_dtype"]), None)
    _vae_dtype = getattr(torch, str(settings["vae_dtype"]), None)
    _text_encoder_dtype = getattr(torch, str(settings["text_encoder_dtype"]), None)
    _fp16_accumulation = settings["fp16_accumulation"] is True


def on_models_parsed(model_data: dict[type[object], tuple[object, list[torch.nn.Module]]]) -> None:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection

    for model_type, (_, meta_models) in model_data.items():
        for meta_model in meta_models:
            if model_type is UNet2DConditionModel and _denoiser_dtype is not None:
                _set_dtype(meta_model, _denoiser_dtype)
            elif model_type is AutoencoderKL and _vae_dtype is not None:
                _set_dtype(meta_model, _vae_dtype)
            elif model_type in (CLIPTextModel, CLIPTextModelWithProjection) and _text_encoder_dtype is not None:
                _set_dtype(meta_model, _text_encoder_dtype)

            _set_fp16_accumulation(meta_model)


def get_module_dtypes(module: torch.nn.Module) -> tuple[dict[str, torch.dtype], dict[str, torch.dtype]]:
    param_dtypes: dict[str, torch.dtype] = {name: parameter.dtype for name, parameter in module.named_parameters()}
    buffer_dtypes: dict[str, torch.dtype] = {name: buffer.dtype for name, buffer in module.named_buffers()}
    return param_dtypes, buffer_dtypes


def _set_dtype(model: torch.nn.Module, dtype_override: torch.dtype) -> None:
    keep_in_fp32_modules: list[str] = list(getattr(model, "_keep_in_fp32_modules", []) or [])

    for name, parameter in model.named_parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(torch.float32 if any(module in name.split(".") for module in keep_in_fp32_modules) else dtype_override)

    for name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(torch.float32 if any(module in name.split(".") for module in keep_in_fp32_modules) else dtype_override)


def _set_fp16_accumulation(model: torch.nn.Module) -> None:
    if _fp16_accumulation and not torch.backends.cuda.matmul.allow_fp16_accumulation and torch.cuda.get_device_capability()[0] >= 7:
        contains_fp16_gemm: bool = any(isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.float16 for module in model.modules())

        if contains_fp16_gemm:
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            print("FP16 Accumulation: Enabled")
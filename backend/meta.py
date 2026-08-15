from __future__ import annotations
import torch
import safetensors

from pathlib import Path
from collections.abc import Callable


class Signals:
    on_meta_parsed: list[Callable[..., object | None]] = []


_model_paths: list[Path] = []
_dtypes_by_model: dict[type[Meta], torch.dtype | None] = {}


def init(settings: dict[str, bool | int | str]) -> None:
    model_dir: Path = Path(str(settings["model_path"]))
    _model_paths.clear()
    _model_paths.extend(model_dir.rglob("*.safetensors"))

    _dtypes_by_model.clear()
    _dtypes_by_model.update({
        MetaUNet2DConditionModel: getattr(torch, str(settings["denoiser_dtype"]), None),
        MetaAutoencoderKL: getattr(torch, str(settings["vae_dtype"]), None),
        MetaCLIPTextModel: getattr(torch, str(settings["text_encoder_dtype"]), None),
        MetaCLIPTextModelWithProjection: getattr(torch, str(settings["text_encoder_dtype"]), None),
    })


def ready() -> None:
    models_by_path: dict[Path, torch.nn.Module] = {}
    for path in _model_paths:
        file: safetensors.safe_open = safetensors.safe_open(path, framework="pt")
        with file:

            meta_type: type[Meta] | None = _get_model_type(file)
            if meta_type is not None:
                models_by_path[path] = meta_type(file, _dtypes_by_model[meta_type]).model

    for receiver in Signals.on_meta_parsed:
        receiver(models_by_path)


def _get_model_type(file: safetensors.safe_open) -> type[Meta] | None:
    keys: list[str] = list(file.keys())

    if (
        "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias" in keys
        and any(key.startswith("model.diffusion_model.") for key in keys)
    ) or (
        "conv_in.weight" in keys
        and "conv_out.weight" in keys
        and "time_embedding.linear_1.weight" in keys
    ):
        return MetaUNet2DConditionModel

    if "encoder.conv_in.weight" in keys and "decoder.conv_in.weight" in keys:
        return MetaAutoencoderKL

    token_embed: str = "text_model.embeddings.token_embedding.weight"
    if token_embed in keys:
        embed_shape: list[int] = list(file.get_slice(token_embed).get_shape())
        size: int = embed_shape[1]
        if size == 768:
            return MetaCLIPTextModel
        elif size == 1280:
            return MetaCLIPTextModelWithProjection

    return None


class Meta:
    import safetensors.torch

    from diffusers.loaders import single_file_utils
    from transformers.models.clip import CLIPTextConfig


    model: torch.nn.Module


    def __init__(self, file: safetensors.safe_open, _dtype: torch.dtype | None = None) -> None:
        self.ckpt_state_dict: dict[str, torch.Tensor] = {}
        keys: list[str] = list(file.keys())

        for key in keys:
            tensors = file.get_slice(key)
            shape: tuple[int, ...] = tuple(tensors.get_shape())
            dtype: torch.dtype = self.safetensors.torch._getdtype(tensors.get_dtype())
            self.ckpt_state_dict[key] = torch.empty(shape, dtype=dtype, device="meta")


    def set_dtype(self, state_dict: dict[str, torch.Tensor], dtype: torch.dtype | None, model_type: type[torch.nn.Module]) -> None:
        if dtype is None:
            return

        keep_in_fp32_modules: list[str] = getattr(model_type, "_keep_in_fp32_modules", None) or []
        for key, tensor in state_dict.items():
            if tensor.is_floating_point():
                if any(module in key for module in keep_in_fp32_modules):
                    state_dict[key] = tensor.to(dtype=torch.float32)
                else:
                    state_dict[key] = tensor.to(dtype=dtype)


class MetaUNet2DConditionModel(Meta):
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


    def __init__(self, file: safetensors.safe_open, dtype: torch.dtype | None = None) -> None:
        super().__init__(file)

        convert_state_dict = self.single_file_utils.convert_ldm_unet_checkpoint(
            self.ckpt_state_dict,
            Configs.SDXL_UNET,
        )
        self.set_dtype(convert_state_dict, dtype, self.UNet2DConditionModel)
        with torch.device("meta"):
            self.model = self.UNet2DConditionModel.from_config(Configs.SDXL_UNET)
        self.model.load_state_dict(convert_state_dict, strict=True, assign=True)


class MetaAutoencoderKL(Meta):
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


    def __init__(self, file: safetensors.safe_open, dtype: torch.dtype | None = None) -> None:
        super().__init__(file)

        convert_state_dict = self.single_file_utils.convert_ldm_vae_checkpoint(
            self.ckpt_state_dict,
            Configs.SDXL_VAE,
        )
        self.set_dtype(convert_state_dict, dtype, self.AutoencoderKL)
        with torch.device("meta"):
            self.model = self.AutoencoderKL.from_config(Configs.SDXL_VAE)
        self.model.load_state_dict(convert_state_dict, strict=True, assign=True)


class MetaCLIPTextModel(Meta):
    from transformers.models.clip import CLIPTextModel


    def __init__(self, file: safetensors.safe_open, dtype: torch.dtype | None = None) -> None:
        super().__init__(file)

        config = self.CLIPTextConfig.from_dict(Configs.SDXL_CLIP_L)
        with torch.device("meta"):
            self.model = self.CLIPTextModel(config)
        state_dict: dict[str, torch.Tensor] = {key.removeprefix("text_model."): tensor for key, tensor in self.ckpt_state_dict.items()}

        self.set_dtype(state_dict, dtype, self.CLIPTextModel)
        self.model.load_state_dict(state_dict, strict=True, assign=True)


class MetaCLIPTextModelWithProjection(Meta):
    from transformers.models.clip import CLIPTextModelWithProjection


    def __init__(self, file: safetensors.safe_open, dtype: torch.dtype | None = None) -> None:
        super().__init__(file)

        config = self.CLIPTextConfig.from_dict(Configs.SDXL_CLIP_G)
        with torch.device("meta"):
            self.model = self.CLIPTextModelWithProjection(config)

        self.set_dtype(self.ckpt_state_dict, dtype, self.CLIPTextModelWithProjection)
        self.model.load_state_dict(self.ckpt_state_dict, strict=True, assign=True)


class Configs:
    SDXL_UNET: dict[str, object] = {
        "act_fn": "silu",
        "addition_embed_type": "text_time",
        "addition_embed_type_num_heads": 64,
        "addition_time_embed_dim": 256,
        "attention_head_dim": [5, 10, 20],
        "block_out_channels": [320, 640, 1280],
        "center_input_sample": False,
        "class_embed_type": None,
        "class_embeddings_concat": False,
        "conv_in_kernel": 3,
        "conv_out_kernel": 3,
        "cross_attention_dim": 2048,
        "cross_attention_norm": None,
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        "downsample_padding": 1,
        "dual_cross_attention": False,
        "encoder_hid_dim": None,
        "encoder_hid_dim_type": None,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "in_channels": 4,
        "layers_per_block": 2,
        "mid_block_only_cross_attention": None,
        "mid_block_scale_factor": 1,
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "norm_eps": 1e-05,
        "norm_num_groups": 32,
        "num_attention_heads": None,
        "num_class_embeds": None,
        "only_cross_attention": False,
        "out_channels": 4,
        "projection_class_embeddings_input_dim": 2816,
        "resnet_out_scale_factor": 1.0,
        "resnet_skip_time_act": False,
        "resnet_time_scale_shift": "default",
        "sample_size": 128,
        "time_cond_proj_dim": None,
        "time_embedding_act_fn": None,
        "time_embedding_dim": None,
        "time_embedding_type": "positional",
        "timestep_post_act": None,
        "transformer_layers_per_block": [1, 2, 10],
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        "upcast_attention": None,
        "use_linear_projection": True,
    }


    SDXL_VAE: dict[str, object] = {
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        "force_upcast": True,
        "in_channels": 3,
        "latent_channels": 4,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 1024,
        "scaling_factor": 0.13025,
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
    }


    SDXL_CLIP_L: dict[str, object] = {
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "dropout": 0.0,
        "eos_token_id": 2,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 77,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 1,
        "projection_dim": 768,
        "vocab_size": 49408,
    }


    SDXL_CLIP_G: dict[str, object] = {
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "dropout": 0.0,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 5120,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 77,
        "num_attention_heads": 20,
        "num_hidden_layers": 32,
        "pad_token_id": 1,
        "projection_dim": 1280,
        "vocab_size": 49408,
    }
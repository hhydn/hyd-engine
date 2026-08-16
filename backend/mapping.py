from __future__ import annotations
import torch
import safetensors
import safetensors.torch

from pathlib import Path

_model_paths: list[Path] = []


def init(settings: dict[str, bool | int | str]) -> None:
    model_dir: Path = Path(str(settings["model_path"]))
    _model_paths.clear()
    _model_paths.extend(model_dir.rglob("*.safetensors"))

def ready() -> None:
    for path in _model_paths:
        file: safetensors.safe_open = safetensors.safe_open(path, framework="pt")
        with file:
            orig_state_dict: dict[str, torch.Tensor] = get_state_dict(file)
            model_type: type[object] | None = _get_model_type(orig_state_dict)

            if model_type is AutoencoderKL:
                converted_state_dict: dict[str, torch.Tensor] = AutoencoderKL(orig_state_dict).state_dict

            elif model_type is UNet2DConditionModel:
                converted_state_dict: dict[str, torch.Tensor] = UNet2DConditionModel(orig_state_dict).state_dict

            elif model_type is CLIPTextModel:
                converted_state_dict: dict[str, torch.Tensor] = CLIPTextModel(orig_state_dict).state_dict

            elif model_type is CLIPTextModelWithProjection:
                converted_state_dict: dict[str, torch.Tensor] = CLIPTextModelWithProjection(orig_state_dict).state_dict


def get_state_dict(file: safetensors.safe_open) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    keys: list[str] = list(file.keys())
    for key in keys:
        tensors = file.get_slice(key)
        shape: tuple[int, ...] = tuple(tensors.get_shape())
        dtype: torch.dtype = safetensors.torch._getdtype(tensors.get_dtype())
        state_dict[key] = torch.empty(shape, dtype=dtype, device="meta")
    return state_dict

def _get_model_type(state_dict: dict[str, torch.Tensor]) -> type[object] | None:
    keys: list[str] = list(state_dict.keys())
    for key in keys:
        if key.endswith("token_embedding.weight"):
            prefix: str = key.removesuffix("token_embedding.weight")
            if prefix.endswith("text_model.embeddings."):
                prefix = prefix.removesuffix("text_model.embeddings.")
            if all(name.startswith(prefix) for name in keys):
                shape: tuple[int, ...] = tuple(state_dict[key].shape)
                if shape[1] == 768:
                    return CLIPTextModel
                elif shape[1] == 1280:
                    return CLIPTextModelWithProjection
    if "encoder.conv_in.weight" in keys and "decoder.conv_in.weight" in keys:
        return AutoencoderKL
    elif "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias" in keys:
        return UNet2DConditionModel
    return None


class UNet2DConditionModel:
    _DIRECT_MAPPING: dict[str, str] = {
        "time_embed.0.weight": "time_embedding.linear_1.weight",
        "time_embed.0.bias": "time_embedding.linear_1.bias",
        "time_embed.2.weight": "time_embedding.linear_2.weight",
        "time_embed.2.bias": "time_embedding.linear_2.bias",
        "input_blocks.0.0.weight": "conv_in.weight",
        "input_blocks.0.0.bias": "conv_in.bias",
        "out.0.weight": "conv_norm_out.weight",
        "out.0.bias": "conv_norm_out.bias",
        "out.2.weight": "conv_out.weight",
        "out.2.bias": "conv_out.bias",
        "label_emb.0.0.weight": "add_embedding.linear_1.weight",
        "label_emb.0.0.bias": "add_embedding.linear_1.bias",
        "label_emb.0.2.weight": "add_embedding.linear_2.weight",
        "label_emb.0.2.bias": "add_embedding.linear_2.bias",
    }
    _RESNET_MAPPING: tuple[tuple[str, str], ...] = (
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"),
        ("skip_connection", "conv_shortcut"),
    )
    state_dict: dict[str, torch.Tensor]


    def __init__(self, orig_state_dict: dict[str, torch.Tensor]) -> None:
        self.state_dict = {}
        self._set_direct_mapping(orig_state_dict)

        for key, tensor in orig_state_dict.items():
            parts: list[str] = key.split(".")
            if parts[0] == "input_blocks" and len(parts) >= 4:
                self._set_down_block(parts, tensor)
            elif parts[0] == "middle_block" and len(parts) >= 3:
                self._set_mid_block(parts, tensor)
            elif parts[0] == "output_blocks" and len(parts) >= 4:
                self._set_up_block(parts, tensor)

    def _set_direct_mapping(self, orig_state_dict: dict[str, torch.Tensor]) -> None:
        keys: list[str] = list(orig_state_dict.keys())
        for key in keys:
            if key.startswith("model.diffusion_model."):
                orig_state_dict[key.removeprefix("model.diffusion_model.")] = orig_state_dict.pop(key)

        for old_key, new_key in self._DIRECT_MAPPING.items():
            if old_key in orig_state_dict:
                self.state_dict[new_key] = orig_state_dict[old_key]

    def _set_down_block(self, parts: list[str], tensor: torch.Tensor) -> None:
        block: int = int(parts[1])
        if block != 0:
            block_id: int = (block - 1) // 3
            layer_id: int = (block - 1) % 3
            module_id: int = int(parts[2])
            suffix: str = ".".join(parts[3:])

            if module_id == 0 and suffix in ("op.weight", "op.bias"):
                self.state_dict[f"down_blocks.{block_id}.downsamplers.0.conv.{suffix.removeprefix('op.')}"] = tensor
            elif module_id == 0:
                suffix = self._get_resnet_mapping(suffix)
                self.state_dict[f"down_blocks.{block_id}.resnets.{layer_id}.{suffix}"] = tensor
            elif module_id == 1:
                self.state_dict[f"down_blocks.{block_id}.attentions.{layer_id}.{suffix}"] = tensor

    def _set_mid_block(self, parts: list[str], tensor: torch.Tensor) -> None:
        block: int = int(parts[1])
        suffix: str = ".".join(parts[2:])

        if block == 1:
            self.state_dict[f"mid_block.attentions.0.{suffix}"] = tensor
        else:
            suffix = self._get_resnet_mapping(suffix)
            layer_id: int = 0 if block == 0 else 1
            self.state_dict[f"mid_block.resnets.{layer_id}.{suffix}"] = tensor

    def _set_up_block(self, parts: list[str], tensor: torch.Tensor) -> None:
        block: int = int(parts[1])
        block_id: int = block // 3
        layer_id: int = block % 3
        module_id: int = int(parts[2])
        suffix: str = ".".join(parts[3:])

        if module_id == 0:
            suffix = self._get_resnet_mapping(suffix)
            self.state_dict[f"up_blocks.{block_id}.resnets.{layer_id}.{suffix}"] = tensor
        elif suffix in ("conv.weight", "conv.bias"):
            self.state_dict[f"up_blocks.{block_id}.upsamplers.0.{suffix}"] = tensor
        elif module_id == 1:
            self.state_dict[f"up_blocks.{block_id}.attentions.{layer_id}.{suffix}"] = tensor

    def _get_resnet_mapping(self, suffix: str) -> str:
        for old, new in self._RESNET_MAPPING:
            suffix = suffix.replace(old, new)
        return suffix


class AutoencoderKL:
    _DIRECT_MAPPING: dict[str, str] = {
        "encoder.conv_in.weight": "encoder.conv_in.weight",
        "encoder.conv_in.bias": "encoder.conv_in.bias",
        "encoder.conv_out.weight": "encoder.conv_out.weight",
        "encoder.conv_out.bias": "encoder.conv_out.bias",
        "encoder.norm_out.weight": "encoder.conv_norm_out.weight",
        "encoder.norm_out.bias": "encoder.conv_norm_out.bias",
        "decoder.conv_in.weight": "decoder.conv_in.weight",
        "decoder.conv_in.bias": "decoder.conv_in.bias",
        "decoder.conv_out.weight": "decoder.conv_out.weight",
        "decoder.conv_out.bias": "decoder.conv_out.bias",
        "decoder.norm_out.weight": "decoder.conv_norm_out.weight",
        "decoder.norm_out.bias": "decoder.conv_norm_out.bias",
        "quant_conv.weight": "quant_conv.weight",
        "quant_conv.bias": "quant_conv.bias",
        "post_quant_conv.weight": "post_quant_conv.weight",
        "post_quant_conv.bias": "post_quant_conv.bias",
    }
    _RESNET_MAPPING: tuple[tuple[str, str], ...] = (
        ("nin_shortcut", "conv_shortcut"),
    )
    _ATTENTION_MAPPING: tuple[tuple[str, str], ...] = (
        ("norm.weight", "group_norm.weight"),
        ("norm.bias", "group_norm.bias"),
        ("q.weight", "to_q.weight"),
        ("q.bias", "to_q.bias"),
        ("k.weight", "to_k.weight"),
        ("k.bias", "to_k.bias"),
        ("v.weight", "to_v.weight"),
        ("v.bias", "to_v.bias"),
        ("proj_out.weight", "to_out.0.weight"),
        ("proj_out.bias", "to_out.0.bias"),
    )
    state_dict: dict[str, torch.Tensor]

    def __init__(self, orig_state_dict: dict[str, torch.Tensor]) -> None:
        self.state_dict = {}
        self._set_direct_mapping(orig_state_dict)

        for key, tensor in orig_state_dict.items():
            parts: list[str] = key.split(".")
            if key.startswith("encoder.down."):
                self._set_down_block(parts, tensor)
            elif key.startswith("encoder.mid.") or key.startswith("decoder.mid."):
                self._set_mid_block(parts, tensor)
            elif key.startswith("decoder.up."):
                self._set_up_block(parts, tensor)

    def _set_direct_mapping(self, orig_state_dict: dict[str, torch.Tensor]) -> None:
        keys: list[str] = list(orig_state_dict.keys())
        for key in keys:
            if key.startswith("first_stage_model."):
                orig_state_dict[key.removeprefix("first_stage_model.")] = orig_state_dict.pop(key)

        for old_key, new_key in self._DIRECT_MAPPING.items():
            if old_key in orig_state_dict:
                self.state_dict[new_key] = orig_state_dict[old_key]

    def _set_down_block(self, parts: list[str], tensor: torch.Tensor) -> None:
        block: int = int(parts[2])

        if parts[3] == "block":
            suffix: str = self._get_resnet_mapping(4, parts)
            self.state_dict[f"encoder.down_blocks.{block}.resnets.{suffix}"] = tensor
        elif parts[3] == "downsample":
            suffix: str = ".".join(parts[4:])
            self.state_dict[f"encoder.down_blocks.{block}.downsamplers.0.{suffix}"] = tensor

    def _set_mid_block(self, parts: list[str], tensor: torch.Tensor) -> None:
        prefix: str = parts[0]

        if parts[2].startswith("block_"):
            block: int = int(parts[2].removeprefix("block_")) - 1
            suffix: str = self._get_resnet_mapping(3, parts)
            self.state_dict[f"{prefix}.mid_block.resnets.{block}.{suffix}"] = tensor
        elif parts[2] == "attn_1":
            suffix: str = self._get_attn_mapping(3, parts)
            if suffix.endswith("weight") and tensor.ndim == 4:
                tensor = tensor[:, :, 0, 0]
            elif suffix.endswith("weight") and tensor.ndim == 3:
                tensor = tensor[:, :, 0]
            self.state_dict[f"{prefix}.mid_block.attentions.0.{suffix}"] = tensor

    def _set_up_block(self, parts: list[str], tensor: torch.Tensor) -> None:
        block: int = int(parts[2])
        block_id: int = 3 - block

        if parts[3] == "block":
            suffix: str = self._get_resnet_mapping(4, parts)
            self.state_dict[f"decoder.up_blocks.{block_id}.resnets.{suffix}"] = tensor
        elif parts[3] == "upsample":
            suffix: str = ".".join(parts[4:])
            self.state_dict[f"decoder.up_blocks.{block_id}.upsamplers.0.{suffix}"] = tensor

    def _get_resnet_mapping(self, join: int, parts: list[str]) -> str:
        suffix: str = ".".join(parts[join:])
        for old, new in self._RESNET_MAPPING:
            suffix = suffix.replace(old, new)
        return suffix

    def _get_attn_mapping(self, join: int, parts: list[str]) -> str:
        suffix: str = ".".join(parts[join:])
        for old, new in self._ATTENTION_MAPPING:
            suffix = suffix.replace(old, new)
        return suffix


class CLIPTextModel:
    state_dict: dict[str, torch.Tensor]

    def __init__(self, orig_state_dict: dict[str, torch.Tensor]) -> None:
        self.state_dict = {}
        for key, tensor in orig_state_dict.items():
            self.state_dict[key.removeprefix("conditioner.embedders.0.transformer.")] = tensor


class CLIPTextModelWithProjection:
    _DIRECT_MAPPING: dict[str, str] = {
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "text_projection": "text_projection.weight",
    }
    _TRANSFORMER_MAPPING: tuple[tuple[str, str], ...] = (
        ("transformer.resblocks.", "text_model.encoder.layers."),
        (".ln_1.", ".layer_norm1."),
        (".ln_2.", ".layer_norm2."),
        (".mlp.c_fc.", ".mlp.fc1."),
        (".mlp.c_proj.", ".mlp.fc2."),
        (".attn.out_proj.", ".self_attn.out_proj."),
    )
    state_dict: dict[str, torch.Tensor]

    def __init__(self, orig_state_dict: dict[str, torch.Tensor]) -> None:
        self.state_dict = {}
        for key, tensor in orig_state_dict.items():
            key = key.removeprefix("conditioner.embedders.1.model.")

            if key in self._DIRECT_MAPPING:
                self.state_dict[self._DIRECT_MAPPING[key]] = tensor
            elif ".attn.in_proj_" in key:
                self._set_attention(key, tensor)
            elif key.startswith("transformer.resblocks."):
                for old, new in self._TRANSFORMER_MAPPING:
                    key = key.replace(old, new)
                self.state_dict[key] = tensor

    def _set_attention(self, key: str, tensor: torch.Tensor) -> None:
        key = key.removeprefix("transformer.resblocks.")
        layer, suffix = key.split(".attn.in_proj_")
        q, k, v = tensor.chunk(3, dim=0)

        self.state_dict[f"text_model.encoder.layers.{layer}.self_attn.q_proj.{suffix}"] = q
        self.state_dict[f"text_model.encoder.layers.{layer}.self_attn.k_proj.{suffix}"] = k
        self.state_dict[f"text_model.encoder.layers.{layer}.self_attn.v_proj.{suffix}"] = v
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
            model_type: type[Mapping] = _get_model_type(orig_state_dict)

            if model_type is AutoencoderKL:
                converted_state_dict: dict[str, torch.Tensor] = AutoencoderKL(orig_state_dict).state_dict
            elif model_type is CLIPTextModel:
                converted_state_dict: dict[str, torch.Tensor] = CLIPTextModel(orig_state_dict).state_dict
            elif model_type is CLIPTextModelWithProjection:
                converted_state_dict: dict[str, torch.Tensor] = CLIPTextModelWithProjection(orig_state_dict).state_dict
            elif model_type is UNet2DConditionModel:
                converted_state_dict: dict[str, torch.Tensor] = UNet2DConditionModel(orig_state_dict).state_dict

def get_state_dict(file: safetensors.safe_open) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    keys: list[str] = list(file.keys())
    for key in keys:
        tensors = file.get_slice(key)
        shape: tuple[int, ...] = tuple(tensors.get_shape())
        dtype: torch.dtype = safetensors.torch._getdtype(tensors.get_dtype())
        state_dict[key] = torch.empty(shape, dtype=dtype, device="meta")
    return state_dict

def _get_model_type(state_dict: dict[str, torch.Tensor]) -> type[Mapping]:
    parameters: int = sum(tensor.numel() for tensor in state_dict.values())
    if 80_000_000 <= parameters <= 90_000_000:
        return AutoencoderKL
    elif 120_000_000 <= parameters <= 130_000_000:
        return CLIPTextModel
    elif 680_000_000 <= parameters <= 710_000_000:
        return CLIPTextModelWithProjection
    elif 2_500_000_000 <= parameters <= 2_700_000_000:
        return UNet2DConditionModel
    else:
        return UNet2DConditionModel


class Mapping:
    state_dict: dict[str, torch.Tensor]


    def get_direct_mapping(self, orig_state_dict: dict[str, torch.Tensor], mapping: dict[str, str]) -> dict[str, torch.Tensor]:
        state_dict: dict[str, torch.Tensor] = orig_state_dict.copy()
        for old_key, new_key in mapping.items():
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        return state_dict

    def get_segment_mapping(self, key: str, replace: dict[str, str]) -> str:
        for old, new in replace.items():
            key = key.replace(old, new)
        return key


class UNet2DConditionModel(Mapping):
    _DIRECT_MAPPING: dict[str, str] = {
        "model.diffusion_model.input_blocks.0.0.weight": "conv_in.weight",
        "model.diffusion_model.input_blocks.0.0.bias": "conv_in.bias",
        "model.diffusion_model.time_embed.0.weight": "time_embedding.linear_1.weight",
        "model.diffusion_model.time_embed.0.bias": "time_embedding.linear_1.bias",
        "model.diffusion_model.time_embed.2.weight": "time_embedding.linear_2.weight",
        "model.diffusion_model.time_embed.2.bias": "time_embedding.linear_2.bias",
        "model.diffusion_model.label_emb.0.0.weight": "add_embedding.linear_1.weight",
        "model.diffusion_model.label_emb.0.0.bias": "add_embedding.linear_1.bias",
        "model.diffusion_model.label_emb.0.2.weight": "add_embedding.linear_2.weight",
        "model.diffusion_model.label_emb.0.2.bias": "add_embedding.linear_2.bias",
        "model.diffusion_model.out.0.weight": "conv_norm_out.weight",
        "model.diffusion_model.out.0.bias": "conv_norm_out.bias",
        "model.diffusion_model.out.2.weight": "conv_out.weight",
        "model.diffusion_model.out.2.bias": "conv_out.bias",
    }
    _SEGMENT_MAPPING: dict[str, str] = {
        "in_layers.0": "norm1",
        "in_layers.2": "conv1",
        "out_layers.0": "norm2",
        "out_layers.3": "conv2",
        "emb_layers.1": "time_emb_proj",
        "skip_connection": "conv_shortcut",
    }


    def __init__(self, orig_state_dict: dict[str, torch.Tensor]) -> None:
        self.state_dict = self.get_direct_mapping(orig_state_dict, self._DIRECT_MAPPING)

        for orig_key, tensor in orig_state_dict.items():
            if not orig_key.startswith("model.diffusion_model."):
                del self.state_dict[orig_key]
            elif orig_key not in self._DIRECT_MAPPING:
                key: str = orig_key.removeprefix("model.diffusion_model.")

                segments: list[str] = key.split(".")
                block_id: int = int(segments[1])
                suffix: str = ".".join(segments[2:])
                if segments[0] == "middle_block":
                    self.state_dict[self._get_mid_block(block_id, suffix)] = tensor
                else:
                    module_id: int = int(segments[2])
                    suffix = ".".join(segments[3:])
                    if segments[0] == "input_blocks":
                        self.state_dict[self._get_down_block(segments, block_id, module_id, suffix)] = tensor
                    elif segments[0] == "output_blocks":
                        self.state_dict[self._get_up_block(segments, block_id, module_id, suffix)] = tensor
                del self.state_dict[orig_key]

    def _get_mid_block(self, block_id: int, suffix: str) -> str:
        if block_id == 1:
            return f"mid_block.attentions.0.{suffix}"
        else:
            suffix = self.get_segment_mapping(suffix, self._SEGMENT_MAPPING)
            layer_id: int = 0 if block_id == 0 else 1
            return f"mid_block.resnets.{layer_id}.{suffix}"

    def _get_down_block(self, segments: list[str], block_id: int, module_id: int, suffix: str) -> str:
        layer_id: int = (block_id - 1) % 3
        block_id = (block_id - 1) // 3

        if module_id == 0 and suffix in ("op.weight", "op.bias"):
            return f"down_blocks.{block_id}.downsamplers.0.conv.{suffix.removeprefix('op.')}"
        elif module_id == 0:
            suffix = self.get_segment_mapping(suffix, self._SEGMENT_MAPPING)
            return f"down_blocks.{block_id}.resnets.{layer_id}.{suffix}"
        elif module_id == 1:
            return f"down_blocks.{block_id}.attentions.{layer_id}.{suffix}"
        return ".".join(segments)

    def _get_up_block(self, segments: list[str], block_id: int, module_id: int, suffix: str) -> str:
        layer_id: int = block_id % 3
        block_id = block_id // 3

        if module_id == 0:
            suffix = self.get_segment_mapping(suffix, self._SEGMENT_MAPPING)
            return f"up_blocks.{block_id}.resnets.{layer_id}.{suffix}"
        elif suffix in ("conv.weight", "conv.bias"):
            return f"up_blocks.{block_id}.upsamplers.0.{suffix}"
        elif module_id == 1:
            return f"up_blocks.{block_id}.attentions.{layer_id}.{suffix}"
        return ".".join(segments)


class AutoencoderKL(Mapping):
    _DIRECT_MAPPING: dict[str, str] = {
        "encoder.norm_out.weight": "encoder.conv_norm_out.weight",
        "encoder.norm_out.bias": "encoder.conv_norm_out.bias",
        "decoder.norm_out.weight": "decoder.conv_norm_out.weight",
        "decoder.norm_out.bias": "decoder.conv_norm_out.bias",
    }
    _SEGMENT_MAPPING: dict[str, str] = {
        "nin_shortcut": "conv_shortcut",
        "norm.weight": "group_norm.weight",
        "norm.bias": "group_norm.bias",
        "q.weight": "to_q.weight",
        "q.bias": "to_q.bias",
        "k.weight": "to_k.weight",
        "k.bias": "to_k.bias",
        "v.weight": "to_v.weight",
        "v.bias": "to_v.bias",
        "proj_out.weight": "to_out.0.weight",
        "proj_out.bias": "to_out.0.bias",
    }


    def __init__(self, orig_state_dict: dict[str, torch.Tensor]) -> None:
        self.state_dict = self.get_direct_mapping(orig_state_dict, self._DIRECT_MAPPING)

        for orig_key, tensor in orig_state_dict.items():
            if orig_key not in self._DIRECT_MAPPING:
                segments: list[str] = orig_key.split(".")
                if orig_key.startswith("encoder.mid.") or orig_key.startswith("decoder.mid."):
                    key, tensor = self._get_mid_block(segments, self.state_dict.pop(orig_key))
                    self.state_dict[key] = tensor
                elif orig_key.startswith("encoder.down.") or orig_key.startswith("decoder.up."):
                    self.state_dict[self._get_block(segments)] = self.state_dict.pop(orig_key)

    def _get_mid_block(self, segments: list[str], tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
        prefix: str = segments[0]
        if segments[2] == "attn_1":
            suffix: str = self.get_segment_mapping(".".join(segments[3:]), self._SEGMENT_MAPPING)
            if suffix.endswith("weight") and tensor.ndim == 4:
                tensor = tensor[:, :, 0, 0]
            elif suffix.endswith("weight") and tensor.ndim == 3:
                tensor = tensor[:, :, 0]
            return f"{prefix}.mid_block.attentions.0.{suffix}", tensor

        else:
            block_id: int = int(segments[2].removeprefix("block_")) - 1
            suffix: str = self.get_segment_mapping(".".join(segments[3:]), self._SEGMENT_MAPPING)
            return f"{prefix}.mid_block.resnets.{block_id}.{suffix}", tensor

    def _get_block(self, segments: list[str]) -> str:
        suffix: str = ".".join(segments[4:])
        block_id: int = int(segments[2]) if segments[0] == "encoder" else 3 - int(segments[2])
        prefix: str = "encoder.down_blocks" if segments[0] == "encoder" else "decoder.up_blocks"
        if segments[3] == "downsample":
            return f"{prefix}.{block_id}.downsamplers.0.{suffix}"
        elif segments[3] == "upsample":
            return f"{prefix}.{block_id}.upsamplers.0.{suffix}"
        else:
            suffix: str = self.get_segment_mapping(suffix, self._SEGMENT_MAPPING)
            return f"{prefix}.{block_id}.resnets.{suffix}"
        

class CLIPTextModel(Mapping):
    def __init__(self, orig_state_dict: dict[str, torch.Tensor]) -> None:
        self.state_dict: dict[str, torch.Tensor] = {}

        for orig_key, tensor in orig_state_dict.items():
            key: str = orig_key.removeprefix("cond_stage_model.transformer.").removeprefix("conditioner.embedders.0.transformer.")
            self.state_dict[key] = tensor


class CLIPTextModelWithProjection(Mapping):
    _DIRECT_MAPPING: dict[str, str] = {
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "text_projection": "text_projection.weight",
    }
    _SEGMENT_MAPPING: dict[str, str] = {
        "transformer.resblocks.": "text_model.encoder.layers.",
        ".ln_1.": ".layer_norm1.",
        ".ln_2.": ".layer_norm2.",
        ".mlp.c_fc.": ".mlp.fc1.",
        ".mlp.c_proj.": ".mlp.fc2.",
        ".attn.out_proj.": ".self_attn.out_proj.",
    }


    def __init__(self, orig_state_dict: dict[str, torch.Tensor]) -> None:
        self.state_dict = self.get_direct_mapping(orig_state_dict, self._DIRECT_MAPPING)

        for orig_key, tensor in orig_state_dict.items():
            if orig_key == "logit_scale":
                del self.state_dict[orig_key]
            elif orig_key.startswith("transformer.resblocks."):
                key: str = self.get_segment_mapping(orig_key, self._SEGMENT_MAPPING)
                if ".attn.in_proj_" in key:
                    for attention_key, attention_tensor in self._get_attention(key, tensor):
                        self.state_dict[attention_key] = attention_tensor
                else:
                    self.state_dict[key] = tensor
                del self.state_dict[orig_key]

    def _get_attention(self, key: str, tensor: torch.Tensor) -> tuple[tuple[str, torch.Tensor], tuple[str, torch.Tensor], tuple[str, torch.Tensor]]:
        layer, suffix = key.split(".attn.in_proj_")
        q, k, v = tensor.chunk(3, dim=0)
        return (
            (f"{layer}.self_attn.q_proj.{suffix}", q),
            (f"{layer}.self_attn.k_proj.{suffix}", k),
            (f"{layer}.self_attn.v_proj.{suffix}", v),
        )
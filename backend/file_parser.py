import safetensors

from pathlib import Path
from collections.abc import Callable

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers.models.clip import CLIPTextModel, CLIPTextModelWithProjection


class Signals:
    on_files_parsed: list[Callable[..., object | None]] = []


def init(settings: dict[str, bool | int | str]) -> None:
    model_dir: Path = Path(str(settings["model_path"]))
    model_paths: list[Path] = list(model_dir.rglob("*.safetensors"))

    paths_by_type: dict[type[object], list[Path]] = _get_types(model_paths)
    [receiver(paths_by_type) for receiver in Signals.on_files_parsed]


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
import torch

from pathlib import Path

from diffusers.models.modeling_utils import ModelMixin
from transformers.modeling_utils import PreTrainedModel


_cuda_stream: list[torch.cuda.Stream] = []

_static_memory: tuple[list[Path], list[torch.nn.Module], list[int]]

_latents: torch.Tensor


def init(settings: dict[str, bool | int | str]) -> None:
    _cuda_stream.clear()
    _cuda_stream.append(torch.cuda.default_stream())

    cuda_stream: int = int(settings["cuda_stream"])
    for _ in range(max(0, cuda_stream - 1)):
        _cuda_stream.append(torch.cuda.Stream())


def ready() -> None:
    enabled = torch.cuda.memory.get_allocator_backend() == "cudaMallocAsync"
    print(f"cudaMallocAsync: {'Enabled' if enabled else 'Disabled'}")

    print(f"CUDA Stream: {len(_cuda_stream)}")


def on_meta_parsed(models: tuple[list[Path], list[ModelMixin | PreTrainedModel]]) -> None:
    global _static_memory

    import copy

    meta_models: list[torch.nn.Module] = [copy.deepcopy(model) for model in models[1]]
    model_sizes: list[int] = [sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) + sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()) for model in meta_models]

    _static_memory = (models[0], meta_models, model_sizes)


def on_latent_changed(batch: int, width: int, height: int) -> None:
    global _latents

    _latents = torch.randn(batch, 4, height // 8, width // 8, device="meta")


def on_model_changed(path: Path) -> None:
    pass


def _empty_cuda_cache() -> None:
    torch.cuda.empty_cache()


class RuntimeMemory:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    from transformers.models.clip import CLIPTextModel, CLIPTextModelWithProjection


    def __init__(self) -> None:
        pass
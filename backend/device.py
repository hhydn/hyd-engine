import torch


_streams: list[torch.cuda.Stream] = []

_model_storage_memory: dict[torch.nn.Module, int] = {}
_model_runtime_memory: dict[torch.nn.Module, int] = {}
_model_priority: dict[torch.nn.Module, int] = {}

_model_device: dict[torch.nn.Module, torch.device] = {}


def init(cuda_malloc: bool, cuda_streams: int) -> None:
    import os
    if cuda_malloc:
        os.environ["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
    else:
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
    enabled = torch.cuda.memory.get_allocator_backend() == "cudaMallocAsync"
    print(f"cudaMallocAsync: {'Enabled' if enabled else 'Disabled'}")

    _streams.clear()
    for _ in range(cuda_streams):
        _streams.append(torch.cuda.Stream(device=torch.device("cuda")))
    print(f"Extra Streams: {len(_streams)}")


def register_model(model: torch.nn.Module, tensor: torch.Tensor, evaluations: int | None = None) -> None:
    if model not in _model_storage_memory:
        _model_storage_memory[model] = _get_storage_size(model)
        _model_runtime_memory[model] = _get_cuda_forward_pass(model, tensor) + _get_storage_size(tensor=tensor)
        _empty_cuda_cache()

    if evaluations is not None:
        _model_priority[model] = evaluations * _model_storage_memory[model]
        _set_devices()
    

def _set_devices() -> None:
    free_memory, allocated_memory = _get_current_global_memory()
    stored_memory: int = 0
    runtime_memory: int = 0
    models_by_priority: list[torch.nn.Module] = sorted(_model_priority, key=lambda model: _model_priority[model], reverse=True)

    for model in models_by_priority:
        proposed_storage: int = stored_memory + _model_storage_memory[model]
        proposed_runtime: int = max(runtime_memory, _model_runtime_memory[model])

        if proposed_storage + proposed_runtime <= free_memory + allocated_memory:
            device = torch.device("cuda")
            stored_memory = proposed_storage
            runtime_memory = proposed_runtime
        else:
            device = torch.device("cpu")

        if model not in _model_device or _model_device[model] != device:
            model.to(device)
            _model_device[model] = device


def _get_cuda_forward_pass(model: torch.nn.Module, tensor: torch.Tensor) -> int:
    device_before: torch.device = _get_model_device(model)
    cuda: torch.device = torch.device("cuda")

    model.to(cuda)
    tensor = tensor.to(cuda)
    model.eval()

    torch.cuda.synchronize(cuda)
    torch.cuda.reset_peak_memory_stats(cuda)
    _, allocated_before = _get_current_global_memory()

    with torch.inference_mode():
        output = model(tensor)

    torch.cuda.synchronize(cuda)
    peak_allocated = torch.cuda.max_memory_allocated(cuda)
    memory_required: int = peak_allocated - allocated_before

    del output
    del tensor
    model.to(torch.device(device_before))
    return memory_required


def _get_current_global_memory(device: torch.device = torch.device("cuda")) -> tuple[int, int]:
    allocated: int = torch.cuda.memory_allocated(device)
    free, _ = torch.cuda.mem_get_info(device)
    return torch.cuda.memory_reserved(device) - allocated + free, allocated


def _get_model_device(model: torch.nn.Module) -> torch.device:
    parameter: torch.nn.Parameter | None = next(model.parameters(), None)
    buffer: torch.Tensor | None = next(model.buffers(), None)
    return parameter.device if parameter is not None else buffer.device if buffer is not None else torch.device("cpu")


def _get_storage_size(module: torch.nn.Module | None = None, tensor: torch.Tensor | None = None) -> int:
    size: int = 0

    if module is not None:
        size += sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
        size += sum(buffer.numel() * buffer.element_size() for buffer in module.buffers())

    if tensor is not None:
        size += tensor.numel() * tensor.element_size()

    return size


def _empty_cuda_cache() -> None:
    torch.cuda.empty_cache()
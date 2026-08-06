from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from pkgutil import walk_packages
from time import perf_counter, sleep


_init_callables: list[Callable[[dict[str, bool | int | str]], object | None]] = []
_ready_callables: list[Callable[[], object | None]] = []
_process_callables: list[Callable[[], object | None]] = []


def _setup(settings: dict[str, bool | int | str]) -> None:
    import os

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    if settings["cuda_malloc"]:
        os.environ["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
    else:
        os.environ.pop("PYTORCH_ALLOC_CONF", None)


def _find_functions() -> None:
    root: Path = Path(__file__).parent

    for module_info in walk_packages([str(root)]):
        if module_info.name == "main":
            continue

        module = import_module(module_info.name)

        init_function = module.__dict__.get("init")
        ready_function = module.__dict__.get("ready")
        process_function = module.__dict__.get("process")

        if callable(init_function):
            _init_callables.append(init_function)
        if callable(ready_function):
            _ready_callables.append(ready_function)
        if callable(process_function):
            _process_callables.append(process_function)


def _init(settings: dict[str, bool | int | str]) -> None:
    for function in _init_callables:
        function(settings)


def _ready() -> None:
    for function in _ready_callables:
        function()


def _process() -> None:
    interval: float = 0.25
    next_process: float = perf_counter()

    while True:
        current_time: float = perf_counter()

        if current_time >= next_process:
            for function in _process_callables:
                function()

            next_process = current_time + interval

        sleep(max(0.0, next_process - perf_counter()))


if __name__ == "__main__":
    import json

    settings_path: Path = Path(__file__).parent / "settings.json"
    settings: dict[str, bool | int | str] = json.loads(settings_path.read_text(encoding="utf-8"))

    _setup(settings)
    _find_functions()
    _init(settings)
    _ready()
    _process()
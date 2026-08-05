from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from pkgutil import walk_packages


_init_callables: list[Callable[[dict[str, bool | int | str]], object | None]] = []
_ready_callables: list[Callable[[], object | None]] = []


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

        if callable(init_function):
            _init_callables.append(init_function)

        if callable(ready_function):
            _ready_callables.append(ready_function)


def _init(settings: dict[str, bool | int | str]) -> None:
    for function in _init_callables:
        function(settings)


def _ready() -> None:
    for function in _ready_callables:
        function()


if __name__ == "__main__":
    import json

    settings_path: Path = Path(__file__).parent / "settings.json"
    settings: dict[str, bool | int | str] = json.loads(settings_path.read_text(encoding="utf-8"))

    _setup(settings)
    _find_functions()
    _init(settings)
    _ready()
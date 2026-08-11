from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from pkgutil import walk_packages
from time import perf_counter, sleep
from types import ModuleType
from typing import cast


_SETTINGS_PATH: Path = Path(__file__).parent / "settings.json"

_callables_by_hook: dict[str, list[Callable[..., object | None]]] = {
    "init": [],  # Passes settings as argument.
    "ready": [],
    "process": [],  # Must be called last.
}


def _setup(settings: dict[str, bool | int | str]) -> None:
    import os

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    if settings["cuda_malloc"]:
        os.environ["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
    else:
        os.environ.pop("PYTORCH_ALLOC_CONF", None)


def _get_modules() -> list[ModuleType]:
    modules: list[ModuleType] = []
    root: Path = Path(__file__).parent

    for module_info in walk_packages([str(root)]):
        if module_info.name != Path(__file__).stem:
            module: ModuleType = import_module(module_info.name)
            modules.append(module)

    return modules


def _register_modules(modules: list[ModuleType]) -> None:
    for module in modules:
        for function_name, functions in _callables_by_hook.items():
            function = module.__dict__.get(function_name)

            if callable(function):
                functions.append(function)

        signals_class: type[object] | None = cast(type[object] | None, module.__dict__.get("Signals"))

        if signals_class is not None:
            for receiver_name, receivers_object in signals_class.__dict__.items():
                if isinstance(receivers_object, list):
                    receivers = cast(list[Callable[..., object | None]], receivers_object)

                    for receiver_module in modules:
                        function = receiver_module.__dict__.get(receiver_name)

                        if callable(function):
                            receivers.append(function)


def _call_hooks(settings: dict[str, bool | int | str]) -> None:
    for name, functions in _callables_by_hook.items():
        for function in functions:
            if name != next(reversed(_callables_by_hook.keys())):
                if name == "init":
                    function(settings)
                else:
                    function()


def _last_hook() -> None:
    functions: list[Callable[..., object | None]] = next(reversed(_callables_by_hook.values()))

    interval: float = 1 / 60
    next_process: float = perf_counter()
    while True:
        current_time: float = perf_counter()

        if current_time >= next_process:
            for function in functions:
                function()

            next_process = current_time + interval

        sleep(max(0.0, next_process - perf_counter()))


if __name__ == "__main__":
    import json

    settings: dict[str, bool | int | str] = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))

    _setup(settings)

    modules: list[ModuleType] = _get_modules()
    _register_modules(modules)
    
    _call_hooks(settings)
    _last_hook()
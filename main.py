from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from pkgutil import walk_packages
from time import perf_counter, sleep
from types import ModuleType


_SETTINGS_PATH: Path = Path(__file__).parent / "settings.json"

_hooks_dict: dict[str, list[Callable[..., object | None]]] = {
    "init": [],  # Passes settings as argument.
    "ready": [],
    "process": [],  # Must be called last.
}
_signals_dict: dict[tuple[str, str], list[Callable[..., object | None]]] = {
    ("inference_changed", "on_inference_changed"): [],
    ("inference_clicked", "on_inference_pressed"): [],
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


def _set_dispatcher(modules: list[ModuleType]) -> dict[str, list[object]]:
    signal_objects: dict[str, list[object]] = {}

    for object_name, _ in _signals_dict:
        signal_objects[object_name] = []

    for module in modules:
        for hook_name, hooks in _hooks_dict.items():
            function = module.__dict__.get(hook_name)

            if callable(function):
                hooks.append(function)

        for (signal_name, hook_name), signal_hooks in _signals_dict.items():
            signal = module.__dict__.get(signal_name)

            if signal is not None:
                signal_objects[signal_name].append(signal)

            function = module.__dict__.get(hook_name)

            if callable(function):
                signal_hooks.append(function)

    return signal_objects


def _call_hooks(settings: dict[str, bool | int | str]) -> None:
    for name, functions in _hooks_dict.items():
        for function in functions:

            if name != next(reversed(_hooks_dict.keys())):
                if name == "init":
                    function(settings)
                else:
                    function()


def _connect_signals(signal_objects: dict[str, list[object]]) -> None:
    from typing import cast
    from PySide6.QtCore import SignalInstance

    for (signal_name, _), signal_hooks in _signals_dict.items():
        for signal_object in signal_objects[signal_name]:
            signal: SignalInstance = cast(SignalInstance, signal_object)

            for function in signal_hooks:
                signal.connect(function)


def _last_hook() -> None:
    functions: list[Callable[..., object | None]] = next(reversed(_hooks_dict.values()))

    interval: float = 0.1
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
    signal_objects: dict[str, list[object]] = _set_dispatcher(modules)

    _call_hooks(settings)
    _connect_signals(signal_objects)
    _last_hook()
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from pkgutil import walk_packages
from time import perf_counter, sleep
from types import ModuleType


_SETTINGS_PATH: Path = Path(__file__).parent / "settings.json"

_hooks_dict: dict[str, list[Callable[..., object | None]]] = {
    "set_emit": [],
    "init": [],  # Passes settings as argument.
    "ready": [],
    "process": [],  # Must be called last.
}
_signals_dict: dict[tuple[str, str], list[Callable[..., object | None]]] = {
    ("generate_clicked", "on_generate_clicked"): [],
    ("model_changed", "on_model_changed"): [],
    ("timesteps_changed", "on_timesteps_changed"): [],

    ("files_parsed", "on_files_parsed"): [],
}


def emit(signal_name: str, *args: object) -> None:
    for (name, _), functions in _signals_dict.items():
        if name == signal_name:
            for function in functions:
                function(*args)


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


def _set_dispatcher(modules: list[ModuleType]) -> None:
    for module in modules:
        for hook_name, hooks in _hooks_dict.items():
            function = module.__dict__.get(hook_name)

            if callable(function):
                hooks.append(function)

        for (_, hook_name), signal_hooks in _signals_dict.items():
            function = module.__dict__.get(hook_name)

            if callable(function):
                signal_hooks.append(function)


def _call_hooks(settings: dict[str, bool | int | str]) -> None:
    for name, functions in _hooks_dict.items():
        for function in functions:
            if name != next(reversed(_hooks_dict.keys())):
                if name == "set_emit":
                    function(emit)
                elif name == "init":
                    function(settings)
                else:
                    function()


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
    _set_dispatcher(modules)
    
    _call_hooks(settings)
    _last_hook()
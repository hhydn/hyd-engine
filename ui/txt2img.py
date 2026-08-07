from collections.abc import Callable
from PySide6 import QtWidgets


emit: Callable[..., object]

_generate_button: QtWidgets.QPushButton = QtWidgets.QPushButton("Generate")
_models_box: QtWidgets.QComboBox = QtWidgets.QComboBox()
_steps_box: QtWidgets.QSpinBox = QtWidgets.QSpinBox()

_txt2img_widget: QtWidgets.QWidget = QtWidgets.QWidget()
_txt2img_layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout(_txt2img_widget)


def set_emit(function: Callable[..., None]) -> None:
    global emit
    emit = function


def ready() -> None:
    from . import app

    app.main_layout.addWidget(_txt2img_widget, 1, 0)

    _txt2img_layout.addWidget(_generate_button)
    _txt2img_layout.addWidget(_models_box)
    _txt2img_layout.addWidget(_steps_box)
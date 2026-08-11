from collections.abc import Callable
from PySide6 import QtWidgets
from pathlib import Path


class Signals:
    on_prompt_changed: list[Callable[..., object | None]] = []
    on_model_changed: list[Callable[..., object | None]] = []
    on_latent_changed: list[Callable[..., object | None]] = []
    on_steps_changed: list[Callable[..., object | None]] = []
    on_generate_clicked: list[Callable[..., object | None]] = []


_txt2img_widget: QtWidgets.QWidget = QtWidgets.QWidget()
_txt2img_layout: QtWidgets.QGridLayout = QtWidgets.QGridLayout(_txt2img_widget)

_prompt_box: QtWidgets.QPlainTextEdit = QtWidgets.QPlainTextEdit()

_denoiser_box: QtWidgets.QComboBox = QtWidgets.QComboBox()
_vae_box: QtWidgets.QComboBox = QtWidgets.QComboBox()
_text_encoders_box: QtWidgets.QComboBox = QtWidgets.QComboBox()

_batch_box: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
_width_box: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
_height_box: QtWidgets.QSpinBox = QtWidgets.QSpinBox()

_steps_box: QtWidgets.QSpinBox = QtWidgets.QSpinBox()

_generate_button: QtWidgets.QPushButton = QtWidgets.QPushButton("Generate")


def ready() -> None:
    from . import app

    _txt2img_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
    app.main_layout.addWidget(_txt2img_widget, 1, 0)

    _set_grid_layout()

    _connect_signals()

    _batch_box.setRange(1, 10)
    _width_box.setRange(64, 8192)
    _height_box.setRange(64, 8192)
    _steps_box.setRange(1, 200)


def on_files_parsed(paths_by_type: dict[type[object], list[Path]]) -> None:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection

    for path in paths_by_type[UNet2DConditionModel]:
        _denoiser_box.addItem(path.name, path)

    for path in paths_by_type[AutoencoderKL]:
        _vae_box.addItem(path.name, path)

    for path in paths_by_type[CLIPTextModel]:
        _text_encoders_box.addItem(path.name, path)

    for path in paths_by_type[CLIPTextModelWithProjection]:
        _text_encoders_box.addItem(path.name, path)


def _connect_signals() -> None:
    _prompt_box.textChanged.connect(lambda: [receiver(_prompt_box.toPlainText()) for receiver in Signals.on_prompt_changed])

    _denoiser_box.currentTextChanged.connect(lambda: [receiver(_denoiser_box.currentData()) for receiver in Signals.on_model_changed])
    _vae_box.currentTextChanged.connect(lambda: [receiver(_vae_box.currentData()) for receiver in Signals.on_model_changed])
    _text_encoders_box.currentTextChanged.connect(lambda: [receiver(_text_encoders_box.currentData()) for receiver in Signals.on_model_changed])

    for tensor in (_batch_box, _width_box, _height_box):
        tensor.valueChanged.connect(lambda: [receiver(_batch_box.value(), _width_box.value(), _height_box.value()) for receiver in Signals.on_latent_changed])
    _steps_box.valueChanged.connect(lambda: [receiver(_steps_box.value()) for receiver in Signals.on_steps_changed])

    _generate_button.clicked.connect(lambda: [receiver() for receiver in Signals.on_generate_clicked])


def _set_grid_layout() -> None:
    _txt2img_layout.addWidget(_prompt_box, 0, 0, 1, 3)

    _txt2img_layout.addWidget(_denoiser_box, 1, 0)
    _txt2img_layout.addWidget(_vae_box, 1, 1)
    _txt2img_layout.addWidget(_text_encoders_box, 1, 2)

    _txt2img_layout.addWidget(_batch_box, 2, 0)
    _txt2img_layout.addWidget(_width_box, 2, 1)
    _txt2img_layout.addWidget(_height_box, 2, 2)
    _txt2img_layout.addWidget(_steps_box, 2, 3)

    _txt2img_layout.addWidget(_generate_button, 3, 0)
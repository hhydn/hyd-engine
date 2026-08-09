from collections.abc import Callable
from PySide6 import QtWidgets
from pathlib import Path


SIGNALS: tuple[str, ...] = (
    "on_model_changed",
    "on_generate_clicked",
    "on_timesteps_changed",
)
emit: Callable[..., None]

_denoiser_box: QtWidgets.QComboBox = QtWidgets.QComboBox()
_vae_box: QtWidgets.QComboBox = QtWidgets.QComboBox()
_text_encoders_box: QtWidgets.QComboBox = QtWidgets.QComboBox()

_generate_button: QtWidgets.QPushButton = QtWidgets.QPushButton("Generate")
_steps_box: QtWidgets.QSpinBox = QtWidgets.QSpinBox()

_txt2img_widget: QtWidgets.QWidget = QtWidgets.QWidget()
_txt2img_layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout(_txt2img_widget)


def ready() -> None:
    from . import app

    _txt2img_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
    app.main_layout.addWidget(_txt2img_widget, 1, 0)

    _txt2img_layout.addWidget(_denoiser_box)
    _txt2img_layout.addWidget(_vae_box)
    _txt2img_layout.addWidget(_text_encoders_box)

    _txt2img_layout.addWidget(_generate_button)
    _txt2img_layout.addWidget(_steps_box)

    _denoiser_box.currentTextChanged.connect(_model_changed)
    _vae_box.currentTextChanged.connect(_model_changed)
    _text_encoders_box.currentTextChanged.connect(_model_changed)

    _generate_button.clicked.connect(_generate_clicked)


def on_files_parsed(models: dict[type[object], list[Path]]) -> None:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection

    for path in models[UNet2DConditionModel]:
        _denoiser_box.addItem(path.name, path)

    for path in models[AutoencoderKL]:
        _vae_box.addItem(path.name, path)

    for path in models[CLIPTextModel]:
        _text_encoders_box.addItem(path.name, path)

    for path in models[CLIPTextModelWithProjection]:
        _text_encoders_box.addItem(path.name, path)


def _model_changed(path_name: str) -> None:
    if _denoiser_box.findText(path_name) != -1:
        emit("on_model_changed", _denoiser_box.currentData())

    elif _vae_box.findText(path_name) != -1:
        emit("on_model_changed", _vae_box.currentData())

    elif _text_encoders_box.findText(path_name) != -1:
        emit("on_model_changed", _text_encoders_box.currentData())


def _generate_clicked() -> None:
    emit("on_generate_clicked")
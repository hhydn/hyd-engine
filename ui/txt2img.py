from collections.abc import Callable
from PySide6 import QtWidgets
from pathlib import Path


emit: Callable[..., object]

_unets_box: QtWidgets.QComboBox = QtWidgets.QComboBox()
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

    _txt2img_layout.addWidget(_unets_box)
    _txt2img_layout.addWidget(_vae_box)
    _txt2img_layout.addWidget(_text_encoders_box)

    _txt2img_layout.addWidget(_generate_button)
    _txt2img_layout.addWidget(_steps_box)


def on_files_parsed(models: dict[type[object], list[Path]]) -> None:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection

    _unets_box.addItems([path.name for path in models[UNet2DConditionModel]])
    _vae_box.addItems([path.name for path in models[AutoencoderKL]])
    _text_encoders_box.addItems([path.name for path in models[CLIPTextModel]])
    _text_encoders_box.addItems([path.name for path in models[CLIPTextModelWithProjection]])
from PySide6 import QtCore, QtWidgets


generate_button: QtWidgets.QPushButton = QtWidgets.QPushButton("Generate")


def ready() -> None:
    from . import app

    layout: QtWidgets.QLayout | None = app.widget.layout()

    if layout:
        layout.addWidget(generate_button)
        layout.setAlignment(generate_button, QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignLeft)
        generate_button.clicked.connect(_generate)


def _generate() -> None:
    print("generate")
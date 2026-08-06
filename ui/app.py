from PySide6 import QtWidgets


app: QtWidgets.QApplication = QtWidgets.QApplication([])
widget: QtWidgets.QWidget = QtWidgets.QWidget()


def init(_settings: dict[str, bool | int | str]) -> None:
    layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
    widget.setLayout(layout)

    widget.setMinimumSize(600, 400)
    widget.resize(1000, 600)


def ready() -> None:
    widget.show()


def process() -> None:
    app.processEvents()
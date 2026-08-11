from PySide6 import QtWidgets


app: QtWidgets.QApplication = QtWidgets.QApplication([])
main_widget: QtWidgets.QWidget = QtWidgets.QWidget()
main_layout: QtWidgets.QGridLayout = QtWidgets.QGridLayout(main_widget)


def init(_settings: dict[str, bool | int | str]) -> None:
    main_widget.setMinimumSize(600, 400)
    main_widget.showMaximized()

    main_layout.setRowStretch(0, 2)
    main_layout.setRowStretch(1, 1)

    main_layout.setColumnStretch(0, 2)
    main_layout.setColumnStretch(1, 1)


def ready() -> None:
    main_widget.show()


def process() -> None:
    app.processEvents()
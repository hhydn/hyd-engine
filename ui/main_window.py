from PySide6 import QtWidgets


def start() -> None:
    app = QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Hyd Engine")
    window.setFixedSize(1000, 600)
    window.show()

    app.exec()
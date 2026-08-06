import sys

from types import SimpleNamespace
from typing import TextIO, cast

from PySide6 import QtCore, QtGui, QtWidgets


_console: QtWidgets.QPlainTextEdit = QtWidgets.QPlainTextEdit()


def init(_settings: dict[str, bool | int | str]) -> None:
    _console.setFixedSize(250, 500)
    _console.setReadOnly(True)

    stream: TextIO = cast(TextIO, SimpleNamespace(write=_write, flush=lambda: None))
    sys.stdout = stream
    sys.stderr = stream


def ready() -> None:
    from . import app

    layout: QtWidgets.QLayout | None = app.widget.layout()

    if layout:
        layout.addWidget(_console)
        layout.setAlignment(_console, QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignRight)


def _write(text: str) -> int:
    _console.moveCursor(QtGui.QTextCursor.MoveOperation.End)
    _console.insertPlainText(text)
    _console.ensureCursorVisible()

    return len(text)
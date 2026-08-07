import sys

from types import SimpleNamespace
from typing import TextIO, cast

from PySide6 import QtGui, QtWidgets


_console: QtWidgets.QPlainTextEdit = QtWidgets.QPlainTextEdit()


def init(_settings: dict[str, bool | int | str]) -> None:
    stream: TextIO = cast(TextIO, SimpleNamespace(write=_write, flush=lambda: None))
    sys.stdout = stream
    sys.stderr = stream


def ready() -> None:
    from . import app

    _console.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
    app.main_layout.addWidget(_console, 1, 1)

    _console.setReadOnly(True)


def _write(text: str) -> int:
    _console.moveCursor(QtGui.QTextCursor.MoveOperation.End)
    _console.insertPlainText(text)
    _console.ensureCursorVisible()

    return len(text)
"""Load only the application's bundled PortAudio on Windows.

sounddevice normally searches PATH (including conda) before its wheel's DLL.
Select the audited normal DLL explicitly, without changing global DLL search
paths or leaving ctypes patched after the import.
"""

from __future__ import annotations

import ctypes.util
from importlib import import_module
from importlib.util import find_spec
import os
from pathlib import Path
import sys
from threading import RLock

from .dependency_policy import ensure_no_prohibited_dependencies


_IMPORT_LOCK = RLock()
_PORTAUDIO_SEARCH_NAMES = frozenset({
    "portaudio", r"bin\libportaudio-2.dll", "lib/libportaudio.dylib",
})


class NativeAudioError(RuntimeError):
    """A bundled native audio component could not be selected safely."""


def _bundled_portaudio() -> Path:
    if getattr(sys, "frozen", False):
        bundle = getattr(sys, "_MEIPASS", None)
        if not bundle:
            raise NativeAudioError("配布物の音声ライブラリの場所を確認できません。ZIPを展開し直してください。")
        package = Path(bundle) / "_sounddevice_data"
    else:
        spec = find_spec("_sounddevice_data")
        locations = None if spec is None else spec.submodule_search_locations
        if not locations:
            raise NativeAudioError("同梱の音声ライブラリがありません。専用の仮想環境でセットアップし直してください。")
        package = Path(next(iter(locations)))
    library = (package / "portaudio-binaries" / "libportaudio64bit.dll").resolve()
    if not library.is_file():
        raise NativeAudioError(
            "同梱のPortAudio DLLがありません。Windows x64用ZIPをすべて展開するか、"
            "専用の仮想環境でセットアップし直してください。外部のDLLは使用しません。"
        )
    return library


def _verify_loaded_library(module, expected: Path) -> None:
    loaded = getattr(module, "_libname", None)
    try:
        path = Path(loaded)
        matches = path.is_absolute() and os.path.normcase(str(path.resolve())) == os.path.normcase(str(expected))
    except (TypeError, ValueError, OSError):
        matches = False
    if not matches:
        raise NativeAudioError(
            "同梱版以外のPortAudioが既に読み込まれています。"
            "このPythonプロセスを終了し、専用の仮想環境か配布EXEから起動し直してください。"
        )


def load_sounddevice():
    """Import sounddevice after policy checks; Windows never searches external DLLs."""
    ensure_no_prohibited_dependencies()
    if sys.platform != "win32":
        return import_module("sounddevice")
    with _IMPORT_LOCK:
        library = _bundled_portaudio()
        existing = sys.modules.get("sounddevice")
        if existing is not None:
            _verify_loaded_library(existing, library)
            return existing
        original_find_library = ctypes.util.find_library

        def bundled_find_library(name):
            if name in _PORTAUDIO_SEARCH_NAMES:
                return str(library)
            return original_find_library(name)

        # sounddevice also has an OSError fallback. Keep that fallback on its
        # normal bundled DLL, even when the caller has enabled optional ASIO.
        asio = os.environ.pop("SD_ENABLE_ASIO", None)
        try:
            ctypes.util.find_library = bundled_find_library
            module = import_module("sounddevice")
        except (ImportError, OSError) as exc:
            raise NativeAudioError(
                "同梱の音声ライブラリを読み込めません。Windows x64用ZIPをすべて展開するか、"
                "専用の仮想環境でセットアップし直してください。"
            ) from exc
        finally:
            ctypes.util.find_library = original_find_library
            if asio is not None:
                os.environ["SD_ENABLE_ASIO"] = asio
        _verify_loaded_library(module, library)
        return module

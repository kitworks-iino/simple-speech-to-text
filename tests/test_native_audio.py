from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_speech_to_text import native_audio as native


@pytest.fixture
def windows(monkeypatch, tmp_path):
    state = SimpleNamespace(platform="win32", modules={}, frozen=False)
    monkeypatch.setattr(native, "sys", state)
    monkeypatch.setattr(native, "ensure_no_prohibited_dependencies", lambda: [])
    package = tmp_path / "site-packages" / "_sounddevice_data"
    dll = package / "portaudio-binaries" / "libportaudio64bit.dll"
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"test DLL placeholder: never loaded")
    monkeypatch.setattr(native, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(package)]))
    return state, dll.resolve()


def test_windows_selects_bundled_dll_and_restores_lookup(monkeypatch, windows):
    state, dll = windows
    lookup_calls = []

    def original(name):
        lookup_calls.append(name)
        return f"external/{name}"

    monkeypatch.setattr(native.ctypes.util, "find_library", original)
    monkeypatch.setenv("SD_ENABLE_ASIO", "1")
    module = SimpleNamespace(_libname=str(dll))

    def importer(name):
        assert name == "sounddevice"
        assert "SD_ENABLE_ASIO" not in native.os.environ
        for library in ("portaudio", r"bin\libportaudio-2.dll", "lib/libportaudio.dylib"):
            assert native.ctypes.util.find_library(library) == str(dll)
        assert native.ctypes.util.find_library("unrelated") == "external/unrelated"
        return module

    monkeypatch.setattr(native, "import_module", importer)
    assert native.load_sounddevice() is module
    assert lookup_calls == ["unrelated"]
    assert native.ctypes.util.find_library is original
    assert native.os.environ["SD_ENABLE_ASIO"] == "1"
    assert not state.modules


@pytest.mark.parametrize("error", [OSError("load failed"), ImportError("missing"), KeyboardInterrupt()])
def test_lookup_and_environment_restored_when_import_fails(monkeypatch, windows, error):
    original = native.ctypes.util.find_library
    monkeypatch.delenv("SD_ENABLE_ASIO", raising=False)

    def fail(name):
        raise error

    monkeypatch.setattr(native, "import_module", fail)
    expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else native.NativeAudioError
    with pytest.raises(expected):
        native.load_sounddevice()
    assert native.ctypes.util.find_library is original
    assert "SD_ENABLE_ASIO" not in native.os.environ


def test_missing_bundled_dll_never_uses_path_or_imports(monkeypatch, windows):
    _, dll = windows
    dll.unlink()
    monkeypatch.setattr(native, "import_module", lambda _: pytest.fail("must not import"))
    monkeypatch.setattr(native.ctypes.util, "find_library", lambda _: pytest.fail("must not search PATH"))
    with pytest.raises(native.NativeAudioError, match="DLLがありません"):
        native.load_sounddevice()


@pytest.mark.parametrize("loaded", [None, "libportaudio64bit.dll", "outside/libportaudio64bit.dll"])
def test_refuses_existing_external_module_without_reload(monkeypatch, windows, loaded):
    state, _ = windows
    module = SimpleNamespace(_libname=loaded)
    state.modules["sounddevice"] = module
    monkeypatch.setattr(native, "import_module", lambda _: pytest.fail("must not reload"))
    with pytest.raises(native.NativeAudioError, match="既に読み込まれ"):
        native.load_sounddevice()
    assert state.modules["sounddevice"] is module


def test_reuses_existing_matching_module(monkeypatch, windows):
    state, dll = windows
    module = SimpleNamespace(_libname=str(dll))
    state.modules["sounddevice"] = module
    monkeypatch.setattr(native, "import_module", lambda _: pytest.fail("must not reload"))
    assert native.load_sounddevice() is module


def test_verifies_newly_imported_library_and_restores_lookup(monkeypatch, windows, tmp_path):
    original = native.ctypes.util.find_library
    monkeypatch.setattr(native, "import_module", lambda _: SimpleNamespace(_libname=str(tmp_path / "external.dll")))
    with pytest.raises(native.NativeAudioError, match="同梱版以外"):
        native.load_sounddevice()
    assert native.ctypes.util.find_library is original


def test_frozen_uses_meipass_without_source_package_lookup(monkeypatch, windows, tmp_path):
    state, _ = windows
    state.frozen = True
    state._MEIPASS = str(tmp_path / "_internal")
    dll = Path(state._MEIPASS) / "_sounddevice_data" / "portaudio-binaries" / "libportaudio64bit.dll"
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"placeholder")
    module = SimpleNamespace(_libname=str(dll.resolve()))
    monkeypatch.setattr(native, "find_spec", lambda _: pytest.fail("must use frozen path"))
    monkeypatch.setattr(native, "import_module", lambda _: module)
    assert native.load_sounddevice() is module


def test_frozen_missing_meipass_fails_without_fallback(monkeypatch, windows):
    state, _ = windows
    state.frozen = True
    monkeypatch.setattr(native, "find_spec", lambda _: pytest.fail("must not fall back"))
    with pytest.raises(native.NativeAudioError, match="場所を確認"):
        native.load_sounddevice()


@pytest.mark.parametrize("spec", [None, SimpleNamespace(submodule_search_locations=None)])
def test_missing_source_data_package_is_clear_error(monkeypatch, windows, spec):
    monkeypatch.setattr(native, "find_spec", lambda _: spec)
    with pytest.raises(native.NativeAudioError, match="同梱の音声ライブラリがありません"):
        native.load_sounddevice()


def test_non_windows_uses_normal_import_after_policy_check(monkeypatch):
    monkeypatch.setattr(native, "sys", SimpleNamespace(platform="linux"))
    calls = []
    monkeypatch.setattr(native, "ensure_no_prohibited_dependencies", lambda: calls.append("policy"))
    monkeypatch.setattr(native, "_bundled_portaudio", lambda: pytest.fail("not Windows"))
    module = SimpleNamespace(_libname="libportaudio.so.2")

    def importer(name):
        calls.append(name)
        return module

    original = native.ctypes.util.find_library
    monkeypatch.setattr(native, "import_module", importer)
    assert native.load_sounddevice() is module
    assert calls == ["policy", "sounddevice"]
    assert native.ctypes.util.find_library is original


def test_policy_failure_stops_before_native_import(monkeypatch, windows):
    def prohibited():
        raise RuntimeError("prohibited dependency")

    monkeypatch.setattr(native, "ensure_no_prohibited_dependencies", prohibited)
    monkeypatch.setattr(native, "import_module", lambda _: pytest.fail("must not import"))
    with pytest.raises(RuntimeError, match="prohibited dependency"):
        native.load_sounddevice()

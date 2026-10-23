from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from simple_speech_to_text import cli


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ensure_no_prohibited_dependencies", lambda: None)
    monkeypatch.setattr(cli, "ensure_models", lambda *a, **k: tmp_path)
    return tmp_path


def test_dependency_rejection_happens_before_model_or_audio_access(monkeypatch, capsys):
    def reject():
        raise cli.ProhibitedDependencyError("blocked dependency")

    monkeypatch.setattr(cli, "ensure_no_prohibited_dependencies", reject)
    monkeypatch.setattr(cli, "ensure_models", lambda *a, **k: pytest.fail("model access"))
    assert cli.main([]) == 1
    assert "blocked dependency" in capsys.readouterr().err


def test_one_utterance_is_transcribed_once(prepared, monkeypatch, capsys):
    from simple_speech_to_text import audio, recognition

    samples = np.ones(16000, dtype=np.float32)
    calls = []

    def capture(directory, **kwargs):
        calls.append(("capture", directory, kwargs["device"]))
        return audio.CaptureResult(samples)

    def transcribe(recording, directory, **kwargs):
        assert recording is samples
        calls.append(("transcribe", directory, kwargs["language"]))
        return "会議は十時からです。"

    monkeypatch.setattr(audio, "capture_utterance", capture)
    monkeypatch.setattr(recognition, "transcribe", transcribe)
    assert cli.main(["--device", "3"]) == 0
    assert calls == [("capture", prepared, 3), ("transcribe", prepared, "ja")]
    assert "会議は十時からです。" in capsys.readouterr().out


def test_empty_transcript_reports_failure(prepared, monkeypatch, capsys):
    from simple_speech_to_text import audio, recognition

    monkeypatch.setattr(audio, "capture_utterance", lambda *a, **k: audio.CaptureResult(np.ones(16000)))
    monkeypatch.setattr(recognition, "transcribe", lambda *a, **k: "")
    assert cli.main([]) == 1
    assert "認識できません" in capsys.readouterr().out


def test_interrupt_does_not_call_recognizer(prepared, monkeypatch):
    from simple_speech_to_text import audio, recognition

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "capture_utterance", interrupt)
    monkeypatch.setattr(recognition, "transcribe", lambda *a, **k: pytest.fail("transcribe"))
    assert cli.main([]) == 130


def test_self_test_never_downloads_or_opens_mic(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ensure_no_prohibited_dependencies", lambda: None)
    calls = []

    def models(directory, **kwargs):
        assert not kwargs["download"]
        return directory

    monkeypatch.setattr(cli, "ensure_models", models)
    monkeypatch.setattr(cli, "self_test", lambda directory: calls.append(directory))
    assert cli.main(["--self-test", "--model-dir", str(tmp_path)]) == 0
    assert calls == [tmp_path]


def test_missing_portable_assets_do_not_download(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ensure_no_prohibited_dependencies", lambda: None)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    def missing(directory, **kwargs):
        assert not kwargs["download"]
        raise cli.ModelError("missing")

    monkeypatch.setattr(cli, "ensure_models", missing)
    assert cli.main(["--no-pause", "--model-dir", str(tmp_path)]) == 1


def test_list_devices_does_not_prepare_model(prepared, monkeypatch, capsys):
    from simple_speech_to_text import native_audio

    monkeypatch.setattr(cli, "ensure_models", lambda *a, **k: pytest.fail("model access"))
    monkeypatch.setattr(native_audio, "load_sounddevice", lambda: SimpleNamespace(query_devices=lambda: [
        {"name": "output only", "max_input_channels": 0},
        {"name": "headset", "max_input_channels": 1},
    ]))
    assert cli.main(["--list-devices"]) == 0
    assert capsys.readouterr().out == "1: headset\n"


def test_device_query_failure_has_no_traceback(prepared, monkeypatch, capsys):
    from simple_speech_to_text import native_audio

    class DeviceError(Exception):
        pass

    def broken():
        raise DeviceError("device unavailable")

    monkeypatch.setattr(native_audio, "load_sounddevice", lambda: SimpleNamespace(
        query_devices=broken, PortAudioError=DeviceError,
    ))
    assert cli.main(["--list-devices"]) == 1
    assert "マイク一覧を取得できません" in capsys.readouterr().err


def test_model_only_mode_exits_without_native_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ensure_no_prohibited_dependencies", lambda: None)

    def models(directory, **kwargs):
        assert directory == Path(tmp_path)
        assert kwargs["download"]
        return directory

    monkeypatch.setattr(cli, "ensure_models", models)
    assert cli.main(["--download-model", "--model-dir", str(tmp_path)]) == 0

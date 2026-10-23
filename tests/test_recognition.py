import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from simple_speech_to_text import recognition
from simple_speech_to_text._vendor.giji_v050 import pipeline


def test_vendor_matches_pinned_source_hashes():
    directory = Path(pipeline.__file__).parent
    provenance = json.loads((directory / "PROVENANCE.json").read_text())
    assert provenance["commit"] == "d5745a3152e7c022704e64d2421196892ebfbe49"
    for filename, info in provenance["files"].items():
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == info["sha256"]


def test_empty_audio_does_not_load_model(monkeypatch):
    def fail(*_args, **_kwargs):
        pytest.fail("empty audio must not load the native model")

    monkeypatch.setattr(pipeline.Pipeline, "_transcribe", fail)
    assert recognition.transcribe(np.empty(0, dtype=np.float32), Path("missing")) == ""


def test_every_pcm16_sample_roundtrips_and_original_entrypoint_runs(monkeypatch):
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    samples = pcm.astype(np.float32) / 32768.0
    visited = []
    expected = [{"start": 0.0, "end": 1.0, "speaker": "SELF", "text": "こんにちは。"}]

    def original(self, directory, settings):
        visited.append(directory)
        assert self.models.whisper_path(settings) == Path("verified-models")
        assert self.secrets is None
        assert settings == {"language": "ja"}
        with wave.open(str(directory / "microphone.wav"), "rb") as source:
            assert source.getframerate() == 48_000
            assert source.getsampwidth() == 2
            assert source.getnchannels() == 1
            assert source.readframes(source.getnframes()) == pcm.astype("<i2").tobytes()
        with wave.open(str(directory / "system.wav"), "rb") as source:
            assert source.getnframes() == 0
            assert source.getcomptype() == "NONE"
        return expected

    monkeypatch.setattr(pipeline.Pipeline, "_transcribe", original)
    actual = recognition.transcribe_segments(samples, Path("verified-models"), sample_rate=48_000)
    assert actual is expected
    assert len(visited) == 1
    assert not visited[0].exists()


def test_no_resampling_before_original_pcm_reader(monkeypatch):
    samples = np.full(44_100 * 2 + 123, 0.25, dtype=np.float32)
    seen_lengths = []

    def original(_self, directory, _settings):
        with wave.open(str(directory / "microphone.wav"), "rb") as source:
            assert source.getnframes() == len(samples)
            assert source.getframerate() == 44_100
        seen_lengths.extend(len(block) for block in pipeline.pcm16_blocks(directory / "microphone.wav"))
        return []

    monkeypatch.setattr(pipeline.Pipeline, "_transcribe", original)
    assert recognition.transcribe_segments(samples, Path("models"), sample_rate=44_100) == []
    assert seen_lengths == [16_000, 16_000, 45]


def test_pipeline_failure_removes_audio_and_sanitizes_message(monkeypatch):
    visited = []

    def fail(_self, directory, _settings):
        visited.append(directory)
        assert (directory / "microphone.wav").exists()
        raise RuntimeError("private audio contents and native details")

    monkeypatch.setattr(pipeline.Pipeline, "_transcribe", fail)
    with pytest.raises(recognition.RecognitionError, match="文字起こしに失敗") as caught:
        recognition.transcribe(np.ones(160, dtype=np.float32), Path("models"))
    assert "private" not in str(caught.value)
    assert len(visited) == 1
    assert not visited[0].exists()


def test_plain_text_comes_from_original_segments(monkeypatch):
    monkeypatch.setattr(pipeline.Pipeline, "_transcribe", lambda *_args: [
        {"text": " こんにちは。"}, {"text": "続きです。 "},
    ])
    assert recognition.transcribe(np.ones(160), Path("models")) == "こんにちは。続きです。"


def test_english_chunks_keep_word_separator():
    assert recognition._join_text(["hello", "world"]) == "hello world"


@pytest.mark.parametrize("samples", [np.array([float("nan")]), np.array([[0.1, 0.2]])], ids=["nan", "stereo"])
def test_invalid_audio_does_not_enter_native_decoder(monkeypatch, samples):
    def fail(*_args, **_kwargs):
        pytest.fail("invalid input reached native decoder")

    monkeypatch.setattr(pipeline.Pipeline, "_transcribe", fail)
    with pytest.raises(recognition.RecognitionError):
        recognition.transcribe(samples, Path("models"))


def test_non_original_thread_setting_is_rejected():
    with pytest.raises(ValueError, match="exactly 4"):
        recognition.transcribe(np.zeros(160), Path("models"), threads=2)


def test_temporary_directory_falls_back_when_tmpfs_unavailable(monkeypatch, tmp_path):
    actual = recognition.tempfile.TemporaryDirectory
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "dir" in kwargs:
            raise PermissionError("tmpfs unavailable")
        return actual(prefix=kwargs["prefix"], dir=tmp_path)

    monkeypatch.setattr(recognition.sys, "platform", "linux")
    monkeypatch.setattr(recognition.Path, "is_dir", lambda _self: True)
    monkeypatch.setattr(recognition.tempfile, "TemporaryDirectory", create)
    with recognition._temporary_audio_directory() as directory:
        assert Path(directory).parent == tmp_path
    assert calls == [{"prefix": "stt-pcm-", "dir": "/dev/shm"}, {"prefix": "stt-pcm-"}]
    assert not Path(directory).exists()

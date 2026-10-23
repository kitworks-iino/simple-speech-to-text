from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from simple_speech_to_text import recognition


def test_long_audio_split_preserves_every_sample_exactly_once():
    samples = np.arange(16_000 * 75, dtype=np.float32)
    chunks = list(recognition.speech_chunks(samples))
    assert len(chunks) >= 3
    assert all(0 < len(chunk) <= 16_000 * 28 for chunk in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), samples)


def test_split_prefers_quiet_boundary():
    samples = np.ones(16_000 * 35, dtype=np.float32)
    samples[16_000 * 26:16_000 * 26 + 640] = 0
    first = next(recognition.speech_chunks(samples))
    assert 26 <= len(first) / 16_000 < 26.04


def test_empty_audio_does_not_load_model():
    assert recognition.transcribe(np.empty(0, dtype=np.float32), Path("missing")) == ""


def install_recognizer(monkeypatch, texts):
    streams = []
    config = {}
    text_iterator = iter(texts)

    class Stream:
        result = None

        def accept_waveform(self, rate, samples):
            assert rate == 16_000
            self.samples = samples.copy()

    class Recognizer:
        def create_stream(self):
            stream = Stream()
            streams.append(stream)
            return stream

        def decode_stream(self, stream):
            stream.result = SimpleNamespace(text=next(text_iterator))

    def factory(**kwargs):
        config.update(kwargs)
        return Recognizer()

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        OfflineRecognizer=SimpleNamespace(from_whisper=factory),
    ))
    return streams, config


def test_japanese_transcription_uses_local_cpu_and_bounded_chunks(monkeypatch):
    streams, config = install_recognizer(monkeypatch, [" こんにちは。", "続きです。 "])
    samples = np.zeros(16_000 * 35, dtype=np.float32)
    result = recognition.transcribe(samples, Path("local-models"))
    assert result == "こんにちは。続きです。"
    assert len(streams) == 2
    assert config["provider"] == "cpu"
    assert config["language"] == "ja"
    assert config["task"] == "transcribe"
    assert config["num_threads"] == 4
    assert config["encoder"] == str(Path("local-models") / "encoder.int8.onnx")
    np.testing.assert_array_equal(np.concatenate([stream.samples for stream in streams]), samples)


def test_alternate_pcm_rate_is_resampled_before_decoding(monkeypatch):
    streams, _ = install_recognizer(monkeypatch, [" hello "])
    result = recognition.transcribe(np.ones(48_000, dtype=np.float32), Path("models"), sample_rate=48_000)
    assert result == "hello"
    assert len(streams[0].samples) == 16_000


def test_english_chunks_keep_word_separator():
    assert recognition._join_text(["hello", "world"]) == "hello world"


def test_decoder_failure_becomes_readable_error(monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("native-library details")

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        OfflineRecognizer=SimpleNamespace(from_whisper=fail),
    ))
    with pytest.raises(recognition.RecognitionError, match="文字起こしに失敗"):
        recognition.transcribe(np.zeros(160), Path("models"))


def test_invalid_audio_does_not_enter_native_decoder():
    with pytest.raises(recognition.RecognitionError):
        recognition.transcribe(np.array([float("nan")]), Path("models"))

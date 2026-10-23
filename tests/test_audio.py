from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from simple_speech_to_text import audio


class FakeStream:
    def __init__(self, *, overflow=False, error=None):
        self.overflow = overflow
        self.error = error
        self.reads = 0
        self.closed = False

    def read(self, frames):
        if self.error:
            raise self.error
        self.reads += 1
        return np.full((frames, 1), 0.05, dtype=np.float32), self.overflow

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True


class FakeVad:
    def __init__(self, states):
        self.states = iter(states)
        self.detected = False
        self.windows = 0

    def accept_waveform(self, samples):
        assert samples.shape == (512,)
        assert samples.dtype == np.float32
        self.detected = next(self.states, self.detected)
        self.windows += 1

    def is_speech_detected(self):
        return self.detected


def capture_fake(stream, vad, **kwargs):
    return audio._capture_stream(
        stream, vad, source_rate=16_000, block_size=320,
        start_timeout=kwargs.pop("start_timeout", 1),
        max_seconds=kwargs.pop("max_seconds", 1), **kwargs,
    )


def test_one_utterance_stops_on_silence_after_speech():
    stream = FakeStream()
    messages = []
    result = capture_fake(stream, FakeVad([False, True, True, False]), on_status=messages.append)
    assert result.sample_rate == 16_000
    assert result.samples.ndim == 1
    assert result.samples.dtype == np.float32
    assert 0 < result.duration_seconds < 1
    assert not result.limit_reached
    assert len(messages) == 1
    assert stream.reads == 7


def test_initial_silence_does_not_finish_successfully():
    with pytest.raises(audio.NoSpeechError):
        capture_fake(FakeStream(), FakeVad([False]), start_timeout=0.1)


def test_continuous_speech_stops_at_safety_limit():
    result = capture_fake(FakeStream(), FakeVad([True]), max_seconds=0.2)
    assert result.limit_reached
    assert 0.2 <= result.duration_seconds < 0.4


def test_long_speech_is_not_stopped_at_whisper_chunk_limit():
    # 26 seconds of continuous speech followed by the VAD endpoint.
    result = capture_fake(FakeStream(), FakeVad([True] * 813 + [False]), max_seconds=40)
    assert result.duration_seconds > 26
    assert not result.limit_reached


def test_initial_wait_is_trimmed_but_onset_preroll_is_preserved():
    result = capture_fake(
        FakeStream(), FakeVad([False] * 63 + [True] * 10 + [False]), start_timeout=5,
    )
    assert 0.5 < result.duration_seconds < 1


def test_overflow_fails_instead_of_silently_dropping_audio():
    with pytest.raises(audio.AudioError, match="欠落"):
        capture_fake(FakeStream(overflow=True), FakeVad([True]))


def test_malformed_microphone_buffer_is_rejected():
    stream = SimpleNamespace(read=lambda count: (np.zeros((count, 2), dtype=np.float32), False))
    with pytest.raises(audio.AudioError):
        capture_fake(stream, FakeVad([True]))


def test_native_rate_capture_returns_16khz():
    result = audio._capture_stream(
        FakeStream(), FakeVad([False, True, True, False]),
        source_rate=48_000, block_size=960, start_timeout=1, max_seconds=1,
    )
    assert result.sample_rate == 16_000
    assert len(result.samples) == 7 * 320


def fake_sounddevice(monkeypatch, stream, *, reject_16k=False):
    calls = []

    class PortAudioError(Exception):
        pass

    def check(**kwargs):
        calls.append(kwargs)
        if reject_16k and kwargs["samplerate"] == 16_000:
            raise PortAudioError("Unsupported rate")

    def create(**kwargs):
        calls.append(kwargs)
        return stream

    module = SimpleNamespace(
        PortAudioError=PortAudioError,
        query_devices=lambda *args: {"max_input_channels": 1, "default_samplerate": 48_000},
        check_input_settings=check, InputStream=create,
    )
    monkeypatch.setattr(audio.native_audio, "load_sounddevice", lambda: module)
    monkeypatch.setattr(audio, "_make_vad", lambda *args: FakeVad([False, True, True, False]))
    return calls


def test_capture_closes_microphone_and_falls_back_to_native_rate(monkeypatch):
    stream = FakeStream()
    calls = fake_sounddevice(monkeypatch, stream, reject_16k=True)
    messages = []
    result = audio.capture_utterance(Path("models"), device=2, on_status=messages.append)
    assert stream.closed
    assert result.sample_rate == 16_000
    assert [call["samplerate"] for call in calls] == [16_000, 48_000, 48_000]
    assert calls[-1]["channels"] == 1
    assert calls[-1]["device"] == 2
    assert len(messages) == 2


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("Disconnected")])
def test_capture_closes_microphone_on_cancel_and_device_error(monkeypatch, error):
    stream = FakeStream(error=error)
    fake_sounddevice(monkeypatch, stream)
    expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else audio.AudioError
    with pytest.raises(expected):
        audio.capture_utterance(Path("models"))
    assert stream.closed


def test_bundled_driver_error_keeps_actionable_message(monkeypatch):
    def fail():
        raise audio.native_audio.NativeAudioError("同梱PortAudioが見つかりません。ZIPを再展開してください。")

    monkeypatch.setattr(audio.native_audio, "load_sounddevice", fail)
    with pytest.raises(audio.AudioError, match="ZIPを再展開"):
        audio.capture_utterance(Path("models"))


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_capture_invalid_limits_do_not_open_device(value):
    with pytest.raises(ValueError):
        audio.capture_utterance(Path("models"), silence_seconds=value)


def write_wav(path, *, sample_rate=44_100, channels=2, width=2):
    samples = (np.sin(np.arange(sample_rate) * 2 * np.pi * 440 / sample_rate) * 16_000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(sample_rate)
        output.writeframes(np.repeat(samples[:, None], channels, axis=1).tobytes())


def test_pcm_wav_downmix_and_resampling(tmp_path):
    path = tmp_path / "stereo.wav"
    write_wav(path)
    samples, rate = audio.read_pcm_wave(path)
    assert rate == 16_000
    assert samples.dtype == np.float32
    assert samples.shape == (16_000,)
    assert np.max(np.abs(samples)) < 0.6
    spectrum = np.abs(np.fft.rfft(samples))
    assert np.argmax(spectrum) == 440


def test_resampling_filters_above_target_nyquist():
    rate = 48_000
    signal = np.sin(np.arange(rate) * 2 * np.pi * 12_000 / rate).astype(np.float32)
    converted = audio.resample(signal, rate)
    assert np.sqrt(np.mean(converted[100:-100] ** 2)) < 0.01


@pytest.mark.parametrize("kind", ["compressed", "wrong_width", "truncated"])
def test_wav_rejects_unsupported_or_broken_input(tmp_path, kind):
    path = tmp_path / "input.wav"
    if kind == "compressed":
        path.write_bytes(b"ID3compressed audio is not PCM")
    elif kind == "wrong_width":
        write_wav(path, width=1)
    else:
        write_wav(path)
        path.write_bytes(path.read_bytes()[:-100])
    with pytest.raises(audio.AudioError):
        audio.read_pcm_wave(path)


def test_vad_min_speech_and_endpoint_configuration(monkeypatch):
    kwargs_seen = {}

    class Config:
        def __init__(self, **kwargs):
            kwargs_seen.update(kwargs)

        def validate(self):
            return True

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        VadModelConfig=Config,
        SileroVadModelConfig=lambda **kwargs: kwargs,
        VoiceActivityDetector=lambda config, **kwargs: kwargs,
    ))
    result = audio._make_vad(Path("models"), 1.0, 120)
    assert kwargs_seen["silero_vad"]["min_speech_duration"] == 0.2
    assert kwargs_seen["silero_vad"]["min_silence_duration"] == 1.0
    assert kwargs_seen["silero_vad"]["max_speech_duration"] > 120
    assert result["buffer_size_in_seconds"] >= 120

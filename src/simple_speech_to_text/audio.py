"""Microphone PCM capture with local, silence-based utterance detection.

No encoded media, external executables, or file recording are involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable
import wave

import numpy as np

from . import native_audio


SAMPLE_RATE = 16_000
VAD_WINDOW = 512


class AudioError(RuntimeError):
    """An actionable capture or PCM input error."""


class NoSpeechError(AudioError):
    """No speech was detected before the initial waiting timeout."""


@dataclass(frozen=True)
class CaptureResult:
    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE
    limit_reached: bool = False

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def resample(samples: np.ndarray, source_rate: int) -> np.ndarray:
    """Convert uncompressed mono float PCM with an anti-aliasing filter."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise AudioError("音声データが不正です。")
    if source_rate <= 0:
        raise AudioError("サンプリング周波数が不正です。")
    if source_rate == SAMPLE_RATE or not len(samples):
        return samples
    from scipy.signal import resample_poly

    divisor = math.gcd(source_rate, SAMPLE_RATE)
    return resample_poly(samples, SAMPLE_RATE // divisor, source_rate // divisor).astype(np.float32)


def _make_vad(model_dir: Path, silence_seconds: float, max_seconds: float):
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=str(model_dir / "silero_vad.onnx"),
            threshold=0.5,
            min_silence_duration=silence_seconds,
            min_speech_duration=0.2,
            window_size=VAD_WINDOW,
            # Endpointing must not be mistaken for Whisper's chunk size limit.
            max_speech_duration=max_seconds + 10,
        ),
        sample_rate=SAMPLE_RATE,
        num_threads=1,
        provider="cpu",
        debug=False,
    )
    if not config.validate():
        raise AudioError("音声検出モデルを読み込めません。モデルを再取得してください。")
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=max_seconds + 20)


def _capture_stream(
    stream,
    vad,
    *,
    source_rate: int,
    block_size: int,
    start_timeout: float,
    max_seconds: float,
    on_status: Callable[[str], None] | None = None,
) -> CaptureResult:
    """Read a blocking stream; separated from device creation for deterministic tests."""
    blocks: list[np.ndarray] = []
    pending = np.empty(0, dtype=np.float32)
    frames = 0
    first_speech_frame: int | None = None
    done = False
    limit_reached = False
    while not done:
        data, overflowed = stream.read(block_size)
        if overflowed:
            raise AudioError("録音が追いつかず音声が欠落しました。他の重い処理を閉じて再実行してください。")
        block = np.asarray(data, dtype=np.float32)
        if block.ndim == 2 and block.shape[1] == 1:
            block = block[:, 0]
        if block.ndim != 1 or len(block) != block_size or not np.isfinite(block).all():
            raise AudioError("マイクから正常な音声を取得できませんでした。")
        blocks.append(block.copy())
        frames += len(block)
        # Per-block resampling is only used to decide speech boundaries. The
        # final audio is resampled once, with a continuous anti-aliasing filter.
        pending = np.concatenate((pending, resample(block, source_rate)))
        while len(pending) >= VAD_WINDOW:
            vad.accept_waveform(pending[:VAD_WINDOW])
            pending = pending[VAD_WINDOW:]
            if vad.is_speech_detected():
                if first_speech_frame is None:
                    first_speech_frame = frames
                    if on_status:
                        on_status("録音中です。話し終わったら、そのまま少しお待ちください。")
            elif first_speech_frame is not None:
                done = True
                break
        if first_speech_frame is None:
            if frames / source_rate >= start_timeout:
                raise NoSpeechError("音声を検出できませんでした。マイクとWindowsのマイク許可を確認してください。")
        elif (frames - first_speech_frame) / source_rate >= max_seconds:
            limit_reached = True
            done = True
    # Preserve the consonants preceding the VAD's minimum-speech confirmation.
    start = max(0, (first_speech_frame or 0) - round(source_rate * 0.5))
    samples = resample(np.concatenate(blocks)[start:], source_rate)
    return CaptureResult(samples=samples, limit_reached=limit_reached)


def capture_utterance(
    model_dir: Path,
    *,
    device: int | str | None = None,
    silence_seconds: float = 1.0,
    start_timeout: float = 30.0,
    max_seconds: float = 120.0,
    on_status: Callable[[str], None] | None = None,
) -> CaptureResult:
    """Capture one utterance from the default microphone, entirely in memory.

    A one-second silence ends capture after speech has started. The initial
    timeout and hard duration limit also apply if the room remains noisy.
    KeyboardInterrupt is intentionally allowed to propagate; context management
    always closes the microphone first.
    """
    for value in (silence_seconds, start_timeout, max_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Capture limits must be positive finite seconds")
    if max_seconds > 600:
        raise ValueError("max_seconds must be at most 600")
    try:
        sd = native_audio.load_sounddevice()
        vad = _make_vad(Path(model_dir), silence_seconds, max_seconds)
        info = sd.query_devices(device, "input")
        if info["max_input_channels"] < 1:
            raise AudioError("入力マイクがありません。マイクを接続してください。")
        source_rate = SAMPLE_RATE
        try:
            sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=source_rate)
        except sd.PortAudioError:
            source_rate = round(info["default_samplerate"])
            sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=source_rate)
        # Standard audio rates yield exact 20ms frames. Other rates are accepted
        # only if this remains an exact conversion, avoiding long-term drift.
        if source_rate % 50:
            raise AudioError("このマイクのサンプリング周波数に対応していません。Windowsで44.1/48 kHzを選んでください。")
        block_size = source_rate // 50
        # Import/initialize the native-rate converter before opening the mic.
        # A cold SciPy import during live reads can otherwise overflow PortAudio.
        resample(np.zeros(block_size, dtype=np.float32), source_rate)
        with sd.InputStream(
            device=device, channels=1, samplerate=source_rate,
            dtype="float32", blocksize=block_size,
        ) as stream:
            if on_status:
                on_status("準備ができました。マイクに向かって話してください。")
            return _capture_stream(
                stream, vad, source_rate=source_rate, block_size=block_size,
                start_timeout=start_timeout, max_seconds=max_seconds, on_status=on_status,
            )
    except (AudioError, KeyboardInterrupt):
        raise
    except native_audio.NativeAudioError as exc:
        raise AudioError(str(exc)) from exc
    except ImportError as exc:
        raise AudioError("録音ライブラリがありません。セットアップを再実行してください。") from exc
    except Exception as exc:
        raise AudioError("録音を開始・継続できませんでした。マイク接続・Windowsのマイク許可・他アプリの占有を確認してください。") from exc


def read_pcm_wave(path: Path) -> tuple[np.ndarray, int]:
    """Read only 16-bit PCM WAV; compressed formats are deliberately unsupported."""
    try:
        with wave.open(str(path), "rb") as recording:
            if recording.getsampwidth() != 2 or recording.getcomptype() != "NONE":
                raise AudioError("16bit PCM形式のWAVファイルを指定してください。")
            channels = recording.getnchannels()
            sample_rate = recording.getframerate()
            frames = recording.getnframes()
            if channels not in (1, 2) or not 8_000 <= sample_rate <= 192_000:
                raise AudioError("WAVはモノラルまたはステレオ、8〜192 kHzで指定してください。")
            if frames > sample_rate * 600:
                raise AudioError("WAVファイルは10分以内にしてください。")
            raw = recording.readframes(frames)
            if len(raw) != frames * channels * 2:
                raise AudioError("WAVファイルの音声データが途中で切れています。")
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels).mean(axis=1)
        return resample(samples / 32768.0, sample_rate), SAMPLE_RATE
    except AudioError:
        raise
    except (OSError, ValueError, EOFError, wave.Error) as exc:
        raise AudioError("WAVファイルを読み込めません。16bit PCM形式を指定してください。") from exc

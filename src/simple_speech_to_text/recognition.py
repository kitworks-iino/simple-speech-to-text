"""Run the unmodified giji-automemo v0.5.0 ASR pipeline on microphone PCM."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import wave

import numpy as np


SAMPLE_RATE = 16_000


class RecognitionError(RuntimeError):
    """A local speech recognition error."""


class _LocalModels:
    """Resolve verified models without invoking the upstream model downloader."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def whisper_path(self, _settings: dict) -> Path:
        return self.directory


def _join_text(parts: list[str]) -> str:
    result = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if result and result[-1].isascii() and result[-1].isalnum() and part[0].isascii() and part[0].isalnum():
            result += " "
        result += part
    return result


def _temporary_audio_directory():
    # Linux demos put the upstream path-based WAV input on tmpfs. Other hosts
    # use their private temporary directory. Both are removed on every exit.
    if sys.platform.startswith("linux") and Path("/dev/shm").is_dir():
        try:
            return tempfile.TemporaryDirectory(prefix="stt-pcm-", dir="/dev/shm")
        except OSError:
            pass
    return tempfile.TemporaryDirectory(prefix="stt-pcm-")


def _write_pcm(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(sample_rate)
        recording.writeframes(pcm)


def transcribe_segments(
    samples: np.ndarray,
    model_dir: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    language: str = "ja",
) -> list[dict]:
    """Call the original Pipeline._transcribe, including its PCM/VAD path.

    Browser PCM16 normalized to float32 round-trips without changing a sample.
    Native microphone float samples are quantized once to the original app's
    PCM16 input format. No resampling takes place before the original pipeline.
    The absent system track is represented by a valid, empty PCM WAV.
    """
    if not isinstance(sample_rate, int) or not 8_000 <= sample_rate <= 192_000:
        raise ValueError("sample_rate must be an integer between 8000 and 192000")
    try:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Expected finite mono PCM samples")
        if not len(samples):
            return []
        # All int16 / 32768 values are exactly representable in float32.
        pcm = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2").tobytes()

        from ._vendor.giji_v050.pipeline import Pipeline

        pipeline = Pipeline(_LocalModels(model_dir), secrets=None)
        with _temporary_audio_directory() as temporary:
            directory = Path(temporary)
            _write_pcm(directory / "microphone.wav", pcm, sample_rate)
            _write_pcm(directory / "system.wav", b"", sample_rate)
            return pipeline._transcribe(directory, {"language": language})
    except ImportError as exc:
        raise RecognitionError("文字起こしライブラリがありません。セットアップを再実行してください。") from exc
    except Exception as exc:
        raise RecognitionError("文字起こしに失敗しました。モデルファイルと空きメモリを確認してください。") from exc


def transcribe(
    samples: np.ndarray,
    model_dir: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    language: str = "ja",
    threads: int = 4,
) -> str:
    """Return the original pipeline's segment text, joined for plain display."""
    if threads != 4:
        raise ValueError("The giji-automemo v0.5.0 pipeline uses exactly 4 threads")
    segments = transcribe_segments(samples, model_dir, sample_rate=sample_rate, language=language)
    return _join_text([segment["text"] for segment in segments])

"""Local ONNX Whisper recognition of uncompressed PCM samples."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .audio import SAMPLE_RATE, resample


MAX_CHUNK_SECONDS = 28


class RecognitionError(RuntimeError):
    """A local speech recognition error."""


def speech_chunks(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Iterator[np.ndarray]:
    """Split long input at the quietest 20ms window in the last four seconds.

    Every sample is preserved exactly once. This avoids an arbitrary 25-second
    recording stop while keeping each decode within Whisper's 30-second window.
    A pause shorter than the capture endpoint may therefore become a chunk edge.
    """
    maximum = sample_rate * MAX_CHUNK_SECONDS
    search = sample_rate * 4
    window = max(1, sample_rate // 50)
    offset = 0
    while len(samples) - offset > maximum:
        begin = offset + maximum - search
        end = offset + maximum
        tail = samples[begin:end]
        frames = tail[:len(tail) // window * window].reshape(-1, window)
        energy = np.mean(np.square(frames, dtype=np.float64), axis=1)
        boundary = begin + int(np.argmin(energy)) * window + window // 2
        yield samples[offset:boundary]
        offset = boundary
    if offset < len(samples):
        yield samples[offset:]


def _join_text(parts: list[str]) -> str:
    result = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Whisper Japanese output is usually unspaced; don't insert artificial
        # spaces between Japanese chunks, but keep Latin words separated.
        if result and result[-1].isascii() and result[-1].isalnum() and part[0].isascii() and part[0].isalnum():
            result += " "
        result += part
    return result


def transcribe(
    samples: np.ndarray,
    model_dir: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    language: str = "ja",
    threads: int = 4,
) -> str:
    """Return plain text. No audio or transcript is sent outside this process."""
    if not 1 <= threads <= 32:
        raise ValueError("threads must be between 1 and 32")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    try:
        samples = resample(samples, sample_rate)
        if not len(samples):
            return ""
        import sherpa_onnx

        model_dir = Path(model_dir)
        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            language=language,
            task="transcribe",
            num_threads=threads,
            provider="cpu",
            debug=False,
        )
        texts = []
        for samples_chunk in speech_chunks(samples):
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples_chunk)
            recognizer.decode_stream(stream)
            texts.append(stream.result.text)
        return _join_text(texts)
    except ImportError as exc:
        raise RecognitionError("文字起こしライブラリがありません。セットアップを再実行してください。") from exc
    except Exception as exc:
        raise RecognitionError("文字起こしに失敗しました。モデルファイルと空きメモリを確認してください。") from exc

"""Optional real-model check using a pinned, public English speech fixture.

Run from an installed source environment:
    python scripts/smoke_speech.py --model-dir build/models/whisper-small

This exercises PCM, actual Silero endpoint detection, and actual Whisper inference.
It does not validate a physical microphone or Japanese recognition accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen

import numpy as np
from scipy.signal import resample_poly

from simple_speech_to_text.audio import NoSpeechError, _capture_stream, _make_vad, read_pcm_wave
from simple_speech_to_text.models import verify_models
from simple_speech_to_text.recognition import transcribe


FIXTURE_URL = (
    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny/resolve/"
    "65176e2deb88badc814a94058666cadccc29b61c/test_wavs/0.wav"
)
FIXTURE_SHA256 = "6bc58a4efdf20daac252b6b1502632601a71efe0308f6757dc1eda34891a7e4f"


class PcmStream:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.position = 0

    def read(self, count: int):
        if self.position >= len(self.samples):
            raise AssertionError("VAD did not stop before the end of the silence tail")
        block = self.samples[self.position:self.position + count]
        self.position += count
        return np.pad(block, (0, count - len(block)))[:, None], False


def check_capture(model_dir: Path, speech: np.ndarray, source_rate: int) -> tuple[dict, np.ndarray]:
    source = np.concatenate((np.zeros(16_000, dtype=np.float32), speech, np.zeros(32_000, dtype=np.float32)))
    if source_rate != 16_000:
        source = resample_poly(source, source_rate // 100, 160).astype(np.float32)
    stream = PcmStream(source)
    result = _capture_stream(
        stream, _make_vad(model_dir, 1.0, 120.0), source_rate=source_rate,
        block_size=source_rate // 50, start_timeout=10, max_seconds=120,
    )
    assert not result.limit_reached
    assert result.duration_seconds >= len(speech) / 16_000 - 0.5
    assert stream.position < len(source), "Capture must finish without reaching the hard timeout"
    return {
        "source_rate": source_rate,
        "speech_seconds": len(speech) / 16_000,
        "captured_seconds": result.duration_seconds,
        "endpoint_seconds": stream.position / source_rate,
        "limit_reached": result.limit_reached,
    }, result.samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, help="Use an already downloaded fixture with the expected SHA-256")
    args = parser.parse_args()
    missing = verify_models(args.model_dir)
    if missing:
        raise SystemExit(f"Missing or invalid models: {', '.join(missing)}")
    if args.fixture:
        raw = args.fixture.read_bytes()
    else:
        with urlopen(FIXTURE_URL, timeout=60) as response:
            raw = response.read(1_000_001)
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, "Fixture SHA-256 mismatch"
    with tempfile.TemporaryDirectory(prefix="simple-stt-spoken-") as directory:
        path = Path(directory) / "speech.wav"
        path.write_bytes(raw)
        speech, _ = read_pcm_wave(path)

    started = time.monotonic()
    capture16, captured = check_capture(args.model_dir, speech, 16_000)
    capture48, _ = check_capture(args.model_dir, speech, 48_000)
    capture44, _ = check_capture(args.model_dir, speech, 44_100)
    # Trim fixture edge silence before repeating; continuous >30-second speech
    # must not be mistaken for the recognizer's shorter decode chunk limit.
    continuous = np.tile(speech[4_800:99_200], 6)
    capture_long, _ = check_capture(args.model_dir, continuous, 16_000)
    assert capture_long["captured_seconds"] > 30
    try:
        _capture_stream(
            PcmStream(np.zeros(32_000, dtype=np.float32)), _make_vad(args.model_dir, 1.0, 120),
            source_rate=16_000, block_size=320, start_timeout=1, max_seconds=120,
        )
    except NoSpeechError:
        silence_rejected = True
    else:
        raise AssertionError("Silence must not be recognized as speech")
    text = transcribe(captured, args.model_dir, language="en")
    normalized = text.lower()
    assert "yellow lamps" in normalized and "quarter" in normalized, f"Unexpected transcription: {text!r}"
    print(json.dumps({
        "fixture_url": FIXTURE_URL,
        "fixture_sha256": FIXTURE_SHA256,
        "captures": [capture16, capture48, capture44, capture_long],
        "silence_rejected": silence_rejected,
        "text": text,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

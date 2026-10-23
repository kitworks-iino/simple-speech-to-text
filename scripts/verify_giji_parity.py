"""Compare the adapter against the pinned original giji v0.5 ASR code.

Uses a local checkout and a pinned public PCM fixture; no network, microphone,
media decoder or speech API is used. This is an optional real-model check:

    python scripts/verify_giji_parity.py --giji-repo ../giji-automemo \
        --model-dir build/models/whisper-small --fixture /path/to/0.wav

The fixture is the same public English recording used by smoke_speech.py.
This checks processing equivalence, not Japanese recognition accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import wave

import numpy as np
from scipy.signal import resample_poly

from simple_speech_to_text._vendor import giji_v050
from simple_speech_to_text.dependency_policy import ensure_no_prohibited_dependencies
from simple_speech_to_text.models import verify_models
from simple_speech_to_text.recognition import transcribe_segments


ORIGINAL_COMMIT = "d5745a3152e7c022704e64d2421196892ebfbe49"
FIXTURE_SHA256 = "6bc58a4efdf20daac252b6b1502632601a71efe0308f6757dc1eda34891a7e4f"


def write_pcm(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        target.writeframes(pcm)


def load_original(repo: Path, destination: Path):
    """Load original files independently of the installed adapter package."""
    vendor = Path(giji_v050.__file__).parent
    provenance = json.loads((vendor / "PROVENANCE.json").read_text(encoding="utf-8"))
    hashes = {}
    package = destination / "_original_giji_parity"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in ("pipeline.py", "models.py", "LICENSE"):
        original_path = name if name == "LICENSE" else f"giji/{name}"
        source = subprocess.check_output([
            "git", "-C", str(repo), "show", f"{ORIGINAL_COMMIT}:{original_path}",
        ])
        if (vendor / name).read_bytes() != source:
            raise AssertionError(f"Vendored {name} differs from the pinned original")
        hashes[name] = hashlib.sha256(source).hexdigest()
        (package / name).write_bytes(source)
    # Retain the recorded metadata in the report while independently checking
    # bytes from the original Git object instead of trusting that metadata.
    spec = importlib.util.spec_from_file_location(
        package.name, package / "__init__.py", submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package.name] = module
    spec.loader.exec_module(module)
    pipeline_spec = importlib.util.spec_from_file_location(f"{package.name}.pipeline", package / "pipeline.py")
    pipeline = importlib.util.module_from_spec(pipeline_spec)
    sys.modules[pipeline_spec.name] = pipeline
    pipeline_spec.loader.exec_module(pipeline)
    return pipeline.Pipeline, hashes, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--giji-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--language", default="en", choices=["en", "ja"])
    args = parser.parse_args()
    ensure_no_prohibited_dependencies()
    invalid = verify_models(args.model_dir)
    if invalid:
        raise SystemExit(f"Missing or invalid models: {', '.join(invalid)}")
    raw = args.fixture.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise SystemExit("Fixture SHA-256 mismatch")
    with wave.open(str(args.fixture), "rb") as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
        speech = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

    started = time.monotonic()
    checks = []
    with tempfile.TemporaryDirectory(prefix="stt-giji-parity-") as temporary:
        base = Path(temporary)
        original_pipeline, hashes, provenance = load_original(args.giji_repo, base)
        recordings = base / "recordings"
        recordings.mkdir()
        original = original_pipeline(SimpleNamespace(whisper_path=lambda _settings: args.model_dir), None)
        for name, rate, samples in (
            ("speech-16000", 16000, speech),
            ("speech-44100", 44100, speech),
            ("speech-48000", 48000, speech),
            ("silence", 16000, np.zeros(16000, dtype=np.float32)),
        ):
            if rate != 16000:
                divisor = math.gcd(rate, 16000)
                samples = resample_poly(samples, rate // divisor, 16000 // divisor)
            pcm = np.clip(np.rint(samples.astype(np.float64) * 32768.0), -32768, 32767).astype("<i2").tobytes()
            # Feed the exact same quantized values to both entry points.
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            write_pcm(recordings / "microphone.wav", pcm, rate)
            write_pcm(recordings / "system.wav", b"", rate)
            expected = original._transcribe(recordings, {"language": args.language})
            actual = transcribe_segments(samples, args.model_dir, sample_rate=rate, language=args.language)
            if actual != expected:
                raise AssertionError(f"Original and adapter segments differ for {name}: {expected!r} != {actual!r}")
            if name == "silence":
                assert not actual, "Digital silence produced a transcript"
            else:
                assert actual and all(item["speaker"] == "SELF" for item in actual)
                if args.language == "en":
                    assert "yellow lamps" in " ".join(item["text"] for item in actual).lower()
            checks.append({"case": name, "sample_rate": rate, "equal": True, "segments": actual})
            print(f"PASS {name}: native segments match", file=sys.stderr, flush=True)
    print(json.dumps({
        "original_commit": ORIGINAL_COMMIT,
        "original_source_sha256": hashes,
        "vendor_provenance": provenance,
        "fixture_sha256": FIXTURE_SHA256,
        "language": args.language,
        "checks": checks,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

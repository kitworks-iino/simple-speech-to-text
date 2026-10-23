"""One utterance, one transcription. No server, accounts or graphical interface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from . import __version__
from .dependency_policy import ProhibitedDependencyError, ensure_no_prohibited_dependencies
from .models import ModelError, default_model_dir, ensure_models


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="喋って、黙ると、文字になる。FFmpeg不要のローカル文字起こし。")
    result.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    actions = result.add_mutually_exclusive_group()
    actions.add_argument("--list-devices", action="store_true", help="利用可能なマイクを表示")
    actions.add_argument("--download-model", action="store_true", help="モデルだけを準備して終了")
    actions.add_argument("--self-test", action="store_true", help="マイクを開かず同梱モデルのCPU推論を検査")
    result.add_argument("--model-dir", type=Path, default=None, help="モデルの保存先")
    result.add_argument("--device", type=int, help="--list-devicesで確認した入力デバイス番号")
    result.add_argument("--language", default="ja", choices=["ja", "en", "auto"], help="認識言語（既定: ja）")
    result.add_argument("--no-pause", action="store_true", help=argparse.SUPPRESS)
    return result


def self_test(directory: Path) -> None:
    """Exercise real native inference, not just import success or a fake model."""
    import numpy as np
    import sherpa_onnx
    from scipy.signal import resample_poly
    from .native_audio import load_sounddevice

    load_sounddevice()  # Load the audited PortAudio; never open a microphone.

    config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=str(directory / "silero_vad.onnx"), threshold=0.5,
            min_silence_duration=1.0, min_speech_duration=0.2,
            window_size=512, max_speech_duration=130,
        ),
        sample_rate=16000, num_threads=1, provider="cpu",
    )
    if not config.validate():
        raise RuntimeError("VAD model configuration failed")
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=5)
    silence = resample_poly(np.zeros(48000, dtype=np.float32), 1, 3)
    for offset in range(0, len(silence) - 511, 512):
        vad.accept_waveform(silence[offset:offset + 512])
    vad.flush()
    if not vad.empty():
        raise RuntimeError("VAD classified digital silence as speech")
    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(directory / "encoder.int8.onnx"),
        decoder=str(directory / "decoder.int8.onnx"),
        tokens=str(directory / "tokens.txt"),
        language="ja", task="transcribe", num_threads=2, provider="cpu", debug=False,
    )
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, silence)
    recognizer.decode_stream(stream)
    if not isinstance(stream.result.text, str):
        raise RuntimeError("Whisper returned an invalid result")
    print("SELF-TEST OK: verified models, PortAudio, resampling, Silero VAD, Whisper CPU inference", flush=True)


def run(args: argparse.Namespace) -> int:
    # Check metadata before importing any native audio library. Do not load the
    # optional ASIO driver; this application uses the normal Windows input path.
    ensure_no_prohibited_dependencies()
    os.environ.pop("SD_ENABLE_ASIO", None)
    if args.list_devices:
        from .native_audio import load_sounddevice

        sd = load_sounddevice()

        found = False
        try:
            devices = sd.query_devices()
        except sd.PortAudioError as exc:
            raise RuntimeError("マイク一覧を取得できません。入力デバイスの接続を確認してください。") from exc
        for index, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                print(f"{index}: {device['name']}")
                found = True
        if not found:
            raise RuntimeError("利用可能なマイクがありません。接続とWindowsのマイク利用許可を確認してください。")
        return 0
    print("準備しています…", flush=True)
    directory = ensure_models(
        args.model_dir or default_model_dir(),
        # Portable runs and self-test must never silently fetch missing assets.
        download=args.download_model or (not getattr(sys, "frozen", False) and not args.self_test),
        on_status=lambda message: print(message, flush=True),
    )
    if args.download_model:
        print(f"モデルの準備ができました: {directory}")
        return 0
    if args.self_test:
        self_test(directory)
        return 0
    from .audio import capture_utterance
    from .recognition import transcribe

    captured = capture_utterance(
        directory, device=args.device, on_status=lambda message: print(message, flush=True),
    )
    if captured.limit_reached:
        print("録音が120秒に達したため、ここまでの音声を文字起こしします。", flush=True)
    print("文字起こし中…", flush=True)
    text = transcribe(
        captured.samples, directory, sample_rate=captured.sample_rate,
        language="" if args.language == "auto" else args.language,
    ).strip()
    if not text:
        print("文字を認識できませんでした。マイクの入力音量を確認し、もう一度話してください。")
        return 1
    print(f"\n{text}\n", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(errors="replace")
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except KeyboardInterrupt:
        print("\n中止しました。", file=sys.stderr)
        result = 130
    except (ModelError, ProhibitedDependencyError, OSError, RuntimeError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        result = 1
    if (getattr(sys, "frozen", False) and sys.stdin.isatty() and not args.no_pause
            and not (args.self_test or args.download_model or args.list_devices)):
        try:
            input("Enterで終了します。")
        except (EOFError, KeyboardInterrupt):
            pass
    return result

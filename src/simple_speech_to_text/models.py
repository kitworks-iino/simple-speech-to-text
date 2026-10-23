"""Pinned model files, downloaded with Python's standard library and verified by SHA-256."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import URLError
from urllib.request import urlopen


REVISION = "8f3c18b358db4d1f2fc1eae49d75cd20989e4309"
BASE = f"https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small/resolve/{REVISION}"
MODEL_FILES = {
    "encoder.int8.onnx": (
        f"{BASE}/small-encoder.int8.onnx",
        "4cbe7b22fa9026b843b60a68640c747de05bafb1a11b57edc0e66c232d9f33a9",
    ),
    "decoder.int8.onnx": (
        f"{BASE}/small-decoder.int8.onnx",
        "acad50b5c782696e91b55914cc5ab4f756f1532f76e22aa6fc615f39fb69a8ee",
    ),
    "tokens.txt": (
        f"{BASE}/small-tokens.txt",
        "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126",
    ),
    "silero_vad.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    ),
}


class ModelError(RuntimeError):
    pass


def default_model_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "models" / "whisper-small"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "simple-speech-to-text" / "models" / "whisper-small"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_models(directory: Path) -> list[str]:
    """Check real contents, not a marker left over from an interrupted download."""
    return [
        name for name, (_, expected) in MODEL_FILES.items()
        if not (directory / name).is_file() or sha256(directory / name) != expected
    ]


def ensure_models(directory: Path, *, download: bool = True, on_status=print) -> Path:
    directory = Path(directory)
    missing = verify_models(directory)
    if not missing:
        return directory
    if not download:
        raise ModelError(
            "同梱モデルが見つからないか破損しています。ZIPをすべて新しいフォルダーへ展開してください。"
            f" 対象: {', '.join(missing)}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    for name in missing:
        url, expected = MODEL_FILES[name]
        on_status(f"モデルを準備しています: {name}（初回のみ）")
        temporary: Path | None = None
        try:
            # The partial file stays on the destination filesystem for atomic replace.
            with tempfile.NamedTemporaryFile(prefix=".download-", dir=directory, delete=False) as output:
                temporary = Path(output.name)
                digest = hashlib.sha256()
                with urlopen(url, timeout=60) as response:
                    if response.geturl().split(":", 1)[0].lower() != "https":
                        raise ModelError("モデルのダウンロード先がHTTPSではありません。")
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise ModelError(f"モデルのSHA-256が一致しません: {name}")
            temporary.replace(directory / name)
        except (OSError, URLError) as exc:
            raise ModelError(
                f"モデルを取得できませんでした: {name}。ネット接続・空き容量を確認してください。"
                " モデル同梱のWindows用ZIPなら取得は不要です。"
            ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return directory

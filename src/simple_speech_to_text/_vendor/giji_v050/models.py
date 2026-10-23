"""Explicit, local model downloads. Importing this module never downloads a model."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Callable

import httpx


WHISPER_ENGINE = "sherpa-onnx-whisper-int8-v1"
WHISPER_MODEL_RELEASES = {
    "tiny": {
        "revision": "65176e2deb88badc814a94058666cadccc29b61c",
        "encoder_sha256": "d24fb083ae3b1041fc24e97971d60e280c9342201fbb67b0ab428a8b4a51a434",
        "decoder_sha256": "d2fece8dd42771f1df975c6c0445770d0c292bf7547c2cae04a6c0cc57540925",
    },
    "base": {
        "revision": "bb53ee204431c90d314c1cc08d28d23e5b7927cc",
        "encoder_sha256": "0b8fb1304b6109976038efff5ace81720e00386f3ff6b54ee8c75291ca0a1e11",
        "decoder_sha256": "9759d217388a01b3a4c7c15533201067b48ae819c4daafc8624e64b9409dc02d",
    },
    "small": {
        "revision": "8f3c18b358db4d1f2fc1eae49d75cd20989e4309",
        "encoder_sha256": "4cbe7b22fa9026b843b60a68640c747de05bafb1a11b57edc0e66c232d9f33a9",
        "decoder_sha256": "acad50b5c782696e91b55914cc5ab4f756f1532f76e22aa6fc615f39fb69a8ee",
    },
    "medium": {
        "revision": "8c31d28503847560985df21f90e14f0c736e075e",
        "encoder_sha256": "1c54582b4d829de0089f6cb63bbbdb3bf7555398bacaf855fbecf1a84dfd193e",
        "decoder_sha256": "595d00a338a365a7bfa0ca7f296cabc639583bef770ab6130df90f49a6412747",
    },
    "large-v3": {
        "revision": "2a6507094dd6020d939d78e3f1834a1d06267fca",
        "encoder_sha256": "d531cf17248acc43e8c09b472a0877055e770877857a5332fc1304b36534ec85",
        "decoder_sha256": "ebc6bfd88e162a46cb3edee8a7e727e1dcbc65cabecb19e2573695e4d495e1af",
    },
}
WHISPER_MODELS = set(WHISPER_MODEL_RELEASES)
WHISPER_TOKENS_SHA256 = "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"
SILERO_VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
SILERO_VAD_SHA256 = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"
SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)
SEGMENTATION_MEMBER = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
WHISPER_REQUIRED_FILES = (".complete", "encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt", "silero_vad.onnx")
DIARIZATION_REQUIRED_FILES = (".complete", "segmentation.onnx", "embedding.onnx")


class ModelError(RuntimeError):
    """A message safe to show in the UI."""


def whisper_sources(model: str) -> dict[str, dict[str, str]]:
    """Only verified multilingual ONNX exports, pinned to immutable HF commits.

    No Hugging Face client or media-decoding dependency is required. Hashes were
    checked against the upstream LFS objects / directly downloaded vocabulary.
    """
    if model not in WHISPER_MODEL_RELEASES:
        raise ModelError("対応していない文字起こしモデルです。")
    release = WHISPER_MODEL_RELEASES[model]
    base = f"https://huggingface.co/csukuangfj/sherpa-onnx-whisper-{model}/resolve/{release['revision']}"
    return {
        "encoder.int8.onnx": {"url": f"{base}/{model}-encoder.int8.onnx", "sha256": release["encoder_sha256"]},
        "decoder.int8.onnx": {"url": f"{base}/{model}-decoder.int8.onnx", "sha256": release["decoder_sha256"]},
        "tokens.txt": {"url": f"{base}/{model}-tokens.txt", "sha256": WHISPER_TOKENS_SHA256},
        "silero_vad.onnx": {"url": SILERO_VAD_URL, "sha256": SILERO_VAD_SHA256},
    }


class ModelManager:
    def __init__(self, data_dir: Path, *, bundled_models: Path | None = None):
        self.directory = Path(data_dir) / "models"
        # PyInstaller onedir layout: App.exe, _internal/, models/. Resolve from
        # the executable, never the working directory or an environment override.
        # Source runs use only user models unless a bundle is injected by tests.
        self.bundled_directory = (
            Path(bundled_models).resolve() if bundled_models is not None
            else Path(sys.executable).resolve().parent / "models" if getattr(sys, "frozen", False)
            else None
        )

    @staticmethod
    def _complete(directory: Path, filenames: tuple[str, ...]) -> bool:
        """Completion markers alone do not establish a usable model."""
        try:
            complete = all((directory / name).is_file() and (directory / name).stat().st_size > 0
                           for name in filenames)
            if complete and filenames == WHISPER_REQUIRED_FILES:
                # A CTranslate2 completion marker or a different model engine
                # must never authorize loading an incompatible cached model.
                marker = directory / ".complete"
                if marker.stat().st_size > 4096:
                    return False
                metadata = json.loads(marker.read_text(encoding="utf-8"))
                return (isinstance(metadata, dict) and metadata.get("engine") == WHISPER_ENGINE
                        and metadata.get("model") == directory.name.removeprefix("whisper-onnx-"))
            return complete
        except (OSError, ValueError, UnicodeError):
            return False

    def _resolve(self, subdirectory: str, required: tuple[str, ...]) -> Path:
        user_directory = self.directory / subdirectory
        if self._complete(user_directory, required):
            return user_directory
        if self.bundled_directory is not None:
            bundled_directory = self.bundled_directory / subdirectory
            if self._complete(bundled_directory, required):
                return bundled_directory
        # Missing/incomplete models always resolve to the writable destination.
        return user_directory

    @property
    def segmentation_path(self) -> Path:
        return self._resolve("diarization", DIARIZATION_REQUIRED_FILES) / "segmentation.onnx"

    @property
    def embedding_path(self) -> Path:
        return self._resolve("diarization", DIARIZATION_REQUIRED_FILES) / "embedding.onnx"

    @staticmethod
    def _model_name(settings: dict) -> str:
        model = settings.get("whisper_model", "small")
        if model not in WHISPER_MODELS:
            raise ModelError("対応していない文字起こしモデルです。")
        return model

    def whisper_path(self, settings: dict) -> Path:
        model = self._model_name(settings)
        return self._resolve(f"whisper-onnx-{model}", WHISPER_REQUIRED_FILES)

    def status(self, settings: dict) -> dict:
        try:
            directory = self.whisper_path(settings)
        except ModelError as exc:
            return {"ready": False, "detail": str(exc)}
        if not self._complete(directory, WHISPER_REQUIRED_FILES):
            return {"ready": False,
                    "detail": f"ONNX文字起こしモデル「{self._model_name(settings)}」がありません。新しい同梱版を利用するか、設定から準備してください。旧版のモデルは再利用できません。"}
        directories = [directory]
        if settings.get("diarization", True):
            diarization_directory = self.segmentation_path.parent
            if not self._complete(diarization_directory, DIARIZATION_REQUIRED_FILES):
                return {"ready": False, "detail": "話者分離モデルがありません。設定からダウンロードしてください。"}
            directories.append(diarization_directory)
        bundled = self.bundled_directory is not None and any(
            path.parent == self.bundled_directory for path in directories
        )
        return {"ready": True,
                "detail": "同梱のローカルモデルを準備済み（ダウンロード不要）" if bundled else "ローカルモデルを準備済み"}

    def download(self, settings: dict, progress_callback: Callable[[str], None]) -> None:
        model = self._model_name(settings)
        try:
            if not self._complete(self.whisper_path(settings), WHISPER_REQUIRED_FILES):
                self._download_whisper(model, progress_callback)
            if settings.get("diarization", True):
                self._download_diarization(progress_callback)
        except ModelError:
            raise
        except ImportError as exc:
            raise ModelError("音声処理ライブラリがありません。セットアップを再実行してください。") from exc
        except Exception as exc:
            raise ModelError(
                "モデルの取得に失敗しました。ネット接続と空き容量を確認し、再実行してください。"
            ) from exc
        progress_callback("ローカルモデルの準備が完了しました。")

    def _download_whisper(self, model: str, callback: Callable[[str], None]) -> None:
        directory = self.directory / f"whisper-onnx-{model}"
        sources = whisper_sources(model)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".complete").unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="giji-whisper-", dir=self.directory) as temporary:
            temporary = Path(temporary)
            for index, (name, source) in enumerate(sources.items(), 1):
                callback(f"ONNX文字起こしモデル「{model}」を取得しています（{index}/{len(sources)}）。")
                downloaded = temporary / name
                self._fetch(source["url"], downloaded)
                with downloaded.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != source["sha256"]:
                    raise ModelError("文字起こしモデルの整合性を確認できません。取得をやり直してください。")
                os.replace(downloaded, directory / name)
        (directory / "sources.json").write_text(
            json.dumps({"engine": WHISPER_ENGINE, "model": model, "files": sources}, indent=2) + "\n",
            encoding="utf-8",
        )
        (directory / ".complete").write_text(
            json.dumps({"engine": WHISPER_ENGINE, "model": model}) + "\n", encoding="utf-8",
        )

    def _download_diarization(self, callback: Callable[[str], None]) -> None:
        if self._complete(self.segmentation_path.parent, DIARIZATION_REQUIRED_FILES):
            return
        # Always write to the user's data directory, even when the installation
        # contains an incomplete or read-only bundle.
        directory = self.directory / "diarization"
        segmentation_path = directory / "segmentation.onnx"
        embedding_path = directory / "embedding.onnx"
        directory.mkdir(parents=True, exist_ok=True)
        callback("話者分離モデルをダウンロードしています。")
        with tempfile.TemporaryDirectory(prefix="giji-model-", dir=self.directory) as temporary:
            temporary = Path(temporary)
            if not self._complete(directory, ("segmentation.onnx",)):
                archive_path = temporary / "segmentation.tar.bz2"
                self._fetch(SEGMENTATION_URL, archive_path)
                with tarfile.open(archive_path, "r:bz2") as archive:
                    member = archive.getmember(SEGMENTATION_MEMBER)
                    if not member.isfile() or member.size > 100_000_000:
                        raise ModelError("話者分離モデルの形式が正しくありません。")
                    # Read one known regular file, never extract archive paths.
                    source = archive.extractfile(member)
                    if source is None:
                        raise ModelError("話者分離モデルを読み取れませんでした。")
                    with source, (temporary / "segmentation.onnx").open("wb") as target:
                        shutil.copyfileobj(source, target)
                    os.replace(temporary / "segmentation.onnx", segmentation_path)
                    for name in ("LICENSE", "README.md"):
                        try:
                            item = archive.getmember(f"sherpa-onnx-pyannote-segmentation-3-0/{name}")
                            if item.isfile() and item.size < 1_000_000:
                                stream = archive.extractfile(item)
                                if stream:
                                    with stream:
                                        (directory / f"segmentation-{name}").write_bytes(stream.read())
                        except KeyError:
                            pass
            if not self._complete(directory, ("embedding.onnx",)):
                self._fetch(EMBEDDING_URL, temporary / "embedding.onnx")
                os.replace(temporary / "embedding.onnx", embedding_path)
        (directory / "sources.json").write_text(
            json.dumps({"segmentation": SEGMENTATION_URL, "embedding": EMBEDDING_URL}, indent=2),
            encoding="utf-8",
        )
        (directory / ".complete").write_text("1\n", encoding="utf-8")

    @staticmethod
    def _fetch(url: str, destination: Path) -> None:
        # Only fixed upstream URLs are passed here. TLS verification stays enabled.
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, read=120)) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with destination.open("wb") as file:
                    for chunk in response.iter_bytes(1024 * 1024):
                        file.write(chunk)
        if destination.stat().st_size < 1024:
            raise ModelError("取得したモデルファイルが不完全です。")

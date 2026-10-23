import hashlib
import io
from pathlib import Path
import sys

import pytest

from simple_speech_to_text import models


class Response(io.BytesIO):
    def geturl(self):
        return "https://example.test/model"


@pytest.fixture
def tiny_manifest(monkeypatch):
    payload = b"verified model fixture"
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(models, "MODEL_FILES", {"fixture.onnx": ("https://example.test/model", expected)})
    return payload


def test_valid_local_model_never_downloads(tmp_path, tiny_manifest, monkeypatch):
    (tmp_path / "fixture.onnx").write_bytes(tiny_manifest)
    monkeypatch.setattr(models, "urlopen", lambda *a, **k: pytest.fail("network access"))
    assert models.ensure_models(tmp_path, download=False) == tmp_path


def test_verified_download_is_atomic_and_reusable(tmp_path, tiny_manifest, monkeypatch):
    monkeypatch.setattr(models, "urlopen", lambda *a, **k: Response(tiny_manifest))
    models.ensure_models(tmp_path, on_status=lambda text: None)
    assert (tmp_path / "fixture.onnx").read_bytes() == tiny_manifest
    assert models.verify_models(tmp_path) == []
    assert list(tmp_path.glob(".download-*")) == []


def test_corrupt_download_preserves_existing_file(tmp_path, tiny_manifest, monkeypatch):
    destination = tmp_path / "fixture.onnx"
    destination.write_bytes(b"previous broken model")
    monkeypatch.setattr(models, "urlopen", lambda *a, **k: Response(b"wrong response"))
    with pytest.raises(models.ModelError, match="SHA-256"):
        models.ensure_models(tmp_path, on_status=lambda text: None)
    assert destination.read_bytes() == b"previous broken model"
    assert list(tmp_path.glob(".download-*")) == []


def test_portable_missing_model_fails_without_download(tmp_path, tiny_manifest, monkeypatch):
    monkeypatch.setattr(models, "urlopen", lambda *a, **k: pytest.fail("network access"))
    with pytest.raises(models.ModelError, match="同梱モデル"):
        models.ensure_models(tmp_path, download=False)


def test_partial_download_is_removed(tmp_path, tiny_manifest, monkeypatch):
    class BrokenResponse(Response):
        def read(self, size=-1):
            raise OSError("disconnected")

    monkeypatch.setattr(models, "urlopen", lambda *a, **k: BrokenResponse(tiny_manifest))
    with pytest.raises(models.ModelError, match="取得できません"):
        models.ensure_models(tmp_path, on_status=lambda text: None)
    assert list(tmp_path.iterdir()) == []


def test_frozen_models_are_relative_to_executable_not_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "app" / "SimpleSpeechToText.exe"))
    assert models.default_model_dir() == tmp_path / "app" / "models" / "whisper-small"


def test_all_model_urls_and_digests_are_fixed():
    assert len(models.MODEL_FILES) == 4
    for filename, (url, digest) in models.MODEL_FILES.items():
        assert Path(filename).name == filename
        assert url.startswith("https://")
        assert len(digest) == 64
        int(digest, 16)
        if filename != "silero_vad.onnx":
            assert models.REVISION in url

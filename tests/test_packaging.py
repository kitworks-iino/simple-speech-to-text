import io
import json
import zipfile

import pytest

from scripts.audit_dependencies import DistributionAudit, audit_archive
from scripts.package_portable import REQUIRED, package
from simple_speech_to_text.dependency_policy import ProhibitedDependencyError


def test_empty_upstream_wheel_marker_is_not_an_embedded_archive():
    audit = DistributionAudit()
    audit.inspect("scipy/scipy-1.16.3-cp313-cp313-win_amd64.whl", b"")
    assert audit.report()["nested_archives_checked"] == 0


def test_empty_prohibited_name_is_still_rejected():
    with pytest.raises(ProhibitedDependencyError):
        DistributionAudit().inspect("avcodec-61.dll", b"")


def test_renamed_native_library_is_rejected_by_function_identifier():
    with pytest.raises(ProhibitedDependencyError, match="native audio identifier"):
        DistributionAudit().inspect("ordinary.dll", b"\0avcodec_version\0")


def test_prohibited_transitive_metadata_is_rejected():
    with pytest.raises(ProhibitedDependencyError, match="package metadata"):
        DistributionAudit().inspect("licenses/ordinary/METADATA.txt", b"Name: innocent\nRequires-Dist: av>=16\n")


def test_nested_prohibited_source_cannot_hide_in_zip(tmp_path):
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("libavcodec/decode.c", "source fixture")
    public = tmp_path / "public.zip"
    with zipfile.ZipFile(public, "w") as archive:
        archive.writestr("application-source.zip", nested.getvalue())
    with pytest.raises(ProhibitedDependencyError, match="Prohibited audio component"):
        audit_archive(public)


def make_distribution(directory):
    for name in REQUIRED:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture")


def test_missing_model_prevents_packaging(tmp_path):
    directory = tmp_path / "app"
    make_distribution(directory)
    (directory / "models/whisper-small/encoder.int8.onnx").unlink()
    with pytest.raises(RuntimeError, match="encoder"):
        package(directory, tmp_path / "public.zip")


def test_recording_is_never_packaged(tmp_path):
    directory = tmp_path / "app"
    make_distribution(directory)
    (directory / "private.wav").write_bytes(b"private audio")
    with pytest.raises(RuntimeError, match="recording"):
        package(directory, tmp_path / "public.zip")


def test_successful_package_contains_audit_and_matching_checksum(tmp_path):
    directory = tmp_path / "app"
    make_distribution(directory)
    output = tmp_path / "public.zip"
    checksum = package(directory, output)
    with zipfile.ZipFile(output) as archive:
        report = json.loads(archive.read("SimpleSpeechToText/DEPENDENCY-AUDIT.json"))
        assert report["passed"] is True
        assert "SimpleSpeechToText/models/whisper-small/silero_vad.onnx" in archive.namelist()
    assert (tmp_path / "SHA256SUMS.txt").read_text() == f"{checksum}  public.zip\n"


def test_prohibited_library_prevents_public_checksum(tmp_path):
    directory = tmp_path / "app"
    make_distribution(directory)
    (directory / "ffmpeg.exe").write_bytes(b"forbidden fixture")
    with pytest.raises(ProhibitedDependencyError):
        package(directory, tmp_path / "public.zip")
    assert not (tmp_path / "SHA256SUMS.txt").exists()

from email.message import Message
from types import SimpleNamespace

import pytest

from simple_speech_to_text import dependency_policy as policy


def distribution(name, version="1.0", requires=()):
    message = Message()
    message["Name"] = name
    for requirement in requires:
        message["Requires-Dist"] = requirement
    return SimpleNamespace(metadata=message, version=version)


@pytest.mark.parametrize("name", [
    "av", "PyAV", "faster_whisper", "CTRANSLATE2", "moviepy", "pydub", "audioread",
    "ffmpeg", "ffmpeg-python", "imageio_ffmpeg", "python.ffmpeg", "TORCHAUDIO",
    "opencv_python", "opencv-contrib-python-headless",
])
def test_rejects_direct_packages_without_importing_them(name):
    with pytest.raises(policy.ProhibitedDependencyError, match="新しい"):
        policy.ensure_no_prohibited_dependencies([distribution(name)])


@pytest.mark.parametrize("requirement", [
    "av>=12", "faster-whisper[all] (>=1)", " imageio_ffmpeg ; sys_platform == 'win32'",
    "ffmpeg-python; extra == 'optional'", "PyAV @ https://example.invalid/pkg.whl",
])
def test_rejects_wrapper_metadata_and_optional_requirements(requirement):
    with pytest.raises(policy.ProhibitedDependencyError, match="wrapper"):
        policy.ensure_no_prohibited_dependencies([distribution("wrapper", requires=[requirement])])


def test_approved_runtime_metadata_is_sorted_and_reported():
    installed = [
        distribution("sounddevice", "0.5.3", ["CFFI>=1.0"]),
        distribution("sherpa-onnx", "1.13.8", ["sherpa-onnx-core==1.13.8"]),
        distribution("sherpa-onnx-core", "1.13.8"),
        distribution("NumPy", "2.3.5"),
        distribution("scipy", "1.16.3", ["numpy<2.6,>=1.25.2"]),
        distribution("cffi", "2.0.0", ["pycparser; implementation_name != 'PyPy'"]),
        distribution("pycparser", "2.23"),
    ]
    records = policy.ensure_no_prohibited_dependencies(installed)
    assert [record["name"] for record in records] == [
        "cffi", "NumPy", "pycparser", "scipy", "sherpa-onnx", "sherpa-onnx-core", "sounddevice",
    ]
    assert next(r for r in records if r["name"] == "sounddevice")["version"] == "0.5.3"


def test_default_checks_actual_metadata_provider(monkeypatch):
    monkeypatch.setattr(policy.metadata, "distributions", lambda: [distribution("av")])
    with pytest.raises(policy.ProhibitedDependencyError):
        policy.ensure_no_prohibited_dependencies()


def test_empty_environment_is_allowed():
    assert policy.ensure_no_prohibited_dependencies([]) == []


def test_similarly_named_unrelated_package_is_allowed():
    assert not policy.prohibited_distribution("available")
    assert not policy.prohibited_distribution("audio-library")
    assert policy.normalized_name(" ImageIO__FFMPEG ") == "imageio-ffmpeg"

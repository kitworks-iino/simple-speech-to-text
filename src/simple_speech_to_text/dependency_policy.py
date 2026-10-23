"""Check package metadata before importing the native audio libraries.

This guard catches common incompatible installations. The build-time auditor
also checks shipped native files and archives; metadata alone is not a binary
or patent audit.
"""

from __future__ import annotations

from importlib import metadata
import re


PROHIBITED_DISTRIBUTIONS = frozenset({
    "av", "pyav", "faster-whisper", "ctranslate2", "moviepy", "pydub", "audioread",
    "opencv-python", "opencv-python-headless", "opencv-contrib-python",
    "opencv-contrib-python-headless", "torchaudio",
})


class ProhibitedDependencyError(RuntimeError):
    """The environment contains a component outside the no-FFmpeg policy."""


def normalized_name(name: str) -> str:
    """Use Python package name normalization, including underscore aliases."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def prohibited_distribution(name: str) -> bool:
    normalized = normalized_name(name)
    return "ffmpeg" in normalized or normalized in PROHIBITED_DISTRIBUTIONS


def ensure_no_prohibited_dependencies(distributions=None) -> list[dict[str, str]]:
    """Fail before audio imports if installed metadata declares a banned stack.

    No package is imported, uninstalled or modified. Optional requirements are
    checked conservatively too, since a shared environment is not an approved
    substitute for the isolated application environment.
    """
    records = []
    installed = metadata.distributions() if distributions is None else distributions
    for dist in installed:
        package_metadata = dist.metadata
        name = package_metadata.get("Name", "")
        if prohibited_distribution(name):
            raise ProhibitedDependencyError(
                f"使用できない音声依存パッケージがあります: {normalized_name(name)}。"
                "配布 ZIP を新しいフォルダーへ展開するか、新しい仮想環境を作成してください。"
                "既存パッケージの更新だけでは古い依存は削除されません。"
            )
        for requirement in package_metadata.get_all("Requires-Dist", []):
            match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
            if match and prohibited_distribution(match.group(1)):
                dependency = normalized_name(match.group(1))
                raise ProhibitedDependencyError(
                    f"{normalized_name(name)} が使用禁止の音声依存 {dependency} を宣言しています。"
                    "このアプリ専用の新しい仮想環境か配布 ZIP を使用してください。"
                )
        records.append({"name": name, "version": dist.version})
    return sorted(records, key=lambda item: normalized_name(item["name"]))

"""Preserve upstream license notices and exact build/model provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
NOTICE = re.compile(r"^(licen[cs]e|copying|copyright|notice|authors)([._-].*)?$", re.I)
# Versioned upstream license sources. Their downloaded SHA256 hashes are also
# recorded in the portable manifest; model files use separate pinned hashes.
UPSTREAM_NOTICES = {
    "sherpa-onnx-LICENSE.txt": (
        "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf/LICENSE",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    "whisper-LICENSE.txt": (
        "https://raw.githubusercontent.com/openai/whisper/31243bad24cc746f07d4c8bfdd2d974872cb1803/LICENSE",
        "b5d65a59060e68c4ff940e1eddfa6f94b2d68fdf58ed7f4dd57721c997e35e9d",
    ),
    "silero-vad-LICENSE.txt": (
        "https://raw.githubusercontent.com/snakers4/silero-vad/6478567951ae5c9979ad7b234185b5515f4be7a1/LICENSE",
        "2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b",
    ),
    "PortAudio-LICENSE.txt": (
        "https://raw.githubusercontent.com/PortAudio/portaudio/147dd722548358763a8b649b3e4b41dfffbcfbb6/LICENSE.txt",
        "ec52a1952d701f94e5135719a47376da4ee0b4a0201f1cafb49f61db6480ac3d",
    ),
    "ONNX-Runtime-LICENSE.txt": (
        "https://raw.githubusercontent.com/microsoft/onnxruntime/33ca9628233dc8f002435e868d4c2e9f82766ca1/LICENSE",
        "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c",
    ),
    "ONNX-Runtime-ThirdPartyNotices.txt": (
        "https://raw.githubusercontent.com/microsoft/onnxruntime/33ca9628233dc8f002435e868d4c2e9f82766ca1/ThirdPartyNotices.txt",
        "0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2",
    ),
}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def collect_python(licenses: Path) -> list[dict]:
    records = []
    for distribution in sorted(metadata.distributions(), key=lambda item: item.metadata.get("Name", "")):
        name = distribution.metadata.get("Name", "unknown")
        component = re.sub(r"[^A-Za-z0-9._-]", "_", f"{name}-{distribution.version}")
        folder = licenses / "python" / component
        folder.mkdir(parents=True, exist_ok=True)
        original = distribution.read_text("METADATA") or distribution.read_text("PKG-INFO")
        if original is None:
            raise RuntimeError(f"Original package metadata missing: {name}")
        (folder / "METADATA.txt").write_text(original, encoding="utf-8")
        copied = []
        for item in distribution.files or []:
            relative = PurePosixPath(str(item).replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            if not NOTICE.match(relative.name) and not any(part.lower() in {"licenses", "licences"}
                                                           for part in relative.parts):
                continue
            source = Path(distribution.locate_file(item))
            if not source.is_file() or source.is_symlink():
                continue
            target = folder / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append({"path": relative.as_posix(), "sha256": digest(target)})
        records.append({
            "name": name, "version": distribution.version,
            "license": distribution.metadata.get("License-Expression") or distribution.metadata.get("License"),
            "requires_dist": distribution.metadata.get_all("Requires-Dist", []),
            "notices": copied,
        })
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        python_license = Path(sys.base_prefix) / "LICENSE"
    if not python_license.is_file():
        raise RuntimeError("Build Python's original license is missing.")
    shutil.copy2(python_license, licenses / "PYTHON-LICENSE.txt")
    return records


def download_notice(url: str, target: Path, expected_sha256: str) -> dict:
    request = Request(url, headers={"User-Agent": "simple-speech-to-text-license-collector/0.1"})
    with urlopen(request, timeout=60) as response:
        if not response.url.startswith("https://"):
            raise RuntimeError("License download redirected to an insecure URL.")
        payload = response.read(8 * 1024 * 1024 + 1)
    if not payload or len(payload) > 8 * 1024 * 1024:
        raise RuntimeError(f"Unexpected license size: {url}")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError(f"License checksum mismatch: {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {"file": target.name, "url": url, "sha256": digest(target)}


def collect_vendored(licenses: Path) -> dict:
    """Keep the unchanged meeting application's license and source provenance."""
    source = ROOT / "src" / "simple_speech_to_text" / "_vendor" / "giji_v050"
    destination = licenses / "giji-automemo"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("LICENSE", "PROVENANCE.json"):
        target = destination / name
        shutil.copy2(source / name, target)
        copied.append({"file": name, "sha256": digest(target)})
    return {"component": "giji-automemo", "notices": copied}


def collect(destination: Path) -> None:
    licenses = destination / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    # Keep the independently collected Microsoft-runtime notices intact.
    for generated in (licenses / "python", licenses / "upstream"):
        if generated.exists():
            shutil.rmtree(generated)
    packages = collect_python(licenses)
    vendored = collect_vendored(licenses)
    notices = [download_notice(url, licenses / "upstream" / name, checksum)
               for name, (url, checksum) in UPSTREAM_NOTICES.items()]
    models = []
    for path in sorted((destination / "models").rglob("*")):
        if path.is_file():
            models.append({"path": path.relative_to(destination).as_posix(),
                           "sha256": digest(path), "bytes": path.stat().st_size})
    if not models:
        raise RuntimeError("Model files must be bundled before license collection.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "source_commit": commit, "python": sys.version,
        "python_distributions_in_build_environment": packages,
        "upstream_notices": notices, "model_files": models,
        "vendored_source": vendored,
        "pyproject_sha256": digest(ROOT / "pyproject.toml"),
    }
    (licenses / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    (licenses / "README.txt").write_text(
        "Simple Speech to Text — third-party notices\n\n"
        "The application is MIT licensed. Runtime components and model weights\n"
        "retain their own license notices. See manifest.json for exact package\n"
        "versions, source commit, upstream notice URLs, and model hashes.\n"
        "The Python directory also records build/test tools; not every listed\n"
        "build dependency is included in the executable.\n"
        "The final native files and imports are listed in DEPENDENCY-AUDIT.json.\n",
        encoding="utf-8",
    )
    subprocess.run([
        "git", "archive", "--format=zip", "--prefix=simple-speech-to-text/",
        "-o", str(destination / "application-source.zip"), "HEAD",
    ], cwd=ROOT, check=True)
    print(f"Collected licenses: {len(packages)} build distributions, {len(models)} model files.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    collect(args.destination.resolve())


if __name__ == "__main__":
    main()

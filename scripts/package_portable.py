"""Package only the audited application folder, never a user's recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.audit_dependencies import audit_archive, audit_directory  # noqa: E402


REQUIRED = (
    "SimpleSpeechToText.exe", "Start.cmd", "LICENSE.txt", "README.md",
    "docs/dependencies.md",
    "models/whisper-small/encoder.int8.onnx", "models/whisper-small/decoder.int8.onnx",
    "models/whisper-small/tokens.txt", "models/whisper-small/silero_vad.onnx",
    "licenses/manifest.json", "licenses/PYTHON-LICENSE.txt", "DEPENDENCY-ENVIRONMENT.json",
)


def package(directory: Path, output: Path) -> str:
    for name in REQUIRED:
        if not (directory / name).is_file() or not (directory / name).stat().st_size:
            raise RuntimeError(f"Incomplete portable distribution: {name}")
    report = audit_directory(directory)
    (directory / "DEPENDENCY-AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                if path.suffix.lower() in {".wav", ".mp3", ".sqlite3"}:
                    raise RuntimeError(f"Unexpected recording or runtime file: {path.name}")
                archive.write(path, "SimpleSpeechToText/" + path.relative_to(directory).as_posix())
    if output.stat().st_size >= 2 * 1024**3:
        raise RuntimeError("The ZIP exceeds GitHub's 2 GiB release asset limit.")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed.")
    audit_archive(output)
    with output.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    (output.parent / "SHA256SUMS.txt").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = package(args.directory, args.output)
    print(f"Verified {args.output.name}: {args.output.stat().st_size:,} bytes; SHA256 {digest}")


if __name__ == "__main__":
    main()

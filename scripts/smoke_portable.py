"""Run native model inference from the exact extracted public ZIP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile


def smoke(archive_path: Path) -> None:
    if os.name != "nt":
        raise RuntimeError("The portable executable must be tested on Windows.")
    with tempfile.TemporaryDirectory(prefix="stt extracted 日本語 ") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted)
        folder = extracted / "SimpleSpeechToText"
        environment = os.environ.copy()
        # A bundled model must work without any network or pre-existing cache.
        environment.update({
            "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9", "NO_PROXY": "",
            "LOCALAPPDATA": str(extracted / "empty-local-appdata"),
        })
        result = subprocess.run(
            [str(folder / "SimpleSpeechToText.exe"), "--self-test"],
            cwd=extracted, env=environment, timeout=240, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        print(result.stdout)
        if result.returncode:
            print(result.stderr)
            raise RuntimeError(f"Extracted executable self-test exited {result.returncode}.")
        print("Verified extracted EXE: bundled model inference, Unicode path, offline startup.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    smoke(args.archive)


if __name__ == "__main__":
    main()

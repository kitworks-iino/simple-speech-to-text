"""Fail builds containing prohibited audio packages, libraries or source archives."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tarfile
import zipfile

# Also usable from a clean checkout before this project is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from simple_speech_to_text.dependency_policy import (  # noqa: E402
    ProhibitedDependencyError, ensure_no_prohibited_dependencies, prohibited_distribution,
)


LIBRARY_NAME = re.compile(
    r"^(?:lib)?(?:ffmpeg|ffprobe|ffplay|avcodec|avformat|avutil|avfilter|avdevice|"
    r"swresample|swscale|postproc)(?:$|[-_.0-9])", re.I,
)
LEGACY_PACKAGE_PATH = re.compile(
    r"^(?:av|pyav|faster[_-]whisper|ctranslate2)(?:$|[._-](?:libs|\d))", re.I,
)
NATIVE_SUFFIXES = {".dll", ".pyd", ".exe", ".so", ".dylib", ".a", ".lib", ".wasm"}
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
ARCHIVE_SUFFIXES = (".zip", ".whl", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
# Exact exported/imported identifiers also catch renamed or statically linked
# libraries. Do not reject broad 'avformat_' substrings: MSVC's std::format_error
# RTTI contains AVformat_error and is unrelated to the prohibited library.
NATIVE_SIGNATURE = re.compile(
    rb"(?<![A-Za-z0-9_])(?:avcodec_version|avcodec_find_decoder|avcodec_send_packet|"
    rb"avcodec_receive_frame|avcodec_configuration|avformat_open_input|avformat_network_init|"
    rb"avutil_version|av_version_info|avfilter_graph_alloc|swresample_version|swr_alloc|"
    rb"swscale_version|sws_getContext|postproc_version|FFmpeg version)(?![A-Za-z0-9_])",
)


def check_path(name: str) -> None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ProhibitedDependencyError("Unsafe path in distribution archive.")
    for part in path.parts:
        if LIBRARY_NAME.match(part.lstrip("@")) or LEGACY_PACKAGE_PATH.match(part):
            raise ProhibitedDependencyError(f"Prohibited audio component in distribution: {name}")


def pe_import_names(data: bytes) -> list[str]:
    """Read normal and delayed PE imports without executing or loading a DLL."""
    if not data.startswith(b"MZ"):
        return []
    try:
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            raise ValueError("PE signature")
        section_count = struct.unpack_from("<H", data, pe + 6)[0]
        optional_size = struct.unpack_from("<H", data, pe + 20)[0]
        optional = pe + 24
        magic = struct.unpack_from("<H", data, optional)[0]
        if magic not in (0x10B, 0x20B) or section_count > 4096:
            raise ValueError("PE format")
        directory = optional + (112 if magic == 0x20B else 96)
        number = struct.unpack_from("<I", data, directory - 4)[0]
        image_base = struct.unpack_from("<Q" if magic == 0x20B else "<I", data,
                                        optional + (24 if magic == 0x20B else 28))[0]
        headers = struct.unpack_from("<I", data, optional + 60)[0]
        sections = []
        for i in range(section_count):
            offset = optional + optional_size + 40 * i
            virtual_size, address, raw_size, raw_offset = struct.unpack_from("<IIII", data, offset + 8)
            sections.append((address, virtual_size, raw_size, raw_offset))

        def file_offset(rva: int) -> int:
            if 0 <= rva < min(headers, len(data)):
                return rva
            for address, virtual_size, raw_size, raw_offset in sections:
                if address <= rva < address + max(virtual_size, raw_size):
                    delta = rva - address
                    if delta < raw_size and raw_offset + delta < len(data):
                        return raw_offset + delta
            raise ValueError("Unmapped PE address")

        def cstring(rva: int) -> str:
            offset = file_offset(rva)
            end = data.find(b"\0", offset, min(len(data), offset + 512))
            if end < 0:
                raise ValueError("Unterminated PE import")
            return data[offset:end].decode("ascii")

        imports = []
        for index, size in ((1, 20), (13, 32)):
            if number <= index or directory + 8 * (index + 1) > optional + optional_size:
                continue
            rva, directory_size = struct.unpack_from("<II", data, directory + index * 8)
            if not rva:
                continue
            if directory_size < size or directory_size > 16 * 1024 * 1024:
                raise ValueError("PE import directory size")
            terminated = False
            for item in range(min(directory_size // size, 4096)):
                offset = file_offset(rva + item * size)
                values = struct.unpack_from("<" + "I" * (size // 4), data, offset)
                if not any(values):
                    terminated = True
                    break
                name_rva = values[3] if index == 1 else values[1]
                if index == 13 and not values[0] & 1:
                    name_rva -= image_base
                imports.append(cstring(name_rva))
            if not terminated:
                raise ValueError("Unterminated PE import directory")
        return sorted(set(imports), key=str.lower)
    except (ValueError, struct.error, UnicodeError, IndexError) as exc:
        raise ProhibitedDependencyError("A native binary's PE imports could not be verified.") from exc


class DistributionAudit:
    def __init__(self):
        self.file_count = 0
        self.archive_count = 0
        self.expanded_bytes = 0
        self.native_files = []

    def inspect(self, name: str, data: bytes, *, depth: int = 0) -> None:
        check_path(name)
        self.file_count += 1
        self.expanded_bytes += len(data)
        if self.file_count > 200000 or self.expanded_bytes > 4 * 1024**3 or depth > 4:
            raise ProhibitedDependencyError("Distribution audit safety limit exceeded.")
        leaf = PurePosixPath(name).name.lower()
        if leaf in {"metadata", "metadata.txt", "pkg-info"}:
            parsed = BytesParser().parsebytes(data, headersonly=True)
            for value in [parsed.get("Name", ""), *parsed.get_all("Requires-Dist", [])]:
                match = re.match(r"\s*([A-Za-z0-9_.-]+)", value)
                if match and prohibited_distribution(match.group(1)):
                    raise ProhibitedDependencyError(f"Prohibited package metadata in distribution: {name}")
        if leaf == "package.json":
            package = json.loads(data)
            names = [package.get("name", "")]
            for kind in ("dependencies", "optionalDependencies"):
                names.extend(package.get(kind, {}))
            if any(prohibited_distribution(item) for item in names):
                raise ProhibitedDependencyError(f"Prohibited frontend package in distribution: {name}")
        native = data.startswith((b"MZ", b"\x7fELF")) or Path(leaf).suffix in NATIVE_SUFFIXES
        if native:
            match = NATIVE_SIGNATURE.search(data)
            if match:
                raise ProhibitedDependencyError(f"Prohibited native audio identifier in: {name}")
            imports = pe_import_names(data)
            for library in imports:
                if LIBRARY_NAME.match(library):
                    raise ProhibitedDependencyError(f"Prohibited PE import {library} in: {name}")
            self.native_files.append({
                "path": name, "sha256": hashlib.sha256(data).hexdigest(), "imports": imports,
            })
        # SciPy's wheel can include an empty wheel-name marker. Empty payloads
        # cannot hide code; path and metadata checks still run above.
        archive = bool(data) and (leaf.endswith(ARCHIVE_SUFFIXES) or data.startswith((b"PK\x03\x04", b"\x1f\x8b")))
        if not archive:
            return
        self.archive_count += 1
        try:
            stream = io.BytesIO(data)
            if zipfile.is_zipfile(stream):
                with zipfile.ZipFile(stream) as nested:
                    self.inspect_zip(nested, prefix=name + "!", depth=depth + 1)
            else:
                with tarfile.open(fileobj=stream, mode="r:*") as nested:
                    for member in nested:
                        check_path(member.name)
                        if member.isdir():
                            continue
                        if not member.isfile() or member.size > MAX_PAYLOAD_BYTES:
                            raise ProhibitedDependencyError("Unsupported member in nested source archive.")
                        source = nested.extractfile(member)
                        if source is None:
                            raise ProhibitedDependencyError("Unreadable source archive member.")
                        with source:
                            self.inspect(name + "!" + member.name, source.read(), depth=depth + 1)
        except (zipfile.BadZipFile, tarfile.TarError) as exc:
            raise ProhibitedDependencyError(f"An embedded archive could not be verified: {name}") from exc

    def inspect_zip(self, archive: zipfile.ZipFile, *, prefix: str = "", depth: int = 0) -> None:
        names = set()
        for member in archive.infolist():
            check_path(member.filename)
            if member.filename in names or member.file_size > 1024**3:
                raise ProhibitedDependencyError("Duplicate or oversized distribution archive member.")
            names.add(member.filename)
            if member.is_dir():
                continue
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ProhibitedDependencyError("Symlinks are not permitted in the distribution.")
            with archive.open(member) as source:
                self.inspect_stream(prefix + member.filename, source, depth=depth)

    def inspect_stream(self, name: str, source, *, depth: int = 0) -> None:
        check_path(name)
        head = source.read(4)
        leaf = name.lower()
        # Weights are data, not executable code. Avoid loading hundreds of MB
        # into memory, while still detecting a native/ZIP file renamed .onnx.
        if leaf.endswith(".onnx") and not head.startswith((b"MZ", b"\x7fELF", b"PK\x03\x04", b"\x1f\x8b")):
            self.file_count += 1
            return
        payload = head + source.read(MAX_PAYLOAD_BYTES + 1)
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise ProhibitedDependencyError("An executable or source payload exceeds the audit size limit.")
        self.inspect(name, payload, depth=depth)

    def report(self) -> dict:
        return {
            "passed": True, "policy": "no-ffmpeg-runtime-or-sources-v1",
            "files_checked": self.file_count, "nested_archives_checked": self.archive_count,
            "native_files": self.native_files,
            "checks": ["prohibited package metadata", "archive member paths", "native identifiers",
                       "PE imports and delayed imports", "recursive source archives"],
        }


def audit_directory(directory: Path) -> dict:
    audit = DistributionAudit()
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ProhibitedDependencyError("Symlinks are not permitted in the distribution.")
        if path.is_file():
            with path.open("rb") as source:
                audit.inspect_stream(path.relative_to(directory).as_posix(), source)
    return audit.report()


def audit_archive(path: Path) -> dict:
    audit = DistributionAudit()
    with zipfile.ZipFile(path) as archive:
        audit.inspect_zip(archive)
    return audit.report()


def audit_environment(*, metadata_only: bool = False) -> dict:
    distributions = list(metadata.distributions())
    records = ensure_no_prohibited_dependencies(distributions)
    audit = DistributionAudit()
    if not metadata_only:
        visited = set()
        for distribution in distributions:
            for relative in distribution.files or []:
                path = Path(distribution.locate_file(relative)).resolve()
                native_name = path.suffix.lower() in NATIVE_SUFFIXES or ".so." in path.name.lower()
                if not native_name or path in visited or not path.is_file():
                    continue
                visited.add(path)
                with path.open("rb") as source:
                    audit.inspect_stream(path.name, source)
    return {**audit.report(), "distributions": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--environment", action="store_true")
    targets.add_argument("--directory", type=Path)
    targets.add_argument("--archive", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.environment:
        result = audit_environment(metadata_only=args.metadata_only)
    elif args.directory:
        result = audit_directory(args.directory)
    else:
        result = audit_archive(args.archive)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Dependency audit passed: {result['files_checked']} files, "
          f"{len(result['native_files'])} native binaries, {result['nested_archives_checked']} nested archives.")


if __name__ == "__main__":
    main()

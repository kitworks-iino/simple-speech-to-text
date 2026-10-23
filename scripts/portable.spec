from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_delvewheel_libs_directory,
    copy_metadata,
)

root = Path(SPECPATH).parent
datas = []
binaries = []
hiddenimports = ["sounddevice", "_sounddevice", "_cffi_backend"]
for package in ("sherpa_onnx", "_sounddevice_data"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hidden)
for package in ("numpy", "scipy", "sherpa_onnx"):
    datas, binaries = collect_delvewheel_libs_directory(
        package, datas=datas, binaries=binaries,
    )
# Preserve the actual runtime dependency declarations for the startup policy.
datas.extend(copy_metadata("simple-speech-to-text", recursive=True))

a = Analysis(
    [str(root / "scripts" / "portable_entry.py")],
    pathex=[str(root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "torch", "tensorflow", "transformers", "matplotlib", "pytest"],
    noarchive=False,
)
# Ordinary Windows microphone capture needs PortAudio's WASAPI/MME support,
# not the optional ASIO build shipped in sounddevice's upstream wheel.
a.binaries = [entry for entry in a.binaries if not entry[0].lower().endswith("-asio.dll")]
a.datas = [entry for entry in a.datas if not entry[0].lower().endswith("-asio.dll")]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="SimpleSpeechToText",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SimpleSpeechToText")

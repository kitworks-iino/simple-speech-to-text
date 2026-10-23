"""Run a single 30-minute, token-protected demo on an ephemeral Linux runner.

Only a SHA256 digest of the random access token is sent to the runner. The bearer
token is never present in the workflow, connection artifact, or launcher logs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent
CLOUDFLARED_VERSION = "2026.10.3"
CLOUDFLARED_URL = (
    f"https://github.com/cloudflare/cloudflared/releases/download/{CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
)
CLOUDFLARED_SHA256 = "053b80b0a23031a038fdbf3c073f59f82b8f1f152aa67971d7383ffafc0dbab0"
TUNNEL_URL = re.compile(r"https://[a-z0-9]+(?:-[a-z0-9]+)*\.trycloudflare\.com\b")
TTL_SECONDS = 1800


def timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="seconds")


def process_identity(pid: int) -> str | None:
    try:
        # /proc starttime (field 22) distinguishes a live child from a reused PID.
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return stat[19]
    except (OSError, IndexError):
        return None


def alive(record: dict) -> bool:
    return bool(record.get("identity")) and process_identity(record["pid"]) == record["identity"]


def write_state(directory: Path, state: dict) -> None:
    path = directory / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    path.chmod(0o600)


def stop(directory: Path) -> None:
    path = directory / "state.json"
    if not path.is_file():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    for record in state.get("processes", []):
        if alive(record):
            try:
                os.killpg(record["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(alive(item) for item in state.get("processes", [])):
        time.sleep(0.2)
    for record in state.get("processes", []):
        if alive(record):
            try:
                os.killpg(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    path.unlink(missing_ok=True)


def download_cloudflared(directory: Path) -> Path:
    binary = directory / "cloudflared"
    digest = hashlib.sha256()
    total = 0
    with urlopen(CLOUDFLARED_URL, timeout=60) as response, binary.open("wb") as output:
        if not response.url.startswith("https://"):
            raise RuntimeError("The tunnel download redirected to an insecure URL.")
        while block := response.read(1024 * 1024):
            total += len(block)
            if total > 80 * 1024 * 1024:
                raise RuntimeError("Unexpected tunnel binary size.")
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != CLOUDFLARED_SHA256:
        binary.unlink(missing_ok=True)
        raise RuntimeError("The tunnel binary checksum does not match the pinned release.")
    binary.chmod(0o700)
    return binary


def launch(command: list[str], log: Path, environment: dict) -> dict:
    with log.open("ab") as output:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
    identity = process_identity(process.pid)
    if not identity:
        raise RuntimeError("A demo child process exited during startup.")
    return {"pid": process.pid, "identity": identity}


def start(directory: Path, model_dir: Path) -> None:
    if sys.platform != "linux":
        raise RuntimeError("The temporary demo launcher is for Linux GitHub Actions runners.")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", os.environ.get("STT_DEMO_TOKEN_SHA256", "")):
        raise RuntimeError("Provide the SHA256 digest of a randomly generated 256-bit access token.")
    directory.mkdir(parents=True, exist_ok=True)
    stop(directory)
    (directory / "connection.json").unlink(missing_ok=True)
    for name in ("server.log", "tunnel.log"):
        (directory / name).unlink(missing_ok=True)
    binary = download_cloudflared(directory)
    started = time.time()
    state = {"expires_at_epoch": started + TTL_SECONDS, "processes": []}
    write_state(directory, state)
    try:
        server_env = os.environ.copy()
        server_env.pop("STT_DEMO_TOKEN", None)
        server = launch(
            [sys.executable, "-u", "scripts/demo_server.py", "--model-dir", str(model_dir),
             "--port", "8765", "--ttl", str(TTL_SECONDS)],
            directory / "server.log", server_env,
        )
        state["processes"].append(server)
        write_state(directory, state)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if not alive(server):
                raise RuntimeError("Demo server exited before readiness.")
            try:
                with socket.create_connection(("127.0.0.1", 8765), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Demo server readiness timed out.")
        tunnel_env = os.environ.copy()
        tunnel_env.pop("STT_DEMO_TOKEN", None)
        tunnel_env.pop("STT_DEMO_TOKEN_SHA256", None)
        tunnel = launch(
            [str(binary), "tunnel", "--protocol", "http2", "--url", "http://127.0.0.1:8765",
             "--no-autoupdate", "--loglevel", "info"],
            directory / "tunnel.log", tunnel_env,
        )
        state["processes"].append(tunnel)
        write_state(directory, state)
        deadline = time.monotonic() + 90
        origin = None
        while time.monotonic() < deadline:
            if not alive(server) or not alive(tunnel):
                raise RuntimeError("A demo process exited while opening the temporary tunnel.")
            log = (directory / "tunnel.log").read_text(encoding="utf-8", errors="replace")
            match = TUNNEL_URL.search(log)
            if match and "Registered tunnel connection" in log:
                origin = match.group(0)
                break
            time.sleep(0.5)
        if origin is None:
            raise RuntimeError("The temporary tunnel did not become ready.")
        connection = {"origin": origin, "created_at": timestamp(started),
                      "expires_at": timestamp(state["expires_at_epoch"])}
        (directory / "connection.json").write_text(json.dumps(connection, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(connection), flush=True)
    except BaseException:
        stop(directory)
        raise


def hold(directory: Path) -> None:
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    try:
        while time.time() < state["expires_at_epoch"]:
            if not all(alive(item) for item in state["processes"]):
                # The server's own deadline may expire just before this loop.
                if state["expires_at_epoch"] - time.time() > 10:
                    raise RuntimeError("The temporary demo ended before its deadline.")
                break
            time.sleep(min(5, max(0, state["expires_at_epoch"] - time.time())))
    finally:
        stop(directory)
    print("Temporary demo closed; server and tunnel processes stopped.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "hold", "stop"))
    parser.add_argument("--directory", type=Path, default=ROOT / "build" / "demo")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "build" / "models" / "whisper-small")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.action == "start":
        start(directory, args.model_dir.resolve())
    elif args.action == "hold":
        hold(directory)
    else:
        stop(directory)


if __name__ == "__main__":
    main()

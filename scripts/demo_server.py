"""Short-lived HTTPS-preview backend for a one-off iPhone microphone demo.

The upstream preview must terminate HTTPS. This process listens on loopback only.
The original ASR reads temporary PCM WAVs (RAM-backed on the Linux demo runner),
which are removed after each request. Requests and transcripts are not logged.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import threading
import time
from urllib.parse import urlsplit
import wave


MAX_UPLOAD_BYTES = 6 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
UPLOAD_TIMEOUT_SECONDS = 30
SOCKET_TIMEOUT_SECONDS = 15
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/pcm-worklet.js": ("pcm-worklet.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class DemoError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def parse_pcm_wave_bytes(body: bytes):
    """Accept bounded, uncompressed PCM16 mono WAV without creating a file."""
    if len(body) > MAX_UPLOAD_BYTES:
        raise DemoError(413, "too_large", "音声データが大きすぎます。30秒以内でお試しください。")
    try:
        with wave.open(io.BytesIO(body), "rb") as source:
            rate, frames = source.getframerate(), source.getnframes()
            if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise DemoError(415, "unsupported_audio", "PCM16・モノラルのWAV音声が必要です。")
            if not 8_000 <= rate <= 96_000:
                raise DemoError(415, "unsupported_rate", "音声のサンプリング周波数に対応していません。")
            if frames <= 0 or frames > rate * MAX_AUDIO_SECONDS:
                raise DemoError(413, "invalid_duration", "音声は0秒より長く、30秒以内にしてください。")
            pcm = source.readframes(frames)
            if len(pcm) != frames * 2:
                raise DemoError(400, "truncated_audio", "音声データが途中で切れています。")
    except DemoError:
        raise
    except (wave.Error, EOFError, ValueError, OSError) as exc:
        raise DemoError(400, "invalid_wav", "音声を読み込めません。もう一度録音してください。") from exc
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return samples, rate, frames / rate


class WhisperEngine:
    """Use giji-automemo v0.5's original PCM/VAD/Whisper implementation."""

    def __init__(self, model_dir: Path):
        from simple_speech_to_text.dependency_policy import ensure_no_prohibited_dependencies
        from simple_speech_to_text.models import verify_models

        ensure_no_prohibited_dependencies()
        invalid = verify_models(model_dir)
        if invalid:
            raise RuntimeError("文字起こしモデルが不足または破損しています。")
        self.model_dir = Path(model_dir)

    def transcribe(self, samples, rate: int) -> str:
        from simple_speech_to_text.recognition import transcribe

        # Keep the browser's original sample rate. The original pipeline owns
        # resampling, Silero segmentation, Whisper decoding and timestamps.
        text = transcribe(samples, self.model_dir, sample_rate=rate, language="ja")
        if not text:
            raise DemoError(422, "no_speech", "声を検出できませんでした。マイクに近づいてもう一度お試しください。")
        return text


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, *, engine, static_dir: Path, ttl: float, token: str = "", token_sha256: str = ""):
        if token_sha256:
            if not re.fullmatch(r"[a-fA-F0-9]{64}", token_sha256):
                raise ValueError("STT_DEMO_TOKEN_SHA256 must be a SHA-256 hex digest")
            digest = token_sha256.lower()
        else:
            if len(token) < 32:
                raise ValueError("STT_DEMO_TOKEN must contain at least 32 characters")
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if not math.isfinite(ttl) or not 0 < ttl <= 3600:
            raise ValueError("ttl must be between 0 and 3600 seconds")
        self.engine = engine
        self.token_digest = digest
        self.static_dir = Path(static_dir).resolve()
        self.expires_at = time.monotonic() + ttl
        self.recognition_lock = threading.Lock()
        super().__init__(address, DemoHandler)

    def handle_error(self, request, client_address):
        # Do not print request paths, addresses, authorization, audio or results.
        pass


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "SpeechDemo"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(SOCKET_TIMEOUT_SECONDS)

    def log_message(self, *_args):
        pass

    def _respond(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "microphone=(self)")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def _json(self, status: int, data: dict):
        self._respond(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, error: DemoError):
        self._json(error.status, {"error": str(error), "code": error.code})

    def _require_auth(self):
        if time.monotonic() >= self.server.expires_at:
            raise DemoError(410, "expired", "この体験用サーバーの公開時間が終了しました。")
        values = self.headers.get_all("Authorization", [])
        supplied = values[0] if len(values) == 1 else ""
        token = supplied.removeprefix("Bearer ") if supplied.startswith("Bearer ") else ""
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if len(token) < 32 or not hmac.compare_digest(digest, self.server.token_digest):
            raise DemoError(401, "unauthorized", "接続用キーがありません。共有されたリンクから開き直してください。")

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/status":
                self._require_auth()
                return self._json(200, {
                    "ready": True,
                    "remaining_seconds": max(0, math.ceil(self.server.expires_at - time.monotonic())),
                })
            if path not in STATIC_FILES:
                raise DemoError(404, "not_found", "ページが見つかりません。")
            name, content_type = STATIC_FILES[path]
            try:
                file_path = self.server.static_dir / name
                if file_path.resolve().parent != self.server.static_dir:
                    raise OSError("Invalid static target")
                body = file_path.read_bytes()
            except OSError as exc:
                raise DemoError(404, "not_found", "ページが見つかりません。") from exc
            self._respond(200, body, content_type)
        except DemoError as exc:
            self._error(exc)

    def do_POST(self):
        locked = False
        try:
            if urlsplit(self.path).path != "/api/transcribe":
                raise DemoError(404, "not_found", "APIが見つかりません。")
            self._require_auth()
            if self.headers.get("Transfer-Encoding") is not None:
                raise DemoError(400, "invalid_length", "分割転送には対応していません。")
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                raise DemoError(411, "length_required", "音声データのサイズが必要です。")
            if len(lengths) != 1 or len(lengths[0]) > 10 or not lengths[0].isascii() or not lengths[0].isdigit():
                raise DemoError(400, "invalid_length", "音声データのサイズが不正です。")
            length = int(lengths[0])
            if not 0 < length <= MAX_UPLOAD_BYTES:
                raise DemoError(413, "too_large", "音声データが大きすぎるか空です。")
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "audio/wav":
                raise DemoError(415, "unsupported_type", "PCM WAV音声を送信してください。")
            locked = self.server.recognition_lock.acquire(blocking=False)
            if not locked:
                raise DemoError(429, "busy", "現在文字起こし中です。少し待ってから再実行してください。")
            deadline = time.monotonic() + UPLOAD_TIMEOUT_SECONDS
            body = bytearray()
            while len(body) < length:
                if time.monotonic() > deadline:
                    raise DemoError(408, "upload_timeout", "音声の送信が時間切れになりました。")
                block = self.rfile.read1(min(64 * 1024, length - len(body)))
                if not block:
                    raise DemoError(400, "truncated_upload", "音声の送信が途中で切れました。")
                body.extend(block)
            samples, rate, seconds = parse_pcm_wave_bytes(body)
            del body
            text = self.server.engine.transcribe(samples, rate)
            del samples
            self._json(200, {"text": text, "seconds": round(seconds, 3)})
        except DemoError as exc:
            self._error(exc)
        except (TimeoutError, socket.timeout):
            self._error(DemoError(408, "upload_timeout", "音声の送信が時間切れになりました。"))
        except Exception:
            self._error(DemoError(500, "processing_failed", "文字起こしに失敗しました。少し待ってから再実行してください。"))
        finally:
            if locked:
                self.server.recognition_lock.release()

    def do_OPTIONS(self):
        self._error(DemoError(405, "method_not_allowed", "この操作には対応していません。"))

    def send_error(self, code, message=None, explain=None):
        self._error(DemoError(code, "request_error", "リクエストを処理できませんでした。"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ttl", type=float, default=1800)
    args = parser.parse_args()
    token = os.environ.get("STT_DEMO_TOKEN", "")
    token_sha256 = os.environ.get("STT_DEMO_TOKEN_SHA256", "")
    if token_sha256 and not re.fullmatch(r"[a-fA-F0-9]{64}", token_sha256):
        parser.error("STT_DEMO_TOKEN_SHA256 must be a SHA-256 hex digest")
    if not token_sha256 and len(token) < 32:
        parser.error("STT_DEMO_TOKEN must contain at least 32 characters")
    if not math.isfinite(args.ttl) or not 0 < args.ttl <= 3600:
        parser.error("--ttl must be between 0 and 3600 seconds")
    engine = WhisperEngine(args.model_dir)
    with DemoServer(
        ("127.0.0.1", args.port), engine=engine, token=token, token_sha256=token_sha256,
        static_dir=Path(__file__).resolve().parents[1] / "demo", ttl=args.ttl,
    ) as server:
        timer = threading.Timer(args.ttl, server.shutdown)
        timer.daemon = True
        timer.start()
        print(f"Temporary speech demo ready on 127.0.0.1:{args.port}; expires in {args.ttl:g}s.", flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            timer.cancel()


if __name__ == "__main__":
    main()

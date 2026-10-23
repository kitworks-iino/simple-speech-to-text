import hashlib
from http.client import HTTPConnection
import importlib.util
import io
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import wave

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("demo_server", Path(__file__).parents[1] / "scripts" / "demo_server.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)
TOKEN = "test-only-random-shaped-token-abcdefghijklmnopqrstuvwxyz"


def pcm_wav(*, rate=16_000, seconds=0.3, channels=1, width=2):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(bytes(round(rate * seconds) * channels * width))
    return output.getvalue()


@pytest.fixture
def running_server(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>Speech demo</title>", encoding="utf-8")
    (tmp_path / "app.js").write_text("'use strict';", encoding="utf-8")
    calls = []

    def transcribe(samples, rate):
        calls.append((samples.copy(), rate))
        return "こんにちは。"

    engine = SimpleNamespace(transcribe=transcribe)
    server = demo.DemoServer(
        ("127.0.0.1", 0), engine=engine, token_sha256=hashlib.sha256(TOKEN.encode()).hexdigest(),
        static_dir=tmp_path, ttl=60,
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield server, calls
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(server, path, *, method="GET", body=None, auth=True, headers=(), automatic_length=True):
    connection = HTTPConnection(*server.server_address, timeout=3)
    connection.putrequest(method, path)
    if auth:
        connection.putheader("Authorization", "Bearer " + TOKEN)
    if body is not None and automatic_length:
        connection.putheader("Content-Length", str(len(body)))
    for key, value in headers:
        connection.putheader(key, value)
    connection.endheaders(body)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_public_static_and_authenticated_status(running_server):
    server, _ = running_server
    status, headers, body = request(server, "/", auth=False)
    assert status == 200 and b"Speech demo" in body
    assert headers["Cache-Control"] == "no-store"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert request(server, "/api/status", auth=False)[0] == 401
    status, headers, body = request(server, "/api/status")
    assert status == 200
    assert b'"ready": true' in body
    assert b'"remaining_seconds":' in body
    assert TOKEN.encode() not in body
    assert not hasattr(server, "token")


def test_bearer_digest_is_not_accepted_as_bearer_token(running_server):
    server, _ = running_server
    status, _, _ = request(
        server, "/api/status", auth=False,
        headers=[("Authorization", "Bearer " + hashlib.sha256(TOKEN.encode()).hexdigest())],
    )
    assert status == 401


def test_successful_upload_keeps_audio_in_memory(running_server, capsys):
    server, calls = running_server
    status, headers, body = request(
        server, "/api/transcribe", method="POST", body=pcm_wav(rate=48_000),
        headers=[("Content-Type", "audio/wav")],
    )
    assert status == 200
    assert "こんにちは。" in body.decode()
    assert b'"seconds": 0.3' in body
    assert len(calls) == 1 and calls[0][1] == 48_000
    assert calls[0][0].dtype == np.float32
    assert sorted(path.name for path in server.static_dir.iterdir()) == ["app.js", "index.html"]
    assert "Access-Control-Allow-Origin" not in headers
    assert capsys.readouterr().out == ""
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("method,path", [("GET", "/api/status"), ("POST", "/api/transcribe")])
def test_duplicate_or_invalid_authorization_rejected(running_server, method, path):
    server, calls = running_server
    status, _, _ = request(server, path, method=method, headers=[("Authorization", "Bearer invalid")])
    assert status == 401
    assert not calls


@pytest.mark.parametrize("headers,expected", [
    ([], 411),
    ([("Content-Length", "-1")], 400),
    ([("Content-Length", "9" * 5000)], 400),
    ([("Content-Length", "10"), ("Content-Length", "10")], 400),
    ([("Content-Length", str(demo.MAX_UPLOAD_BYTES + 1))], 413),
    ([("Content-Length", "0")], 413),
    ([("Content-Length", "10"), ("Transfer-Encoding", "chunked")], 400),
    ([("Content-Length", "10"), ("Content-Type", "audio/mpeg")], 415),
])
def test_invalid_upload_headers_rejected_before_reading(running_server, headers, expected):
    server, calls = running_server
    status, response_headers, body = request(
        server, "/api/transcribe", method="POST", headers=headers, automatic_length=False,
    )
    assert status == expected
    assert response_headers["Content-Type"].startswith("application/json")
    assert b'"error":' in body
    assert not calls


def test_busy_recognition_returns_429(running_server):
    server, calls = running_server
    server.recognition_lock.acquire()
    try:
        status, _, body = request(server, "/api/transcribe", method="POST", body=pcm_wav(), headers=[("Content-Type", "audio/wav")])
    finally:
        server.recognition_lock.release()
    assert status == 429 and b'"busy"' in body
    assert not calls


def test_model_error_is_json_and_releases_lock(running_server):
    server, _ = running_server

    def fail(*_args):
        raise RuntimeError("private transcript and native details")

    server.engine.transcribe = fail
    status, _, body = request(server, "/api/transcribe", method="POST", body=pcm_wav(), headers=[("Content-Type", "audio/wav")])
    assert status == 500
    assert b"private" not in body
    assert not server.recognition_lock.locked()


def test_actual_silence_error_is_preserved(running_server):
    server, _ = running_server

    def no_speech(*_args):
        raise demo.DemoError(422, "no_speech", "声を検出できませんでした。")

    server.engine.transcribe = no_speech
    status, _, body = request(server, "/api/transcribe", method="POST", body=pcm_wav(), headers=[("Content-Type", "audio/wav")])
    assert status == 422 and b'"no_speech"' in body
    assert not server.recognition_lock.locked()


def test_no_cross_origin_access_or_preflight(running_server):
    server, _ = running_server
    status, headers, body = request(server, "/api/transcribe", method="OPTIONS", auth=False, headers=[("Origin", "https://other.invalid")])
    assert status == 405
    assert not any(name.lower().startswith("access-control-") for name in headers)
    assert b'"error":' in body


@pytest.mark.parametrize("path", ["/../pyproject.toml", "/%2e%2e/pyproject.toml", "/api/token", "/demo_server.py"])
def test_static_allowlist_rejects_other_paths(running_server, path):
    assert request(running_server[0], path, auth=False)[0] == 404


def test_expired_api_returns_410(running_server):
    server, _ = running_server
    server.expires_at = time.monotonic() - 1
    assert request(server, "/api/status")[0] == 410


@pytest.mark.parametrize("body", [
    # Pytest copies node IDs into PYTEST_CURRENT_TEST. Raw WAV representations
    # exceed Windows' environment-variable limit and obscure the CI traceback.
    pytest.param(b"compressed audio", id="compressed"),
    pytest.param(pcm_wav(channels=2), id="stereo"),
    pytest.param(pcm_wav(width=1), id="8bit"),
    pytest.param(pcm_wav(rate=4000), id="unsupported-rate"),
    pytest.param(pcm_wav(seconds=30.01), id="too-long"),
    pytest.param(pcm_wav(seconds=0), id="empty"),
    pytest.param(pcm_wav()[:-20], id="truncated"),
])
def test_invalid_wav_is_rejected(body):
    with pytest.raises(demo.DemoError):
        demo.parse_pcm_wave_bytes(body)


def test_30_second_mono_pcm_is_accepted():
    samples, rate, seconds = demo.parse_pcm_wave_bytes(pcm_wav(rate=96_000, seconds=30))
    assert len(samples) == 2_880_000 and rate == 96_000 and seconds == 30


@pytest.mark.parametrize("kwargs", [
    {"token": "short"}, {"token_sha256": "not-a-hash"}, {"token": TOKEN, "ttl": -1},
])
def test_server_rejects_bad_security_config(tmp_path, kwargs):
    options = {"engine": None, "static_dir": tmp_path, "ttl": 60, **kwargs}
    with pytest.raises(ValueError):
        demo.DemoServer(("127.0.0.1", 0), **options)


def test_main_stops_after_ttl_without_user_input(monkeypatch):
    monkeypatch.setenv("STT_DEMO_TOKEN_SHA256", hashlib.sha256(TOKEN.encode()).hexdigest())
    monkeypatch.delenv("STT_DEMO_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["demo_server.py", "--model-dir", "unused", "--port", "0", "--ttl", "0.1"])
    monkeypatch.setattr(demo, "WhisperEngine", lambda directory: object())
    started = time.monotonic()
    demo.main()
    assert time.monotonic() - started < 2

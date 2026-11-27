"""Synthetic transcription endpoint used by client and fresh-process tests."""
import argparse
import io
import json
import struct
import time
import wave
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        message = BytesParser(policy=default).parsebytes(
            b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
        parts = {part.get_param("name", header="Content-Disposition"): part.get_payload(decode=True)
                 for part in message.iter_parts()}
        self.server.requests.append((self.path, self.headers.get("X-Request-ID"), parts))
        with wave.open(io.BytesIO(parts["file"]), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
            first_sample = struct.unpack("<h", audio.readframes(1))[0]
        settings = {"model": parts["model"].decode(), "model_revision": "fixture-rev", "dtype": "fp32",
                    "quant": "none", "engine_version": "v1", "temperature": float(parts["temperature"]),
                    "language": parts["language"].decode(), "cache": False, "audio_duration_s": duration,
                    "beam_size": 1, "detected_language": parts["language"].decode(), "language_confidence": 0.99}
        mode = self.server.mode
        text = "hello world" if first_sample < 500 else "second phrase"
        if mode == "wrong" or mode == "heldout" and first_sample >= 500:
            text = "different transcription"
        if mode == "empty":
            text = ""
        if mode == "cache":
            settings["cache"] = True
        if mode == "duration":
            settings["audio_duration_s"] = duration * 2
        if mode == "missing":
            settings.pop("model_revision")
        if mode == "perturbation" and parts["file"] not in self.server.originals:
            text = "degraded from noise"
        response = {"text": text, "effective": settings}
        streaming = parts.get("stream") == b"true"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        self.end_headers()
        if not streaming:
            self.wfile.write(json.dumps(response).encode())
            return
        time.sleep(0.02)
        delta = {"type": "transcript.text.delta", "delta": text}
        self.wfile.write(("data: " + json.dumps(delta) + "\r\n\r\n").encode())
        self.wfile.flush()
        time.sleep(0.02)
        if mode != "truncated":
            self.wfile.write(("data: " + json.dumps({"type": "transcript.text.done", **response}) + "\n\n").encode())


def server(port=0, mode="ok"):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.mode, srv.requests, srv.originals = mode, [], set()
    return srv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", default="ok")
    args = parser.parse_args()
    with server(args.port, args.mode) as httpd:
        httpd.serve_forever()

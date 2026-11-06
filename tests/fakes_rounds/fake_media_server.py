"""Synthetic fresh-process media endpoint; no model or accelerator dependencies."""
import argparse
import base64
import hashlib
import json
import os
import struct
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def png(width, height, seed):
    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value) & 0xffffffff)
    color = hashlib.sha256(str(seed).encode()).digest()[:3]
    raw = (b"\x00" + color * width) * height
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--mode", choices=("audio", "image"), required=True)
parser.add_argument("--fail", action="store_true")
args = parser.parse_args()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ready")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if args.fail:
            self.send_response(500)
            self.end_headers()
            return
        time.sleep(0.01)
        self.send_response(200)
        if args.mode == "audio":
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            pcm = struct.pack("<h", 2000) * body["sample_rate"] * body["channels"]
            events = [{"type": "metadata", "encoding": "pcm_s16le", "sample_rate": body["sample_rate"],
                       "channels": body["channels"]},
                      {"type": "audio", "pcm_b64": base64.b64encode(pcm).decode()},
                      {"type": "final", "sample_count": body["sample_rate"],
                       "effective": {"model": body["model"], "seed": body["seed"], **body["settings"]}}]
            self.wfile.write(b"".join(json.dumps(event).encode() + b"\n" for event in events))
        else:
            effective = {k: v for k, v in body.items() if k not in ("prompt", "size", "steps")}
            effective["steps_run"] = body["steps"]
            raw = json.dumps({"format": "png", "cache": [], "effective": effective,
                              "image_b64": base64.b64encode(png(body["width"], body["height"], body["seed"])).decode()}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)


print(f"pid={os.getpid()}", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

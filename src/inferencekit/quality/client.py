"""A tiny OpenAI-compatible HTTP client (stdlib only; no streaming: quality runs need whole responses).

The API key, if any, is read from an environment variable whose *name* the caller gives
(default ``OPENAI_API_KEY``); its value is sent as a bearer token and never recorded.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from inferencekit.registry import OpError


class HTTPError(OpError):
    """A non-2xx answer from the server; ``status`` and ``message`` come from its body when it has one."""

    def __init__(self, url: str, status: int, message: str):
        super().__init__(f"{url}: HTTP {status}: {message[:500]}")
        self.status = status
        self.message = message


class Client:
    def __init__(self, base_url: str, api_key_env: str | None = "OPENAI_API_KEY", timeout: float = 600):
        self.base = base_url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.timeout = timeout
        self._key = os.environ.get(api_key_env) if api_key_env else None

    def url(self, path: str) -> str:
        return self.base + path

    def _request(self, path: str, body: dict | None) -> dict:
        url = self.url(path)
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
        req.add_header("Content-Type", "application/json")
        if self._key:
            req.add_header("Authorization", f"Bearer {self._key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                obj = json.loads(raw)
                err = obj.get("error", obj)
                msg = err.get("message") if isinstance(err, dict) else str(err)
                msg = msg or obj.get("detail") or raw
            except (ValueError, AttributeError):
                msg = raw
            raise HTTPError(url, e.code, str(msg)) from None
        except (urllib.error.URLError, OSError) as e:
            raise OpError(f"{url}: cannot connect ({getattr(e, 'reason', e)})") from None
        except ValueError:
            raise OpError(f"{url}: response is not JSON") from None

    def post(self, path: str, body: dict) -> dict:
        return self._request(path, body)

    def get(self, path: str) -> dict:
        return self._request(path, None)

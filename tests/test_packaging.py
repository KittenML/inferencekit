"""Packaging: versions agree everywhere, the wheel carries its data, and the installed wheel works.

The build and install tests call ``uv`` (and may need the package index). They skip, with the reason,
when ``uv`` is missing or the index cannot be reached, so the suite still passes offline.
"""

from __future__ import annotations

import importlib.util
import json
import queue
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

import inferencekit
from inferencekit import cli, registry

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "scripts" / "release"
UV = shutil.which("uv")
needs_uv = pytest.mark.skipif(UV is None, reason="uv is not installed")


def _load_check_versions():
    spec = importlib.util.spec_from_file_location("check_versions", RELEASE / "check_versions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check_versions = _load_check_versions()


def _run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=600, **kw)


def _skip_unless_ok(r: subprocess.CompletedProcess, what: str) -> None:
    """Skip when a step failed for lack of network; fail on anything else."""
    if r.returncode == 0:
        return
    err = (r.stderr or "") + (r.stdout or "")
    offline = ("Failed to fetch", "Could not connect", "dns error", "Network is unreachable", "offline",
               "Temporary failure in name resolution", "error sending request")
    if any(s.lower() in err.lower() for s in offline):
        pytest.skip(f"{what} needs the package index, which is unreachable: {err.strip()[-300:]}")
    pytest.fail(f"{what} failed ({r.returncode}):\n{err[-3000:]}")


# ---------------------------------------------------------------- versions


def test_every_version_source_agrees():
    found, errors = check_versions.check(ROOT)
    assert not errors, errors
    assert set(found.values()) == {inferencekit.__version__}
    for source in ("pyproject.toml [project].version", "plugin/.claude-plugin/plugin.json version",
                   ".claude-plugin/marketplace.json metadata.version",
                   ".claude-plugin/marketplace.json plugins[inferencekit].version", "plugin/.mcp.json uvx pin"):
        assert source in found


def _copy_version_sources(dst: Path) -> None:
    for rel in ("pyproject.toml", "src/inferencekit/__init__.py", "plugin/.claude-plugin/plugin.json",
                ".claude-plugin/marketplace.json", "plugin/.mcp.json"):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dst / rel)


def test_check_versions_catches_drift(tmp_path):
    _copy_version_sources(tmp_path)
    assert check_versions.check(tmp_path)[1] == []
    mcp = tmp_path / "plugin/.mcp.json"
    mcp.write_text(mcp.read_text().replace(f"=={inferencekit.__version__}", "==0.0.1"))
    readme = tmp_path / "README.md"
    readme.write_text("uvx --from 'inferencekit-cli[mcp]==9.9.9' inferencekit mcp serve\n"
                      "pin it as inferencekit-cli[mcp]==X\n")  # a placeholder, not a version
    errors = check_versions.check(tmp_path)[1]
    assert any("plugin/.mcp.json uvx pin = 0.0.1" in e for e in errors), errors
    assert any("README.md:1 pin = 9.9.9" in e for e in errors), errors
    assert not any("README.md:2" in e for e in errors), errors
    assert check_versions.main(["--root", str(tmp_path), "--json"]) == 1


def test_check_versions_release_mode_rejects_the_placeholder_url(tmp_path):
    _copy_version_sources(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    text = pyproject.read_text()
    if "github.com/ORG/" not in text:
        text += '\n[project.urls]\nHomepage = "https://github.com/ORG/inferencekit"\n'
        pyproject.write_text(text)
    assert check_versions.check(tmp_path)[1] == []
    errors = check_versions.check(tmp_path, release=True)[1]
    assert errors and all("placeholder repository URL" in e for e in errors)
    pyproject.write_text(text.replace("github.com/ORG/", "github.com/some-org/"))
    assert check_versions.check(tmp_path, release=True)[1] == []


def test_check_versions_release_mode_finds_org_in_readme_docs_and_plugin(tmp_path):
    _copy_version_sources(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace("github.com/ORG/", "github.com/some-org/"))
    files = {
        "README.md": "claude plugin marketplace add ORG/inferencekit\n"
                     '<img src="https://github.com/ORG/inferencekit/raw/main/assets/chart.png">\n'
                     "the placeholder ORG/ is replaced at release\n",  # prose, not a repo reference
        "AGENTS.md": "see <org>/inferencekit and ORGANISATION/inferencekit\n",  # not the placeholder
        "docs/guide.md": "[engines](https://github.com/ORG/inferencekit-engines/tree/main)\n",
        "plugin/skills/x/SKILL.md": "codex plugin marketplace add ORG/inferencekit\n",
        "tests/fixture.md": "ORG/inferencekit\n",  # outside the scanned files
    }
    for rel, body in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(body)
    assert check_versions.check(tmp_path)[1] == []  # only --release looks for the placeholder
    errors = check_versions.check(tmp_path, release=True)[1]
    where = sorted(e.split(": ", 1)[0] for e in errors)
    assert where == ["README.md:1", "README.md:2", "docs/guide.md:1", "plugin/skills/x/SKILL.md:1"], errors
    assert all("placeholder repository URL (ORG)" in e for e in errors)
    for rel, body in files.items():
        (tmp_path / rel).write_text(body.replace("ORG/inferencekit", "some-org/inferencekit"))
    assert check_versions.check(tmp_path, release=True)[1] == []
    assert check_versions.main(["--root", str(tmp_path), "--release", "--json"]) == 0


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"inferencekit {inferencekit.__version__}"
    r = _run([sys.executable, "-m", "inferencekit", "--version"])
    assert r.returncode == 0 and r.stdout.strip() == f"inferencekit {inferencekit.__version__}"


def test_mcp_launcher_is_pinned_to_this_version():
    server = json.loads((ROOT / "plugin/.mcp.json").read_text())["mcpServers"]["inferencekit"]
    assert server["command"] == "uvx"
    assert server["args"] == ["--from", f"inferencekit-cli[mcp]=={inferencekit.__version__}",
                              "inferencekit", "mcp", "serve"]


# ---------------------------------------------------------------- the wheel


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    """A wheel built from the working tree, exactly as `uv build` would publish it."""
    if UV is None:
        pytest.skip("uv is not installed")
    out = tmp_path_factory.mktemp("wheel")
    r = _run([UV, "build", "--wheel", "--out-dir", out, ROOT], cwd=ROOT)
    _skip_unless_ok(r, "uv build")
    (whl,) = out.glob("inferencekit_cli-*.whl")
    return whl


def test_wheel_ships_schemas_data_and_licenses(wheel):
    names = set(zipfile.ZipFile(wheel).namelist())
    assert "inferencekit/schemas/evidence.v1.json" in names
    for schema in (ROOT / "schemas").glob("*.json"):  # every schema version, including later ones
        assert f"inferencekit/schemas/{schema.name}" in names
    prompts = ROOT / "src/inferencekit/prompts"
    if prompts.is_dir():  # the dev prompt set is package data, not only Python files
        for f in prompts.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                assert f"inferencekit/prompts/{f.relative_to(prompts).as_posix()}" in names
    dist_info = f"inferencekit_cli-{inferencekit.__version__}.dist-info"
    assert {f"{dist_info}/licenses/LICENSE", f"{dist_info}/licenses/NOTICE"} <= names
    meta = zipfile.ZipFile(wheel).read(f"{dist_info}/METADATA").decode()
    assert f"Version: {inferencekit.__version__}\n" in meta
    assert "License-Expression: Apache-2.0" in meta
    for dep in ("Requires-Dist: jsonschema", "Requires-Dist: pyyaml", "Provides-Extra: mcp"):
        assert dep in meta
    assert not any(n.startswith(("tests/", "benchmarks/", "evals/", "examples/")) for n in names)


@pytest.fixture(scope="module")
def installed(wheel, tmp_path_factory) -> Path:
    """A fresh venv with the built wheel and its mcp extra installed; returns its bin directory."""
    venv = tmp_path_factory.mktemp("venv") / ".venv"
    r = _run([UV, "venv", "--quiet", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", venv])
    _skip_unless_ok(r, "uv venv")
    r = _run([UV, "pip", "install", "--quiet", "--python", venv / "bin/python", f"{wheel}[mcp]"])
    _skip_unless_ok(r, "uv pip install")
    return venv / "bin"


def test_installed_wheel_reports_its_version(installed):
    r = _run([installed / "inferencekit", "--version"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"inferencekit {inferencekit.__version__}"
    r = _run([installed / "python", "-c",
              "import importlib.resources as r; print(r.files('inferencekit').joinpath('schemas/evidence.v1.json')"
              ".is_file())"])
    assert r.stdout.strip() == "True", r.stderr


def _rpc(proc: subprocess.Popen, lines: queue.Queue, message: dict, timeout: float = 30.0) -> dict | None:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    if "id" not in message:
        return None
    while True:
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            pytest.fail(f"no MCP response to {message['method']} within {timeout}s")
        if line is None:
            pytest.fail(f"MCP server exited before answering {message['method']}: {proc.stderr.read()[-2000:]}")
        reply = json.loads(line)
        if reply.get("id") == message["id"]:
            return reply


def test_installed_wheel_serves_mcp_over_stdio(installed):
    proc = subprocess.Popen([str(installed / "inferencekit"), "mcp", "serve"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines: queue.Queue = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    try:
        init = _rpc(proc, lines, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                  "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "clientInfo": {"name": "test_packaging", "version": "0"}}})
        assert init["result"]["serverInfo"] == {"name": "inferencekit", "version": inferencekit.__version__}
        assert "tools" in init["result"]["capabilities"]
        _rpc(proc, lines, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools, cursor, request_id = [], None, 2
        while True:
            reply = _rpc(proc, lines, {"jsonrpc": "2.0", "id": request_id, "method": "tools/list",
                                       "params": {"cursor": cursor} if cursor else {}})
            tools += [t["name"] for t in reply["result"]["tools"]]
            cursor, request_id = reply["result"].get("nextCursor"), request_id + 1
            if not cursor:
                break
        # The installed wheel exposes exactly the operations this source tree registers.
        assert sorted(tools) == sorted(registry.load_all())
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------- the release build


@needs_uv
@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="build_wheel.sh builds from git archive HEAD")
def test_release_build_is_reproducible_and_hash_installable(tmp_path):
    outs = [tmp_path / "a", tmp_path / "b"]
    for out in outs:
        r = _run(["bash", RELEASE / "build_wheel.sh", out], cwd=ROOT)
        _skip_unless_ok(r, "build_wheel.sh")
    sums = [(o / "SHA256SUMS").read_text() for o in outs]
    assert sums[0] == sums[1], "two builds of the same commit differ"
    (whl,) = outs[0].glob("inferencekit_cli-*.whl")
    assert f"  {whl.name}\n" in sums[0] and "  requirements-hashed.txt\n" in sums[0]
    reqs = (outs[0] / "requirements-hashed.txt").read_text()
    wheel_sha = next(line.split()[0] for line in sums[0].splitlines() if line.endswith(whl.name))
    assert reqs.rstrip().endswith(f"./{whl.name} \\\n    --hash=sha256:{wheel_sha}")
    for pkg in ("jsonschema==", "pyyaml==", "mcp=="):
        assert f"\n{pkg}" in reqs
    assert str(tmp_path) not in reqs and str(ROOT) not in reqs  # no local paths leak into the file

    # The GPU-host flow: install everything hash-checked from inside the output directory.
    venv = tmp_path / "host-venv"
    _skip_unless_ok(_run([UV, "venv", "--quiet", venv]), "uv venv")
    r = _run([UV, "pip", "install", "--quiet", "--python", venv / "bin/python", "--require-hashes",
              "-r", "requirements-hashed.txt"], cwd=outs[0])
    _skip_unless_ok(r, "uv pip install --require-hashes")
    r = _run([venv / "bin/inferencekit", "--version"])
    assert r.returncode == 0 and r.stdout.strip().startswith("inferencekit ")

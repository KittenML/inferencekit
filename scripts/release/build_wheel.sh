#!/usr/bin/env bash
# Reproducible release build of the inferencekit-cli wheel, plus what a GPU host needs to install it
# with every byte hash-checked.
#
# Usage: scripts/release/build_wheel.sh [OUT_DIR]        (default OUT_DIR: <repo>/dist)
#
# Writes to OUT_DIR:
#   inferencekit_cli-<version>-py3-none-any.whl
#   requirements-hashed.txt  every runtime dependency, including the mcp extra, pinned from uv.lock with
#                            sha256 hashes, then the wheel itself by relative path and sha256
#   SHA256SUMS               sha256 of the wheel and of requirements-hashed.txt (`sha256sum -c` format)
#
# Reproducible: the wheel is built from `git archive HEAD` (uncommitted and untracked files never get
# in), with SOURCE_DATE_EPOCH set to the last commit's time, a pinned Python and a hash-pinned build
# backend (build-constraints.txt). The same commit therefore gives the same wheel bytes.
#
# On the GPU host, copy OUT_DIR over and run, from inside it:
#   sha256sum -c SHA256SUMS
#   uv venv && uv pip install --require-hashes -r requirements-hashed.txt
#   .venv/bin/inferencekit --version
# (`python -m pip install --require-hashes -r requirements-hashed.txt` works the same way.)
#
# Environment: PYTHON_VERSION (default 3.12) is the interpreter the build backend runs on.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(git -C "$here" rev-parse --show-toplevel)"
out="${1:-$root/dist}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
py="${PYTHON_VERSION:-3.12}"

die() { echo "build_wheel: $*" >&2; exit 1; }
command -v uv >/dev/null || die "uv not found on PATH (https://docs.astral.sh/uv/)"
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

cd "$root"
commit="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "build_wheel: warning: uncommitted changes are NOT in the wheel; building HEAD ${commit:0:12}" >&2
fi

# Versions must agree before anything is built.
uv run --no-project --python "$py" python "$here/check_versions.py" --root "$root" >/dev/null \
  || die "versions disagree: run scripts/release/check_versions.py"

src="$out/.build-src"
rm -rf "$src"
trap 'rm -rf "$src"' EXIT
mkdir -p "$src"
git archive --format=tar "$commit" | tar -x -C "$src"

export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct "$commit")"
rm -f "$out"/inferencekit_cli-*.whl "$out/requirements-hashed.txt" "$out/SHA256SUMS"
uv build --wheel --python "$py" --out-dir "$out" \
  --build-constraints "$here/build-constraints.txt" --require-hashes "$src" >&2

shopt -s nullglob
wheels=("$out"/inferencekit_cli-*.whl)
[ "${#wheels[@]}" -eq 1 ] || die "expected one wheel in $out, found ${#wheels[@]}"
wheel="$(basename "${wheels[0]}")"
wheel_sha="$(sha256 "$out/$wheel" | cut -d' ' -f1)"

# Runtime dependencies exactly as locked (universal markers, so any Linux / Python >= 3.10 host works),
# then the wheel itself. uv resolves the relative path from the directory the install runs in.
{
  echo "# inferencekit-cli runtime requirements, hash-pinned from uv.lock at commit ${commit:0:12}."
  echo "# Install from this directory: uv pip install --require-hashes -r requirements-hashed.txt"
  uv export --project "$src" --frozen --format requirements.txt --no-dev --extra mcp \
    --no-emit-project --no-header --no-annotate
  echo "./$wheel \\"
  echo "    --hash=sha256:$wheel_sha"
} > "$out/requirements-hashed.txt"

(cd "$out" && sha256 "$wheel" requirements-hashed.txt > SHA256SUMS)

echo "commit           $commit"
echo "SOURCE_DATE_EPOCH $SOURCE_DATE_EPOCH"
echo "wheel            $out/$wheel"
echo "wheel sha256     $wheel_sha"
echo "requirements     $out/requirements-hashed.txt"
echo "checksums        $out/SHA256SUMS"

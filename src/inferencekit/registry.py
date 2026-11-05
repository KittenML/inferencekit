"""The operation registry: the single source of truth for what inferencekit can do.

Each operation is a plain Python function decorated with :func:`operation`. The
decorator records its name (the MCP tool name), its CLI path (``("nsys", "analyze")``
becomes ``inferencekit nsys analyze``), a one-line summary and an argument list. The
CLI (``cli.py``) and the MCP server (``mcp_server.py``) are generated from this
registry, so the two interfaces cannot drift apart.

Handlers are called as ``handler(call, **validated_args)`` and return a JSON-able
dict (usually an evidence record). ``call`` carries what the evidence layer needs
to write provenance: which operation ran, with which arguments, and when.
"""

from __future__ import annotations

import copy
import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Callable

from inferencekit.canonical import digest

# JSON-schema type names used for arguments.
_TYPES = {"string", "integer", "number", "boolean", "array"}


class OpError(Exception):
    """A user-facing failure (bad input, missing tool). Printed without a traceback."""


@dataclass(frozen=True)
class Arg:
    """One argument of an operation.

    ``positional`` only affects the CLI (``inferencekit compare A B`` instead of flags).
    ``sensitive`` marks command lines and similar free text: such values are used,
    but only their digest is ever written to an evidence record.
    """

    name: str
    type: str
    help: str
    required: bool = False
    default: Any = None
    choices: tuple | None = None
    positional: bool = False
    items: str = "string"  # element type when type == "array"
    sensitive: bool = False

    def __post_init__(self):
        if self.type not in _TYPES:
            raise ValueError(f"arg {self.name}: unknown type {self.type}")

    def json_schema(self) -> dict:
        if self.type == "array":
            schema: dict = {"type": "array", "items": {"type": self.items}}
            if self.choices:
                schema["items"]["enum"] = list(self.choices)
        else:
            schema = {"type": self.type}
            if self.choices:
                schema["enum"] = list(self.choices)
        schema["description"] = self.help
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass(frozen=True)
class Operation:
    name: str
    cli: tuple[str, ...]
    summary: str
    args: tuple[Arg, ...]
    handler: Callable[..., dict]
    passthrough: bool = False  # True: takes a command after "--" as the "argv" argument

    def input_schema(self) -> dict:
        """JSON Schema for the MCP tool's input (same names and defaults as the CLI)."""
        return {
            "type": "object",
            "properties": {a.name: a.json_schema() for a in self.args},
            "required": [a.name for a in self.args if a.required],
            "additionalProperties": False,
        }

    def validate(self, raw: dict | None) -> dict:
        """Check types, fill defaults, reject unknown names. Same rules for CLI and MCP."""
        raw = dict(raw or {})
        known = {a.name for a in self.args}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise OpError(f"{self.name}: unknown argument(s) {unknown}; expected {sorted(known)}")
        out = {}
        for a in self.args:
            value = raw.get(a.name)
            if value is None:
                if a.required:
                    raise OpError(f"{self.name}: missing required argument '{a.name}'")
                value = copy.deepcopy(a.default)
            else:
                value = _coerce(self.name, a, value)
            if a.choices and value is not None:
                for v in value if isinstance(value, list) else [value]:
                    if v not in a.choices:
                        raise OpError(f"{self.name}: {a.name}={v!r} is not one of {list(a.choices)}")
            out[a.name] = value
        return out


def _coerce(op: str, a: Arg, value: Any) -> Any:
    """Convert CLI strings (and lenient MCP inputs) to the declared type."""
    try:
        if a.type == "integer":
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError
            return int(value)
        if a.type == "number":
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if a.type == "boolean":
            if isinstance(value, bool):
                return value
            if str(value).lower() in ("1", "true", "yes", "on"):
                return True
            if str(value).lower() in ("0", "false", "no", "off"):
                return False
            raise ValueError
        if a.type == "array":
            if isinstance(value, (str, int, float)):
                value = [value]
            sub = Arg(a.name, a.items, a.help)
            return [_coerce(op, sub, v) for v in value]
        return str(value)
    except (TypeError, ValueError):
        raise OpError(f"{op}: argument '{a.name}' expects {a.type}, got {value!r}") from None


@dataclass
class Call:
    """One invocation of an operation; handed to the handler for provenance."""

    op: Operation
    args: dict
    started: str = field(default_factory=lambda: utc_now())

    def argv_digest(self) -> str:
        # Digest of the full invocation (operation path + every argument, including
        # sensitive ones). Lets you check "was this the same command?" without storing it.
        return digest(["inferencekit", *self.op.cli, self.args])

    def request(self) -> dict:
        """The invocation with sensitive values replaced by digests (safe to store)."""
        safe = {}
        for a in self.op.args:
            v = self.args.get(a.name)
            if a.sensitive and v not in (None, [], ""):
                v = {"digest": digest(v)}
            safe[a.name] = v
        return {"op": self.op.name, "cli": "inferencekit " + " ".join(self.op.cli), "args": safe}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


REGISTRY: dict[str, Operation] = {}


def operation(name: str, cli: tuple[str, ...], summary: str, args: list[Arg] | tuple[Arg, ...] = (),
              passthrough: bool = False):
    """Decorator: register ``fn`` as operation ``name``."""

    def register(fn: Callable[..., dict]) -> Callable[..., dict]:
        if name in REGISTRY:
            raise ValueError(f"operation {name} registered twice")
        all_args = tuple(args)
        if passthrough:
            all_args += (Arg("argv", "array", "Command to run (after '--' on the CLI).", required=True,
                             sensitive=True),)
        REGISTRY[name] = Operation(name, tuple(cli), summary, all_args, fn, passthrough)
        return fn

    return register


def load_all() -> dict[str, Operation]:
    """Import every module in inferencekit.ops so their decorators run."""
    import inferencekit.ops  # noqa: F401  (the package __init__ imports each op module)

    return REGISTRY


def get(name: str) -> Operation:
    load_all()
    if name not in REGISTRY:
        raise OpError(f"unknown operation '{name}'; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def call(name: str, raw_args: dict | None = None) -> dict:
    """Validate arguments and run an operation. Both front ends end up here."""
    op = get(name)
    args = op.validate(raw_args)
    return op.handler(Call(op, args), **args)

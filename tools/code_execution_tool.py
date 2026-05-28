#!/usr/bin/env python3
"""
Code Execution Tool -- Programmatic Tool Calling (PTC)

Lets the LLM write a Python script that calls Hermes tools via RPC,
collapsing multi-step tool chains into a single inference turn.

Architecture (two transports):

  **Local backend (UDS):**
  1. Parent generates a `hermes_tools.py` stub module with UDS RPC functions
  2. Parent opens a Unix domain socket and starts an RPC listener thread
  3. Parent spawns a child process that runs the LLM's script
  4. Tool calls travel over the UDS back to the parent for dispatch

  **Remote backends (file-based RPC):**
  1. Parent generates `hermes_tools.py` with file-based RPC stubs
  2. Parent ships both files to the remote environment
  3. Script runs inside the terminal backend (Docker/SSH/Modal/Daytona/etc.)
  4. Tool calls are written as request files; a polling thread on the parent
     reads them via env.execute(), dispatches, and writes response files
  5. The script polls for response files and continues

In both cases, only the script's stdout is returned to the LLM; intermediate
tool results never enter the context window.

Platform: Linux / macOS only (Unix domain sockets for local). Disabled on Windows.
Remote execution additionally requires Python 3 in the terminal backend.
"""

import base64
import functools
import json
import keyword as _keyword
import logging
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid

_IS_WINDOWS = platform.system() == "Windows"
from typing import Any, Dict, List, Optional, Set, Tuple

# Availability gate.  On Windows we fall back to loopback TCP for the
# sandbox RPC transport (AF_UNIX is unreliable on Windows Python) — see
# ``_use_tcp_rpc`` in ``_execute_local`` below.  That makes execute_code
# available on every platform Hermes itself runs on.
logger = logging.getLogger(__name__)

SANDBOX_AVAILABLE = True

# The 7 tools allowed inside the sandbox. The intersection of this list
# and the session's enabled tools determines which stubs are generated.
SANDBOX_ALLOWED_TOOLS = frozenset([
    "web_search",
    "web_extract",
    "read_file",
    "write_file",
    "search_files",
    "patch",
    "terminal",
])

# Experimental: when ``code_execution.expose_mcp_tools=true``, connected
# MCP-server tools become importable inside the sandbox as
# ``from hermes_mcp.<server> import <tool>``.  The RPC dispatch path in
# this module bypasses ``check_all_command_guards()`` (issues #4146 and
# #30882), so MCP calls originating from sandbox scripts inherit that
# bypass.  Keep the flag off in production until those land upstream.

# Resource limit defaults (overridable via config.yaml → code_execution.*)
DEFAULT_TIMEOUT = 300        # 5 minutes
DEFAULT_MAX_TOOL_CALLS = 50
MAX_STDOUT_BYTES = 50_000    # 50 KB
MAX_STDERR_BYTES = 10_000    # 10 KB

# Environment variable scrubbing rules (shared between the local + remote
# backends).  Secret-substring block is applied first; anything left must
# match either a safe prefix or, on Windows, an OS-essential name.
_SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                      "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                      "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA",
                      "HERMES_")
_SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                      "PASSWD", "AUTH")

# Windows-only: a handful of variables are required by the OS/CRT itself.
# Without them, even stdlib calls like ``socket.socket()`` fail with
# WinError 10106 (Winsock can't locate mswsock.dll) and ``subprocess``
# can't resolve cmd.exe.  These are well-known OS paths, not secrets, so
# we allow them through by exact name.  The _SECRET_SUBSTRINGS block
# still runs as a safety net (none of these names match those substrings).
_WINDOWS_ESSENTIAL_ENV_VARS = frozenset({
    "SYSTEMROOT",       # %SYSTEMROOT%\System32 — Winsock needs this
    "SYSTEMDRIVE",      # C: (or wherever Windows lives)
    "WINDIR",           # usually same as SYSTEMROOT
    "COMSPEC",          # cmd.exe path — subprocess shell=True needs it
    "PATHEXT",          # .COM;.EXE;.BAT;... — shell lookup
    "OS",               # "Windows_NT" — some tools gate on this
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "PUBLIC",           # C:\Users\Public
    "ALLUSERSPROFILE",  # C:\ProgramData — some stdlib paths use it
    "PROGRAMDATA",      # C:\ProgramData
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "APPDATA",          # %USERPROFILE%\AppData\Roaming — Python uses it
    "LOCALAPPDATA",     # %USERPROFILE%\AppData\Local
    "USERPROFILE",      # C:\Users\<name> — Python's expanduser uses it
    "USERDOMAIN",
    "USERNAME",
    "HOMEDRIVE",        # C:
    "HOMEPATH",         # \Users\<name>
    "COMPUTERNAME",
})


def _scrub_child_env(source_env, is_passthrough=None, is_windows=None):
    """Produce the scrubbed child-process env for execute_code.

    Rules (order matters):
      1. Passthrough vars (skill- or config-declared) always pass.
      2. Secret-substring names (KEY/TOKEN/etc.) are blocked.
      3. Names matching a safe prefix pass.
      4. On Windows, a small OS-essential allowlist passes by exact name
         — without these the child can't even create a socket or spawn a
         subprocess.

    Extracted into a helper so tests can exercise the logic without
    spawning a subprocess.
    """
    if is_passthrough is None:
        try:
            from tools.env_passthrough import is_env_passthrough as _ep
        except Exception:
            _ep = lambda _: False  # noqa: E731
        is_passthrough = _ep
    if is_windows is None:
        is_windows = _IS_WINDOWS

    scrubbed = {}
    for k, v in source_env.items():
        if is_passthrough(k):
            scrubbed[k] = v
            continue
        if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
            continue
        if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
            scrubbed[k] = v
            continue
        if is_windows and k.upper() in _WINDOWS_ESSENTIAL_ENV_VARS:
            scrubbed[k] = v
    return scrubbed


def check_sandbox_requirements() -> bool:
    """Code execution sandbox requires a POSIX OS for Unix domain sockets."""
    if not SANDBOX_AVAILABLE:
        return False
    return True


# ---------------------------------------------------------------------------
# hermes_tools.py code generator
# ---------------------------------------------------------------------------

# Per-tool stub templates: (function_name, signature, docstring, args_dict_expr)
# The args_dict_expr builds the JSON payload sent over the RPC socket.
_TOOL_STUBS = {
    "web_search": (
        "web_search",
        "query: str, limit: int = 5",
        '"""Search the web. Returns dict with data.web list of {url, title, description}."""',
        '{"query": query, "limit": limit}',
    ),
    "web_extract": (
        "web_extract",
        "urls: list",
        '"""Extract content from URLs. Returns dict with results list of {url, title, content, error}."""',
        '{"urls": urls}',
    ),
    "read_file": (
        "read_file",
        "path: str, offset: int = 1, limit: int = 500",
        '"""Read a file (1-indexed lines). Returns dict with "content" and "total_lines"."""',
        '{"path": path, "offset": offset, "limit": limit}',
    ),
    "write_file": (
        "write_file",
        "path: str, content: str, cross_profile: bool = False",
        '"""Write content to a file (always overwrites). Returns dict with status. cross_profile=True opts out of the cross-Hermes-profile soft guard."""',
        '{"path": path, "content": content, "cross_profile": cross_profile}',
    ),
    "search_files": (
        "search_files",
        'pattern: str, target: str = "content", path: str = ".", file_glob: str = None, limit: int = 50, offset: int = 0, output_mode: str = "content", context: int = 0',
        '"""Search file contents (target="content") or find files by name (target="files"). Returns dict with "matches"."""',
        '{"pattern": pattern, "target": target, "path": path, "file_glob": file_glob, "limit": limit, "offset": offset, "output_mode": output_mode, "context": context}',
    ),
    "patch": (
        "patch",
        'path: str = None, old_string: str = None, new_string: str = None, replace_all: bool = False, mode: str = "replace", patch: str = None, cross_profile: bool = False',
        '"""Targeted find-and-replace (mode="replace") or V4A multi-file patches (mode="patch"). Returns dict with status. cross_profile=True opts out of the cross-Hermes-profile soft guard."""',
        '{"path": path, "old_string": old_string, "new_string": new_string, "replace_all": replace_all, "mode": mode, "patch": patch, "cross_profile": cross_profile}',
    ),
    "terminal": (
        "terminal",
        "command: str, timeout: int = None, workdir: str = None",
        '"""Run a shell command (foreground only). Returns dict with "output" and "exit_code"."""',
        '{"command": command, "timeout": timeout, "workdir": workdir}',
    ),
}


def generate_hermes_tools_module(enabled_tools: List[str],
                                 transport: str = "uds",
                                 mcp_enabled: bool = False) -> str:
    """
    Build the source code for the hermes_tools.py stub module.

    Only tools in both SANDBOX_ALLOWED_TOOLS and enabled_tools get stubs.

    Args:
        enabled_tools: Tool names enabled in the current session.
        transport: ``"uds"`` for Unix domain socket (local backend) or
                   ``"file"`` for file-based RPC (remote backends).
        mcp_enabled: When True, appends the shared MCP validator
            infrastructure (jsonschema import, the validator registries,
            and the ``_validate_input`` / ``_validate_output`` helpers)
            so generated ``hermes_mcp/<server>.py`` modules don't have to
            duplicate them.  Keep False when no MCP wrappers will be
            shipped — avoids pulling jsonschema into the sandbox env.
    """
    tools_to_generate = sorted(SANDBOX_ALLOWED_TOOLS & set(enabled_tools))

    stub_functions = []
    export_names = []
    for tool_name in tools_to_generate:
        if tool_name not in _TOOL_STUBS:
            continue
        func_name, sig, doc, args_expr = _TOOL_STUBS[tool_name]
        stub_functions.append(
            f"def {func_name}({sig}):\n"
            f"    {doc}\n"
            f"    return _call({func_name!r}, {args_expr})\n"
        )
        export_names.append(func_name)

    if transport == "file":
        header = _FILE_TRANSPORT_HEADER
    else:
        header = _UDS_TRANSPORT_HEADER

    body = header + "\n".join(stub_functions)
    if mcp_enabled:
        body += _MCP_VALIDATOR_INFRASTRUCTURE
    return body


# Shared MCP validator infrastructure — appended to hermes_tools.py once
# per execute_code run whenever MCP wrappers are present.  Lets each
# generated ``hermes_mcp/<server>.py`` module simply
# ``from hermes_tools import _validate_input, _validate_output,
# _register_input_schema, _register_output_schema`` instead of carrying
# its own copy.  Keyed on the full ``mcp_<server>_<tool>`` registry
# name so two servers with the same short tool name don't collide in
# the shared dicts.
_MCP_VALIDATOR_INFRASTRUCTURE = '''
# ---------------------------------------------------------------------------
# MCP input/output schema validators (shared across hermes_mcp/<server>.py).
# ---------------------------------------------------------------------------
import warnings as _warnings
import jsonschema as _jsonschema

_INPUT_VALIDATORS = {}
_OUTPUT_VALIDATORS = {}


def _register_input_schema(registry_name, schema):
    _INPUT_VALIDATORS[registry_name] = _jsonschema.Draft202012Validator(schema)


def _register_output_schema(registry_name, schema):
    _OUTPUT_VALIDATORS[registry_name] = _jsonschema.Draft202012Validator(schema)


def _validate_input(registry_name, args):
    """Raise ValueError listing every input-side schema violation."""
    v = _INPUT_VALIDATORS.get(registry_name)
    if v is None:
        return
    errors = sorted(v.iter_errors(args), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    lines = [f"MCP tool {registry_name!r} input validation failed:"]
    for e in errors:
        path = ".".join(str(p) for p in e.absolute_path) or "(root)"
        lines.append(f"  - {path}: {e.message}")
    raise ValueError("\\n".join(lines))


def _validate_output(registry_name, result):
    """Warn (don't raise) on output-side drift — return result unchanged."""
    v = _OUTPUT_VALIDATORS.get(registry_name)
    if v is None:
        return result
    errors = sorted(v.iter_errors(result), key=lambda e: list(e.absolute_path))
    if not errors:
        return result
    lines = [f"MCP tool {registry_name!r} output does not match its declared schema:"]
    for e in errors:
        path = ".".join(str(p) for p in e.absolute_path) or "(root)"
        lines.append(f"  - {path}: {e.message}")
    _warnings.warn("\\n".join(lines), RuntimeWarning, stacklevel=3)
    return result
'''


# ---------------------------------------------------------------------------
# Experimental: MCP-server stubs (opt-in via code_execution.expose_mcp_tools)
# ---------------------------------------------------------------------------

# Names we must not shadow in the generated wrapper bodies.  Combined with
# Python's reserved words, used to filter unsafe parameter names.
_RESERVED_PY_NAMES: frozenset = (
    frozenset(_keyword.kwlist)
    | frozenset(_keyword.softkwlist)
    | frozenset({"_call", "_args", "_result", "_validate_input",
                 "_validate_output", "kwargs", "match", "case"})
)


def _is_safe_py_identifier(name: str) -> bool:
    """Return True if ``name`` is a valid Python identifier that isn't reserved."""
    return (
        isinstance(name, str)
        and name.isidentifier()
        and name not in _RESERVED_PY_NAMES
    )


def _build_defs_map(schema) -> Dict[str, dict]:
    """Build a flat ``$defs`` / ``definitions`` lookup table from a schema."""
    if not isinstance(schema, dict):
        return {}
    defs = schema.get("$defs") or schema.get("definitions") or {}
    return defs if isinstance(defs, dict) else {}


def _resolve_local_ref(ref: str, defs: Dict[str, dict]) -> Optional[dict]:
    """Resolve ``#/$defs/<name>`` or ``#/definitions/<name>`` against ``defs``."""
    if not isinstance(ref, str):
        return None
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            target = defs.get(ref[len(prefix):])
            return target if isinstance(target, dict) else None
    return None


def _schema_to_python_hint(
    node,
    defs: Dict[str, dict],
    _seen: Optional[Set[str]] = None,
) -> str:
    """Map a JSON Schema 2020-12 node to a Python type-hint expression.

    The returned string is valid Python source assuming
    ``from typing import Any, Literal, Optional, Union`` is in scope.
    Falls back to ``Any`` for unresolvable refs, recursive structures, and
    constructs that have no faithful type-system representation
    (``allOf``, ``if`` / ``then`` / ``else``).
    """
    if _seen is None:
        _seen = set()
    if not isinstance(node, dict):
        return "Any"

    # $ref — resolve once, fall back on recursion.
    if "$ref" in node:
        ref = node["$ref"]
        if ref in _seen:
            return "Any"
        resolved = _resolve_local_ref(ref, defs)
        if resolved is None:
            return "Any"
        return _schema_to_python_hint(resolved, defs, _seen | {ref})

    # const / enum → Literal.
    if "const" in node:
        return f"Literal[{node['const']!r}]"
    if "enum" in node:
        values = node["enum"]
        if isinstance(values, list) and values:
            return "Literal[" + ", ".join(repr(v) for v in values) + "]"
        return "Any"

    # oneOf / anyOf → Union, collapsing the ``null`` arm to Optional.
    for combinator in ("oneOf", "anyOf"):
        arms = node.get(combinator)
        if not isinstance(arms, list) or not arms:
            continue
        non_null: List[str] = []
        has_null = False
        for arm in arms:
            if isinstance(arm, dict) and arm.get("type") == "null":
                has_null = True
                continue
            non_null.append(_schema_to_python_hint(arm, defs, _seen))
        non_null = list(dict.fromkeys(non_null))  # dedupe preserving order
        if not non_null:
            return "None"
        if len(non_null) == 1:
            return f"Optional[{non_null[0]}]" if has_null else non_null[0]
        joined = ", ".join(non_null)
        return f"Union[{joined}, None]" if has_null else f"Union[{joined}]"

    # allOf and conditionals — no faithful single-hint representation.
    if "allOf" in node or "if" in node or "then" in node or "else" in node:
        return "Any"

    # ``type`` can be a list (``["string", "null"]`` form).
    t = node.get("type")
    if isinstance(t, list):
        has_null = "null" in t
        non_null_types = [x for x in t if x != "null"]
        if not non_null_types:
            return "None"
        if len(non_null_types) == 1:
            inner = _schema_to_python_hint(
                {**node, "type": non_null_types[0]}, defs, _seen
            )
            return f"Optional[{inner}]" if has_null else inner
        parts = [
            _schema_to_python_hint({**node, "type": tt}, defs, _seen)
            for tt in non_null_types
        ]
        parts = list(dict.fromkeys(parts))
        joined = ", ".join(parts)
        return f"Union[{joined}, None]" if has_null else f"Union[{joined}]"

    if t == "string":
        return "str"
    if t == "integer":
        return "int"
    if t == "number":
        return "float"
    if t == "boolean":
        return "bool"
    if t == "null":
        return "None"
    if t == "array":
        items = node.get("items")
        if items is None:
            return "list[Any]"
        return f"list[{_schema_to_python_hint(items, defs, _seen)}]"
    if t == "object":
        return "dict[str, Any]"

    return "Any"


def _render_field_constraints(node) -> str:
    """Return a ``(constraint: v, ...)`` suffix for a property node, or ``""``.

    Surfaces per-field constraints the Python type system can't encode, so
    the model still reads them when scanning the wrapper docstring.
    """
    if not isinstance(node, dict):
        return ""
    parts: List[str] = []
    for key in ("pattern", "format"):
        if key in node:
            parts.append(f"{key}: {node[key]}")
    for key in ("minLength", "maxLength",
                "minimum", "maximum",
                "exclusiveMinimum", "exclusiveMaximum",
                "multipleOf",
                "minItems", "maxItems"):
        if key in node:
            parts.append(f"{key}: {node[key]}")
    if node.get("uniqueItems"):
        parts.append("uniqueItems")
    if "default" in node:
        parts.append(f"default: {node['default']!r}")
    return f"({', '.join(parts)})" if parts else ""


def _render_cross_field_constraints(schema) -> List[str]:
    """Return bullet strings for object-level constraints.

    These go in the ``Constraints:`` section of the docstring — the rules that
    cross multiple properties and can't sit next to a single ``Args:`` line.
    """
    if not isinstance(schema, dict):
        return []
    bullets: List[str] = []

    deps = schema.get("dependentRequired")
    if isinstance(deps, dict):
        for trigger, required_others in deps.items():
            if isinstance(required_others, list) and required_others:
                tail = ", ".join(f"`{r}`" for r in required_others)
                bullets.append(
                    f"When `{trigger}` is provided, {tail} also required."
                )

    if isinstance(schema.get("dependentSchemas"), dict):
        bullets.append(
            "Has `dependentSchemas` — see Input schema (JSON) below for the rules."
        )

    if "if" in schema:
        bullets.append(
            "Conditional shape (`if`/`then`/`else`) — see Input schema (JSON) below."
        )

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of and schema.get("type") == "object":
        bullets.append(
            f"One of {len(one_of)} mutually-exclusive shapes "
            f"(`oneOf` at object level) — see Input schema (JSON)."
        )

    return bullets


def _partition_properties(
    schema,
) -> Tuple[List[Tuple[str, str, dict]], List[Tuple[str, str, dict]], bool, bool]:
    """Split a schema into (required, optional, accepts_kwargs, has_any_props).

    Returns:
        (required, optional, accepts_kwargs, has_any_props), where each entry
        in ``required`` / ``optional`` is ``(json_name, py_name, node)``.
        Properties whose JSON name can't be expressed as a Python identifier
        are dropped from these lists — they'll still be accepted via
        ``**kwargs`` and validated at runtime.  ``accepts_kwargs`` is False
        only when ``additionalProperties: false`` AND every property was
        py-safe (so the signature can be closed).
    """
    if not isinstance(schema, dict):
        return [], [], True, False

    props = schema.get("properties")
    if not isinstance(props, dict):
        return [], [], schema.get("additionalProperties", True) is not False, False

    required_set = set(
        r for r in (schema.get("required") or []) if isinstance(r, str)
    )

    required: List[Tuple[str, str, dict]] = []
    optional: List[Tuple[str, str, dict]] = []
    used: Set[str] = set()
    dropped_any = False

    for json_name, node in props.items():
        if not isinstance(node, dict):
            node = {}
        py_name = json_name if _is_safe_py_identifier(json_name) else None
        if py_name is None and isinstance(json_name, str) and json_name in _RESERVED_PY_NAMES:
            candidate = json_name + "_"
            if _is_safe_py_identifier(candidate) and candidate not in used:
                py_name = candidate
        if py_name is None or py_name in used:
            dropped_any = True
            continue
        used.add(py_name)
        entry = (json_name, py_name, node)
        if json_name in required_set:
            required.append(entry)
        else:
            optional.append(entry)

    accepts_kwargs = schema.get("additionalProperties", True) is not False or dropped_any
    return required, optional, accepts_kwargs, bool(props)


def _emit_signature_params(input_schema) -> Tuple[str, List[Tuple[str, str]]]:
    """Build the parameter list for the generated def line.

    Returns:
        ``(signature_string, py_to_json_map)``.  The map preserves the
        original JSON property names so the wrapper body can rebuild the
        args dict before validation / dispatch.
    """
    defs = _build_defs_map(input_schema)
    required, optional, accepts_kwargs, _ = _partition_properties(input_schema)
    py_to_json: List[Tuple[str, str]] = []
    parts: List[str] = []

    for json_name, py_name, node in required:
        hint = _schema_to_python_hint(node, defs)
        parts.append(f"{py_name}: {hint}")
        py_to_json.append((py_name, json_name))

    for json_name, py_name, node in optional:
        hint = _schema_to_python_hint(node, defs)
        # Optional[T] for non-nullable hints; the wrapper body strips
        # None values before validation, so passing ``state=None`` means
        # "field omitted" — the JSON Schema idiom for absence.
        if hint != "Any" and not hint.startswith("Optional["):
            hint = f"Optional[{hint}]"
        parts.append(f"{py_name}: {hint} = None")
        py_to_json.append((py_name, json_name))

    if accepts_kwargs:
        parts.append("**kwargs")

    return ", ".join(parts), py_to_json


def emit_short_signature(input_schema) -> str:
    """Return a compact ``name: Type, name: Type`` string for the README catalog.

    Lists only the required parameters with their type hints — the README is
    an index, not a reference, so optional params and defaults are out of
    scope here.  Reused by ``mcp_code_discovery._render_readme_markdown`` so
    catalog signatures stay aligned with the generated wrapper signatures.
    """
    defs = _build_defs_map(input_schema)
    required, _, _, _ = _partition_properties(input_schema)
    parts = [
        f"{py_name}: {_schema_to_python_hint(node, defs)}"
        for _, py_name, node in required
    ]
    return ", ".join(parts)


def _emit_return_hint(output_schema) -> str:
    """Build the return-type annotation for a tool wrapper.

    ``-> Any`` when no outputSchema was advertised — an explicit ``Any`` is the
    honest signal that the server didn't promise a shape.
    """
    if not isinstance(output_schema, dict):
        return "Any"
    defs = _build_defs_map(output_schema)
    return _schema_to_python_hint(output_schema, defs)


def _render_args_section(input_schema) -> List[str]:
    """Build the ``Args:`` docstring section as a list of indented lines."""
    if not isinstance(input_schema, dict):
        return []
    props = input_schema.get("properties")
    if not isinstance(props, dict) or not props:
        return []

    required_set = set(
        r for r in (input_schema.get("required") or []) if isinstance(r, str)
    )
    defs = _build_defs_map(input_schema)
    lines = ["Args:"]
    for json_name, node in props.items():
        if not isinstance(node, dict):
            node = {}
        hint = _schema_to_python_hint(node, defs)
        desc = (node.get("description") or "").strip().splitlines()
        desc_first = desc[0] if desc else ""
        constraints = _render_field_constraints(node)
        req_tag = "" if json_name in required_set else " (optional)"
        head = f"    {json_name} ({hint}){req_tag}:"
        trail = " ".join(filter(None, [desc_first, constraints]))
        lines.append(f"{head} {trail}".rstrip())
    return lines


def _render_returns_section(output_schema) -> List[str]:
    """Build the ``Returns:`` docstring section — unfolds one level of object shape."""
    if not isinstance(output_schema, dict):
        return []
    defs = _build_defs_map(output_schema)
    hint = _schema_to_python_hint(output_schema, defs)
    desc = (output_schema.get("description") or "").strip().splitlines()
    desc_first = desc[0] if desc else ""
    constraints = _render_field_constraints(output_schema)
    head_trail = " ".join(filter(None, [desc_first, constraints]))
    lines = [f"Returns:", f"    {hint}: {head_trail}".rstrip()]

    # Unfold one level: object → properties, array-of-object → properties.
    nested = output_schema
    if output_schema.get("type") == "array":
        items = output_schema.get("items")
        if isinstance(items, dict):
            nested = items

    if isinstance(nested, dict) and isinstance(nested.get("properties"), dict):
        sub_props = nested["properties"]
        if sub_props:
            lines.append("        Fields:")
            for sub_name, sub_node in sub_props.items():
                if not isinstance(sub_node, dict):
                    sub_node = {}
                sub_hint = _schema_to_python_hint(sub_node, defs)
                sub_desc = (sub_node.get("description") or "").strip().splitlines()
                sub_desc_first = sub_desc[0] if sub_desc else ""
                sub_constraints = _render_field_constraints(sub_node)
                trailer = " ".join(filter(None, [sub_desc_first, sub_constraints]))
                lines.append(
                    f"            {sub_name} ({sub_hint}): {trailer}".rstrip()
                )
    return lines


def _render_constraints_section(input_schema, output_schema) -> List[str]:
    """Build the ``Constraints:`` section — cross-field rules from both schemas."""
    bullets = (
        _render_cross_field_constraints(input_schema)
        + _render_cross_field_constraints(output_schema)
    )
    if not bullets:
        return []
    lines = ["Constraints:"]
    for b in bullets:
        lines.append(f"    - {b}")
    return lines


def _render_tool_docstring(
    description: str,
    input_schema,
    output_schema,
    server_name: str,
    tool_name: str,
) -> str:
    """Build the full multi-section docstring for one generated wrapper.

    Sections (in order):
      <description>
      Args:
      Returns:
      Constraints:
      MCP server / tool metadata
      Input schema (JSON) and Output schema (JSON) fallbacks
    """
    parts: List[str] = []
    desc = (description or "").strip()
    if desc:
        parts.append(desc)

    args_lines = _render_args_section(input_schema)
    if args_lines:
        if parts:
            parts.append("")
        parts.extend(args_lines)

    returns_lines = _render_returns_section(output_schema)
    if returns_lines:
        if parts:
            parts.append("")
        parts.extend(returns_lines)

    constraint_lines = _render_constraints_section(input_schema, output_schema)
    if constraint_lines:
        if parts:
            parts.append("")
        parts.extend(constraint_lines)

    if parts:
        parts.append("")
    parts.append(f"MCP server: {server_name}")
    parts.append(f"MCP tool:   {tool_name}")
    # Machine-readable schema fallback lives at module scope as the
    # ``_INPUT_SCHEMA_<tool>`` / ``_OUTPUT_SCHEMA_<tool>`` Python literals
    # the validators are built from — same bytes, same read-depth, no
    # duplication.  See ``_emit_tool_module``.

    # Escape any embedded triple-quotes that would close the docstring early.
    text = "\n".join(parts).replace('"""', "''")
    return text


# Per-server module preamble — small now that the validator infrastructure
# lives in the shared ``hermes_tools`` module.  Just the typing imports and
# the dispatch / validator function imports.
_MCP_WRAPPER_HEADER = '''\
"""Auto-generated stubs for MCP server {server_name!r}.

Function signatures and docstrings are derived from each tool's inputSchema
/ outputSchema; ``_validate_input`` / ``_validate_output`` (imported from
``hermes_tools``) enforce the full JSON Schema 2020-12 contract at runtime.
Input violations raise ``ValueError``; output violations warn via
``RuntimeWarning`` so call sites stay robust to server drift.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from hermes_tools import (
    _call,
    _register_input_schema,
    _register_output_schema,
    _validate_input,
    _validate_output,
)


'''


def _emit_tool_module(server_name: str, mcp_tools) -> Tuple[str, Set[str], List[str]]:
    """Build the source of one ``hermes_mcp/<server>.py`` module.

    Returns ``(source, allowed_names, emitted_tool_names)``.
    """
    from tools.mcp_tool import sanitize_mcp_name_component  # local import: avoid cycle

    safe_server = sanitize_mcp_name_component(server_name)

    lines: List[str] = [_MCP_WRAPPER_HEADER.format(server_name=server_name)]
    allowed_names: Set[str] = set()
    emitted_tools: List[str] = []

    for mcp_tool in mcp_tools:
        tool_name = getattr(mcp_tool, "name", None)
        if not tool_name:
            continue
        safe_tool = sanitize_mcp_name_component(tool_name)
        registry_name = f"mcp_{safe_server}_{safe_tool}"

        description = getattr(mcp_tool, "description", "") or ""
        input_schema = getattr(mcp_tool, "inputSchema", None)
        # outputSchema lands here when the server is post-SEP-2106 (JSON
        # Schema 2020-12 alignment, merged 2026-05).  Carrying it through
        # gives the model a typed return + runtime drift detection.
        output_schema = getattr(mcp_tool, "outputSchema", None)

        sig_params, py_to_json = _emit_signature_params(input_schema)
        return_hint = _emit_return_hint(output_schema)
        docstring = _render_tool_docstring(
            description, input_schema, output_schema, server_name, tool_name
        )

        # The signature line is one Python statement — wrap it so long
        # generated lines stay readable when the agent reads the file.
        sig_line = f"def {safe_tool}({sig_params}) -> {return_hint}:"

        # Raw docstring (``r"""..."""``) so regex patterns embedded in
        # the schema (``\d+``, ``\w*``) don't trip Python's
        # SyntaxWarning for unknown escape sequences at module import.
        # Edge case: a raw string can't end in a single backslash —
        # append a space when it does.
        safe_doc = docstring.replace("\n", "\n    ")
        if safe_doc.endswith("\\"):
            safe_doc += " "
        body_lines: List[str] = [
            '    r"""' + safe_doc + '\n    """'
        ]

        # Build dispatch dict.  Drop None-valued optional params so the
        # server sees "field omitted" rather than "field provided as null",
        # matching the JSON Schema 2020-12 idiom for absence.
        body_lines.append("    _args = {")
        for py_name, json_name in py_to_json:
            body_lines.append(f"        {json_name!r}: {py_name},")
        body_lines.append("    }")
        body_lines.append("    _args = {_k: _v for _k, _v in _args.items() if _v is not None}")
        body_lines.append("    _args.update(kwargs)" if "**kwargs" in sig_params else "")

        # Validator lookups key on the full registry name so two servers
        # with the same short tool name (e.g. github+gitlab both expose
        # ``list_issues``) don't clobber each other in the shared
        # hermes_tools validator dicts.
        body_lines.append(f"    _validate_input({registry_name!r}, _args)")
        body_lines.append(f"    _result = _call({registry_name!r}, _args)")
        body_lines.append(f"    return _validate_output({registry_name!r}, _result)")

        # Validator registration at module scope — after the function so
        # the per-tool schema constants stay grouped with the function.
        # Skip emission entirely when a schema isn't JSON-serializable
        # (e.g. a circular reference or a stray object from a misbehaving
        # server) — passing ``None`` to ``Draft202012Validator`` would
        # crash at module import and take all sibling tools with it.
        in_schema_repr = (
            _safe_python_literal(input_schema)
            if _is_json_serializable(input_schema)
            else None
        )
        out_schema_repr = (
            _safe_python_literal(output_schema)
            if output_schema is not None and _is_json_serializable(output_schema)
            else None
        )

        lines.append(sig_line)
        lines.extend(filter(None, body_lines))
        lines.append("")
        if in_schema_repr is not None:
            lines.append(f"_INPUT_SCHEMA_{safe_tool} = {in_schema_repr}")
            lines.append(
                f"_register_input_schema({registry_name!r}, _INPUT_SCHEMA_{safe_tool})"
            )
        if out_schema_repr is not None:
            lines.append(f"_OUTPUT_SCHEMA_{safe_tool} = {out_schema_repr}")
            lines.append(
                f"_register_output_schema({registry_name!r}, _OUTPUT_SCHEMA_{safe_tool})"
            )
        lines.append("")

        allowed_names.add(registry_name)
        emitted_tools.append(safe_tool)

    return "\n".join(lines) + "\n", allowed_names, emitted_tools


def _is_json_serializable(value) -> bool:
    """Return True if ``value`` survives a ``json.dumps`` round-trip."""
    if value is None:
        return True
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _safe_python_literal(value) -> str:
    """Render a JSON-Schema-ish value as a Python literal in generated code.

    Caller is expected to have checked ``_is_json_serializable`` first; this
    helper only handles the happy path.  Round-trips through
    ``json.dumps`` → ``json.loads`` to coerce non-JSON-native types (tuples,
    sets) into list/dict form, then uses ``repr`` for stable, parse-safe
    Python source.
    """
    if value is None:
        return "None"
    return repr(json.loads(json.dumps(value)))


def _build_mcp_sandbox_bundle(cfg: dict) -> Tuple[Dict[str, str], Set[str]]:
    """Return ``(files, allowed_names)`` for MCP exposure in the sandbox.

    Walks ``tools.mcp_tool._servers`` under that module's ``_lock`` and
    emits one Python submodule per connected server, plus a
    ``hermes_mcp/__init__.py`` that lists the exposed servers in its
    docstring.  Each generated function dispatches via
    ``_call("mcp_<server>_<tool>", kwargs)`` — the same name MCP tools
    are registered under in the Hermes registry
    (``tools/mcp_tool.py:2833``), so ``handle_function_call`` already
    routes them without any dispatcher changes.

    Args:
        cfg: The ``code_execution`` config dict (as returned by
             ``_load_config``).  Reads ``expose_mcp_tools`` and
             ``mcp_servers_allowlist``.

    Returns:
        ``(files, allowed_names)``:
          - ``files``: dict mapping relative path → file source.  Empty
            when the feature is disabled or no servers match.
          - ``allowed_names``: set of ``mcp_<server>_<tool>`` names to
            add to the RPC dispatch allowlist.
    """
    if not cfg.get("expose_mcp_tools"):
        return {}, set()
    try:
        from tools.mcp_tool import (
            _lock as _mcp_lock,
            _servers as _mcp_servers,
            sanitize_mcp_name_component,
        )
    except Exception:
        logger.debug("MCP module unavailable; skipping sandbox bundle", exc_info=True)
        return {}, set()

    allowlist_raw = cfg.get("mcp_servers_allowlist")
    allowlist = None
    if isinstance(allowlist_raw, list):
        allowlist = {str(name) for name in allowlist_raw}

    files: Dict[str, str] = {}
    allowed_names: Set[str] = set()
    server_modules: List[str] = []

    with _mcp_lock:
        items = list(_mcp_servers.items())

    for server_name, server_task in items:
        if allowlist is not None and server_name not in allowlist:
            continue
        tools = getattr(server_task, "_tools", None) or []
        if not tools:
            continue
        safe_server = sanitize_mcp_name_component(server_name)

        source, server_allowed, emitted_tools = _emit_tool_module(
            server_name, tools
        )
        if not emitted_tools:
            continue
        files[f"hermes_mcp/{safe_server}.py"] = source
        allowed_names |= server_allowed
        server_modules.append(safe_server)

    if not files:
        return {}, set()

    init_lines: List[str] = ['"""Auto-generated wrappers for connected MCP servers.', ""]
    init_lines.append("Available submodules:")
    for safe in sorted(server_modules):
        init_lines.append(f"  - hermes_mcp.{safe}")
    init_lines.append('"""')
    init_lines.append(f"__all__ = {sorted(server_modules)!r}")
    files["hermes_mcp/__init__.py"] = "\n".join(init_lines) + "\n"
    return files, allowed_names


# ---- Shared helpers section (embedded in both transport headers) ----------

_COMMON_HELPERS = '''\

# ---------------------------------------------------------------------------
# Convenience helpers (avoid common scripting pitfalls)
# ---------------------------------------------------------------------------

def json_parse(text: str):
    """Parse JSON tolerant of control characters (strict=False).
    Use this instead of json.loads() when parsing output from terminal()
    or web_extract() that may contain raw tabs/newlines in strings."""
    return json.loads(text, strict=False)


def shell_quote(s: str) -> str:
    """Shell-escape a string for safe interpolation into commands.
    Use this when inserting dynamic content into terminal() commands:
        terminal(f"echo {shell_quote(user_input)}")
    """
    return shlex.quote(s)


def retry(fn, max_attempts=3, delay=2):
    """Retry a function up to max_attempts times with exponential backoff.
    Use for transient failures (network errors, API rate limits):
        result = retry(lambda: terminal("gh issue list ..."))
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (2 ** attempt))
    raise last_err

'''

# ---- UDS transport (local backend) ---------------------------------------

_UDS_TRANSPORT_HEADER = '''\
"""Auto-generated Hermes tools RPC stubs."""
import json, os, socket, shlex, threading, time

_sock = None
# The RPC server handles a single client connection serially and has no
# request-id in the protocol, so concurrent _call() invocations from multiple
# threads (e.g. ThreadPoolExecutor) would race on the shared socket and get
# each other's responses. Serialize the entire send+recv round-trip.
_call_lock = threading.Lock()
''' + _COMMON_HELPERS + '''\

def _connect():
    """Connect to the parent's RPC server via the transport it picked.

    HERMES_RPC_SOCKET can be either:
      - a filesystem path (POSIX Unix domain socket — the default on
        Linux and macOS)
      - a string of the form ``tcp://127.0.0.1:<port>`` (Windows, where
        AF_UNIX is unreliable — the parent falls back to loopback TCP)
    """
    global _sock
    if _sock is None:
        endpoint = os.environ["HERMES_RPC_SOCKET"]
        if endpoint.startswith("tcp://"):
            # tcp://host:port  (host is always 127.0.0.1 in practice — we
            # only bind loopback server-side)
            _host_port = endpoint[len("tcp://"):]
            _host, _, _port = _host_port.rpartition(":")
            _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _sock.connect((_host or "127.0.0.1", int(_port)))
        else:
            _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            _sock.connect(endpoint)
        _sock.settimeout(300)
    return _sock

def _call(tool_name, args):
    """Send a tool call to the parent process and return the parsed result."""
    request = json.dumps({"tool": tool_name, "args": args}) + "\\n"
    with _call_lock:
        conn = _connect()
        conn.sendall(request.encode())
        buf = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                raise RuntimeError("Agent process disconnected")
            buf += chunk
            if buf.endswith(b"\\n"):
                break
    raw = buf.decode().strip()
    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result

'''

# ---- File-based transport (remote backends) -------------------------------

_FILE_TRANSPORT_HEADER = '''\
"""Auto-generated Hermes tools RPC stubs (file-based transport)."""
import json, os, shlex, tempfile, threading, time

_RPC_DIR = os.environ.get("HERMES_RPC_DIR") or os.path.join(tempfile.gettempdir(), "hermes_rpc")
_seq = 0
# `_seq += 1` is not atomic (read-modify-write), so concurrent _call()
# invocations from multiple threads could allocate the same sequence number
# and clobber each other's request files. Guard seq allocation with a lock.
_seq_lock = threading.Lock()
''' + _COMMON_HELPERS + '''\

def _call(tool_name, args):
    """Send a tool call request via file-based RPC and wait for response."""
    global _seq
    with _seq_lock:
        _seq += 1
        seq = _seq
    seq_str = f"{seq:06d}"
    req_file = os.path.join(_RPC_DIR, f"req_{seq_str}")
    res_file = os.path.join(_RPC_DIR, f"res_{seq_str}")

    # Write request atomically (write to .tmp, then rename).
    # encoding="utf-8" is critical: on Windows-hosted remote backends
    # (or any non-UTF-8 locale) the default open() mode would mangle
    # non-ASCII chars in tool args when encoding them as JSON.
    tmp = req_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"tool": tool_name, "args": args, "seq": seq}, f)
    os.rename(tmp, req_file)

    # Wait for response with adaptive polling
    deadline = time.monotonic() + 300  # 5-minute timeout per tool call
    poll_interval = 0.05  # Start at 50ms
    while not os.path.exists(res_file):
        if time.monotonic() > deadline:
            raise RuntimeError(f"RPC timeout: no response for {tool_name} after 300s")
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.2, 0.25)  # Back off to 250ms

    with open(res_file, encoding="utf-8") as f:
        raw = f.read()

    # Clean up response file
    try:
        os.unlink(res_file)
    except OSError:
        pass

    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result

'''


# ---------------------------------------------------------------------------
# RPC server (runs in a thread inside the parent process)
# ---------------------------------------------------------------------------

# Terminal parameters that must not be used from ephemeral sandbox scripts
_TERMINAL_BLOCKED_PARAMS = {"background", "pty", "notify_on_complete", "watch_patterns"}


def _rpc_server_loop(
    server_sock: socket.socket,
    task_id: str,
    tool_call_log: list,
    tool_call_counter: list,   # mutable [int] so the thread can increment
    max_tool_calls: int,
    allowed_tools: frozenset,
):
    """
    Accept one client connection and dispatch tool-call requests until
    the client disconnects or the call limit is reached.
    """
    from model_tools import handle_function_call

    conn = None
    try:
        server_sock.settimeout(5)
        conn, _ = server_sock.accept()
        conn.settimeout(300)

        buf = b""
        while True:
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk

            # Process all complete newline-delimited messages in the buffer
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue

                call_start = time.monotonic()
                try:
                    request = json.loads(line.decode())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    resp = tool_error(f"Invalid RPC request: {exc}")
                    conn.sendall((resp + "\n").encode())
                    continue

                tool_name = request.get("tool", "")
                tool_args = request.get("args", {})

                # Enforce the allow-list
                if tool_name not in allowed_tools:
                    available = ", ".join(sorted(allowed_tools))
                    resp = json.dumps({
                        "error": (
                            f"Tool '{tool_name}' is not available in execute_code. "
                            f"Available: {available}"
                        )
                    })
                    conn.sendall((resp + "\n").encode())
                    continue

                # Enforce tool call limit
                if tool_call_counter[0] >= max_tool_calls:
                    resp = json.dumps({
                        "error": (
                            f"Tool call limit reached ({max_tool_calls}). "
                            "No more tool calls allowed in this execution."
                        )
                    })
                    conn.sendall((resp + "\n").encode())
                    continue

                # Strip forbidden terminal parameters
                if tool_name == "terminal" and isinstance(tool_args, dict):
                    for param in _TERMINAL_BLOCKED_PARAMS:
                        tool_args.pop(param, None)

                # Dispatch through the standard tool handler.
                # Suppress stdout/stderr from internal tool handlers so
                # their status prints don't leak into the CLI spinner.
                try:
                    _real_stdout, _real_stderr = sys.stdout, sys.stderr
                    devnull = open(os.devnull, "w", encoding="utf-8")
                    try:
                        sys.stdout = devnull
                        sys.stderr = devnull
                        result = handle_function_call(
                            tool_name, tool_args, task_id=task_id
                        )
                    finally:
                        sys.stdout, sys.stderr = _real_stdout, _real_stderr
                        devnull.close()
                except Exception as exc:
                    logger.error("Tool call failed in sandbox: %s", exc, exc_info=True)
                    result = tool_error(str(exc))

                tool_call_counter[0] += 1
                call_duration = time.monotonic() - call_start

                # Log for observability
                args_preview = str(tool_args)[:80]
                tool_call_log.append({
                    "tool": tool_name,
                    "args_preview": args_preview,
                    "duration": round(call_duration, 2),
                })

                conn.sendall((result + "\n").encode())

    except socket.timeout:
        logger.debug("RPC listener socket timeout")
    except OSError as e:
        logger.debug("RPC listener socket error: %s", e, exc_info=True)
    finally:
        if conn:
            try:
                conn.close()
            except OSError as e:
                logger.debug("RPC conn close error: %s", e)


# ---------------------------------------------------------------------------
# Remote execution support (file-based RPC via terminal backend)
# ---------------------------------------------------------------------------

def _get_or_create_env(task_id: str):
    """Get or create the terminal environment for *task_id*.

    Reuses the same environment (container/sandbox/SSH session) that the
    terminal and file tools use, creating one if it doesn't exist yet.
    Returns ``(env, env_type)`` tuple.
    """
    from tools.terminal_tool import (
        _active_environments, _env_lock, _create_environment,
        _get_env_config, _last_activity, _start_cleanup_thread,
        _creation_locks, _creation_locks_lock, _task_env_overrides,
        _resolve_container_task_id,
    )

    effective_task_id = _resolve_container_task_id(task_id)

    # Fast path: environment already exists
    with _env_lock:
        if effective_task_id in _active_environments:
            _last_activity[effective_task_id] = time.time()
            return _active_environments[effective_task_id], _get_env_config()["env_type"]

    # Slow path: create environment (same pattern as file_tools._get_file_ops)
    with _creation_locks_lock:
        if effective_task_id not in _creation_locks:
            _creation_locks[effective_task_id] = threading.Lock()
        task_lock = _creation_locks[effective_task_id]

    with task_lock:
        with _env_lock:
            if effective_task_id in _active_environments:
                _last_activity[effective_task_id] = time.time()
                return _active_environments[effective_task_id], _get_env_config()["env_type"]

        config = _get_env_config()
        env_type = config["env_type"]
        overrides = _task_env_overrides.get(effective_task_id, {})

        if env_type == "docker":
            image = overrides.get("docker_image") or config["docker_image"]
        elif env_type == "singularity":
            image = overrides.get("singularity_image") or config["singularity_image"]
        elif env_type == "modal":
            image = overrides.get("modal_image") or config["modal_image"]
        elif env_type == "daytona":
            image = overrides.get("daytona_image") or config["daytona_image"]
        else:
            image = ""

        cwd = overrides.get("cwd") or config["cwd"]

        container_config = None
        if env_type in {"docker", "singularity", "modal", "daytona"}:
            container_config = {
                "container_cpu": config.get("container_cpu", 1),
                "container_memory": config.get("container_memory", 5120),
                "container_disk": config.get("container_disk", 51200),
                "container_persistent": config.get("container_persistent", True),
                "docker_volumes": config.get("docker_volumes", []),
                "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
            }

        ssh_config = None
        if env_type == "ssh":
            ssh_config = {
                "host": config.get("ssh_host", ""),
                "user": config.get("ssh_user", ""),
                "port": config.get("ssh_port", 22),
                "key": config.get("ssh_key", ""),
                "persistent": config.get("ssh_persistent", False),
            }

        local_config = None
        if env_type == "local":
            local_config = {
                "persistent": config.get("local_persistent", False),
            }

        logger.info("Creating new %s environment for execute_code task %s...",
                     env_type, effective_task_id[:8])
        env = _create_environment(
            env_type=env_type,
            image=image,
            cwd=cwd,
            timeout=config["timeout"],
            ssh_config=ssh_config,
            container_config=container_config,
            local_config=local_config,
            task_id=effective_task_id,
            host_cwd=config.get("host_cwd"),
        )

        with _env_lock:
            _active_environments[effective_task_id] = env
            _last_activity[effective_task_id] = time.time()

        _start_cleanup_thread()
        logger.info("%s environment ready for execute_code task %s",
                     env_type, effective_task_id[:8])
        return env, env_type


def _ship_file_to_remote(env, remote_path: str, content: str) -> None:
    """Write *content* to *remote_path* on the remote environment.

    Uses ``echo … | base64 -d`` rather than stdin piping because some
    backends (Modal) don't reliably deliver stdin_data to chained
    commands.  Base64 output is shell-safe ([A-Za-z0-9+/=]) so single
    quotes are fine.
    """
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    quoted_remote_path = shlex.quote(remote_path)
    env.execute(
        f"echo '{encoded}' | base64 -d > {quoted_remote_path}",
        cwd="/",
        timeout=30,
    )


def _env_temp_dir(env: Any) -> str:
    """Return a writable temp dir for env-backed execute_code sandboxes."""
    get_temp_dir = getattr(env, "get_temp_dir", None)
    if callable(get_temp_dir):
        try:
            temp_dir = get_temp_dir()
            if isinstance(temp_dir, str) and temp_dir.startswith("/"):
                return temp_dir.rstrip("/") or "/"
        except Exception as exc:
            logger.debug("Could not resolve execute_code env temp dir: %s", exc)
    candidate = tempfile.gettempdir()
    if isinstance(candidate, str) and candidate.startswith("/"):
        return candidate.rstrip("/") or "/"
    return "/tmp"


def _rpc_poll_loop(
    env,
    rpc_dir: str,
    task_id: str,
    tool_call_log: list,
    tool_call_counter: list,
    max_tool_calls: int,
    allowed_tools: frozenset,
    stop_event: threading.Event,
):
    """Poll the remote filesystem for tool call requests and dispatch them.

    Runs in a background thread.  Each ``env.execute()`` spawns an
    independent process, so these calls run safely concurrent with the
    script-execution thread.
    """
    from model_tools import handle_function_call

    poll_interval = 0.1  # 100 ms

    quoted_rpc_dir = shlex.quote(rpc_dir)
    while not stop_event.is_set():
        try:
            # List pending request files (skip .tmp partials)
            ls_result = env.execute(
                f"ls -1 {quoted_rpc_dir}/req_* 2>/dev/null || true",
                cwd="/",
                timeout=10,
            )
            output = ls_result.get("output", "").strip()
            if not output:
                stop_event.wait(poll_interval)
                continue

            req_files = sorted([
                f.strip() for f in output.split("\n")
                if f.strip()
                and not f.strip().endswith(".tmp")
                and "/req_" in f.strip()
            ])

            for req_file in req_files:
                if stop_event.is_set():
                    break

                call_start = time.monotonic()

                quoted_req_file = shlex.quote(req_file)
                # Read request
                read_result = env.execute(
                    f"cat {quoted_req_file}",
                    cwd="/",
                    timeout=10,
                )
                try:
                    request = json.loads(read_result.get("output", ""))
                except (json.JSONDecodeError, ValueError):
                    logger.debug("Malformed RPC request in %s", req_file)
                    # Remove bad request to avoid infinite retry
                    env.execute(f"rm -f {quoted_req_file}", cwd="/", timeout=5)
                    continue

                tool_name = request.get("tool", "")
                tool_args = request.get("args", {})
                seq = request.get("seq", 0)
                seq_str = f"{seq:06d}"
                res_file = f"{rpc_dir}/res_{seq_str}"
                quoted_res_file = shlex.quote(res_file)

                # Enforce allow-list
                if tool_name not in allowed_tools:
                    available = ", ".join(sorted(allowed_tools))
                    tool_result = json.dumps({
                        "error": (
                            f"Tool '{tool_name}' is not available in execute_code. "
                            f"Available: {available}"
                        )
                    })
                # Enforce tool call limit
                elif tool_call_counter[0] >= max_tool_calls:
                    tool_result = json.dumps({
                        "error": (
                            f"Tool call limit reached ({max_tool_calls}). "
                            "No more tool calls allowed in this execution."
                        )
                    })
                else:
                    # Strip forbidden terminal parameters
                    if tool_name == "terminal" and isinstance(tool_args, dict):
                        for param in _TERMINAL_BLOCKED_PARAMS:
                            tool_args.pop(param, None)

                    # Dispatch through the standard tool handler
                    try:
                        _real_stdout, _real_stderr = sys.stdout, sys.stderr
                        devnull = open(os.devnull, "w", encoding="utf-8")
                        try:
                            sys.stdout = devnull
                            sys.stderr = devnull
                            tool_result = handle_function_call(
                                tool_name, tool_args, task_id=task_id
                            )
                        finally:
                            sys.stdout, sys.stderr = _real_stdout, _real_stderr
                            devnull.close()
                    except Exception as exc:
                        logger.error("Tool call failed in remote sandbox: %s",
                                     exc, exc_info=True)
                        tool_result = tool_error(str(exc))

                    tool_call_counter[0] += 1
                    call_duration = time.monotonic() - call_start
                    tool_call_log.append({
                        "tool": tool_name,
                        "args_preview": str(tool_args)[:80],
                        "duration": round(call_duration, 2),
                    })

                # Write response atomically (tmp + rename).
                # Use echo piping (not stdin_data) because Modal doesn't
                # reliably deliver stdin to chained commands.
                encoded_result = base64.b64encode(
                    tool_result.encode("utf-8")
                ).decode("ascii")
                env.execute(
                    f"echo '{encoded_result}' | base64 -d > {quoted_res_file}.tmp"
                    f" && mv {quoted_res_file}.tmp {quoted_res_file}",
                    cwd="/",
                    timeout=60,
                )

                # Remove the request file
                env.execute(f"rm -f {quoted_req_file}", cwd="/", timeout=5)

        except Exception as e:
            if not stop_event.is_set():
                logger.debug("RPC poll error: %s", e, exc_info=True)

        if not stop_event.is_set():
            stop_event.wait(poll_interval)


def _execute_remote(
    code: str,
    task_id: Optional[str],
    enabled_tools: Optional[List[str]],
) -> str:
    """Run a script on the remote terminal backend via file-based RPC.

    The script and the generated hermes_tools.py module are shipped to
    the remote environment, and tool calls are proxied through a polling
    thread that communicates via request/response files.
    """

    _cfg = _load_config()
    timeout = _cfg.get("timeout", DEFAULT_TIMEOUT)
    max_tool_calls = _cfg.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)

    session_tools = set(enabled_tools) if enabled_tools else set()
    sandbox_tools = frozenset(SANDBOX_ALLOWED_TOOLS & session_tools)
    if not sandbox_tools:
        sandbox_tools = SANDBOX_ALLOWED_TOOLS

    effective_task_id = task_id or "default"
    env, env_type = _get_or_create_env(effective_task_id)

    sandbox_id = uuid.uuid4().hex[:12]
    temp_dir = _env_temp_dir(env)
    sandbox_dir = f"{temp_dir}/hermes_exec_{sandbox_id}"
    quoted_sandbox_dir = shlex.quote(sandbox_dir)
    quoted_rpc_dir = shlex.quote(f"{sandbox_dir}/rpc")

    tool_call_log: list = []
    tool_call_counter = [0]
    exec_start = time.monotonic()
    stop_event = threading.Event()
    rpc_thread = None

    try:
        # Verify Python is available on the remote
        py_check = env.execute(
            "command -v python3 >/dev/null 2>&1 && echo OK",
            cwd="/", timeout=15,
        )
        if "OK" not in py_check.get("output", ""):
            return json.dumps({
                "status": "error",
                "error": (
                    f"Python 3 is not available in the {env_type} terminal "
                    "environment. Install Python to use execute_code with "
                    "remote backends."
                ),
                "tool_calls_made": 0,
                "duration_seconds": 0,
            })

        # Create sandbox directory on remote
        env.execute(
            f"mkdir -p {quoted_rpc_dir}", cwd="/", timeout=10,
        )

        # Experimental: build hermes_mcp/<server>.py stubs first so we know
        # whether hermes_tools.py needs the shared MCP validator block.
        # See _build_mcp_sandbox_bundle.
        mcp_files, mcp_allowed_names = _build_mcp_sandbox_bundle(_cfg)

        # Generate and ship files
        tools_src = generate_hermes_tools_module(
            list(sandbox_tools), transport="file", mcp_enabled=bool(mcp_files),
        )
        _ship_file_to_remote(env, f"{sandbox_dir}/hermes_tools.py", tools_src)
        _ship_file_to_remote(env, f"{sandbox_dir}/script.py", code)

        for rel_path, content in mcp_files.items():
            remote_target = f"{sandbox_dir}/{rel_path}"
            remote_dir = remote_target.rsplit("/", 1)[0]
            env.execute(f"mkdir -p {shlex.quote(remote_dir)}", cwd="/", timeout=10)
            _ship_file_to_remote(env, remote_target, content)
        effective_allowlist = frozenset(sandbox_tools | mcp_allowed_names)

        # Generated MCP wrappers ``import jsonschema`` for the local
        # input-validation gate.  Local UDS transport inherits it from the
        # parent interpreter; remote sandboxes may not have it yet.
        # Best-effort install — silent success when already present, warning
        # on failure (the script will surface a clearer ImportError if so).
        if mcp_files:
            install_res = env.execute(
                "python3 -m pip install --quiet --disable-pip-version-check "
                "'jsonschema>=4.23.0,<5.0.0'",
                cwd="/", timeout=60,
            )
            if install_res.get("returncode", -1) != 0:
                logger.warning(
                    "execute_code: could not install jsonschema in remote "
                    "%s sandbox — MCP wrappers will fail at import. "
                    "pip output: %s",
                    env_type, (install_res.get("output", "") or "")[:500],
                )

        # Start RPC polling thread
        rpc_thread = threading.Thread(
            target=_rpc_poll_loop,
            args=(
                env, f"{sandbox_dir}/rpc", effective_task_id,
                tool_call_log, tool_call_counter, max_tool_calls,
                effective_allowlist, stop_event,
            ),
            daemon=True,
        )
        rpc_thread.start()

        # Build environment variable prefix for the script
        env_prefix = (
            f"HERMES_RPC_DIR={shlex.quote(f'{sandbox_dir}/rpc')} "
            f"PYTHONDONTWRITEBYTECODE=1"
        )
        tz = os.getenv("HERMES_TIMEZONE", "").strip()
        if tz:
            env_prefix += f" TZ={tz}"

        # Execute the script on the remote backend
        logger.info("Executing code on %s backend (task %s)...",
                     env_type, effective_task_id[:8])
        script_result = env.execute(
            f"cd {quoted_sandbox_dir} && {env_prefix} python3 script.py",
            timeout=timeout,
        )

        stdout_text = script_result.get("output", "")
        exit_code = script_result.get("returncode", -1)
        status = "success"

        # Check for timeout/interrupt from the backend
        if exit_code == 124:
            status = "timeout"
        elif exit_code == 130:
            status = "interrupted"

    except Exception as exc:
        duration = round(time.monotonic() - exec_start, 2)
        logger.error(
            "execute_code remote failed after %ss with %d tool calls: %s: %s",
            duration, tool_call_counter[0], type(exc).__name__, exc,
            exc_info=True,
        )
        return json.dumps({
            "status": "error",
            "error": str(exc),
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }, ensure_ascii=False)

    finally:
        # Stop the polling thread
        stop_event.set()
        if rpc_thread is not None:
            rpc_thread.join(timeout=5)

        # Clean up remote sandbox dir
        try:
            env.execute(
                f"rm -rf {quoted_sandbox_dir}", cwd="/", timeout=15,
            )
        except Exception:
            logger.debug("Failed to clean up remote sandbox %s", sandbox_dir)

    duration = round(time.monotonic() - exec_start, 2)

    # --- Post-process output (same as local path) ---

    # Truncate stdout to cap
    if len(stdout_text) > MAX_STDOUT_BYTES:
        head_bytes = int(MAX_STDOUT_BYTES * 0.4)
        tail_bytes = MAX_STDOUT_BYTES - head_bytes
        head = stdout_text[:head_bytes]
        tail = stdout_text[-tail_bytes:]
        omitted = len(stdout_text) - len(head) - len(tail)
        stdout_text = (
            head
            + f"\n\n... [OUTPUT TRUNCATED - {omitted:,} chars omitted "
            f"out of {len(stdout_text):,} total] ...\n\n"
            + tail
        )

    # Strip ANSI escape sequences
    from tools.ansi_strip import strip_ansi
    stdout_text = strip_ansi(stdout_text)

    # Redact secrets
    from agent.redact import redact_sensitive_text
    stdout_text = redact_sensitive_text(stdout_text)

    # Build response
    result: Dict[str, Any] = {
        "status": status,
        "output": stdout_text,
        "tool_calls_made": tool_call_counter[0],
        "duration_seconds": duration,
    }

    if status == "timeout":
        timeout_msg = f"Script timed out after {timeout}s and was killed."
        result["error"] = timeout_msg
        # Include timeout message in output so the LLM always surfaces it
        # to the user (see local path comment — same reasoning, #10807).
        if stdout_text:
            result["output"] = stdout_text + f"\n\n⏰ {timeout_msg}"
        else:
            result["output"] = f"⏰ {timeout_msg}"
        logger.warning(
            "execute_code (remote) timed out after %ss (limit %ss) with %d tool calls",
            duration, timeout, tool_call_counter[0],
        )
    elif status == "interrupted":
        result["output"] = (
            stdout_text + "\n[execution interrupted — user sent a new message]"
        )
    elif exit_code != 0:
        result["status"] = "error"
        result["error"] = f"Script exited with code {exit_code}"

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def execute_code(
    code: str,
    task_id: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
) -> str:
    """
    Run a Python script in a sandboxed child process with RPC access
    to a subset of Hermes tools.

    Dispatches to the local (UDS) or remote (file-based RPC) path
    depending on the configured terminal backend.

    Args:
        code:          Python source code to execute.
        task_id:       Session task ID for tool isolation (terminal env, etc.).
        enabled_tools: Tool names enabled in the current session. The sandbox
                       gets the intersection with SANDBOX_ALLOWED_TOOLS.

    Returns:
        JSON string with execution results.
    """
    if not SANDBOX_AVAILABLE:
        return json.dumps({
            "error": "execute_code sandbox is unavailable in this environment. "
                     "Use normal tool calls (terminal, read_file, write_file, ...) instead."
        })

    if not code or not code.strip():
        return tool_error("No code provided.")

    # Dispatch: remote backends use file-based RPC, local uses UDS
    from tools.terminal_tool import _get_env_config
    env_type = _get_env_config()["env_type"]
    if env_type != "local":
        return _execute_remote(code, task_id, enabled_tools)

    # --- Local execution path (UDS) --- below this line is unchanged ---

    # Import per-thread interrupt check (cooperative cancellation)
    from tools.interrupt import is_interrupted as _is_interrupted

    # Resolve config
    _cfg = _load_config()
    timeout = _cfg.get("timeout", DEFAULT_TIMEOUT)
    max_tool_calls = _cfg.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)

    # Determine which tools the sandbox can call
    session_tools = set(enabled_tools) if enabled_tools else set()
    sandbox_tools = frozenset(SANDBOX_ALLOWED_TOOLS & session_tools)

    if not sandbox_tools:
        sandbox_tools = SANDBOX_ALLOWED_TOOLS

    # --- Set up temp directory with hermes_tools.py and script.py ---
    tmpdir = tempfile.mkdtemp(prefix="hermes_sandbox_")
    # Use /tmp on macOS to avoid the long /var/folders/... path that pushes
    # Unix domain socket paths past the 104-byte macOS AF_UNIX limit.
    # On Linux, tempfile.gettempdir() already returns /tmp.
    #
    # Windows: Python 3.9+ added partial AF_UNIX support but the file-backed
    # variant is flaky across Windows builds (requires Windows 10 1803+,
    # still fails under some configurations, and the socket file can't live
    # on the same temp drive as the script).  Fall back to loopback TCP —
    # same ephemeral port, same 1-connection listen queue, same serialized
    # request/response framing.  The generated client reads the transport
    # selector from HERMES_RPC_SOCKET (path vs. ``tcp://host:port``).
    _sock_tmpdir = "/tmp" if sys.platform == "darwin" else tempfile.gettempdir()
    _use_tcp_rpc = _IS_WINDOWS
    if _use_tcp_rpc:
        sock_path = None  # not used on Windows; TCP endpoint stored below
        rpc_endpoint = None  # set after bind()
    else:
        sock_path = os.path.join(_sock_tmpdir, f"hermes_rpc_{uuid.uuid4().hex}.sock")
        rpc_endpoint = sock_path

    tool_call_log: list = []
    tool_call_counter = [0]  # mutable so the RPC thread can increment
    exec_start = time.monotonic()
    server_sock = None

    try:
        # Write the auto-generated hermes_tools module.
        # encoding="utf-8" is required on Windows — the stub and user code
        # both contain non-ASCII characters (em-dashes in docstrings, plus
        # whatever the user script carries).  Python's default open() uses
        # the system locale on Windows (cp1252 typically), which corrupts
        # those bytes; the child then fails to import with a SyntaxError
        # ("'utf-8' codec can't decode byte 0x97 in position ...") because
        # Python source files are decoded as UTF-8 by default (PEP 3120).
        # sandbox_tools is already the correct set (intersection with session
        # tools, or SANDBOX_ALLOWED_TOOLS as fallback — see lines above).
        # Experimental: build hermes_mcp/<server>.py stubs for connected
        # MCP servers when code_execution.expose_mcp_tools=true.  The MCP
        # tool names are added to the dispatch allowlist below so the RPC
        # loop will route them through handle_function_call.  Built first
        # so hermes_tools.py knows whether to include the shared MCP
        # validator infrastructure.
        mcp_files, mcp_allowed_names = _build_mcp_sandbox_bundle(_cfg)

        tools_src = generate_hermes_tools_module(
            list(sandbox_tools), mcp_enabled=bool(mcp_files),
        )
        with open(os.path.join(tmpdir, "hermes_tools.py"), "w", encoding="utf-8") as f:
            f.write(tools_src)

        for rel_path, content in mcp_files.items():
            target = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
        effective_allowlist = frozenset(sandbox_tools | mcp_allowed_names)

        # Write the user's script
        with open(os.path.join(tmpdir, "script.py"), "w", encoding="utf-8") as f:
            f.write(code)

        # --- Start RPC server ---
        # Two transports:
        #   POSIX: AF_UNIX stream socket on sock_path, chmod 0600 for
        #   owner-only access.  Filesystem permissions gate the socket.
        #   Windows: AF_INET stream socket on 127.0.0.1 with an ephemeral
        #   port.  No filesystem permission story, but loopback-only bind
        #   means only the current user's processes (not remote) can
        #   connect.  HERMES_RPC_SOCKET is set to ``tcp://127.0.0.1:<port>``
        #   which the generated client parses to pick AF_INET.
        if _use_tcp_rpc:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.bind(("127.0.0.1", 0))  # ephemeral port
            _host, _port = server_sock.getsockname()[:2]
            rpc_endpoint = f"tcp://{_host}:{_port}"
        else:
            server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_sock.bind(sock_path)
            os.chmod(sock_path, 0o600)
        server_sock.listen(1)

        rpc_thread = threading.Thread(
            target=_rpc_server_loop,
            args=(
                server_sock, task_id, tool_call_log,
                tool_call_counter, max_tool_calls, effective_allowlist,
            ),
            daemon=True,
        )
        rpc_thread.start()

        # --- Spawn child process ---
        # Build a minimal environment for the child. We intentionally exclude
        # API keys and tokens to prevent credential exfiltration from LLM-
        # generated scripts. The child accesses tools via RPC, not direct API.
        # Exception: env vars declared by loaded skills (via env_passthrough
        # registry) or explicitly allowed by the user in config.yaml
        # (terminal.env_passthrough) are passed through.  On Windows, a small
        # OS-essential allowlist (SYSTEMROOT, WINDIR, COMSPEC, ...) is also
        # passed through — without those, the child can't create a socket
        # or spawn a subprocess.  See ``_scrub_child_env`` for the rules.
        child_env = _scrub_child_env(os.environ)
        child_env["HERMES_RPC_SOCKET"] = rpc_endpoint
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Force UTF-8 for the child's stdio and default file encoding.
        #
        # Without this, on Windows sys.stdout is bound to the console code
        # page (cp1252 on US-locale installs), and any script that does
        # ``print("café")`` or ``print("→")`` crashes with:
        #
        #   UnicodeEncodeError: 'charmap' codec can't encode character
        #   '\u2192' in position N: character maps to <undefined>
        #
        # PYTHONIOENCODING fixes sys.stdin/stdout/stderr.
        # PYTHONUTF8=1 enables "UTF-8 mode" (PEP 540) which additionally
        # makes ``open()``'s default encoding UTF-8, so user scripts that
        # write files without specifying encoding= also work correctly.
        #
        # On POSIX both values usually match the locale default already,
        # so setting them is harmless belt-and-suspenders for environments
        # with a C/POSIX locale (containers, minimal base images).
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        # Ensure the hermes-agent root is importable in the sandbox so
        # repo-root modules are available to child scripts.  We also prepend
        # the staging tmpdir so ``from hermes_tools import ...`` resolves even
        # when the subprocess CWD is not tmpdir (project mode).
        #
        # The stable hermes_mcp/ wrappers (Slice A of
        # code_execution.expose_mcp_tools) deliberately stay OUT of this
        # PYTHONPATH: the per-call tmpdir already has a content-identical
        # copy AND its regular __init__.py shadows the stable copy for
        # submodule lookup anyway (Python won't merge a regular package
        # with sibling PYTHONPATH entries).  The stable path's purpose is
        # between-turn browseability via read_file / search_files, not
        # in-sandbox imports.
        _hermes_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _existing_pp = child_env.get("PYTHONPATH", "")
        _pp_parts = [tmpdir, _hermes_root]
        if _existing_pp:
            _pp_parts.append(_existing_pp)
        child_env["PYTHONPATH"] = os.pathsep.join(_pp_parts)
        # Inject user's configured timezone so datetime.now() in sandboxed
        # code reflects the correct wall-clock time.  Only TZ is set —
        # HERMES_TIMEZONE is an internal Hermes setting and must not leak
        # into child processes.
        _tz_name = os.getenv("HERMES_TIMEZONE", "").strip()
        if _tz_name:
            child_env["TZ"] = _tz_name
        child_env.pop("HERMES_TIMEZONE", None)

        # Per-profile HOME isolation: redirect system tool configs into
        # {HERMES_HOME}/home/ when that directory exists.
        from hermes_constants import get_subprocess_home
        _profile_home = get_subprocess_home()
        if _profile_home:
            child_env["HOME"] = _profile_home

        # Resolve interpreter + CWD based on execute_code mode.
        #   - strict : today's behavior (sys.executable + tmpdir CWD).
        #   - project: user's venv python + session's working directory, so
        #              project deps like pandas and user files resolve.
        # Env scrubbing and tool whitelist apply identically in both modes.
        _mode = _get_execution_mode()
        _child_python = _resolve_child_python(_mode)
        _child_cwd = _resolve_child_cwd(_mode, tmpdir)
        _script_path = os.path.join(tmpdir, "script.py")

        proc = subprocess.Popen(
            [_child_python, _script_path],
            cwd=_child_cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            preexec_fn=None if _IS_WINDOWS else os.setsid,
            creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
        )

        # --- Poll loop: watch for exit, timeout, and interrupt ---
        deadline = time.monotonic() + timeout
        stderr_chunks: list = []

        # Background readers to avoid pipe buffer deadlocks.
        # For stdout we use a head+tail strategy: keep the first HEAD_BYTES
        # and a rolling window of the last TAIL_BYTES so the final print()
        # output is never lost.  Stderr keeps head-only (errors appear early).
        _STDOUT_HEAD_BYTES = int(MAX_STDOUT_BYTES * 0.4)   # 40% head
        _STDOUT_TAIL_BYTES = MAX_STDOUT_BYTES - _STDOUT_HEAD_BYTES  # 60% tail

        def _drain(pipe, chunks, max_bytes):
            """Simple head-only drain (used for stderr)."""
            total = 0
            try:
                while True:
                    data = pipe.read(4096)
                    if not data:
                        break
                    if total < max_bytes:
                        keep = max_bytes - total
                        chunks.append(data[:keep])
                    total += len(data)
            except (ValueError, OSError) as e:
                logger.debug("Error reading process output: %s", e, exc_info=True)

        stdout_total_bytes = [0]  # mutable ref for total bytes seen

        def _drain_head_tail(pipe, head_chunks, tail_chunks, head_bytes, tail_bytes, total_ref):
            """Drain stdout keeping both head and tail data."""
            head_collected = 0
            from collections import deque
            tail_buf = deque()
            tail_collected = 0
            try:
                while True:
                    data = pipe.read(4096)
                    if not data:
                        break
                    total_ref[0] += len(data)
                    # Fill head buffer first
                    if head_collected < head_bytes:
                        keep = min(len(data), head_bytes - head_collected)
                        head_chunks.append(data[:keep])
                        head_collected += keep
                        data = data[keep:]  # remaining goes to tail
                        if not data:
                            continue
                    # Everything past head goes into rolling tail buffer
                    tail_buf.append(data)
                    tail_collected += len(data)
                    # Evict old tail data to stay within tail_bytes budget
                    while tail_collected > tail_bytes and tail_buf:
                        oldest = tail_buf.popleft()
                        tail_collected -= len(oldest)
            except (ValueError, OSError):
                pass
            # Transfer final tail to output list
            tail_chunks.extend(tail_buf)

        stdout_head_chunks: list = []
        stdout_tail_chunks: list = []

        stdout_reader = threading.Thread(
            target=_drain_head_tail,
            args=(proc.stdout, stdout_head_chunks, stdout_tail_chunks,
                  _STDOUT_HEAD_BYTES, _STDOUT_TAIL_BYTES, stdout_total_bytes),
            daemon=True
        )
        stderr_reader = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_chunks, MAX_STDERR_BYTES), daemon=True
        )
        stdout_reader.start()
        stderr_reader.start()

        status = "success"
        _activity_state = {
            "last_touch": time.monotonic(),
            "start": exec_start,
        }
        while proc.poll() is None:
            if _is_interrupted():
                _kill_process_group(proc)
                status = "interrupted"
                break
            if time.monotonic() > deadline:
                _kill_process_group(proc, escalate=True)
                status = "timeout"
                break
            # Periodic activity touch so the gateway's inactivity timeout
            # doesn't kill the agent during long code execution (#10807).
            try:
                from tools.environments.base import touch_activity_if_due
                touch_activity_if_due(_activity_state, "execute_code running")
            except Exception:
                pass
            time.sleep(0.2)

        # Wait for readers to finish draining
        stdout_reader.join(timeout=3)
        stderr_reader.join(timeout=3)

        stdout_head = b"".join(stdout_head_chunks).decode("utf-8", errors="replace")
        stdout_tail = b"".join(stdout_tail_chunks).decode("utf-8", errors="replace")
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        # Assemble stdout with head+tail truncation
        total_stdout = stdout_total_bytes[0]
        if total_stdout > MAX_STDOUT_BYTES and stdout_tail:
            omitted = total_stdout - len(stdout_head) - len(stdout_tail)
            truncated_notice = (
                f"\n\n... [OUTPUT TRUNCATED - {omitted:,} chars omitted "
                f"out of {total_stdout:,} total] ...\n\n"
            )
            stdout_text = stdout_head + truncated_notice + stdout_tail
        else:
            stdout_text = stdout_head + stdout_tail

        exit_code = proc.returncode if proc.returncode is not None else -1
        duration = round(time.monotonic() - exec_start, 2)

        # Wait for RPC thread to finish
        server_sock.close()  # break accept() so thread exits promptly
        server_sock = None  # prevent double close in finally
        rpc_thread.join(timeout=3)

        # Strip ANSI escape sequences so the model never sees terminal
        # formatting — prevents it from copying escapes into file writes.
        from tools.ansi_strip import strip_ansi
        stdout_text = strip_ansi(stdout_text)
        stderr_text = strip_ansi(stderr_text)

        # Redact secrets (API keys, tokens, etc.) from sandbox output.
        # The sandbox env-var filter (lines 434-454) blocks os.environ access,
        # but scripts can still read secrets from disk (e.g. open('~/.hermes/.env')).
        # This ensures leaked secrets never enter the model context.
        from agent.redact import redact_sensitive_text
        stdout_text = redact_sensitive_text(stdout_text)
        stderr_text = redact_sensitive_text(stderr_text)

        # Build response
        result: Dict[str, Any] = {
            "status": status,
            "output": stdout_text,
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }

        if status == "timeout":
            timeout_msg = f"Script timed out after {timeout}s and was killed."
            result["error"] = timeout_msg
            # Include timeout message in output so the LLM always surfaces it
            # to the user.  When output is empty, models often treat the result
            # as "nothing happened" and produce an empty response, which the
            # gateway stream consumer silently drops (#10807).
            if stdout_text:
                result["output"] = stdout_text + f"\n\n⏰ {timeout_msg}"
            else:
                result["output"] = f"⏰ {timeout_msg}"
            logger.warning(
                "execute_code timed out after %ss (limit %ss) with %d tool calls",
                duration, timeout, tool_call_counter[0],
            )
        elif status == "interrupted":
            result["output"] = stdout_text + "\n[execution interrupted — user sent a new message]"
        elif exit_code != 0:
            result["status"] = "error"
            result["error"] = stderr_text or f"Script exited with code {exit_code}"
            # Include stderr in output so the LLM sees the traceback
            if stderr_text:
                result["output"] = stdout_text + "\n--- stderr ---\n" + stderr_text

        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        duration = round(time.monotonic() - exec_start, 2)
        logger.error(
            "execute_code failed after %ss with %d tool calls: %s: %s",
            duration,
            tool_call_counter[0],
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return json.dumps({
            "status": "error",
            "error": str(exc),
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }, ensure_ascii=False)

    finally:
        # Cleanup temp dir and socket
        if server_sock is not None:
            try:
                server_sock.close()
            except OSError as e:
                logger.debug("Server socket close error: %s", e)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            # Only UDS has a filesystem socket to unlink; TCP sockets are
            # freed by server_sock.close() above.
            if sock_path:
                os.unlink(sock_path)
        except OSError:
            pass  # already cleaned up or never created


def _kill_process_group(proc, escalate: bool = False):
    """Kill the child and its entire process tree (cross-platform via psutil)."""
    import psutil
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
    except psutil.NoSuchProcess:
        pass
    except (PermissionError, OSError) as e:
        logger.debug("Could not terminate process tree: %s", e, exc_info=True)
        try:
            proc.kill()
        except Exception as e2:
            logger.debug("Could not kill process: %s", e2, exc_info=True)

    if escalate:
        # Give the process 5s to exit after SIGTERM, then SIGKILL
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                parent = psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
            except psutil.NoSuchProcess:
                pass
            except (PermissionError, OSError) as e:
                logger.debug("Could not kill process tree: %s", e, exc_info=True)
                try:
                    proc.kill()
                except Exception as e2:
                    logger.debug("Could not kill process: %s", e2, exc_info=True)


def _load_config() -> dict:
    """Load code_execution config without importing the interactive CLI.

    This helper is called while building the module-level execute_code schema
    during tool discovery.  Importing ``cli`` here pulls prompt_toolkit/Rich and
    a large chunk of the classic REPL onto every agent startup path, including
    ``hermes --tui`` where it is never used.  Read the lightweight raw config
    instead; the config layer already caches by (mtime, size), and an absent
    key cleanly falls back to DEFAULT_EXECUTION_MODE.
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config().get("code_execution", {})
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Execution mode resolution (strict vs project)
# ---------------------------------------------------------------------------

# Valid values for code_execution.mode. Kept as a module constant so tests
# and the config layer can reference the canonical set.
EXECUTION_MODES = ("project", "strict")
DEFAULT_EXECUTION_MODE = "project"


def _get_execution_mode() -> str:
    """Return the active execute_code mode — 'project' or 'strict'.

    Reads ``code_execution.mode`` from config.yaml; invalid values fall back
    to ``DEFAULT_EXECUTION_MODE`` ('project') with a log warning.

    Mode semantics:
      - ``project`` (default): scripts run in the session's working directory
        with the active virtual environment's python, so project dependencies
        (pandas, torch, project packages) and files resolve naturally.
      - ``strict``: scripts run in an isolated temp directory with
        ``sys.executable`` (hermes-agent's python). Reproducible and the
        interpreter is guaranteed to work, but project deps and relative paths
        won't resolve.

    Env scrubbing and tool whitelist apply identically in both modes.
    """
    cfg_value = str(_load_config().get("mode", DEFAULT_EXECUTION_MODE)).strip().lower()
    if cfg_value in EXECUTION_MODES:
        return cfg_value
    logger.warning(
        "Ignoring code_execution.mode=%r (expected one of %s), falling back to %r",
        cfg_value, EXECUTION_MODES, DEFAULT_EXECUTION_MODE,
    )
    return DEFAULT_EXECUTION_MODE


@functools.lru_cache(maxsize=32)
def _is_usable_python(python_path: str) -> bool:
    """Check whether a candidate Python interpreter is usable for execute_code.

    Requires Python 3.8+ (f-strings and stdlib modules the RPC stubs need).
    Cached so we don't fork a subprocess on every execute_code call.
    """
    try:
        result = subprocess.run(
            [python_path, "-c",
             "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"],
            timeout=5,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False


def _resolve_child_python(mode: str) -> str:
    """Pick the Python interpreter for the execute_code subprocess.

    In ``strict`` mode, always ``sys.executable`` — guaranteed to work and
    keeps behavior fully reproducible across sessions.

    In ``project`` mode, prefer the user's active virtualenv/conda env's
    python so ``import pandas`` etc. work. Falls back to ``sys.executable``
    if no venv is detected, the candidate binary is missing/not executable,
    or it fails a Python 3.8+ version check.
    """
    if mode != "project":
        return sys.executable

    if _IS_WINDOWS:
        exe_names = ("python.exe", "python3.exe")
        subdirs = ("Scripts",)
    else:
        exe_names = ("python", "python3")
        subdirs = ("bin",)

    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        root = os.environ.get(var, "").strip()
        if not root:
            continue
        for subdir in subdirs:
            for exe in exe_names:
                candidate = os.path.join(root, subdir, exe)
                if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
                    continue
                if _is_usable_python(candidate):
                    return candidate
                # Found the interpreter but it failed the version check —
                # log once and fall through to sys.executable.
                logger.info(
                    "execute_code: skipping %s=%s (Python version < 3.8 or broken). "
                    "Using sys.executable instead.", var, candidate,
                )
                return sys.executable

    return sys.executable


def _resolve_child_cwd(mode: str, staging_dir: str) -> str:
    """Resolve the working directory for the execute_code subprocess.

    - ``strict``: the staging tmpdir (today's behavior).
    - ``project``: the session's TERMINAL_CWD (same as the terminal tool), or
      ``os.getcwd()`` if TERMINAL_CWD is unset or doesn't point at a real dir.
      Falls back to the staging tmpdir as a last resort so we never invoke
      Popen with a nonexistent cwd.
    """
    if mode != "project":
        return staging_dir
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if raw:
        expanded = os.path.expanduser(raw)
        if os.path.isdir(expanded):
            return expanded
    here = os.getcwd()
    if os.path.isdir(here):
        return here
    return staging_dir


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

# Per-tool documentation lines for the execute_code description.
# Ordered to match the canonical display order.
_TOOL_DOC_LINES = [
    ("web_search",
     "  web_search(query: str, limit: int = 5) -> dict\n"
     "    Returns {\"data\": {\"web\": [{\"url\", \"title\", \"description\"}, ...]}}"),
    ("web_extract",
     "  web_extract(urls: list[str]) -> dict\n"
     "    Returns {\"results\": [{\"url\", \"title\", \"content\", \"error\"}, ...]} where content is markdown"),
    ("read_file",
     "  read_file(path: str, offset: int = 1, limit: int = 500) -> dict\n"
     "    Lines are 1-indexed. Returns {\"content\": \"...\", \"total_lines\": N}"),
    ("write_file",
     "  write_file(path: str, content: str) -> dict\n"
     "    Always overwrites the entire file."),
    ("search_files",
     "  search_files(pattern: str, target=\"content\", path=\".\", file_glob=None, limit=50) -> dict\n"
     "    target: \"content\" (search inside files) or \"files\" (find files by name). Returns {\"matches\": [...]}"),
    ("patch",
     "  patch(path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict\n"
     "    Replaces old_string with new_string in the file."),
    ("terminal",
     "  terminal(command: str, timeout=None, workdir=None) -> dict\n"
     "    Foreground only (no background/pty). Returns {\"output\": \"...\", \"exit_code\": N}"),
]


def build_execute_code_schema(enabled_sandbox_tools: set = None,
                              mode: str = None) -> dict:
    """Build the execute_code schema with description listing only enabled tools.

    When tools are disabled via ``hermes tools`` (e.g. web is turned off),
    the schema description should NOT mention web_search / web_extract —
    otherwise the model thinks they are available and keeps trying to use them.

    ``mode`` controls the working-directory sentence in the description:
      - ``'strict'``: scripts run in a temp dir (not the session's CWD)
      - ``'project'`` (default): scripts run in the session's CWD with the
        active venv's python
    If ``mode`` is None, the current ``code_execution.mode`` config is read.
    """
    if enabled_sandbox_tools is None:
        enabled_sandbox_tools = SANDBOX_ALLOWED_TOOLS
    if mode is None:
        mode = _get_execution_mode()

    # Build tool documentation lines for only the enabled tools
    tool_lines = "\n".join(
        doc for name, doc in _TOOL_DOC_LINES if name in enabled_sandbox_tools
    )

    # Build example import list from enabled tools
    import_examples = [n for n in ("web_search", "terminal") if n in enabled_sandbox_tools]
    if not import_examples:
        import_examples = sorted(enabled_sandbox_tools)[:2]
    if import_examples:
        import_str = ", ".join(import_examples) + ", ..."
    else:
        import_str = "..."

    # Mode-specific CWD guidance. Project mode is the default and matches
    # terminal()'s filesystem/interpreter; strict mode retains the isolated
    # temp-dir staging and hermes-agent's own python.
    if mode == "strict":
        cwd_note = (
            "Scripts run in their own temp dir, not the session's CWD — use absolute paths "
            "(os.path.expanduser('~/.hermes/.env')) or terminal()/read_file() for user files."
        )
    else:
        cwd_note = (
            "Scripts run in the session's working directory with the active venv's python, "
            "so project deps (pandas, etc.) and relative paths work like in terminal()."
        )

    description = (
        "Run a Python script that can call Hermes tools programmatically. "
        "Use this when you need 3+ tool calls with processing logic between them, "
        "need to filter/reduce large tool outputs before they enter your context, "
        "need conditional branching (if X then Y else Z), or need to loop "
        "(fetch N pages, process N files, retry on failure).\n\n"
        "Use normal tool calls instead when: single tool call with no processing, "
        "you need to see the full result and apply complex reasoning, "
        "or the task requires interactive user input.\n\n"
        f"Available via `from hermes_tools import ...`:\n\n"
        f"{tool_lines}\n\n"
        "Limits: 5-minute timeout, 50KB stdout cap, max 50 tool calls per script. "
        "terminal() is foreground-only (no background or pty).\n\n"
        f"{cwd_note}\n\n"
        "Print your final result to stdout. Use Python stdlib (json, re, math, csv, "
        "datetime, collections, etc.) for processing between tool calls.\n\n"
        "Also available (no import needed — built into hermes_tools):\n"
        "  json_parse(text: str) — json.loads with strict=False; use for terminal() output with control chars\n"
        "  shell_quote(s: str) — shlex.quote(); use when interpolating dynamic strings into shell commands\n"
        "  retry(fn, max_attempts=3, delay=2) — retry with exponential backoff for transient failures"
    )

    return {
        "name": "execute_code",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute. Import tools with "
                        f"`from hermes_tools import {import_str}` "
                        "and print your final result to stdout."
                    ),
                },
            },
            "required": ["code"],
        },
    }


# Default schema used at registration time (all sandbox tools listed,
# current configured mode).  model_tools.py rebuilds per-session anyway.
EXECUTE_CODE_SCHEMA = build_execute_code_schema()


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="execute_code",
    toolset="code_execution",
    schema=EXECUTE_CODE_SCHEMA,
    handler=lambda args, **kw: execute_code(
        code=args.get("code", ""),
        task_id=kw.get("task_id"),
        enabled_tools=kw.get("enabled_tools")),
    check_fn=check_sandbox_requirements,
    emoji="🐍",
    max_result_size_chars=100_000,
)

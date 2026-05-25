#!/usr/bin/env python3
"""
Tests for the experimental MCP-stub generation in the execute_code sandbox.

Covers the opt-in path enabled by ``code_execution.expose_mcp_tools=true``:
the ``_build_mcp_sandbox_bundle`` helper, the dispatch-allowlist plumbing,
and the end-to-end round-trip of ``from hermes_mcp.<server> import <tool>``
calls through the mocked tool dispatcher.

Run with:  python -m pytest tests/tools/test_code_execution_mcp.py -v
"""

import json
import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    """Match test_code_execution.py: ensure each test starts with TERMINAL_ENV=local."""
    monkeypatch.setenv("TERMINAL_ENV", "local")


from tools.code_execution_tool import (  # noqa: E402
    SANDBOX_ALLOWED_TOOLS,
    _build_mcp_sandbox_bundle,
    execute_code,
)


def _fake_tool(name, description="", input_schema=None, output_schema=None):
    """Build a stand-in for an MCP `Tool` object.

    ``output_schema`` defaults to ``None`` (attribute set to None) — mirrors
    pre-SEP-2106 reality where most servers don't yet emit one.  Pass a
    dict to simulate a post-SEP server that declares its return shape.
    """
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema if input_schema is not None else {"type": "object", "properties": {}},
        outputSchema=output_schema,
    )


def _fake_server_task(tools):
    """Build a stand-in for an MCPServerTask. Only ._tools is consulted."""
    return SimpleNamespace(_tools=list(tools))


@pytest.fixture
def fake_mcp_servers(monkeypatch):
    """Replace tools.mcp_tool._servers with a deterministic fake set.

    _build_mcp_sandbox_bundle does a lazy ``from tools.mcp_tool import ...``
    inside the function, so patching the source attribute (not a name
    cached in code_execution_tool's globals) is what the helper observes.
    """
    fake = {
        "github": _fake_server_task([
            _fake_tool(
                "list_issues",
                description="List issues in a repo.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "state": {"type": "string", "enum": ["open", "closed"]},
                    },
                    "required": ["owner", "repo"],
                },
            ),
            _fake_tool("search_code", description="Search across code."),
        ]),
        "notion": _fake_server_task([
            _fake_tool("query_database", description="Query a Notion DB."),
        ]),
    }
    monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
    monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
    return fake


# ---------------------------------------------------------------------------
# Unit tests: _build_mcp_sandbox_bundle
# ---------------------------------------------------------------------------


class TestBuildMcpSandboxBundle:
    def test_disabled_returns_empty(self, fake_mcp_servers):
        files, names = _build_mcp_sandbox_bundle({})
        assert files == {}
        assert names == set()

    def test_explicit_false_returns_empty(self, fake_mcp_servers):
        files, names = _build_mcp_sandbox_bundle({"expose_mcp_tools": False})
        assert files == {}
        assert names == set()

    def test_enabled_emits_one_module_per_server(self, fake_mcp_servers):
        files, names = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        assert "hermes_mcp/__init__.py" in files
        assert "hermes_mcp/github.py" in files
        assert "hermes_mcp/notion.py" in files
        assert names == {
            "mcp_github_list_issues",
            "mcp_github_search_code",
            "mcp_notion_query_database",
        }

    def test_init_lists_submodules(self, fake_mcp_servers):
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        init = files["hermes_mcp/__init__.py"]
        assert "hermes_mcp.github" in init
        assert "hermes_mcp.notion" in init
        assert "__all__" in init

    def test_stub_dispatches_to_prefixed_registry_name(self, fake_mcp_servers):
        """The generated function must dispatch as ``mcp_<server>_<tool>``
        — that's the registry name in tools/mcp_tool.py:2833, so
        handle_function_call routes them without dispatcher changes.

        Post-fidelity-binding rewrite: the signature carries typed params
        derived from inputSchema, and dispatch goes through a ``_args``
        dict built from the named params (so jsonschema can validate the
        full payload before the RPC fires).
        """
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        github_mod = files["hermes_mcp/github.py"]
        # Server modules now do a multi-line ``from hermes_tools import (...)``
        # block — assert on the import target rather than the exact form.
        assert "from hermes_tools import" in github_mod
        assert "_call," in github_mod
        # Typed signature: required params positional, optional keyword w/ default.
        assert "def list_issues(owner: str, repo: str" in github_mod
        # state is enum on inputSchema → Literal in the signature.
        assert "Literal['open', 'closed']" in github_mod
        # Dispatch site uses the prefixed registry name with the validated args.
        assert "_call('mcp_github_list_issues', _args)" in github_mod
        # search_code has an empty inputSchema (default fixture) → kwargs-only.
        assert "def search_code(" in github_mod
        assert "_call('mcp_github_search_code', _args)" in github_mod

    def test_stub_docstring_contains_schema(self, fake_mcp_servers):
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        github_mod = files["hermes_mcp/github.py"]
        assert "List issues in a repo." in github_mod
        # Schema now lives at module scope as the ``_INPUT_SCHEMA_<tool>``
        # Python literal that the validator is built from — same bytes,
        # same read-depth, no docstring duplication.
        assert "_INPUT_SCHEMA_list_issues" in github_mod
        assert "'owner'" in github_mod
        assert "'required'" in github_mod

    def test_output_schema_embedded_when_server_provides_one(self, monkeypatch):
        """SEP-2106 enables servers to declare outputSchema (arrays /
        primitives / compositions, not just object).  When present, embed
        it so the model knows the response shape — avoids the
        ``result[0]`` vs ``result["result"][0]`` guessing the E2E hit
        against the (pre-SEP) github MCP server."""
        fake = {
            "demo": _fake_server_task([
                _fake_tool(
                    "get_weather_forecast",
                    description="Hourly forecast.",
                    output_schema={
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "hour": {"type": "string"},
                                "temp": {"type": "number"},
                            },
                        },
                    },
                ),
            ]),
        }
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        mod = files["hermes_mcp/demo.py"]
        # Output schema lands at module scope (drives both return-type
        # rendering and the runtime output validator) — the docstring
        # carries the human-readable Returns: section, not a JSON blob.
        assert "_OUTPUT_SCHEMA_get_weather_forecast" in mod
        assert "_register_output_schema('mcp_demo_get_weather_forecast'" in mod
        assert "'hour'" in mod
        # Return annotation reflects the outputSchema (array of objects).
        assert "-> list[dict[str, Any]]" in mod
        # Input schema constant also lives at module scope.
        assert "_INPUT_SCHEMA_get_weather_forecast" in mod

    def test_no_output_validator_when_server_omits_output_schema(self, fake_mcp_servers):
        """Silent degradation: tools without outputSchema (pre-SEP-2106
        majority today) get stubs without an output validator
        registration and without an ``_OUTPUT_SCHEMA_*`` constant.
        Return annotation falls back to ``Any``."""
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        github_mod = files["hermes_mcp/github.py"]
        # fake_mcp_servers fixture's tools don't set output_schema.
        assert "_register_output_schema('mcp_github_list_issues'" not in github_mod
        assert "_OUTPUT_SCHEMA_list_issues" not in github_mod
        # Return annotation is honestly ``Any``.
        assert "-> Any:" in github_mod
        # Input schema constant still emitted.
        assert "_INPUT_SCHEMA_list_issues" in github_mod

    def test_unserializable_output_schema_falls_back_to_omitting(self, monkeypatch):
        """A garbage outputSchema (e.g. a circular reference, a custom
        object) must not crash bundle generation — just skip the line."""
        class Unserializable:
            def __repr__(self):
                return "<unserializable>"

        fake = {
            "demo": _fake_server_task([
                _fake_tool("x", output_schema=Unserializable()),
            ]),
        }
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        mod = files["hermes_mcp/demo.py"]
        # Stub still generated, just without the output validator
        # registration (an unserializable schema can't be passed to
        # ``Draft202012Validator`` at import without taking the whole
        # module down).
        assert "def x(" in mod
        assert "_register_output_schema('mcp_demo_x'" not in mod
        assert "_OUTPUT_SCHEMA_x" not in mod
        # Input validator still registered (input schema is fine).
        assert "_register_input_schema('mcp_demo_x'" in mod

    def test_allowlist_filters_servers(self, fake_mcp_servers):
        files, names = _build_mcp_sandbox_bundle({
            "expose_mcp_tools": True,
            "mcp_servers_allowlist": ["notion"],
        })
        assert "hermes_mcp/notion.py" in files
        assert "hermes_mcp/github.py" not in files
        assert names == {"mcp_notion_query_database"}

    def test_empty_allowlist_filters_everything(self, fake_mcp_servers):
        files, names = _build_mcp_sandbox_bundle({
            "expose_mcp_tools": True,
            "mcp_servers_allowlist": [],
        })
        assert files == {}
        assert names == set()

    def test_unknown_server_in_allowlist_is_skipped_silently(self, fake_mcp_servers):
        files, names = _build_mcp_sandbox_bundle({
            "expose_mcp_tools": True,
            "mcp_servers_allowlist": ["does-not-exist"],
        })
        assert files == {}
        assert names == set()

    def test_name_sanitization_for_hyphenated_server(self, monkeypatch):
        fake = {
            "my-server": _fake_server_task([
                _fake_tool("do.thing", description="Hyphenated names."),
            ]),
        }
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, names = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        assert "hermes_mcp/my_server.py" in files
        assert names == {"mcp_my_server_do_thing"}
        module_src = files["hermes_mcp/my_server.py"]
        assert "def do_thing(" in module_src
        assert "_call('mcp_my_server_do_thing', _args)" in module_src

    def test_server_with_no_tools_is_skipped(self, monkeypatch):
        fake = {"empty": _fake_server_task([])}
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, names = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        assert files == {}
        assert names == set()


# ---------------------------------------------------------------------------
# Runtime: exec the generated wrappers and exercise the validation gate.
# Structural assertions above only check that the right text is emitted.
# These tests confirm the emitted code actually validates as designed.
# ---------------------------------------------------------------------------


def _exec_generated_module(module_src):
    """Compile + exec a generated wrapper module against a real hermes_tools.

    Builds the genuine ``hermes_tools.py`` with ``mcp_enabled=True`` so the
    validator infrastructure (``_validate_input``, ``_register_input_schema``,
    etc.) is in place — the wrapper module's ``from hermes_tools import ...``
    line resolves to real implementations.  Returns the module object so
    callers can rebind ``_call`` on it to script different server responses.
    """
    import types
    from tools.code_execution_tool import generate_hermes_tools_module

    ht_src = generate_hermes_tools_module(
        enabled_tools=[], transport="uds", mcp_enabled=True
    )
    ht = types.ModuleType("hermes_tools")
    # Provide HERMES_RPC_SOCKET so the (unused) connect helper doesn't KeyError.
    os.environ.setdefault("HERMES_RPC_SOCKET", "/tmp/_unused")
    exec(compile(ht_src, "<hermes_tools>", "exec"), ht.__dict__)
    # Tests never actually round-trip to a real RPC socket — stub _call.
    ht._call = lambda name, args: None
    sys.modules["hermes_tools"] = ht

    mod = types.ModuleType("hermes_mcp.test_module")
    exec(compile(module_src, "<emitted>", "exec"), mod.__dict__)
    return mod


class TestGeneratedWrapperRuntime:
    """End-to-end: exec the generated wrappers and verify the validators fire."""

    def test_input_validation_catches_wrong_type(self, monkeypatch):
        fake = {"github": _fake_server_task([
            _fake_tool(
                "list_issues",
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo":  {"type": "string"},
                    },
                    "required": ["owner", "repo"],
                },
            ),
        ])}
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        mod = _exec_generated_module(files["hermes_mcp/github.py"])

        with pytest.raises(ValueError) as exc_info:
            mod.list_issues(owner="oct", repo=123)
        msg = str(exc_info.value)
        assert "list_issues" in msg
        assert "repo" in msg
        assert "string" in msg

    def test_input_validation_enumerates_multiple_failures(self, monkeypatch):
        fake = {"github": _fake_server_task([
            _fake_tool(
                "list_issues",
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo":  {"type": "string", "pattern": "^[a-z]+$"},
                        "state": {"type": "string", "enum": ["open", "closed"]},
                    },
                    "required": ["owner", "repo"],
                },
            ),
        ])}
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        mod = _exec_generated_module(files["hermes_mcp/github.py"])

        # owner: int (bad type), repo: 'BAD' (pattern), state: 'weird' (enum) -
        # the model should see all three at once so it can correct in one turn.
        with pytest.raises(ValueError) as exc_info:
            mod.list_issues(owner=42, repo="BAD", state="weird")
        msg = str(exc_info.value)
        for token in ("owner", "repo", "state"):
            assert token in msg, f"missing field {token} in: {msg}"

    def test_happy_path_passes_validated_args_to_call(self, monkeypatch):
        fake = {"github": _fake_server_task([
            _fake_tool(
                "list_issues",
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo":  {"type": "string"},
                        "state": {"type": "string", "enum": ["open", "closed"]},
                    },
                    "required": ["owner", "repo"],
                },
            ),
        ])}
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        mod = _exec_generated_module(files["hermes_mcp/github.py"])

        captured = []
        def _spy(name, args):
            captured.append((name, args))
            return [{"id": 1}]
        mod._call = _spy

        result = mod.list_issues(owner="oct", repo="foo", state="open")
        assert result == [{"id": 1}]
        assert len(captured) == 1
        name, args = captured[0]
        assert name == "mcp_github_list_issues"
        # None-valued optionals are dropped before validation/dispatch — the
        # server sees "omitted", which is the JSON Schema idiom for absence.
        assert args == {"owner": "oct", "repo": "foo", "state": "open"}

    def test_optional_none_dropped_from_dispatch(self, monkeypatch):
        fake = {"github": _fake_server_task([
            _fake_tool(
                "list_issues",
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "labels": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["owner"],
                },
            ),
        ])}
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        mod = _exec_generated_module(files["hermes_mcp/github.py"])

        captured = []
        mod._call = lambda n, a: captured.append((n, a)) or None
        mod.list_issues(owner="oct")  # labels left at default None
        assert captured[-1][1] == {"owner": "oct"}, captured

    def test_output_validation_warns_but_returns(self, monkeypatch):
        import warnings
        fake = {"demo": _fake_server_task([
            _fake_tool(
                "fetch",
                input_schema={"type": "object", "properties": {}},
                output_schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                    },
                },
            ),
        ])}
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        mod = _exec_generated_module(files["hermes_mcp/demo.py"])

        # Server returns malformed data — missing required `id` on the item.
        mod._call = lambda n, a: [{"wrong_key": "no id"}]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mod.fetch()
        # Result still returned — don't punish the model for server drift.
        assert result == [{"wrong_key": "no id"}]
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert runtime_warnings, "expected a RuntimeWarning on output drift"
        msg = str(runtime_warnings[0].message)
        assert "fetch" in msg
        assert "id" in msg

    def test_typed_signature_carries_literal_and_optional(self, monkeypatch):
        """The wrapper's __annotations__ reflect JSON Schema enum/nullability."""
        import typing
        fake = {"demo": _fake_server_task([
            _fake_tool(
                "act",
                input_schema={
                    "type": "object",
                    "properties": {
                        "id":   {"type": "string"},
                        "mode": {"type": "string", "enum": ["one", "two"]},
                    },
                    "required": ["id"],
                },
            ),
        ])}
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        mod = _exec_generated_module(files["hermes_mcp/demo.py"])

        annotations = typing.get_type_hints(mod.act)
        # Required string param → bare str
        assert annotations["id"] is str
        # Optional enum → Optional[Literal[...]] — origin is Union, args
        # include None and the Literal.
        mode_t = annotations["mode"]
        origin = typing.get_origin(mode_t)
        assert origin is typing.Union
        args = typing.get_args(mode_t)
        assert type(None) in args
        literal_arg = [a for a in args if a is not type(None)][0]
        assert typing.get_origin(literal_arg) is typing.Literal
        assert set(typing.get_args(literal_arg)) == {"one", "two"}


# ---------------------------------------------------------------------------
# Integration: end-to-end through the real RPC path (UDS / TCP fallback)
# ---------------------------------------------------------------------------


def _mock_handle_function_call(function_name, function_args, task_id=None, user_task=None):
    """Mock dispatcher that recognizes built-in stubs and MCP-prefixed names."""
    if function_name == "mcp_fake_tool_a":
        return json.dumps({"result": {"echoed": function_args, "from": "mcp_fake_tool_a"}})
    if function_name == "mcp_fake_tool_b":
        return json.dumps({"result": {"count": len(function_args)}})
    if function_name == "terminal":
        return json.dumps({"output": "mock", "exit_code": 0})
    return json.dumps({"error": f"Unknown tool in mock: {function_name}"})


@pytest.fixture
def fake_one_server(monkeypatch):
    fake = {
        "fake": _fake_server_task([
            _fake_tool("tool_a", description="Echo back args."),
            _fake_tool("tool_b", description="Count args."),
        ]),
    }
    monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
    monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
    return fake


@unittest.skipIf(sys.platform == "win32", "UDS not available on Windows in CI")
class TestExecuteCodeWithMcpStubs:
    """Integration tests: child process imports hermes_mcp and round-trips through RPC."""

    def _run(self, code, *, cfg, enabled_tools=None):
        with patch("tools.code_execution_tool._load_config", return_value=cfg), \
             patch("model_tools.handle_function_call", side_effect=_mock_handle_function_call):
            raw = execute_code(
                code=code,
                task_id="test-mcp-task",
                enabled_tools=enabled_tools or list(SANDBOX_ALLOWED_TOOLS),
            )
        return json.loads(raw)

    def test_disabled_flag_blocks_import(self, fake_one_server):
        """expose_mcp_tools=false ⇒ hermes_mcp package not written, import fails."""
        code = "from hermes_mcp.fake import tool_a\nprint(tool_a(x=1))\n"
        result = self._run(code, cfg={"timeout": 30, "max_tool_calls": 5})
        assert result["status"] == "error"
        assert "ModuleNotFoundError" in result.get("error", "") or \
               "ModuleNotFoundError" in result.get("output", "")

    def test_enabled_flag_round_trips_through_rpc(self, fake_one_server):
        """expose_mcp_tools=true ⇒ script imports stub, RPC routes mcp_fake_tool_a."""
        code = (
            "from hermes_mcp.fake import tool_a\n"
            "print(tool_a(x=1, y='hello'))\n"
        )
        result = self._run(code, cfg={
            "timeout": 30,
            "max_tool_calls": 5,
            "expose_mcp_tools": True,
        })
        assert result["status"] == "success"
        assert "mcp_fake_tool_a" in result["output"]
        assert "'x': 1" in result["output"]
        assert "'y': 'hello'" in result["output"]
        assert result["tool_calls_made"] == 1

    def test_allowlist_at_call_level_blocks_unlisted_server(self, monkeypatch):
        """If a server is registered but not in mcp_servers_allowlist, the
        stub isn't generated AND the dispatch allowlist excludes it — so even
        a hand-crafted _call('mcp_<server>_<tool>', ...) gets rejected."""
        fake = {
            "fake": _fake_server_task([_fake_tool("tool_a")]),
            "other": _fake_server_task([_fake_tool("tool_x")]),
        }
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        code = (
            "from hermes_tools import _call\n"
            "print(_call('mcp_other_tool_x', {}))\n"
        )
        result = self._run(code, cfg={
            "timeout": 30,
            "max_tool_calls": 5,
            "expose_mcp_tools": True,
            "mcp_servers_allowlist": ["fake"],
        })
        assert result["status"] == "success"
        # The dispatch gate rejected it with the "not available" error
        assert "is not available in execute_code" in result["output"]

    def test_per_call_tmpdir_wrappers_still_take_precedence(
        self, tmp_path, monkeypatch, fake_one_server,
    ):
        """Slice A regression guard: the per-call tmpdir copy of hermes_mcp/
        must keep being the package Python picks up inside the sandbox,
        independent of whatever's at the stable HERMES_HOME path.  The
        stable path is for between-turn read_file/search_files only — it
        deliberately does NOT compete on PYTHONPATH (a regular __init__.py
        in tmpdir shadows the stable copy anyway, by Python's package
        resolution rules)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Write a stable hermes_mcp/fake.py with a poison marker — if it
        # ever overrode the per-call copy, the script would print this.
        stable_root = tmp_path / "code-execution" / "mcp"
        (stable_root / "hermes_mcp").mkdir(parents=True)
        (stable_root / "hermes_mcp" / "__init__.py").write_text("", encoding="utf-8")
        (stable_root / "hermes_mcp" / "fake.py").write_text(
            "def tool_a(**kwargs):\n"
            "    return {'POISON': 'stable-path-wins-which-is-bad'}\n",
            encoding="utf-8",
        )

        code = (
            "from hermes_mcp.fake import tool_a\n"
            "print(tool_a(x=1))\n"
        )
        result = self._run(code, cfg={
            "timeout": 30,
            "max_tool_calls": 5,
            "expose_mcp_tools": True,
        })
        assert result["status"] == "success"
        # The per-call tmpdir copy routes through RPC to the mock dispatcher,
        # which returns {'echoed': ..., 'from': 'mcp_fake_tool_a'}.  If the
        # stable poison ever won, we'd see 'POISON' instead.
        assert "POISON" not in result["output"]
        assert "mcp_fake_tool_a" in result["output"]

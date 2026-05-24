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


def _fake_tool(name, description="", input_schema=None):
    """Build a stand-in for an MCP `Tool` object."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema if input_schema is not None else {"type": "object", "properties": {}},
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
        """The generated function must _call("mcp_<server>_<tool>", kwargs)
        — that's the name MCP tools are registered under in the Hermes registry
        (tools/mcp_tool.py:2833), so handle_function_call routes them
        without dispatcher changes."""
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        github_mod = files["hermes_mcp/github.py"]
        assert "from hermes_tools import _call" in github_mod
        assert "def list_issues(**kwargs):" in github_mod
        assert "return _call('mcp_github_list_issues', kwargs)" in github_mod
        assert "def search_code(**kwargs):" in github_mod
        assert "return _call('mcp_github_search_code', kwargs)" in github_mod

    def test_stub_docstring_contains_schema(self, fake_mcp_servers):
        files, _ = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        github_mod = files["hermes_mcp/github.py"]
        assert "List issues in a repo." in github_mod
        # inputSchema embedded as JSON so the LLM can read it via help()
        assert '"owner"' in github_mod
        assert '"required"' in github_mod

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
        assert "def do_thing(**kwargs):" in module_src
        assert "return _call('mcp_my_server_do_thing', kwargs)" in module_src

    def test_server_with_no_tools_is_skipped(self, monkeypatch):
        fake = {"empty": _fake_server_task([])}
        monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        files, names = _build_mcp_sandbox_bundle({"expose_mcp_tools": True})
        assert files == {}
        assert names == set()


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

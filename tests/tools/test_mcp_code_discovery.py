#!/usr/bin/env python3
"""
Tests for the post-MCP-discovery hooks (Slice 2 of code_execution.expose_mcp_tools).

Covers:
  * _categorize_tool_by_name — the verb-prefix heuristic
  * _write_stable_mcp_wrappers — Slice A wipe-and-regenerate of the stable
    ``~/.hermes/code-execution/mcp/hermes_mcp/`` package
  * _write_mcp_auto_skills — Slice C wipe-and-regenerate of the per-server
    ``~/.hermes/skills/mcp-auto/mcp-<server>/SKILL.md`` files
  * apply_post_discovery_hooks — flag gating, error-swallowing posture
"""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mcp_code_discovery import (
    AUTO_SKILL_SUBDIR,
    STABLE_WRAPPER_SUBDIR,
    _categorize_tool_by_name,
    _render_skill_markdown,
    _write_mcp_auto_skills,
    _write_stable_mcp_wrappers,
    apply_post_discovery_hooks,
    auto_skill_root,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME to a per-test tmp dir.

    Patches both the env var (so anything that reads it after the fact picks
    up the override) AND the in-module references in tools.mcp_code_discovery
    (which captured ``get_hermes_home`` at import time).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _fake_tool(name, description="", input_schema=None):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema if input_schema is not None
        else {"type": "object", "properties": {}},
    )


def _fake_server_task(tools):
    return SimpleNamespace(_tools=list(tools))


@pytest.fixture
def fake_mcp_servers(monkeypatch):
    fake = {
        "github": _fake_server_task([
            _fake_tool("list_issues", description="List issues in a repo."),
            _fake_tool("search_code", description="Search across code."),
            _fake_tool("create_pull_request", description="Open a new PR."),
            _fake_tool("delete_branch", description="Delete a branch."),
            _fake_tool("do_weird_thing", description="Verb is unclassifiable."),
        ]),
        "notion": _fake_server_task([
            _fake_tool("query_database", description="Query a Notion DB."),
            _fake_tool("update_page", description="Update a page."),
        ]),
    }
    monkeypatch.setattr("tools.mcp_tool._servers", fake, raising=True)
    monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
    return fake


# ---------------------------------------------------------------------------
# Heuristic categorization
# ---------------------------------------------------------------------------


class TestCategorizeToolByName:
    @pytest.mark.parametrize("name", [
        "list_issues", "get_user", "search_code", "find_file", "read_log",
        "show_diff", "view_pr", "query_database", "count_results",
        "describe_table", "inspect_session", "fetch_changes", "lookup_user",
    ])
    def test_read_prefixes(self, name):
        assert _categorize_tool_by_name(name) == "read"

    @pytest.mark.parametrize("name", [
        "delete_branch", "remove_user", "drop_table", "erase_history",
        "clear_cache", "purge_records", "destroy_session", "revoke_token",
        "uninstall_package", "unassign_role",
    ])
    def test_destructive_prefixes(self, name):
        assert _categorize_tool_by_name(name) == "destroy"

    @pytest.mark.parametrize("name", [
        "create_issue", "update_user", "write_file", "set_label",
        "post_comment", "put_object", "patch_doc", "add_member",
        "assign_reviewer", "rename_file", "move_card", "copy_object",
        "merge_pr", "send_message", "install_app", "enable_feature",
        "disable_feature",
    ])
    def test_mutating_prefixes(self, name):
        assert _categorize_tool_by_name(name) == "mutate"

    @pytest.mark.parametrize("name", ["do_weird_thing", "ping", "noop", ""])
    def test_other_or_unknown(self, name):
        assert _categorize_tool_by_name(name) == "other"

    def test_destroy_beats_mutate_for_safety(self):
        """A name starting with a destructive prefix wins over an embedded
        mutating prefix, so dangerous operations don't get hidden in the
        mutating bucket."""
        # delete_and_recreate starts with delete_, so it's destroy
        assert _categorize_tool_by_name("delete_and_recreate") == "destroy"

    def test_case_insensitive(self):
        assert _categorize_tool_by_name("LIST_THINGS") == "read"
        assert _categorize_tool_by_name("Delete_X") == "destroy"


# ---------------------------------------------------------------------------
# Slice A: stable wrappers
# ---------------------------------------------------------------------------


class TestWriteStableMcpWrappers:
    def test_disabled_returns_none_no_write(self, tmp_hermes_home, fake_mcp_servers):
        result = _write_stable_mcp_wrappers({})
        assert result is None
        assert not (tmp_hermes_home / STABLE_WRAPPER_SUBDIR / "hermes_mcp").exists()

    def test_enabled_writes_package(self, tmp_hermes_home, fake_mcp_servers):
        result = _write_stable_mcp_wrappers({"expose_mcp_tools": True})
        assert result == tmp_hermes_home / STABLE_WRAPPER_SUBDIR
        pkg = result / "hermes_mcp"
        assert (pkg / "__init__.py").is_file()
        assert (pkg / "github.py").is_file()
        assert (pkg / "notion.py").is_file()
        gh = (pkg / "github.py").read_text(encoding="utf-8")
        assert "def list_issues(**kwargs):" in gh
        assert "from hermes_tools import _call" in gh

    def test_regenerate_wipes_stale_servers(self, tmp_hermes_home, fake_mcp_servers, monkeypatch):
        # First pass: both github and notion exist
        _write_stable_mcp_wrappers({"expose_mcp_tools": True})
        pkg = tmp_hermes_home / STABLE_WRAPPER_SUBDIR / "hermes_mcp"
        assert (pkg / "notion.py").exists()

        # Remove notion from the connected set and regenerate
        smaller = {"github": fake_mcp_servers["github"]}
        monkeypatch.setattr("tools.mcp_tool._servers", smaller, raising=True)
        _write_stable_mcp_wrappers({"expose_mcp_tools": True})

        assert (pkg / "github.py").exists()
        assert not (pkg / "notion.py").exists(), \
            "stale wrapper must be wiped or the agent will import a tool that no longer dispatches"


# ---------------------------------------------------------------------------
# Slice C: auto-generated skills
# ---------------------------------------------------------------------------


class TestWriteMcpAutoSkills:
    def test_disabled_returns_empty(self, tmp_hermes_home, fake_mcp_servers):
        result = _write_mcp_auto_skills({})
        assert result == []
        assert not auto_skill_root().exists()

    def test_enabled_writes_one_skill_per_server(self, tmp_hermes_home, fake_mcp_servers):
        result = _write_mcp_auto_skills({"expose_mcp_tools": True})
        paths = sorted(p.relative_to(tmp_hermes_home).as_posix() for p in result)
        assert paths == [
            f"skills/{AUTO_SKILL_SUBDIR}/mcp-github/SKILL.md",
            f"skills/{AUTO_SKILL_SUBDIR}/mcp-notion/SKILL.md",
        ]

    def test_skill_frontmatter_shape(self, tmp_hermes_home, fake_mcp_servers):
        _write_mcp_auto_skills({"expose_mcp_tools": True})
        gh = (auto_skill_root() / "mcp-github" / "SKILL.md").read_text(encoding="utf-8")
        # YAML frontmatter required fields per skills_tool.py
        assert "name: mcp-github" in gh
        assert "description:" in gh
        # Description includes the categorized counts so the user can see the
        # surface from `skills_list` without opening the skill.
        # github fake = list_issues, search_code, create_pull_request, delete_branch, do_weird_thing
        #             = 2 read / 1 mutate / 1 destroy / 1 other = 5 total
        assert "5 tools" in gh
        assert "2 read-only" in gh
        assert "1 mutating" in gh
        assert "1 destructive" in gh

    def test_skill_categorizes_tools_correctly(self, tmp_hermes_home, fake_mcp_servers):
        _write_mcp_auto_skills({"expose_mcp_tools": True})
        gh = (auto_skill_root() / "mcp-github" / "SKILL.md").read_text(encoding="utf-8")
        # Section ordering and presence
        read_idx = gh.find("### Read-only")
        mutate_idx = gh.find("### Mutating")
        destroy_idx = gh.find("### Destructive")
        other_idx = gh.find("### Other")
        assert 0 < read_idx < mutate_idx < destroy_idx < other_idx
        # list_issues lives in Read-only
        assert "list_issues" in gh[read_idx:mutate_idx]
        # create_pull_request lives in Mutating
        assert "create_pull_request" in gh[mutate_idx:destroy_idx]
        # delete_branch lives in Destructive
        assert "delete_branch" in gh[destroy_idx:other_idx]
        # do_weird_thing lives in Other
        assert "do_weird_thing" in gh[other_idx:]

    def test_skill_example_uses_a_read_tool(self, tmp_hermes_home, fake_mcp_servers):
        """The example block should pick a read-only tool when one exists —
        safer default than auto-suggesting a destructive op."""
        _write_mcp_auto_skills({"expose_mcp_tools": True})
        gh = (auto_skill_root() / "mcp-github" / "SKILL.md").read_text(encoding="utf-8")
        assert "## Example" in gh
        example_block = gh.split("## Example", 1)[1]
        # First read-only alphabetically is list_issues
        assert "from hermes_mcp.github import list_issues" in example_block

    def test_regenerate_wipes_stale_servers(self, tmp_hermes_home, fake_mcp_servers, monkeypatch):
        _write_mcp_auto_skills({"expose_mcp_tools": True})
        assert (auto_skill_root() / "mcp-notion" / "SKILL.md").exists()

        smaller = {"github": fake_mcp_servers["github"]}
        monkeypatch.setattr("tools.mcp_tool._servers", smaller, raising=True)
        _write_mcp_auto_skills({"expose_mcp_tools": True})

        assert (auto_skill_root() / "mcp-github" / "SKILL.md").exists()
        assert not (auto_skill_root() / "mcp-notion").exists()

    def test_does_not_touch_unrelated_skills(self, tmp_hermes_home, fake_mcp_servers):
        # Pre-existing hand-authored skill under skills/my-skill/ must survive
        # wipe-and-regenerate of mcp-auto/.
        user_skill = tmp_hermes_home / "skills" / "my-skill" / "SKILL.md"
        user_skill.parent.mkdir(parents=True, exist_ok=True)
        user_skill.write_text("---\nname: my-skill\ndescription: mine.\n---\nhi\n",
                              encoding="utf-8")
        _write_mcp_auto_skills({"expose_mcp_tools": True})
        assert user_skill.exists(), "auto-skill writer must never touch sibling skill dirs"

    def test_allowlist_filters_servers(self, tmp_hermes_home, fake_mcp_servers):
        result = _write_mcp_auto_skills({
            "expose_mcp_tools": True,
            "mcp_servers_allowlist": ["notion"],
        })
        names = [p.parent.name for p in result]
        assert names == ["mcp-notion"]
        assert not (auto_skill_root() / "mcp-github").exists()


# ---------------------------------------------------------------------------
# _render_skill_markdown — unit-level rendering checks
# ---------------------------------------------------------------------------


class TestRenderSkillMarkdown:
    def test_all_empty_buckets_still_emits_header(self):
        md = _render_skill_markdown("empty", "empty", {
            "read": [], "mutate": [], "destroy": [], "other": [],
        })
        assert "name: mcp-empty" in md
        assert "0 tools" in md
        # No section bodies, but the header table is still there
        assert "## Tools" in md

    def test_no_example_when_no_tools(self):
        md = _render_skill_markdown("empty", "empty", {
            "read": [], "mutate": [], "destroy": [], "other": [],
        })
        assert "## Example" not in md

    def test_server_name_with_hyphen_uses_safe_form_in_import(self):
        # Caller (apply_post_discovery_hooks) is responsible for sanitization,
        # so we exercise _render with already-sanitized inputs.
        md = _render_skill_markdown("my-server", "my_server", {
            "read": [("get_x", "get_x", "Get X.")],
            "mutate": [], "destroy": [], "other": [],
        })
        assert "from hermes_mcp.my_server import get_x" in md
        # Original server name preserved in human-readable header for clarity
        assert "MCP server: my-server" in md


# ---------------------------------------------------------------------------
# Entry point: apply_post_discovery_hooks
# ---------------------------------------------------------------------------


class TestApplyPostDiscoveryHooks:
    def test_disabled_short_circuits(self, tmp_hermes_home, fake_mcp_servers):
        apply_post_discovery_hooks({"expose_mcp_tools": False})
        assert not (tmp_hermes_home / STABLE_WRAPPER_SUBDIR / "hermes_mcp").exists()
        assert not auto_skill_root().exists()

    def test_enabled_runs_both_hooks(self, tmp_hermes_home, fake_mcp_servers):
        apply_post_discovery_hooks({"expose_mcp_tools": True})
        assert (tmp_hermes_home / STABLE_WRAPPER_SUBDIR / "hermes_mcp" / "github.py").exists()
        assert (auto_skill_root() / "mcp-github" / "SKILL.md").exists()

    def test_loader_path_when_cfg_omitted(self, tmp_hermes_home, fake_mcp_servers, monkeypatch):
        """cfg=None ⇒ calls tools.code_execution_tool._load_config."""
        called = {}

        def _fake_loader():
            called["yes"] = True
            return {"expose_mcp_tools": True}

        monkeypatch.setattr("tools.code_execution_tool._load_config", _fake_loader)
        apply_post_discovery_hooks(None)
        assert called.get("yes") is True

    def test_swallows_inner_exceptions(self, tmp_hermes_home, fake_mcp_servers, monkeypatch):
        """A failing inner helper must not break MCP discovery overall."""
        def _boom(_cfg):
            raise RuntimeError("simulated explosion")

        monkeypatch.setattr(
            "tools.mcp_code_discovery._write_stable_mcp_wrappers", _boom,
        )
        # Should not raise — failure is logged and the other helper still runs
        apply_post_discovery_hooks({"expose_mcp_tools": True})
        # Slice C still completed despite Slice A failing
        assert (auto_skill_root() / "mcp-github" / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# Slice B: three-gate conditional for the MCP-as-code prompt nudge.
# Lives in agent/system_prompt.py:_should_inject_mcp_as_code.  We test it
# here (rather than in test_prompt_builder.py) because all three gates are
# tied to the same Slice 2 feature flag the rest of this file exercises.
# ---------------------------------------------------------------------------


class TestShouldInjectMcpAsCode:
    """Each test must flip exactly one gate; if any single gate fails the
    helper must return False so the prompt nudge never lands in a session
    where it would be misleading."""

    @pytest.fixture
    def _import(self):
        from agent.system_prompt import _should_inject_mcp_as_code
        return _should_inject_mcp_as_code

    def test_all_gates_pass(self, _import, monkeypatch, fake_mcp_servers):
        monkeypatch.setattr(
            "tools.code_execution_tool._load_config",
            lambda: {"expose_mcp_tools": True},
        )
        assert _import({"execute_code", "terminal"}) is True

    def test_gate_1_fails_without_execute_code(self, _import, monkeypatch, fake_mcp_servers):
        """Gate 1: if execute_code isn't in valid_tool_names, the model has
        no way to act on the guidance — short-circuit BEFORE the cfg read
        so we don't pay for it on every session."""
        monkeypatch.setattr(
            "tools.code_execution_tool._load_config",
            lambda: {"expose_mcp_tools": True},
        )
        assert _import({"terminal", "memory"}) is False

    def test_gate_2_fails_when_flag_off(self, _import, monkeypatch, fake_mcp_servers):
        """Gate 2: feature flag off ⇒ guidance must not land even if MCP
        servers are connected."""
        monkeypatch.setattr(
            "tools.code_execution_tool._load_config",
            lambda: {"expose_mcp_tools": False},
        )
        assert _import({"execute_code"}) is False

    def test_gate_2_fails_when_cfg_key_missing(self, _import, monkeypatch, fake_mcp_servers):
        monkeypatch.setattr(
            "tools.code_execution_tool._load_config", lambda: {},
        )
        assert _import({"execute_code"}) is False

    def test_gate_3_fails_when_no_mcp_servers(self, _import, monkeypatch):
        """Gate 3: even with the flag on, no connected MCP servers ⇒ the
        wrapper imports the guidance recommends resolve to nothing."""
        monkeypatch.setattr(
            "tools.code_execution_tool._load_config",
            lambda: {"expose_mcp_tools": True},
        )
        monkeypatch.setattr("tools.mcp_tool._servers", {}, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        assert _import({"execute_code"}) is False

    def test_config_load_failure_short_circuits_to_false(self, _import, monkeypatch, fake_mcp_servers):
        """A transient _load_config failure must not propagate — the prompt
        nudge is an enhancement, never a requirement for system-prompt
        assembly to succeed."""
        def _boom():
            raise RuntimeError("config read exploded")
        monkeypatch.setattr("tools.code_execution_tool._load_config", _boom)
        assert _import({"execute_code"}) is False

    def test_mcp_module_failure_short_circuits_to_false(self, _import, monkeypatch):
        monkeypatch.setattr(
            "tools.code_execution_tool._load_config",
            lambda: {"expose_mcp_tools": True},
        )
        # Simulate the import path failing — set _servers to something that
        # raises when accessed.  Easier: delete the attribute and let the
        # import succeed but the bool() coerce something explosive.
        class _ExplodingDict:
            def __bool__(self):
                raise RuntimeError("simulated dict failure")
        monkeypatch.setattr("tools.mcp_tool._servers", _ExplodingDict(), raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        assert _import({"execute_code"}) is False

#!/usr/bin/env python3
"""
Tests for the post-MCP-discovery hooks (Slice 4 — README catalog version).

Covers:
  * _categorize_tool_by_name — the verb-prefix heuristic
  * _write_stable_mcp_wrappers — wipe-and-regenerate of the stable
    ``~/.hermes/code-execution/mcp/hermes_mcp/`` package
  * _write_wrapper_readme — per-server categorized index inside the
    package + diff-since-last-launch section
  * _cleanup_legacy_auto_skills — one-time removal of the dropped
    ``~/.hermes/skills/mcp-auto/`` directory
  * apply_post_discovery_hooks — flag gating + error-swallowing posture
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mcp_code_discovery import (
    STABLE_WRAPPER_SUBDIR,
    _LEGACY_AUTO_SKILL_SUBDIR,
    _categorize_tool_by_name,
    _cleanup_legacy_auto_skills,
    _diff_manifests,
    _readme_path,
    _required_args,
    _schema_hash,
    _write_stable_mcp_wrappers,
    _write_wrapper_readme,
    apply_post_discovery_hooks,
    stable_wrapper_root,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_hermes_home(tmp_path, monkeypatch):
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
            _fake_tool(
                "list_issues",
                description="List issues in a repo.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                    },
                    "required": ["owner", "repo"],
                },
            ),
            _fake_tool("search_code", description="Search across code."),
            _fake_tool(
                "create_pull_request",
                description="Open a new PR.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "required": ["owner", "repo", "title"],
                },
            ),
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
    ])
    def test_destructive_prefixes(self, name):
        assert _categorize_tool_by_name(name) == "destroy"

    @pytest.mark.parametrize("name", [
        "create_issue", "update_user", "write_file", "set_label",
        "post_comment", "put_object", "patch_doc", "add_member",
    ])
    def test_mutating_prefixes(self, name):
        assert _categorize_tool_by_name(name) == "mutate"

    @pytest.mark.parametrize("name", ["do_weird_thing", "ping", "noop", ""])
    def test_other_or_unknown(self, name):
        assert _categorize_tool_by_name(name) == "other"

    def test_destroy_beats_mutate_for_safety(self):
        # delete_and_recreate starts with delete_, so it's destroy — the
        # safer-to-flag category wins when the name mixes verbs.
        assert _categorize_tool_by_name("delete_and_recreate") == "destroy"

    def test_case_insensitive(self):
        assert _categorize_tool_by_name("LIST_THINGS") == "read"
        assert _categorize_tool_by_name("Delete_X") == "destroy"


# ---------------------------------------------------------------------------
# _required_args + _schema_hash
# ---------------------------------------------------------------------------


class TestSchemaHelpers:
    def test_required_args_returns_required_field(self):
        tool = _fake_tool("x", input_schema={
            "type": "object",
            "required": ["a", "b"],
        })
        assert _required_args(tool) == ["a", "b"]

    def test_required_args_empty_when_missing(self):
        tool = _fake_tool("x", input_schema={"type": "object"})
        assert _required_args(tool) == []

    def test_required_args_filters_non_strings(self):
        tool = _fake_tool("x", input_schema={"required": ["a", 42, None, "b"]})
        assert _required_args(tool) == ["a", "b"]

    def test_schema_hash_stable_for_same_input(self):
        # Same schema → same hash, regardless of dict iteration order
        s1 = {"type": "object", "properties": {"a": {"type": "string"}}}
        s2 = {"properties": {"a": {"type": "string"}}, "type": "object"}
        h1 = _schema_hash(_fake_tool("x", input_schema=s1))
        h2 = _schema_hash(_fake_tool("x", input_schema=s2))
        assert h1 == h2 and h1 != ""

    def test_schema_hash_changes_when_schema_changes(self):
        h1 = _schema_hash(_fake_tool("x", input_schema={"required": ["a"]}))
        h2 = _schema_hash(_fake_tool("x", input_schema={"required": ["a", "b"]}))
        assert h1 != h2

    def test_schema_hash_empty_for_missing_or_unserializable(self):
        # `_fake_tool` defaults to `{"type": "object", "properties": {}}` —
        # bypass it to get a truly inputSchema=None tool.
        missing = SimpleNamespace(name="x", description="", inputSchema=None)
        assert _schema_hash(missing) == ""

        class _Bad:
            pass
        bad = SimpleNamespace(name="x", description="", inputSchema=_Bad())
        assert _schema_hash(bad) == ""


# ---------------------------------------------------------------------------
# Stable wrapper path (unchanged behavior from earlier slices)
# ---------------------------------------------------------------------------


class TestWriteStableMcpWrappers:
    def test_disabled_returns_none(self, tmp_hermes_home, fake_mcp_servers):
        assert _write_stable_mcp_wrappers({}) is None
        assert not (tmp_hermes_home / STABLE_WRAPPER_SUBDIR / "hermes_mcp").exists()

    def test_enabled_writes_package(self, tmp_hermes_home, fake_mcp_servers):
        root = _write_stable_mcp_wrappers({"expose_mcp_tools": True})
        assert root == tmp_hermes_home / STABLE_WRAPPER_SUBDIR
        pkg = root / "hermes_mcp"
        assert (pkg / "__init__.py").is_file()
        assert (pkg / "github.py").is_file()
        gh = (pkg / "github.py").read_text(encoding="utf-8")
        assert "def list_issues(**kwargs):" in gh
        assert "from hermes_tools import _call" in gh

    def test_regenerate_wipes_stale(self, tmp_hermes_home, fake_mcp_servers, monkeypatch):
        _write_stable_mcp_wrappers({"expose_mcp_tools": True})
        pkg = tmp_hermes_home / STABLE_WRAPPER_SUBDIR / "hermes_mcp"
        assert (pkg / "notion.py").exists()

        smaller = {"github": fake_mcp_servers["github"]}
        monkeypatch.setattr("tools.mcp_tool._servers", smaller, raising=True)
        _write_stable_mcp_wrappers({"expose_mcp_tools": True})

        assert (pkg / "github.py").exists()
        assert not (pkg / "notion.py").exists()


# ---------------------------------------------------------------------------
# _write_wrapper_readme — Slice 4's catalog file
# ---------------------------------------------------------------------------


class TestWriteWrapperReadme:
    def test_disabled_writes_nothing(self, tmp_hermes_home, fake_mcp_servers):
        assert _write_wrapper_readme({}) is None
        assert not _readme_path().exists()

    def test_writes_readme_at_expected_path(self, tmp_hermes_home, fake_mcp_servers):
        result = _write_wrapper_readme({"expose_mcp_tools": True})
        expected = tmp_hermes_home / STABLE_WRAPPER_SUBDIR / "hermes_mcp" / "README.md"
        assert result == expected
        assert expected.is_file()

    def test_readme_contains_per_server_sections(self, tmp_hermes_home, fake_mcp_servers):
        _write_wrapper_readme({"expose_mcp_tools": True})
        md = _readme_path().read_text(encoding="utf-8")
        assert "## github" in md
        assert "## notion" in md
        # The intro example block and import hint
        assert "from hermes_mcp.<server> import <tool>" in md
        assert "Import with: `from hermes_mcp.github import <tool>`" in md
        assert "Import with: `from hermes_mcp.notion import <tool>`" in md

    def test_readme_categorizes_tools(self, tmp_hermes_home, fake_mcp_servers):
        _write_wrapper_readme({"expose_mcp_tools": True})
        md = _readme_path().read_text(encoding="utf-8")
        # github fake: list_issues, search_code (read), create_pull_request (mutate),
        # delete_branch (destroy), do_weird_thing (other)
        gh_section = md.split("## github", 1)[1].split("## notion", 1)[0]
        read_idx = gh_section.find("### Read-only")
        mutate_idx = gh_section.find("### Mutating")
        destroy_idx = gh_section.find("### Destructive")
        other_idx = gh_section.find("### Other")
        assert 0 < read_idx < mutate_idx < destroy_idx < other_idx
        assert "list_issues" in gh_section[read_idx:mutate_idx]
        assert "create_pull_request" in gh_section[mutate_idx:destroy_idx]
        assert "delete_branch" in gh_section[destroy_idx:other_idx]
        assert "do_weird_thing" in gh_section[other_idx:]

    def test_signatures_include_required_args(self, tmp_hermes_home, fake_mcp_servers):
        """The categorized list shows the call shape inline so the model
        doesn't have to read the wrapper file to learn required args."""
        _write_wrapper_readme({"expose_mcp_tools": True})
        md = _readme_path().read_text(encoding="utf-8")
        # list_issues has required owner, repo
        assert "`list_issues(owner, repo)`" in md
        # create_pull_request has required owner, repo, title
        assert "`create_pull_request(owner, repo, title)`" in md
        # search_code (no required args in fixture) → bare parens
        assert "`search_code()`" in md

    def test_header_summarizes_counts(self, tmp_hermes_home, fake_mcp_servers):
        _write_wrapper_readme({"expose_mcp_tools": True})
        md = _readme_path().read_text(encoding="utf-8")
        # github: 5 total — 2 read / 1 mutate / 1 destroy / 1 other
        assert "## github (5 tools — 2 read-only, 1 mutating, 1 destructive)" in md
        # notion: 2 total — 1 read (query_database), 1 mutate (update_page)
        assert "## notion (2 tools — 1 read-only, 1 mutating, 0 destructive)" in md

    def test_allowlist_filters_servers(self, tmp_hermes_home, fake_mcp_servers):
        _write_wrapper_readme({
            "expose_mcp_tools": True,
            "mcp_servers_allowlist": ["notion"],
        })
        md = _readme_path().read_text(encoding="utf-8")
        assert "## notion" in md
        assert "## github" not in md

    def test_no_servers_no_readme_and_stale_cleaned(self, tmp_hermes_home, monkeypatch):
        # Pretend a stale README + manifest exists from a previous launch
        readme = _readme_path()
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text("stale content", encoding="utf-8")
        from tools.mcp_code_discovery import _manifest_path
        manifest = _manifest_path()
        manifest.write_text("{}", encoding="utf-8")

        # No servers now
        monkeypatch.setattr("tools.mcp_tool._servers", {}, raising=True)
        monkeypatch.setattr("tools.mcp_tool._lock", threading.Lock(), raising=True)
        assert _write_wrapper_readme({"expose_mcp_tools": True}) is None
        # Stale README and manifest both gone — README shouldn't lie when
        # nothing's actually connected
        assert not readme.exists()
        assert not manifest.exists()

    def test_writes_sidecar_manifest(self, tmp_hermes_home, fake_mcp_servers):
        _write_wrapper_readme({"expose_mcp_tools": True})
        from tools.mcp_code_discovery import _manifest_path
        manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert "github" in manifest["servers"]
        assert "list_issues" in manifest["servers"]["github"]
        # Schema hash recorded as a string
        assert isinstance(
            manifest["servers"]["github"]["list_issues"]["schema_hash"], str
        )


# ---------------------------------------------------------------------------
# Diff-since-last-launch
# ---------------------------------------------------------------------------


class TestDiffSinceLastLaunch:
    def test_first_run_no_diff_section(self, tmp_hermes_home, fake_mcp_servers):
        # No prior manifest → no diff section even though everything is "new"
        _write_wrapper_readme({"expose_mcp_tools": True})
        md = _readme_path().read_text(encoding="utf-8")
        assert "Recent changes" not in md

    def test_no_changes_omits_section(self, tmp_hermes_home, fake_mcp_servers):
        # Two writes with identical state → second should have no diff section
        _write_wrapper_readme({"expose_mcp_tools": True})
        _write_wrapper_readme({"expose_mcp_tools": True})
        md = _readme_path().read_text(encoding="utf-8")
        assert "Recent changes" not in md

    def test_added_tool_shows_in_diff(self, tmp_hermes_home, fake_mcp_servers, monkeypatch):
        # First run baseline
        _write_wrapper_readme({"expose_mcp_tools": True})
        # Add a tool to github
        new_github = _fake_server_task(
            list(fake_mcp_servers["github"]._tools) + [
                _fake_tool("get_repo_topics", description="Get repo topics."),
            ]
        )
        monkeypatch.setattr(
            "tools.mcp_tool._servers",
            {"github": new_github, "notion": fake_mcp_servers["notion"]},
            raising=True,
        )
        _write_wrapper_readme({"expose_mcp_tools": True})
        md = _readme_path().read_text(encoding="utf-8")
        assert "## Recent changes" in md
        assert "**Added:**" in md
        assert "`mcp_github_get_repo_topics`" in md

    def test_removed_tool_shows_in_diff(self, tmp_hermes_home, fake_mcp_servers, monkeypatch):
        _write_wrapper_readme({"expose_mcp_tools": True})
        # Drop one tool from github
        kept = [t for t in fake_mcp_servers["github"]._tools if t.name != "delete_branch"]
        smaller_github = _fake_server_task(kept)
        monkeypatch.setattr(
            "tools.mcp_tool._servers",
            {"github": smaller_github, "notion": fake_mcp_servers["notion"]},
            raising=True,
        )
        _write_wrapper_readme({"expose_mcp_tools": True})
        md = _readme_path().read_text(encoding="utf-8")
        assert "**Removed:**" in md
        assert "`mcp_github_delete_branch`" in md

    def test_schema_change_detected(self, tmp_hermes_home, fake_mcp_servers, monkeypatch):
        _write_wrapper_readme({"expose_mcp_tools": True})
        # Same tool name, different inputSchema → should land in
        # schema_changed bucket, not added/removed
        new_list_issues = _fake_tool(
            "list_issues",
            description="List issues in a repo (NEW SCHEMA).",
            input_schema={
                "type": "object",
                "properties": {"owner": {"type": "string"}, "repo": {"type": "string"},
                              "state": {"type": "string"}},
                "required": ["owner", "repo", "state"],  # added required field
            },
        )
        rest = [t for t in fake_mcp_servers["github"]._tools if t.name != "list_issues"]
        mutated_github = _fake_server_task([new_list_issues] + rest)
        monkeypatch.setattr(
            "tools.mcp_tool._servers",
            {"github": mutated_github, "notion": fake_mcp_servers["notion"]},
            raising=True,
        )
        _write_wrapper_readme({"expose_mcp_tools": True})
        md = _readme_path().read_text(encoding="utf-8")
        assert "**Schema changed:**" in md
        assert "`mcp_github_list_issues`" in md
        assert "**Added:**" not in md
        assert "**Removed:**" not in md

    def test_diff_manifests_pure_helper(self):
        # Direct unit test for the diff function — first-run posture
        empty = _diff_manifests(None, {"github": {"foo": "abc"}})
        assert empty == {"added": [], "removed": [], "schema_changed": []}

    def test_diff_manifests_mixed_changes(self):
        prev = {
            "github": {"foo": "hash1", "bar": "hash2", "baz": "hash3"},
            "notion": {"page": "n1"},
        }
        curr = {
            "github": {"foo": "hash1", "bar": "DIFFERENT", "newtool": "hash4"},
            # notion server entirely gone
        }
        diff = _diff_manifests(prev, curr)
        assert diff["added"] == ["mcp_github_newtool"]
        assert diff["removed"] == ["mcp_github_baz", "mcp_notion_page"]
        assert diff["schema_changed"] == ["mcp_github_bar"]


# ---------------------------------------------------------------------------
# Legacy auto-skill cleanup
# ---------------------------------------------------------------------------


class TestLegacyCleanup:
    def test_removes_existing_legacy_dir(self, tmp_hermes_home):
        legacy = tmp_hermes_home / "skills" / _LEGACY_AUTO_SKILL_SUBDIR / "mcp-github"
        legacy.mkdir(parents=True)
        (legacy / "SKILL.md").write_text("stale", encoding="utf-8")
        _cleanup_legacy_auto_skills()
        assert not (tmp_hermes_home / "skills" / _LEGACY_AUTO_SKILL_SUBDIR).exists()

    def test_noop_when_legacy_missing(self, tmp_hermes_home):
        # Should silently do nothing — no crash, no log spam
        _cleanup_legacy_auto_skills()
        assert not (tmp_hermes_home / "skills" / _LEGACY_AUTO_SKILL_SUBDIR).exists()

    def test_does_not_touch_other_skills(self, tmp_hermes_home):
        # Pre-existing user skill in skill space must survive
        user_skill = tmp_hermes_home / "skills" / "my-real-skill" / "SKILL.md"
        user_skill.parent.mkdir(parents=True, exist_ok=True)
        user_skill.write_text("hand-authored", encoding="utf-8")
        # And a legacy auto-skill exists too
        legacy = tmp_hermes_home / "skills" / _LEGACY_AUTO_SKILL_SUBDIR / "mcp-github"
        legacy.mkdir(parents=True)
        (legacy / "SKILL.md").write_text("stale", encoding="utf-8")

        _cleanup_legacy_auto_skills()

        assert user_skill.exists()
        assert not (tmp_hermes_home / "skills" / _LEGACY_AUTO_SKILL_SUBDIR).exists()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


class TestApplyPostDiscoveryHooks:
    def test_disabled_short_circuits_but_still_cleans_legacy(
        self, tmp_hermes_home, fake_mcp_servers,
    ):
        # Stale legacy dir from a previous run
        legacy = tmp_hermes_home / "skills" / _LEGACY_AUTO_SKILL_SUBDIR / "mcp-github"
        legacy.mkdir(parents=True)
        (legacy / "SKILL.md").write_text("x", encoding="utf-8")

        apply_post_discovery_hooks({"expose_mcp_tools": False})

        # README not written (feature off)
        assert not _readme_path().exists()
        # But legacy cleanup STILL fires — users who disable the flag
        # shouldn't be stuck with stale auto-skills forever
        assert not (tmp_hermes_home / "skills" / _LEGACY_AUTO_SKILL_SUBDIR).exists()

    def test_enabled_writes_wrappers_and_readme(
        self, tmp_hermes_home, fake_mcp_servers,
    ):
        apply_post_discovery_hooks({"expose_mcp_tools": True})
        assert (tmp_hermes_home / STABLE_WRAPPER_SUBDIR / "hermes_mcp" / "github.py").exists()
        assert _readme_path().exists()

    def test_loader_path_when_cfg_omitted(
        self, tmp_hermes_home, fake_mcp_servers, monkeypatch,
    ):
        called = {}

        def _fake_loader():
            called["yes"] = True
            return {"expose_mcp_tools": True}

        monkeypatch.setattr("tools.code_execution_tool._load_config", _fake_loader)
        apply_post_discovery_hooks(None)
        assert called.get("yes") is True

    def test_swallows_inner_exceptions(
        self, tmp_hermes_home, fake_mcp_servers, monkeypatch,
    ):
        """A failing inner helper must not break MCP discovery overall."""
        def _boom(_cfg):
            raise RuntimeError("simulated explosion")

        monkeypatch.setattr(
            "tools.mcp_code_discovery._write_stable_mcp_wrappers", _boom,
        )
        # Should not raise — failure is logged and the other helper still runs
        apply_post_discovery_hooks({"expose_mcp_tools": True})
        # README still written despite the wrappers helper failing
        assert _readme_path().exists()

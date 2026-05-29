"""Tests for SEP-2640 MCP-served skill root discovery in agent/skill_utils.py.

Covers:
- ``get_mcp_skills_dirs`` enumerates ``~/.hermes/mcp-skills/<server>/`` subdirs
- ``get_disabled_mcp_servers`` reads ``skills.disabled_mcp_servers`` from config
- ``get_all_skills_dirs`` ordering: local → MCP → external_dirs
- Disabled MCP servers are excluded from discovery
- Empty / missing MCP root → empty list (backward compatible)
"""

import os
from pathlib import Path


def _hermes_home() -> Path:
    """Resolve the per-test HERMES_HOME the autouse conftest sets up."""
    return Path(os.environ["HERMES_HOME"])


def _make_mcp_skill(server: str, skill: str) -> Path:
    """Create a SKILL.md under ~/.hermes/mcp-skills/<server>/<skill>/."""
    skill_dir = _hermes_home() / "mcp-skills" / server / skill
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill}\ndescription: From MCP server {server}.\n---\n\n# {skill}\n",
        encoding="utf-8",
    )
    return skill_dir


class TestGetMcpSkillsDirs:
    def test_returns_empty_when_root_missing(self):
        from agent.skill_utils import get_mcp_skills_dirs
        assert get_mcp_skills_dirs() == []

    def test_returns_each_server_subdir(self):
        _make_mcp_skill("alpha", "refunds")
        _make_mcp_skill("beta", "summarize")

        from agent.skill_utils import get_mcp_skills_dirs
        dirs = get_mcp_skills_dirs()

        names = sorted(d.name for d in dirs)
        assert names == ["alpha", "beta"]

    def test_skips_non_directories(self):
        root = _hermes_home() / "mcp-skills"
        root.mkdir(parents=True, exist_ok=True)
        (root / "stray.txt").write_text("not a dir", encoding="utf-8")
        _make_mcp_skill("github", "issues")

        from agent.skill_utils import get_mcp_skills_dirs
        dirs = get_mcp_skills_dirs()
        assert [d.name for d in dirs] == ["github"]


class TestGetDisabledMcpServers:
    def test_empty_when_no_config(self):
        from agent.skill_utils import get_disabled_mcp_servers
        assert get_disabled_mcp_servers() == set()

    def test_reads_disabled_list(self):
        (_hermes_home() / "config.yaml").write_text(
            "skills:\n  disabled_mcp_servers:\n    - alpha\n    - beta\n",
            encoding="utf-8",
        )
        from agent.skill_utils import get_disabled_mcp_servers
        assert get_disabled_mcp_servers() == {"alpha", "beta"}

    def test_handles_scalar_string(self):
        (_hermes_home() / "config.yaml").write_text(
            "skills:\n  disabled_mcp_servers: only-one\n",
            encoding="utf-8",
        )
        from agent.skill_utils import get_disabled_mcp_servers
        assert get_disabled_mcp_servers() == {"only-one"}


class TestGetAllSkillsDirsOrdering:
    def test_local_first_then_mcp_then_external(self, tmp_path):
        # Local dir is always first.
        local = _hermes_home() / "skills"
        local.mkdir(parents=True, exist_ok=True)

        # MCP-served skills come from the cache dir.
        _make_mcp_skill("alpha", "refunds")

        # External dir configured via config.yaml.
        ext = tmp_path / "team-skills"
        ext.mkdir()
        (_hermes_home() / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {ext}\n",
            encoding="utf-8",
        )

        # Bust the external-dirs cache that survives within a test process.
        from agent.skill_utils import _external_dirs_cache_clear, get_all_skills_dirs
        _external_dirs_cache_clear()

        dirs = get_all_skills_dirs()

        assert dirs[0] == local
        assert dirs[1].name == "alpha"  # MCP root next
        assert dirs[-1] == ext.resolve()  # External last

    def test_disabled_server_skipped(self):
        _make_mcp_skill("alpha", "skill1")
        _make_mcp_skill("beta", "skill2")
        (_hermes_home() / "config.yaml").write_text(
            "skills:\n  disabled_mcp_servers:\n    - beta\n",
            encoding="utf-8",
        )

        from agent.skill_utils import _external_dirs_cache_clear, get_all_skills_dirs
        _external_dirs_cache_clear()

        dirs = get_all_skills_dirs()
        mcp_names = [d.name for d in dirs if (d.parent.name == "mcp-skills")]
        assert mcp_names == ["alpha"]

    def test_no_mcp_dirs_means_backward_compatible(self, tmp_path):
        # Backward compatibility: with no MCP cache at all, get_all_skills_dirs
        # returns the same shape as before (local + external_dirs).
        ext = tmp_path / "team-skills"
        ext.mkdir()
        (_hermes_home() / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {ext}\n",
            encoding="utf-8",
        )

        from agent.skill_utils import (
            _external_dirs_cache_clear, get_all_skills_dirs, get_skills_dir,
        )
        _external_dirs_cache_clear()

        dirs = get_all_skills_dirs()
        assert dirs == [get_skills_dir(), ext.resolve()]

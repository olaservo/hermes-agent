"""Tests for SEP-2640 provenance suffix in build_skills_system_prompt.

A skill materialized under ``~/.hermes/mcp-skills/<server>/<name>/`` should
appear in the rendered system prompt with a trailing ``(via MCP: <server>)``
marker so the model knows the skill came from a remote source.
"""

import json
import os
from pathlib import Path


def _hermes_home() -> Path:
    return Path(os.environ["HERMES_HOME"])


def _make_mcp_skill(server: str, skill: str, description: str) -> Path:
    skill_dir = _hermes_home() / "mcp-skills" / server / skill
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill}\ndescription: {description}\n---\n\n# {skill}\n\nDo {skill}.\n",
        encoding="utf-8",
    )
    (skill_dir / ".mcp-source.json").write_text(
        json.dumps({"server": server, "skill_uri_root": f"skill://acme/{skill}"}),
        encoding="utf-8",
    )
    return skill_dir


def _make_local_skill(skill: str, description: str) -> Path:
    skill_dir = _hermes_home() / "skills" / skill
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill}\ndescription: {description}\n---\n\n# {skill}\n",
        encoding="utf-8",
    )
    return skill_dir


def _bust_caches():
    """Drop both layers of the prompt-builder cache between tests."""
    from agent.skill_utils import _external_dirs_cache_clear
    _external_dirs_cache_clear()
    from agent import prompt_builder
    with prompt_builder._SKILLS_PROMPT_CACHE_LOCK:
        prompt_builder._SKILLS_PROMPT_CACHE.clear()


def test_mcp_skill_shown_with_via_marker():
    _make_mcp_skill("alpha", "refunds", "Process refund requests.")
    _bust_caches()

    from agent.prompt_builder import build_skills_system_prompt
    rendered = build_skills_system_prompt()

    assert "refunds" in rendered
    assert "(via MCP: alpha)" in rendered


def test_local_skill_has_no_marker():
    _make_local_skill("local-only", "A purely local skill.")
    _bust_caches()

    from agent.prompt_builder import build_skills_system_prompt
    rendered = build_skills_system_prompt()

    # Local skill present, no provenance suffix.
    assert "local-only" in rendered
    assert "(via MCP" not in rendered


def test_local_skill_wins_over_mcp_skill_with_same_name():
    # Local precedence: same skill name in both sources → local wins, no
    # provenance suffix should appear since the local copy is what renders.
    _make_local_skill("shared", "Local copy of the skill.")
    _make_mcp_skill("alpha", "shared", "Remote MCP copy of the skill.")
    _bust_caches()

    from agent.prompt_builder import build_skills_system_prompt
    rendered = build_skills_system_prompt()

    assert "shared" in rendered
    assert "(via MCP" not in rendered

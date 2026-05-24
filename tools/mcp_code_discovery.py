"""Post-MCP-discovery hooks for the experimental code-execution-with-MCP feature.

Runs immediately after ``tools.mcp_tool.register_mcp_servers`` has populated
``_servers[name]._tools``, gated on ``code_execution.expose_mcp_tools=true``:

  * **Slice A** — writes a stable, browseable copy of the generated
    ``hermes_mcp/`` wrapper package under ``~/.hermes/code-execution/mcp/``
    so the agent can ``read_file`` / ``search_files`` it between turns
    (the per-call execute_code tmpdir copy is content-identical but
    short-lived).
  * **Slice C** — writes one auto-generated Hermes skill per connected
    MCP server under ``~/.hermes/skills/mcp-auto/mcp-<server>/``
    indexing the server's tools with heuristic read-only/destructive/
    mutating hints so the agent finds the catalog through the normal
    skills_list / skill_view surface.

Both surfaces wipe-and-regenerate on each call; the source of truth is
always the live MCP server state. None of this changes behavior when
``expose_mcp_tools`` is false — the entry point short-circuits early.

The same security caveat that applies to Slice 1 applies here: the RPC
dispatch path in ``code_execution_tool`` does not currently re-apply
``check_all_command_guards()`` (issues #4146 / #30882), so MCP calls
originating from the sandbox inherit that bypass.  Do not enable in
production until those land.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


# Subdirectory under HERMES_HOME that holds the stable wrapper package.
# Parent of the ``hermes_mcp/`` package — that's the PYTHONPATH-able root.
STABLE_WRAPPER_SUBDIR = "code-execution/mcp"

# Subdirectory under HERMES_HOME/skills/ that holds the auto-generated
# per-server skills.  Marks them as wipe-and-regenerate so they can't be
# mistaken for hand-authored skills.
AUTO_SKILL_SUBDIR = "mcp-auto"

# Verb-prefix heuristics for tool categorization.  Names are lowercased
# and the leading ``<server>_`` strip is applied before matching.
_READ_PREFIXES = (
    "get_", "list_", "search_", "find_", "read_", "show_", "view_",
    "query_", "count_", "describe_", "inspect_", "fetch_", "lookup_",
)
_DESTRUCTIVE_PREFIXES = (
    "delete_", "remove_", "drop_", "erase_", "clear_", "purge_",
    "destroy_", "revoke_", "uninstall_", "unassign_",
)
_MUTATING_PREFIXES = (
    "create_", "update_", "write_", "set_", "post_", "put_", "patch_",
    "add_", "assign_", "rename_", "move_", "copy_", "merge_", "send_",
    "install_", "enable_", "disable_",
)


def _categorize_tool_by_name(tool_name: str) -> str:
    """Return ``"read"`` / ``"mutate"`` / ``"destroy"`` / ``"other"``.

    Pure name-heuristic — no MCP-spec field exists for this, and inspecting
    the inputSchema for ``write``-like behavior is unreliable.  When the name
    starts with one of the destructive prefixes that wins, so a tool named
    ``delete_and_recreate`` gets flagged ``destroy`` (the more dangerous
    category) rather than ``mutate``.
    """
    if not tool_name:
        return "other"
    lowered = tool_name.lower()
    if any(lowered.startswith(p) for p in _DESTRUCTIVE_PREFIXES):
        return "destroy"
    if any(lowered.startswith(p) for p in _READ_PREFIXES):
        return "read"
    if any(lowered.startswith(p) for p in _MUTATING_PREFIXES):
        return "mutate"
    return "other"


# ---------------------------------------------------------------------------
# Slice A: stable wrapper path
# ---------------------------------------------------------------------------


def stable_wrapper_root() -> Path:
    """Return the parent dir to add to PYTHONPATH (contains ``hermes_mcp/``)."""
    return get_hermes_home() / STABLE_WRAPPER_SUBDIR


def _write_stable_mcp_wrappers(cfg: dict) -> Optional[Path]:
    """Write hermes_mcp/ to a stable path so the agent can browse between turns.

    Reuses :func:`tools.code_execution_tool._build_mcp_sandbox_bundle` so the
    stable copy is byte-identical to what the sandbox sees per-call.  Returns
    the parent path (suitable for PYTHONPATH) on success, or ``None`` when
    the feature is disabled / no wrappers were produced.
    """
    try:
        from tools.code_execution_tool import _build_mcp_sandbox_bundle
    except Exception:
        logger.debug("code_execution_tool unavailable; skipping stable wrappers",
                     exc_info=True)
        return None

    files, _names = _build_mcp_sandbox_bundle(cfg)
    root = stable_wrapper_root()
    pkg = root / "hermes_mcp"

    # Wipe before regenerating so a server removed from config disappears
    # cleanly — leftover stubs that no longer route would silently fail at
    # dispatch time and confuse the agent.
    if pkg.exists():
        shutil.rmtree(pkg, ignore_errors=True)

    if not files:
        return None

    root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    logger.debug("Wrote %d stable MCP wrapper files under %s", len(files), root)
    return root


# ---------------------------------------------------------------------------
# Slice C: per-server auto-generated Hermes skill
# ---------------------------------------------------------------------------


def auto_skill_root() -> Path:
    """Return the root directory that holds the auto-generated skills."""
    return get_hermes_home() / "skills" / AUTO_SKILL_SUBDIR


def _required_args(mcp_tool) -> List[str]:
    """Return the list of required arg names from an MCP tool's inputSchema.

    Empty list when the schema is missing/non-conforming.  The order is
    preserved from the schema's ``required`` field (JSON Schema convention
    is to list them in the order the tool expects).
    """
    schema = getattr(mcp_tool, "inputSchema", None)
    if not isinstance(schema, dict):
        return []
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    return [str(r) for r in required if isinstance(r, str)]


def _bucket_tools(
    server_task,
    sanitize,
) -> Dict[str, List[Tuple[str, str, str, List[str]]]]:
    """Group a server's tools into (read/mutate/destroy/other) buckets.

    Each bucket entry is
    ``(safe_tool_name, original_tool_name, description, required_args)``.
    The ``required_args`` list (empty if none) is plumbed through so the
    example block can render placeholder kwargs that the model can fill
    in, rather than a misleading zero-arg call.

    Skips tools with falsy names.
    """
    buckets: Dict[str, List[Tuple[str, str, str, List[str]]]] = {
        "read": [], "mutate": [], "destroy": [], "other": [],
    }
    for mcp_tool in getattr(server_task, "_tools", None) or []:
        original = getattr(mcp_tool, "name", None)
        if not original:
            continue
        safe = sanitize(original)
        desc = (getattr(mcp_tool, "description", "") or "").strip()
        # First line only — keeps the skill skimmable; full schema is in
        # the wrapper module's docstring (see Slice 1).
        desc_line = desc.splitlines()[0] if desc else ""
        required = _required_args(mcp_tool)
        category = _categorize_tool_by_name(safe)
        buckets[category].append((safe, original, desc_line, required))
    for items in buckets.values():
        items.sort()
    return buckets


def _render_skill_markdown(
    server_name: str,
    safe_server: str,
    buckets: Dict[str, List[Tuple[str, str, str, List[str]]]],
) -> str:
    """Render the SKILL.md text for one MCP server."""
    total = sum(len(v) for v in buckets.values())
    n_read = len(buckets["read"])
    n_mutate = len(buckets["mutate"])
    n_destroy = len(buckets["destroy"])

    description = (
        f"Index of MCP server '{server_name}' tools — {total} tools, "
        f"{n_read} read-only / {n_mutate} mutating / {n_destroy} destructive. "
        f"Use when calling this server's tools, especially via "
        f"`from hermes_mcp.{safe_server} import <tool>` inside execute_code."
    )

    lines: List[str] = [
        "---",
        f"name: mcp-{safe_server}",
        f"description: {description!r}",
        "metadata:",
        "  hermes:",
        "    generated_by: mcp-auto",
        f"    server: {server_name}",
        f"    tool_count: {total}",
        "---",
        "",
        f"# MCP server: {server_name}",
        "",
        f"Wrappers live at `~/.hermes/code-execution/mcp/hermes_mcp/{safe_server}.py` —",
        f"importable as `from hermes_mcp.{safe_server} import <tool>` inside execute_code.",
        f"Each call routes through the existing MCP dispatcher; intermediate results stay",
        f"out of the LLM context.",
        "",
        "## Tools",
        "",
    ]

    section_titles = {
        "read": "### Read-only",
        "mutate": "### Mutating",
        "destroy": "### Destructive",
        "other": "### Other",
    }
    for key in ("read", "mutate", "destroy", "other"):
        items = buckets[key]
        if not items:
            continue
        lines.append(section_titles[key])
        for safe_tool, original_tool, desc_line, _required in items:
            label = safe_tool if safe_tool == original_tool else f"{safe_tool} ({original_tool})"
            if desc_line:
                lines.append(f"- `{label}` — {desc_line}")
            else:
                lines.append(f"- `{label}`")
        lines.append("")

    # Best-effort example: pick the first read-only tool if there is one,
    # else the first tool from any non-empty bucket.  Render required args
    # as ``<arg>="..."`` placeholders so the example is a usable template
    # rather than a misleading zero-arg call that always fails.
    example_tool = None
    example_required: List[str] = []
    for key in ("read", "other", "mutate", "destroy"):
        if buckets[key]:
            example_tool, _orig, _desc, example_required = buckets[key][0]
            break
    if example_tool:
        if example_required:
            call_args = ", ".join(f'{arg}="..."' for arg in example_required)
            note_line = ""
        else:
            call_args = ""
            note_line = "# tool takes no required args"
        call = f"result = {example_tool}({call_args})"
        example_lines = [
            "## Example",
            "",
            "```python",
            f"from hermes_mcp.{safe_server} import {example_tool}",
        ]
        if note_line:
            example_lines.append(note_line)
        example_lines.extend([
            call,
            "print(result)",
            "```",
            "",
        ])
        lines.extend(example_lines)

    return "\n".join(lines)


def _write_mcp_auto_skills(cfg: dict) -> List[Path]:
    """Write one SKILL.md per connected MCP server under HERMES_HOME/skills/mcp-auto/.

    Returns the list of SKILL.md paths written.  Wipes the ``mcp-auto/``
    container before regenerating so a server removed from config disappears
    cleanly.  Honors ``code_execution.mcp_servers_allowlist``: only listed
    servers (when the list is set) get skills generated.
    """
    if not cfg.get("expose_mcp_tools"):
        return []

    try:
        from tools.mcp_tool import (
            _lock as _mcp_lock,
            _servers as _mcp_servers,
            sanitize_mcp_name_component,
        )
    except Exception:
        logger.debug("MCP module unavailable; skipping auto-skill generation",
                     exc_info=True)
        return []

    allowlist_raw = cfg.get("mcp_servers_allowlist")
    allowlist = None
    if isinstance(allowlist_raw, list):
        allowlist = {str(name) for name in allowlist_raw}

    root = auto_skill_root()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)

    with _mcp_lock:
        items = list(_mcp_servers.items())

    written: List[Path] = []
    for server_name, server_task in items:
        if allowlist is not None and server_name not in allowlist:
            continue
        tools = getattr(server_task, "_tools", None) or []
        if not tools:
            continue
        safe_server = sanitize_mcp_name_component(server_name)
        buckets = _bucket_tools(server_task, sanitize_mcp_name_component)
        if not any(buckets.values()):
            continue
        md = _render_skill_markdown(server_name, safe_server, buckets)
        skill_dir = root / f"mcp-{safe_server}"
        skill_dir.mkdir(parents=True, exist_ok=True)
        target = skill_dir / "SKILL.md"
        target.write_text(md, encoding="utf-8")
        written.append(target)

    if written:
        logger.debug("Wrote %d MCP auto-skills under %s", len(written), root)
    return written


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_post_discovery_hooks(cfg: Optional[dict] = None) -> None:
    """Run all post-MCP-discovery side-effects for the code-execution feature.

    Called from ``tools.mcp_tool.register_mcp_servers`` after MCP server
    discovery completes.  Cheap when ``code_execution.expose_mcp_tools`` is
    falsy — short-circuits without touching the filesystem.

    Args:
        cfg: The ``code_execution`` config dict.  If ``None``, loads it
             via :func:`tools.code_execution_tool._load_config` (the same
             helper Slice 1 uses).  Tests pass ``cfg`` directly so they
             don't have to monkeypatch the loader.
    """
    if cfg is None:
        try:
            from tools.code_execution_tool import _load_config
            cfg = _load_config()
        except Exception:
            logger.debug("Could not load code_execution config; skipping hooks",
                         exc_info=True)
            return
    if not cfg.get("expose_mcp_tools"):
        return
    try:
        _write_stable_mcp_wrappers(cfg)
    except Exception:
        # The feature is experimental and behind a flag — never let a
        # generation failure bubble up and break MCP discovery itself.
        logger.warning("Failed to write stable MCP wrappers", exc_info=True)
    try:
        _write_mcp_auto_skills(cfg)
    except Exception:
        logger.warning("Failed to write MCP auto-skills", exc_info=True)

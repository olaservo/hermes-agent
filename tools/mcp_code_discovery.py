"""Post-MCP-discovery hooks for the experimental code-execution-with-MCP feature.

Runs at the end of ``tools.mcp_tool.register_mcp_servers`` (gated on
``code_execution.expose_mcp_tools=true``):

  * **Wrapper package** — writes a stable, browseable copy of the
    generated ``hermes_mcp/`` package under
    ``~/.hermes/code-execution/mcp/`` so scripts can
    ``from hermes_mcp.<server> import <tool>`` and the agent can
    ``read_file`` / ``search_files`` the source between turns.
  * **Catalog README** — writes ``hermes_mcp/README.md`` with a
    human-readable per-server tool index, categorized by verb-prefix
    heuristic (read-only / mutating / destructive / other), and a
    "Recent changes (since last launch)" diff section computed against
    a sidecar manifest.

An earlier iteration also auto-generated one Hermes skill per server
under ``~/.hermes/skills/mcp-auto/``.  That has been dropped — the
ownership story was awkward (one skill with auto-content the agent
might want to patch, and the wipe-and-regenerate behavior would
clobber any patches).  The skill namespace is now exclusively for
agent / bg-review authored content.  Legacy ``mcp-auto/`` directories
are cleaned up on first run after upgrade.

Both surfaces wipe-and-regenerate on each call; the source of truth is
always the live MCP server state.  All side-effects no-op when
``expose_mcp_tools`` is false.

Security caveat (same as Slice 1): the RPC dispatch path in
``code_execution_tool`` does not currently re-apply
``check_all_command_guards()`` (issues #4146 / #30882), so MCP calls
originating from the sandbox inherit that bypass.  Do not enable in
production until those land.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


# Subdirectory under HERMES_HOME that holds the stable wrapper package.
# Parent of the ``hermes_mcp/`` package — that's the PYTHONPATH-able root.
STABLE_WRAPPER_SUBDIR = "code-execution/mcp"

# Legacy: dropped from Slice 4.  Path kept here only so the cleanup hook
# below can find and remove it on first run after upgrade.
_LEGACY_AUTO_SKILL_SUBDIR = "mcp-auto"

# Verb-prefix heuristics for tool categorization.  Names are lowercased
# before matching.  Destructive prefixes win ties: a tool named
# ``delete_and_recreate`` lands in the dangerous bucket, not the safer
# mutating one.
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
    """Return ``"read"`` / ``"mutate"`` / ``"destroy"`` / ``"other"``."""
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


def _required_args(mcp_tool) -> List[str]:
    """Return the list of required arg names from an MCP tool's inputSchema."""
    schema = getattr(mcp_tool, "inputSchema", None)
    if not isinstance(schema, dict):
        return []
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    return [str(r) for r in required if isinstance(r, str)]


def _schema_hash(mcp_tool) -> str:
    """8-char stable hash of inputSchema, for diff detection.

    Empty string when no inputSchema or non-serializable.  Stable across
    runs because we ``sort_keys=True`` and use a fixed separator pair.
    """
    schema = getattr(mcp_tool, "inputSchema", None)
    if schema is None:
        return ""
    try:
        canon = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return ""
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def _bucket_tools(
    server_task,
    sanitize,
) -> Dict[str, List[Tuple[str, str, str, List[str]]]]:
    """Group a server's tools into (read/mutate/destroy/other) buckets.

    Entry shape: ``(safe_tool_name, original_tool_name, description, required_args)``.
    Skips tools with falsy names.  Sorted alphabetically within each bucket.
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
        desc_line = desc.splitlines()[0] if desc else ""
        required = _required_args(mcp_tool)
        category = _categorize_tool_by_name(safe)
        buckets[category].append((safe, original, desc_line, required))
    for items in buckets.values():
        items.sort()
    return buckets


# ---------------------------------------------------------------------------
# Stable wrapper path (unchanged from earlier slices)
# ---------------------------------------------------------------------------


def stable_wrapper_root() -> Path:
    """Return the parent dir to add to PYTHONPATH (contains ``hermes_mcp/``)."""
    return get_hermes_home() / STABLE_WRAPPER_SUBDIR


def _write_stable_mcp_wrappers(cfg: dict) -> Optional[Path]:
    """Write hermes_mcp/ to a stable path so the agent can browse between turns."""
    try:
        from tools.code_execution_tool import _build_mcp_sandbox_bundle
    except Exception:
        logger.debug("code_execution_tool unavailable; skipping stable wrappers",
                     exc_info=True)
        return None

    files, _names = _build_mcp_sandbox_bundle(cfg)
    root = stable_wrapper_root()
    pkg = root / "hermes_mcp"

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
# Catalog README + diff-since-last-launch
# ---------------------------------------------------------------------------


def _readme_path() -> Path:
    """Where the catalog README lives — inside the importable package."""
    return stable_wrapper_root() / "hermes_mcp" / "README.md"


def _manifest_path() -> Path:
    """Sidecar JSON manifest used to compute diffs across runs."""
    return stable_wrapper_root() / ".last-catalog.json"


def _build_current_manifest(server_items) -> Dict[str, Dict[str, str]]:
    """Snapshot the current server→tool→schema_hash mapping for diffing.

    server_items is a list of (server_name, server_task) tuples as
    returned by walking ``_servers`` under the lock.
    """
    manifest: Dict[str, Dict[str, str]] = {}
    for server_name, server_task in server_items:
        tools = getattr(server_task, "_tools", None) or []
        per_server: Dict[str, str] = {}
        for mcp_tool in tools:
            name = getattr(mcp_tool, "name", None)
            if not name:
                continue
            per_server[name] = _schema_hash(mcp_tool)
        if per_server:
            manifest[server_name] = per_server
    return manifest


def _load_previous_manifest() -> Optional[Dict[str, Dict[str, str]]]:
    """Read the sidecar manifest from the previous launch, if any."""
    path = _manifest_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not parse %s; treating as no prior manifest", path)
        return None
    servers = data.get("servers")
    if not isinstance(servers, dict):
        return None
    # Defensive normalisation: drop entries we can't make sense of
    clean: Dict[str, Dict[str, str]] = {}
    for srv, tools in servers.items():
        if not isinstance(tools, dict):
            continue
        clean[str(srv)] = {
            str(name): str(info.get("schema_hash", ""))
            for name, info in tools.items()
            if isinstance(info, dict)
        }
    return clean


def _save_manifest(manifest: Dict[str, Dict[str, str]]) -> None:
    """Persist the manifest sidecar so the next launch can diff against it."""
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "servers": {
            srv: {name: {"schema_hash": h} for name, h in tools.items()}
            for srv, tools in manifest.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _diff_manifests(
    previous: Optional[Dict[str, Dict[str, str]]],
    current: Dict[str, Dict[str, str]],
) -> Dict[str, List[str]]:
    """Return added / removed / schema_changed tool-name lists.

    Empty lists when nothing changed.  When ``previous`` is None (first
    run after upgrade) all returned lists are empty too — we don't claim
    "everything is new" because there's no baseline to compare against.
    """
    diff = {"added": [], "removed": [], "schema_changed": []}
    if previous is None:
        return diff

    def _flat(m: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        flat: Dict[str, str] = {}
        for srv, tools in m.items():
            for tool, h in tools.items():
                flat[f"mcp_{srv}_{tool}"] = h
        return flat

    prev_flat = _flat(previous)
    curr_flat = _flat(current)

    prev_names = set(prev_flat)
    curr_names = set(curr_flat)
    diff["added"] = sorted(curr_names - prev_names)
    diff["removed"] = sorted(prev_names - curr_names)
    diff["schema_changed"] = sorted(
        name for name in (prev_names & curr_names)
        if prev_flat[name] and curr_flat[name] and prev_flat[name] != curr_flat[name]
    )
    return diff


def _render_readme_markdown(
    server_items,
    sanitize,
    diff: Dict[str, List[str]],
) -> str:
    """Render the catalog README covering all connected servers."""
    lines: List[str] = [
        "# Hermes MCP wrapper catalog",
        "",
        "These wrappers expose connected MCP-server tools as importable Python:",
        "",
        "```python",
        "from hermes_mcp.<server> import <tool>",
        "result = <tool>(arg1=\"...\", arg2=\"...\")",
        "print(result)",
        "```",
        "",
        ("Each call routes through the same RPC path as the built-in `hermes_tools` "
         "stubs, so intermediate results stay out of the LLM context — only "
         "your script's `print()` output comes back."),
        "",
        ("Generated from the live MCP server state at every Hermes startup. "
         "Per-server sections below. Tools are categorised by verb-prefix "
         "heuristic — `destructive` wins on ties (`delete_and_recreate` is "
         "destructive, not mutating)."),
        "",
    ]

    if any(diff.values()):
        lines.append("## Recent changes (since last launch)")
        lines.append("")
        if diff["added"]:
            lines.append("- **Added:** " + ", ".join(f"`{n}`" for n in diff["added"]))
        if diff["removed"]:
            lines.append("- **Removed:** " + ", ".join(f"`{n}`" for n in diff["removed"]))
        if diff["schema_changed"]:
            lines.append(
                "- **Schema changed:** "
                + ", ".join(f"`{n}`" for n in diff["schema_changed"])
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    section_titles = {
        "read": "### Read-only",
        "mutate": "### Mutating",
        "destroy": "### Destructive",
        "other": "### Other",
    }

    rendered_any_server = False
    for server_name, server_task in sorted(server_items):
        tools = getattr(server_task, "_tools", None) or []
        if not tools:
            continue
        safe_server = sanitize(server_name)
        buckets = _bucket_tools(server_task, sanitize)
        if not any(buckets.values()):
            continue
        total = sum(len(v) for v in buckets.values())
        n_read = len(buckets["read"])
        n_mutate = len(buckets["mutate"])
        n_destroy = len(buckets["destroy"])

        header_suffix = (
            f"{total} tool{'s' if total != 1 else ''} — "
            f"{n_read} read-only, {n_mutate} mutating, {n_destroy} destructive"
        )
        lines.append(f"## {server_name} ({header_suffix})")
        lines.append("")
        lines.append(f"Import with: `from hermes_mcp.{safe_server} import <tool>`")
        lines.append("")

        for key in ("read", "mutate", "destroy", "other"):
            items = buckets[key]
            if not items:
                continue
            lines.append(section_titles[key])
            for safe_tool, original_tool, desc_line, required in items:
                # Render call signature with required args as placeholders so
                # the model sees the call shape inline rather than having to
                # cross-reference the wrapper file.
                if required:
                    sig = f"{safe_tool}({', '.join(required)})"
                else:
                    sig = f"{safe_tool}()"
                label = sig if safe_tool == original_tool else f"{sig} ← {original_tool}"
                if desc_line:
                    lines.append(f"- `{label}` — {desc_line}")
                else:
                    lines.append(f"- `{label}`")
            lines.append("")
        rendered_any_server = True

    if not rendered_any_server:
        lines.append("_(No connected MCP servers exposed any tools.)_")
        lines.append("")

    return "\n".join(lines)


def _write_wrapper_readme(cfg: dict) -> Optional[Path]:
    """Write hermes_mcp/README.md with the categorised catalog + diff section.

    Returns the README path on success, ``None`` when disabled or no servers
    produced any tools.  Updates the sidecar manifest atomically so the next
    launch can compute its diff section.
    """
    if not cfg.get("expose_mcp_tools"):
        return None

    try:
        from tools.mcp_tool import (
            _lock as _mcp_lock,
            _servers as _mcp_servers,
            sanitize_mcp_name_component,
        )
    except Exception:
        logger.debug("MCP module unavailable; skipping wrapper README", exc_info=True)
        return None

    allowlist_raw = cfg.get("mcp_servers_allowlist")
    allowlist = None
    if isinstance(allowlist_raw, list):
        allowlist = {str(name) for name in allowlist_raw}

    with _mcp_lock:
        items = list(_mcp_servers.items())

    if allowlist is not None:
        items = [(name, task) for name, task in items if name in allowlist]

    if not items:
        # Don't write a stub README when there's nothing to catalog — and
        # actively clean up an old one so it doesn't lie about state.
        for stale in (_readme_path(), _manifest_path()):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.debug("Could not clean stale %s", stale, exc_info=True)
        return None

    previous = _load_previous_manifest()
    current = _build_current_manifest(items)
    diff = _diff_manifests(previous, current)
    markdown = _render_readme_markdown(items, sanitize_mcp_name_component, diff)

    readme = _readme_path()
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(markdown, encoding="utf-8")
    _save_manifest(current)

    logger.debug(
        "Wrote MCP wrapper catalog README (%d servers; +%d / -%d / Δ%d)",
        len(items), len(diff["added"]), len(diff["removed"]),
        len(diff["schema_changed"]),
    )
    return readme


# ---------------------------------------------------------------------------
# One-time legacy cleanup
# ---------------------------------------------------------------------------


def _cleanup_legacy_auto_skills() -> None:
    """Remove the dropped ``~/.hermes/skills/mcp-auto/`` directory if present.

    A previous slice (since reverted) wrote one auto-generated Hermes skill
    per connected MCP server here.  Leaving the directory around would
    pollute ``skills_list`` with stale entries that no live code maintains.
    Idempotent: no-op when the directory doesn't exist.
    """
    legacy = get_hermes_home() / "skills" / _LEGACY_AUTO_SKILL_SUBDIR
    if not legacy.exists():
        return
    try:
        shutil.rmtree(legacy, ignore_errors=True)
        logger.info(
            "Removed legacy auto-skill directory %s (replaced by "
            "code-execution/mcp/hermes_mcp/README.md)", legacy,
        )
    except Exception:
        logger.debug("Failed to clean legacy auto-skills at %s", legacy, exc_info=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_post_discovery_hooks(cfg: Optional[dict] = None) -> None:
    """Run all post-MCP-discovery side-effects for the code-execution feature.

    Called from ``tools.mcp_tool.register_mcp_servers`` after MCP server
    discovery completes.  Cheap when ``code_execution.expose_mcp_tools`` is
    falsy — short-circuits without touching the filesystem (except for the
    one-time legacy cleanup, which runs unconditionally so old auto-skills
    don't linger after the user opts out).
    """
    # Legacy cleanup runs regardless of the flag — leaving stale skills
    # around would silently surface in skills_list even after the user
    # disables expose_mcp_tools.
    try:
        _cleanup_legacy_auto_skills()
    except Exception:
        logger.debug("Legacy auto-skill cleanup failed", exc_info=True)

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
        logger.warning("Failed to write stable MCP wrappers", exc_info=True)
    try:
        _write_wrapper_readme(cfg)
    except Exception:
        logger.warning("Failed to write MCP wrapper README", exc_info=True)

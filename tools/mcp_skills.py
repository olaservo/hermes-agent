#!/usr/bin/env python3
"""
Skills-over-MCP support (SEP-2640).

Consumes Agent Skills served by MCP servers that advertise the
``io.modelcontextprotocol/skills`` capability extension. Skills are
discovered via the ``skill://index.json`` resource, fetched via standard
``resources/read``, materialized to ``~/.hermes/mcp-skills/<server>/<skill>/``,
and surfaced through the normal skill discovery/loader pipeline.

Layout::

    ~/.hermes/mcp-skills/
    └── <safe-server-name>/
        └── <safe-skill-name>/
            ├── SKILL.md             # frontmatter + body fetched at connect
            ├── .mcp-source.json     # provenance sidecar (server, root URI, ts)
            └── scripts/, refs/, …   # supporting files fetched lazily on first read

Trust: scanned by ``tools.skills_guard.scan_skill`` at the ``trusted`` tier
(same as openai/anthropic/huggingface official repos). Dangerous verdicts are
rejected and the skill is not materialized.

Discovery channels in v1: ``skill://index.json`` only. ``resources/list``
scanning and ``instructions``-field references are deferred to v2.
Archives and RFC-6570 parameterized templates are out of scope.

Feature gate: ``mcp.skills_extension: experimental`` in ``config.yaml``
(default ``off``). The SEP is on a feature branch and shapes may shift.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


SKILLS_EXTENSION_CAPABILITY = "io.modelcontextprotocol/skills"
INDEX_URI = "skill://index.json"
SIDECAR_FILENAME = ".mcp-source.json"


# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------


def get_mcp_skills_root() -> Path:
    """Return ``~/.hermes/mcp-skills/`` — the parent of all per-server caches."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "mcp-skills"


def get_server_skills_dir(server_name: str) -> Path:
    """Return the materialization root for a given MCP server.

    The directory is created lazily — callers that need it to exist should
    call ``mkdir(parents=True, exist_ok=True)`` themselves.
    """
    from tools.mcp_tool import sanitize_mcp_name_component
    safe = sanitize_mcp_name_component(server_name)
    return get_mcp_skills_root() / safe


def iter_mcp_server_dirs() -> List[Path]:
    """Return every existing ``~/.hermes/mcp-skills/<server>/`` directory.

    Used by ``agent.skill_utils.get_all_skills_dirs`` to add MCP-served
    skill roots to the discovery walk.
    """
    root = get_mcp_skills_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def server_name_from_skill_dir(skill_dir: Path) -> Optional[str]:
    """If *skill_dir* lives under an MCP server cache, return the server name.

    Walks the sidecar from the skill directory upward; returns ``None`` for
    skills outside the MCP cache tree (bundled, hub-installed, external).
    """
    try:
        sidecar = skill_dir / SIDECAR_FILENAME
        if sidecar.is_file():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            value = data.get("server")
            if isinstance(value, str) and value.strip():
                return value
    except Exception as exc:
        logger.debug("Failed to read MCP sidecar at %s: %s", skill_dir, exc)
    # Fallback: infer from path layout.
    try:
        mcp_root = get_mcp_skills_root().resolve()
        resolved = skill_dir.resolve()
        if mcp_root in resolved.parents:
            # ~/.hermes/mcp-skills/<server>/<skill>/...
            rel = resolved.relative_to(mcp_root)
            if rel.parts:
                return rel.parts[0]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------


def skills_extension_enabled() -> bool:
    """Return True if ``mcp.skills_extension`` is set to ``experimental``.

    Reads ``config.yaml`` directly to stay independent of the CLI config
    layer (which imports a much heavier stack).
    """
    try:
        from hermes_constants import get_config_path
        from agent.skill_utils import yaml_load
    except Exception:
        return False
    path = get_config_path()
    if not path.exists():
        return False
    try:
        parsed = yaml_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Could not read config.yaml for skills_extension gate: %s", exc)
        return False
    if not isinstance(parsed, dict):
        return False
    mcp_cfg = parsed.get("mcp")
    if not isinstance(mcp_cfg, dict):
        return False
    flag = str(mcp_cfg.get("skills_extension", "off")).strip().lower()
    return flag in {"experimental", "on", "true", "yes", "1"}


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


def server_advertises_skills_extension(initialize_result: Any) -> bool:
    """Check the ``InitializeResult`` for the SEP-2640 capability key.

    SEP-2640 advertises support via
    ``capabilities.extensions["io.modelcontextprotocol/skills"] = {}``.
    The MCP Python SDK may or may not model ``capabilities.extensions`` yet,
    so we look at both the typed attribute and the raw dict
    (``model_dump()``) — whichever the SDK exposes.
    """
    if initialize_result is None:
        return False
    capabilities = getattr(initialize_result, "capabilities", None)
    if capabilities is None:
        return False

    # Path 1: typed attribute (future SDK).
    extensions = getattr(capabilities, "extensions", None)
    if isinstance(extensions, dict) and SKILLS_EXTENSION_CAPABILITY in extensions:
        return True

    # Path 2: dump-based fallback. SDK models usually expose model_dump().
    for src in (capabilities, initialize_result):
        for dump_method in ("model_dump", "dict"):
            fn = getattr(src, dump_method, None)
            if not callable(fn):
                continue
            try:
                dumped = fn()
            except Exception:
                continue
            if not isinstance(dumped, dict):
                continue
            caps = dumped if src is capabilities else dumped.get("capabilities") or {}
            if not isinstance(caps, dict):
                continue
            exts = caps.get("extensions")
            if isinstance(exts, dict) and SKILLS_EXTENSION_CAPABILITY in exts:
                return True
    return False


# ---------------------------------------------------------------------------
# Index + resource fetch helpers
# ---------------------------------------------------------------------------


async def _read_resource_text(server, uri: str) -> Optional[str]:
    """Call ``resources/read`` on the server and return the concatenated text.

    Returns ``None`` if the resource has no textual content (binary-only
    responses are not supported in v1 — they have no use for SKILL.md or
    text-based supporting files).
    """
    async with server._rpc_lock:
        result = await server.session.read_resource(uri)
    parts: List[str] = []
    contents = getattr(result, "contents", None) or []
    for block in contents:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    if not parts:
        return None
    return "\n".join(parts)


_NAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_skill_dir_name(value: str) -> str:
    """Return a filesystem-safe directory name for a skill.

    Preserves dots and dashes (common in skill names like ``billing.refunds``)
    while replacing path-traversal / separator characters with ``_``.
    """
    cleaned = _NAME_SANITIZE_RE.sub("_", str(value or "").strip())
    cleaned = cleaned.strip("._-") or "skill"
    return cleaned[:128]  # bound long names


def _parse_index(text: str) -> List[Dict[str, Any]]:
    """Parse ``skill://index.json`` and return its concrete-skill entries.

    The SEP defines three entry types: ``concrete``, ``archive``, and
    parameterized templates. v1 keeps only concrete entries — archives are
    out of scope, templates need RFC-6570 expansion which is deferred.

    Tolerant of multiple shapes:
      - ``{"skills": [...]}`` — flat concrete list
      - ``{"concrete": [...], "archives": [...]}`` — typed groups
    """
    try:
        data = json.loads(text)
    except Exception as exc:
        logger.warning("MCP skills: failed to parse %s: %s", INDEX_URI, exc)
        return []
    if not isinstance(data, dict):
        return []

    raw_entries: List[Any] = []
    if isinstance(data.get("concrete"), list):
        raw_entries.extend(data["concrete"])
    if isinstance(data.get("skills"), list):
        raw_entries.extend(data["skills"])

    entries: List[Dict[str, Any]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        # Skip archive entries even if mixed into the concrete list.
        kind = str(item.get("kind") or item.get("type") or "concrete").lower()
        if kind in {"archive", "template", "parameterized"}:
            continue
        # An entry must point at SOMETHING readable. Accept any of these
        # equivalent shapes:
        #   {"uri": "skill://acme/billing/refunds/SKILL.md"}
        #   {"root": "skill://acme/billing/refunds"}
        #   {"path": "acme/billing/refunds"}
        uri = item.get("uri")
        root = item.get("root")
        path = item.get("path")
        if not (uri or root or path):
            continue
        entries.append(item)
    return entries


def _resolve_skill_md_uri(entry: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Return ``(skill_md_uri, skill_root_uri)`` for an index entry.

    Returns ``None`` if the entry can't be normalized.
    """
    uri = entry.get("uri")
    if isinstance(uri, str) and uri.endswith("/SKILL.md"):
        return uri, uri[: -len("/SKILL.md")]
    root = entry.get("root")
    if isinstance(root, str) and root:
        root = root.rstrip("/")
        return f"{root}/SKILL.md", root
    path = entry.get("path")
    if isinstance(path, str) and path:
        path = path.strip("/")
        root = f"skill://{path}"
        return f"{root}/SKILL.md", root
    if isinstance(uri, str) and uri.startswith("skill://"):
        # Bare ``skill://acme/refunds`` — assume root and append SKILL.md.
        root = uri.rstrip("/")
        return f"{root}/SKILL.md", root
    return None


def _frontmatter_name(content: str) -> Optional[str]:
    """Extract the ``name:`` field from a SKILL.md's YAML frontmatter."""
    try:
        from agent.skill_utils import parse_frontmatter
        frontmatter, _ = parse_frontmatter(content)
    except Exception as exc:
        logger.debug("Failed to parse SKILL.md frontmatter: %s", exc)
        return None
    name = frontmatter.get("name") if isinstance(frontmatter, dict) else None
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


# ---------------------------------------------------------------------------
# Materialization + scan
# ---------------------------------------------------------------------------


def _write_sidecar(skill_dir: Path, server_name: str, skill_root_uri: str) -> None:
    """Write the provenance sidecar that ties this skill back to its MCP source.

    Used by ``server_name_from_skill_dir`` (provenance suffix) and by
    ``ensure_mcp_skill_file_present`` (lazy supporting-file fetch).
    """
    payload = {
        "server": server_name,
        "skill_uri_root": skill_root_uri,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (skill_dir / SIDECAR_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("MCP skills: failed to write sidecar at %s: %s", skill_dir, exc)


def _scan_and_enforce(skill_dir: Path, server_name: str, skill_name: str) -> bool:
    """Scan a freshly materialized skill at the ``trusted`` tier.

    Returns True if the skill passes policy, False if it was rejected and
    the directory has been removed.
    """
    try:
        from tools.skills_guard import scan_skill, should_allow_install
    except Exception as exc:
        logger.warning("MCP skills: skills_guard unavailable, skipping scan: %s", exc)
        return True
    try:
        result = scan_skill(skill_dir, source=f"mcp/{server_name}")
        allowed, reason = should_allow_install(result)
    except Exception:
        logger.exception("MCP skills: scan failed for '%s' from '%s'", skill_name, server_name)
        return True  # Fail open so a scanner bug doesn't drop trusted skills.
    if allowed is False:
        logger.warning(
            "MCP skills: rejecting skill '%s' from server '%s' — %s",
            skill_name, server_name, reason,
        )
        shutil.rmtree(skill_dir, ignore_errors=True)
        return False
    if allowed is None:
        logger.warning(
            "MCP skills: skill '%s' from server '%s' needs confirmation (%s); "
            "not auto-installing",
            skill_name, server_name, reason,
        )
        shutil.rmtree(skill_dir, ignore_errors=True)
        return False
    if result.verdict == "caution":
        logger.info(
            "MCP skills: '%s' from '%s' passed with caution-level findings (%d)",
            skill_name, server_name, len(result.findings),
        )
    return True


async def _materialize_one_skill(
    server,
    server_name: str,
    entry: Dict[str, Any],
    server_dir: Path,
) -> Optional[str]:
    """Fetch + write one skill from the index. Returns the skill name on success."""
    resolved = _resolve_skill_md_uri(entry)
    if not resolved:
        logger.debug("MCP skills: skipping unresolvable index entry: %r", entry)
        return None
    skill_md_uri, skill_root_uri = resolved

    try:
        content = await _read_resource_text(server, skill_md_uri)
    except Exception:
        logger.exception(
            "MCP skills: read_resource failed for %s on server '%s'",
            skill_md_uri, server_name,
        )
        return None
    if not content:
        logger.warning(
            "MCP skills: empty/binary SKILL.md from %s on server '%s'",
            skill_md_uri, server_name,
        )
        return None

    skill_name = _frontmatter_name(content)
    if not skill_name:
        # Last-resort name: trailing segment of the root URI.
        skill_name = skill_root_uri.rsplit("/", 1)[-1] or "skill"
    safe_name = _sanitize_skill_dir_name(skill_name)
    skill_dir = server_dir / safe_name

    # Clean any previous version of this skill so stale files don't linger.
    shutil.rmtree(skill_dir, ignore_errors=True)
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    _write_sidecar(skill_dir, server_name, skill_root_uri)

    if not _scan_and_enforce(skill_dir, server_name, skill_name):
        return None
    return skill_name


async def discover_and_materialize(server, server_name: str) -> List[str]:
    """Discover SEP-2640 skills on *server* and materialize them on disk.

    Returns the list of skill names that were successfully installed.
    Safe to call when the capability isn't advertised or the feature is
    disabled — it returns an empty list and logs at debug.
    """
    if not skills_extension_enabled():
        logger.debug(
            "MCP skills: feature flag mcp.skills_extension is off — "
            "skipping discovery for '%s'",
            server_name,
        )
        return []

    init_result = getattr(server, "initialize_result", None)
    if not server_advertises_skills_extension(init_result):
        return []

    try:
        index_text = await _read_resource_text(server, INDEX_URI)
    except Exception as exc:
        # Most servers without an index will fail here with a not-found.
        logger.info(
            "MCP skills: server '%s' advertises skills but %s is unreadable (%s); "
            "no skills installed",
            server_name, INDEX_URI, exc,
        )
        return []
    if not index_text:
        logger.info(
            "MCP skills: server '%s' returned empty %s — no skills installed",
            server_name, INDEX_URI,
        )
        return []

    entries = _parse_index(index_text)
    if not entries:
        logger.info(
            "MCP skills: server '%s' index has no concrete entries", server_name,
        )
        return []

    server_dir = get_server_skills_dir(server_name)
    server_dir.mkdir(parents=True, exist_ok=True)

    installed: List[str] = []
    for entry in entries:
        name = await _materialize_one_skill(server, server_name, entry, server_dir)
        if name:
            installed.append(name)

    logger.info(
        "MCP skills: server '%s' provided %d skill(s): %s",
        server_name, len(installed), ", ".join(installed) or "(none)",
    )
    return installed


async def refresh_for_server(server, server_name: str) -> List[str]:
    """Re-run discovery for a single server (used on ``resources/list_changed``).

    Nuke-and-repave: the server's cache subdir is wiped before discovery so
    a skill that disappeared upstream also disappears locally.
    """
    server_dir = get_server_skills_dir(server_name)
    if server_dir.exists():
        shutil.rmtree(server_dir, ignore_errors=True)
    return await discover_and_materialize(server, server_name)


def cleanup_for_server(server_name: str) -> bool:
    """Remove a server's cache subdir on disconnect/removal. Idempotent."""
    server_dir = get_server_skills_dir(server_name)
    if not server_dir.exists():
        return False
    try:
        shutil.rmtree(server_dir, ignore_errors=True)
    except Exception as exc:
        logger.warning("MCP skills: failed to clean %s: %s", server_dir, exc)
        return False
    return True


# ---------------------------------------------------------------------------
# Lazy supporting-file fetch (used by skill_view)
# ---------------------------------------------------------------------------


def _read_sidecar(skill_dir: Path) -> Optional[Dict[str, Any]]:
    sidecar = skill_dir / SIDECAR_FILENAME
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _is_safe_relative(skill_dir: Path, candidate: Path) -> bool:
    """Reject paths that escape the skill root (path-traversal guard)."""
    try:
        resolved = candidate.resolve()
        root = skill_dir.resolve()
    except Exception:
        return False
    try:
        resolved.relative_to(root)
    except ValueError:
        return False
    return True


def _scan_file_bytes(path: Path) -> str:
    """Scan a single freshly written file and return a verdict string."""
    try:
        from tools.skills_guard import scan_file, _determine_verdict
    except Exception:
        return "safe"
    try:
        findings = scan_file(path, str(path.name))
    except Exception:
        return "safe"
    return _determine_verdict(findings)


def ensure_mcp_skill_file_present(skill_dir: Path, file_path: str) -> Optional[str]:
    """Materialize a supporting file from the MCP server if it isn't local yet.

    Called from ``skill_view`` before reading a file under an MCP-backed skill
    so the bytes are present on disk by the time the read happens.

    Returns:
        ``None``    when no MCP action is needed (not an MCP skill, or file
                    already on disk) — caller proceeds with normal disk read.
        ``"ok"``    when the file was fetched and written.
        ``"blocked: <reason>"``  when policy refused to write the file.
        ``"missing: <reason>"``  when the fetch failed.

    The function is intentionally lenient: if anything goes wrong, it returns
    ``None`` (or a structured error) and the existing skill_view error path
    handles user messaging.
    """
    try:
        if not file_path or not isinstance(file_path, str):
            return None
        rel = file_path.lstrip("/").replace("\\", "/")
        target = skill_dir / rel
        if target.exists():
            return None
        if not _is_safe_relative(skill_dir, target):
            return "blocked: path traversal rejected"

        sidecar = _read_sidecar(skill_dir)
        if not sidecar:
            return None  # Not an MCP-backed skill — normal disk read will 404.

        server_name = sidecar.get("server")
        root_uri = sidecar.get("skill_uri_root")
        if not (server_name and root_uri):
            return None

        # Late import to avoid circular dependency at module load.
        from tools.mcp_tool import _servers, _run_on_mcp_loop

        server = _servers.get(server_name)
        if not server or not getattr(server, "session", None):
            return "missing: MCP server not connected"

        uri = f"{root_uri.rstrip('/')}/{rel}"

        async def _do_read():
            return await _read_resource_text(server, uri)

        try:
            content = _run_on_mcp_loop(_do_read, timeout=float(getattr(server, "tool_timeout", 60.0)))
        except Exception as exc:
            return f"missing: {type(exc).__name__}: {exc}"
        if content is None:
            return "missing: empty or non-text response"

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        verdict = _scan_file_bytes(target)
        if verdict == "dangerous":
            try:
                target.unlink()
            except Exception:
                pass
            return "blocked: scanner flagged dangerous content"
        return "ok"
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("ensure_mcp_skill_file_present failed: %s", exc)
        return None

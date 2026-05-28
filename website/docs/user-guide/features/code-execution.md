---
sidebar_position: 8
title: "Code Execution"
description: "Programmatic Python execution with RPC tool access — collapse multi-step workflows into a single turn"
---

# Code Execution (Programmatic Tool Calling)

The `execute_code` tool lets the agent write Python scripts that call Hermes tools programmatically, collapsing multi-step workflows into a single LLM turn. The script runs in a child process on the agent host, communicating with Hermes over a Unix domain socket RPC.

## How It Works

1. The agent writes a Python script using `from hermes_tools import ...`
2. Hermes generates a `hermes_tools.py` stub module with RPC functions
3. Hermes opens a Unix domain socket and starts an RPC listener thread
4. The script runs in a child process — tool calls travel over the socket back to Hermes
5. Only the script's `print()` output is returned to the LLM; intermediate tool results never enter the context window

```python
# The agent can write scripts like:
from hermes_tools import web_search, web_extract

results = web_search("Python 3.13 features", limit=5)
for r in results["data"]["web"]:
    content = web_extract([r["url"]])
    # ... filter and process ...
print(summary)
```

**Available tools inside scripts:** `web_search`, `web_extract`, `read_file`, `write_file`, `search_files`, `patch`, `terminal` (foreground only).

## When the Agent Uses This

The agent uses `execute_code` when there are:

- **3+ tool calls** with processing logic between them
- Bulk data filtering or conditional branching
- Loops over results

The key benefit: intermediate tool results never enter the context window — only the final `print()` output comes back, dramatically reducing token usage.

## Practical Examples

### Data Processing Pipeline

```python
from hermes_tools import search_files, read_file
import json

# Find all config files and extract database settings
matches = search_files("database", path=".", file_glob="*.yaml", limit=20)
configs = []
for match in matches.get("matches", []):
    content = read_file(match["path"])
    configs.append({"file": match["path"], "preview": content["content"][:200]})

print(json.dumps(configs, indent=2))
```

### Multi-Step Web Research

```python
from hermes_tools import web_search, web_extract
import json

# Search, extract, and summarize in one turn
results = web_search("Rust async runtime comparison 2025", limit=5)
summaries = []
for r in results["data"]["web"]:
    page = web_extract([r["url"]])
    for p in page.get("results", []):
        if p.get("content"):
            summaries.append({
                "title": r["title"],
                "url": r["url"],
                "excerpt": p["content"][:500]
            })

print(json.dumps(summaries, indent=2))
```

### Bulk File Refactoring

```python
from hermes_tools import search_files, read_file, patch

# Find all Python files using deprecated API and fix them
matches = search_files("old_api_call", path="src/", file_glob="*.py")
fixed = 0
for match in matches.get("matches", []):
    result = patch(
        path=match["path"],
        old_string="old_api_call(",
        new_string="new_api_call(",
        replace_all=True
    )
    if "error" not in str(result):
        fixed += 1

print(f"Fixed {fixed} files out of {len(matches.get('matches', []))} matches")
```

### Build and Test Pipeline

```python
from hermes_tools import terminal, read_file
import json

# Run tests, parse results, and report
result = terminal("cd /project && python -m pytest --tb=short -q 2>&1", timeout=120)
output = result.get("output", "")

# Parse test output
passed = output.count(" passed")
failed = output.count(" failed")
errors = output.count(" error")

report = {
    "passed": passed,
    "failed": failed,
    "errors": errors,
    "exit_code": result.get("exit_code", -1),
    "summary": output[-500:] if len(output) > 500 else output
}

print(json.dumps(report, indent=2))
```

## Execution Mode

`execute_code` has two execution modes controlled by `code_execution.mode` in `~/.hermes/config.yaml`:

| Mode | Working directory | Python interpreter |
|------|-------------------|--------------------|
| **`project`** (default) | The session's working directory (same as `terminal()`) | Active `VIRTUAL_ENV` / `CONDA_PREFIX` python, falling back to Hermes's own python |
| `strict` | A temp staging directory isolated from the user's project | `sys.executable` (Hermes's own python) |

**When to leave it on `project`:** you want `import pandas`, `from my_project import foo`, or relative paths like `open(".env")` to work the same way they do in `terminal()`. This is almost always what you want.

**When to flip to `strict`:** you need maximum reproducibility — you want the same interpreter every session regardless of which venv the user activated, and you want scripts quarantined from the project tree (no risk of accidentally reading project files through a relative path).

```yaml
# ~/.hermes/config.yaml
code_execution:
  mode: project   # or "strict"
```

Fallback behavior in `project` mode: if `VIRTUAL_ENV` / `CONDA_PREFIX` is unset, broken, or points at a Python older than 3.8, the resolver falls back cleanly to `sys.executable` — it never leaves the agent without a working interpreter.

Security-critical invariants are identical across both modes:

- environment scrubbing (API keys, tokens, credentials stripped)
- tool whitelist (scripts cannot call `execute_code` recursively or `delegate_task`; MCP tools require the experimental `expose_mcp_tools` opt-in described below)
- resource limits (timeout, stdout cap, tool-call cap)

Switching mode changes where scripts run and which interpreter runs them, not what credentials they can see or which tools they can call.

## Resource Limits

| Resource | Limit | Notes |
|----------|-------|-------|
| **Timeout** | 5 minutes (300s) | Script is killed with SIGTERM, then SIGKILL after 5s grace |
| **Stdout** | 50 KB | Output truncated with `[output truncated at 50KB]` notice |
| **Stderr** | 10 KB | Included in output on non-zero exit for debugging |
| **Tool calls** | 50 per execution | Error returned when limit reached |

All limits are configurable via `config.yaml`:

```yaml
# In ~/.hermes/config.yaml
code_execution:
  mode: project      # project (default) | strict
  timeout: 300       # Max seconds per script (default: 300)
  max_tool_calls: 50 # Max tool calls per execution (default: 50)
```

## How Tool Calls Work Inside Scripts

When your script calls a function like `web_search("query")`:

1. The call is serialized to JSON and sent over a Unix domain socket to the parent process
2. The parent dispatches through the standard `handle_function_call` handler
3. The result is sent back over the socket
4. The function returns the parsed result

This means tool calls inside scripts behave identically to normal tool calls — same rate limits, same error handling, same capabilities. The only restriction is that `terminal()` is foreground-only (no `background` or `pty` parameters).

## Experimental: MCP tools in the sandbox

:::warning Experimental — security caveat
The RPC dispatch path in `execute_code` does not currently re-apply `check_all_command_guards()` (tracked in upstream issues [#4146](https://github.com/NousResearch/hermes-agent/issues/4146) and [#30882](https://github.com/NousResearch/hermes-agent/issues/30882)). MCP tools called from inside the sandbox inherit that bypass. **Do not enable this in production until those issues land.** It exists today for prototyping the [Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) pattern against a real workload.
:::

When `code_execution.expose_mcp_tools=true`, Hermes generates a `hermes_mcp/` package alongside `hermes_tools.py` at the start of each `execute_code` call, with one submodule per connected MCP server. Scripts can then import MCP tools as ordinary Python functions:

```python
from hermes_mcp.github import list_issues, search_code

issues = list_issues(owner="anthropics", repo="claude-code", state="open")
filtered = [i for i in issues["result"] if "bug" in i.get("title", "").lower()]
print(len(filtered), "open bug-tagged issues")
```

The same dispatch path is used as for the built-in seven tools — the generated stub calls back to the parent via RPC, and the parent routes to the existing MCP client. Intermediate MCP responses never enter the LLM context; only the script's `print()` output does.

### Config

```yaml
# ~/.hermes/config.yaml
code_execution:
  expose_mcp_tools: true                 # master switch, default false
  mcp_servers_allowlist: [github, notion] # optional; omit for all connected servers
```

| Key | Default | Behavior |
|-----|---------|----------|
| `code_execution.expose_mcp_tools` | `false` | Master switch. When `false`, behavior is unchanged. |
| `code_execution.mcp_servers_allowlist` | `null` | Optional list of server names. When set, only those servers' tools become importable. |

### How stubs are generated

For each connected (and allowlisted) MCP server, a submodule like `hermes_mcp/github.py` is written into the per-call temp dir. Each MCP tool becomes a `**kwargs`-only function whose docstring carries the original description plus the JSON `inputSchema`, so an agent that wants to inspect parameters can call `help(list_issues)`. The stub dispatches via the same registry name the tool is already registered under (`mcp_github_list_issues`), so no new dispatcher code is involved.

### Listing what's available

```python
import hermes_mcp
print(hermes_mcp.__all__)
# ['github', 'notion']
import hermes_mcp.github
print([n for n in dir(hermes_mcp.github) if not n.startswith("_")])
```

### Discovery surface (between turns)

In addition to the per-call sandbox copy, Hermes writes the same wrapper package to a stable path at MCP-discovery time, plus a categorized `README.md` inside the package so the agent can browse the catalog without grepping individual `.py` files:

```
~/.hermes/code-execution/mcp/hermes_mcp/
  __init__.py
  README.md       # categorized catalog (all servers) + "Recent changes" diff
  github.py       # one module per connected server, byte-identical to the sandbox copy
  notion.py
```

`README.md` is regenerated every time `register_mcp_servers()` runs. It contains a per-server section with tools grouped by a verb-prefix heuristic (read-only / mutating / destructive / other), call signatures with required-arg names rendered inline, and a "Recent changes (since last launch)" section that diffs against a sidecar `.last-catalog.json` manifest — so when an MCP server adds / removes / changes the schema of a tool, the agent sees it on the next launch and can proactively update recipes that depend on it.

The heuristic classifies tool names by leading verb (`list_`/`get_`/`search_`/... → read-only; `delete_`/`remove_`/`purge_`/... → destructive; `create_`/`update_`/`patch_`/... → mutating; otherwise → other). Destructive wins ties, so `delete_and_recreate_thing` is flagged destructive rather than mutating.

The package directory is wiped and regenerated each launch (with the README + manifest preserved across runs for the diff). Removing a server from `config.yaml` makes its wrapper and section in the README disappear cleanly on next launch.

**The skill namespace is left alone.** An earlier iteration of this feature auto-generated one Hermes skill per connected server under `~/.hermes/skills/mcp-auto/`. That created an awkward two-owner conflict — the auto-generated content would clobber any patches the [background-review fork](#persistence-recipe-save-is-handled-out-of-band) made to refine the skill with session-learned gotchas. Slice 4 drops auto-skill generation entirely; on first launch after upgrade, `~/.hermes/skills/mcp-auto/` is removed if present.

### Prompt nudge

When all three of the following are true, the system prompt picks up a short `MCP_AS_CODE_GUIDANCE` block:

1. `code_execution.expose_mcp_tools=true`
2. at least one MCP server is connected
3. `execute_code` is in the session's enabled tools

The block teaches the model to prefer `import hermes_mcp.<server>` *specifically* when batching, filtering, or looping over MCP results — one-shot calls stay on the direct MCP tool path. Without all three gates met, the prompt is byte-identical to main and the model never sees the block.

### Persistence (recipe-save) is handled out of band

The reusable-recipe loop — "agent writes a useful `hermes_mcp` script, that script becomes a skill the next session finds" — is handled by Hermes's existing background-review fork (`agent/background_review.py`), **not** by a foreground prompt addition. After every Nth turn (`skills.creation_nudge_interval`, default 10), the main agent spawns a separate AIAgent with a narrowed memory+skill toolset and an aggressive review prompt that proposes skill creates and patches based on the conversation snapshot. The foreground prompt above stops at "use the wrappers when it helps" — what's worth saving is the background review's job to decide.

For non-interactive callers (`cli.py -q`, `mcp_serve`, batch runners, cron) the background review thread is joined on process exit (`HERMES_BG_REVIEW_TIMEOUT_SEC`, default 30) so the review actually completes before the process tears down. See `tools/code_execution_tool.py` and `agent/conversation_loop.py` for details.

## Error Handling

When a script fails, the agent receives structured error information:

- **Non-zero exit code**: stderr is included in the output so the agent sees the full traceback
- **Timeout**: Script is killed and the agent sees `"Script timed out after 300s and was killed."`
- **Interruption**: If the user sends a new message during execution, the script is terminated and the agent sees `[execution interrupted — user sent a new message]`
- **Tool call limit**: When the 50-call limit is hit, subsequent tool calls return an error message

The response always includes `status` (success/error/timeout/interrupted), `output`, `tool_calls_made`, and `duration_seconds`.

## Security

:::danger Security Model
The child process runs with a **minimal environment**. API keys, tokens, and credentials are stripped by default. The script accesses tools exclusively via the RPC channel — it cannot read secrets from environment variables unless explicitly allowed.
:::

Environment variables containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PASSWD`, or `AUTH` in their names are excluded. Only safe system variables (`PATH`, `HOME`, `LANG`, `SHELL`, `PYTHONPATH`, `VIRTUAL_ENV`, etc.) are passed through.

### Skill Environment Variable Passthrough

When a skill declares `required_environment_variables` in its frontmatter, those variables are **automatically passed through** to both `execute_code` and `terminal` child processes after the skill is loaded. This lets skills use their declared API keys without weakening the security posture for arbitrary code.

For non-skill use cases, you can explicitly allowlist variables in `config.yaml`:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

See the [Security guide](/user-guide/security#environment-variable-passthrough) for full details.

Hermes always writes the script and the auto-generated `hermes_tools.py` RPC stub into a temp staging directory that is cleaned up after execution. In `strict` mode the script also *runs* there; in `project` mode it runs in the session's working directory (the staging directory stays on `PYTHONPATH` so imports still resolve). The child process runs in its own process group so it can be cleanly killed on timeout or interruption.

## execute_code vs terminal

| Use Case | execute_code | terminal |
|----------|-------------|----------|
| Multi-step workflows with tool calls between | ✅ | ❌ |
| Simple shell command | ❌ | ✅ |
| Filtering/processing large tool outputs | ✅ | ❌ |
| Running a build or test suite | ❌ | ✅ |
| Looping over search results | ✅ | ❌ |
| Interactive/background processes | ❌ | ✅ |
| Needs API keys in environment | ⚠️ Only via [passthrough](/user-guide/security#environment-variable-passthrough) | ✅ (most pass through) |

**Rule of thumb:** Use `execute_code` when you need to call Hermes tools programmatically with logic between calls. Use `terminal` for running shell commands, builds, and processes.

## Platform Support

Code execution requires Unix domain sockets and is available on **Linux and macOS only**. It is automatically disabled on Windows — the agent falls back to regular sequential tool calls.

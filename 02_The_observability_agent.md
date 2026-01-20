```python
import os
import shutil
import subprocess
from typing import Any

from dotenv import load_dotenv
from IPython.display import Markdown, display
from utils.agent_visualizer import (
    display_agent_response,
    print_activity,
    reset_activity_context,
    visualize_conversation,
)

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
```

# 02 - The Observability Agent

In the previous notebooks we have built a basic research agent and a Chief of Staff multi-agent framework. While the agents we have built are already powerful, they were still limited in what they could do: the web search agent is limited to searching the internet and our Chief of Staff agent was limited to interacting with its own filesystem.

This is a serious constraint: real-world agents often need to interact with other systems like databases, APIs, file systems, and other specialized services. [MCP (Model Context Protocol)](https://modelcontextprotocol.io/docs/getting-started/intro) is an open-source standard for AI-tool integrations that allows for an easy connection between our agents and these external systems. In this notebook, we will explore how to connect MCP servers to our agent.

**Need more details on MCP?** For comprehensive setup instructions, configuration best practices, and troubleshooting tips, see the [Claude Code MCP documentation](https://docs.claude.com/en/docs/claude-code/mcp).

## Introduction to the MCP Server
### 1. The Git MCP server

Let's first give our agent the ability to understand and work with Git repositories. By adding the [Git MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/git) to our agent, it gains access to 13 Git-specific tools that let it examine commit history, check file changes, create branches, and even make commits. This transforms our agent from a passive observer into an active participant in your development workflow. In this example, we'll configure the agent to explore a repository's history using only Git tools. This is pretty simple, but knowing this, it is not difficult to imagine agents that can automatically create pull requests, analyze code evolution patterns, or help manage complex Git workflows across multiple repositories.


```python
# Get the git repository root (mcp_server_git requires a valid git repo path)
# os.getcwd() may return a subdirectory, so we find the actual repo root
git_executable = shutil.which("git")
if git_executable is None:
    raise RuntimeError("Git executable not found in PATH")

git_repo_root = subprocess.run(  # noqa: S603
    [git_executable, "rev-parse", "--show-toplevel"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()

# Define our git MCP server (installed via uv sync from pyproject.toml)
git_mcp: dict[str, Any] = {
    "git": {
        "command": "uv",
        "args": ["run", "python", "-m", "mcp_server_git", "--repository", git_repo_root],
    }
}
```


```python
messages = []
async with ClaudeSDKClient(
    options=ClaudeAgentOptions(
        model="claude-opus-4-5",
        mcp_servers=git_mcp,
        allowed_tools=["mcp__git"],
        # disallowed_tools ensures the agent ONLY uses MCP tools, not Bash with git commands
        disallowed_tools=["Bash", "Task", "WebSearch", "WebFetch"],
        permission_mode="acceptEdits",
    )
) as agent:
    await agent.query(
        "Explore this repo's git history and provide a brief summary of recent activity."
    )
    async for msg in agent.receive_response():
        print_activity(msg)
        messages.append(msg)
```

[🤖 Using: mcp__git__git_log(), ..., 🤖 Thinking...]



```python
display(Markdown(f"\nResult:\n{messages[-1].result}"))
```



Result:
## Git Repository Summary

### Current Branch
You're on the **`upstream-contribution`** branch (up to date with origin), with `main` also available locally.

---

### Recent Commit Activity (Last ~5 Days)

| Date | Author | Summary |
|------|--------|---------|
| **Nov 27, 2025** | costiash | 3 commits enhancing the **Claude Agent SDK** - improved chief of staff agent, notebooks, observability agent, research agent, documentation, and utilities |
| **Nov 26, 2025** | Pedram Navid | Added GitHub issue templates, `/review-issue` command, `/add-registry` slash command, and new cookbook entries |
| **Nov 25, 2025** | Elie Schoppik | Renamed PTC notebook to `programmatic_tool_calling_ptc.ipynb` for clarity |
| **Nov 24, 2025** | henrykeetay | Added **tool search cookbook** |
| **Nov 24, 2025** | Alex Notov | Multiple merges consolidating cookbooks for Opus 4.5, dependency updates |
| **Nov 23, 2025** | Cal Rueb | Simplified crop tool notebook with Claude Agent SDK section |
| **Nov 23, 2025** | Pedram Navid | PR comment fixes and lint cleanup |

---

### Key Themes in Recent Development
1. **Claude Agent SDK enhancements** - Major work on agent implementations (research, chief of staff, observability agents)
2. **New cookbooks** - Tool search, crop tool, programmatic tool calling
3. **CI/CD improvements** - PR review workflows, issue templates, slash commands
4. **Documentation** - Added troubleshooting guides, codebase overviews

---

### Working Directory Status
There are **uncommitted changes** in your working directory:
- **22 modified files** (mostly in `claude_agent_sdk/`)
- **4 deleted files** (documentation files in `docs/`)
- **6 untracked files** (new reports, plans, VS Code config)

These changes appear to be further work on the Claude Agent SDK agents, notebooks, and utilities that haven't been staged or committed yet.


### 2. The GitHub MCP server

Now let's level up from local Git operations to full GitHub platform integration. By switching to the [official GitHub MCP server](https://github.com/github/github-mcp-server/tree/main), our agent gains access to over 100 tools that interact with GitHub's entire ecosystem – from managing issues and pull requests to monitoring CI/CD workflows and analyzing code security alerts. This server can work with both public and private repositories, giving your agent the ability to automate complex GitHub workflows that would typically require multiple manual steps.

#### Step 1: Set up your GitHub Token

You need a GitHub Personal Access Token. Get one [here](https://github.com/settings/personal-access-tokens/new) and put in the .env file as ```GITHUB_TOKEN="<token>"```
> Note: When getting your token, select "Fine-grained" token with the default options (i.e., public repos, no account permissions), that'll be the easiest way to get this demo working.

Also, for this example you will have to have [Docker](https://www.docker.com/products/docker-desktop/) running on your machine. Docker is required because the GitHub MCP server runs in a containerized environment for security and isolation.

**Docker Quick Setup:**
- Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)
- Ensure Docker is running (you'll see the Docker icon in your system tray)
- Verify with `docker --version` in your terminal
- **Troubleshooting:** If Docker won't start, check that virtualization is enabled in your BIOS. For detailed setup instructions, see the [Docker documentation](https://docs.docker.com/get-docker/)

#### Step 2: Define the mcp server and start the agent loop!


```python
# define our github mcp server
load_dotenv(override=True)
github_mcp: dict[str, Any] = {
    "github": {
        "command": "docker",
        "args": [
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
        ],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_TOKEN")},
    }
}
```


```python
# run our agent
messages = []
async with ClaudeSDKClient(
    options=ClaudeAgentOptions(
        model="claude-opus-4-5",
        mcp_servers=github_mcp,
        allowed_tools=["mcp__github"],
        # disallowed_tools ensures the agent ONLY uses MCP tools, not Bash with gh CLI
        disallowed_tools=["Bash", "Task", "WebSearch", "WebFetch"],
        permission_mode="acceptEdits",
    )
) as agent:
    await agent.query(
        "Search for the anthropics/claude-agent-sdk-python repository and give me a few key facts about it."
    )
    async for msg in agent.receive_response():
        print_activity(msg)
        messages.append(msg)
```

[🤖 Using: mcp__github__search_repositories(), ..., 🤖 Thinking...]



```python
display(Markdown(f"\nResult:\n{messages[-1].result}"))
```



Result:
Here are the key facts about the **anthropics/claude-agent-sdk-python** repository:

| Fact | Details |
|------|---------|
| **Full Name** | anthropics/claude-agent-sdk-python |
| **URL** | https://github.com/anthropics/claude-agent-sdk-python |
| **Language** | Python |
| **Stars** | ⭐ 3,357 |
| **Forks** | 🍴 435 |
| **Open Issues** | 149 |
| **Created** | June 11, 2025 |
| **Last Updated** | December 4, 2025 |
| **Default Branch** | main |
| **Visibility** | Public |
| **Archived** | No |

This is the official Python SDK for building Claude agents, maintained by Anthropic. It's quite popular with over 3,300 stars and has an active community with 435 forks. The repository is actively maintained (recently updated) and has a notable number of open issues (149), which suggests active development and community engagement.


## Real use case: An observability agent

Now, with such simple setup we can already have an agent acting as self-healing software system!


```python
load_dotenv(override=True)

prompt = """Analyze the CI health for facebook/react repository.

Examine the most recent runs of the 'CI' workflow and provide:
1. Current status and what triggered the run (push, PR, schedule, etc.)
2. If failing: identify the specific failing jobs/tests and assess severity
3. If passing: note any concerning patterns (long duration, flaky history)
4. Recommended actions with priority (critical/high/medium/low)

Provide a concise operational summary suitable for an on-call engineer.
Do not create issues or PRs - this is a read-only analysis."""

github_mcp: dict[str, Any] = {
    "github": {
        "command": "docker",
        "args": [
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
        ],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_TOKEN")},
    }
}

messages = []
async with ClaudeSDKClient(
    options=ClaudeAgentOptions(
        model="claude-opus-4-5",
        mcp_servers=github_mcp,
        allowed_tools=["mcp__github"],
        # IMPORTANT: disallowed_tools is required to actually RESTRICT tool usage.
        # Without this, allowed_tools only controls permission prompting, not availability.
        # The agent would still have access to Bash (and could use `gh` CLI instead of MCP).
        disallowed_tools=["Bash", "Task", "WebSearch", "WebFetch"],
        permission_mode="acceptEdits",
    )
) as agent:
    await agent.query(prompt)
    async for msg in agent.receive_response():
        print_activity(msg)
        messages.append(msg)
```

[🤖 Using: mcp__github__get_file_contents(), ..., 🤖 Thinking...]



```python
display(Markdown(f"\nResult:\n{messages[-1].result}"))
```



Result:
Based on my comprehensive analysis of the facebook/react repository CI infrastructure, here is the operational summary:

---

# CI Health Analysis: facebook/react

## Executive Summary
**Overall Status: 🟢 HEALTHY**

The React repository's CI appears to be in good health. Recent commits to `main` have been successfully merged, and active PRs show passing CodeSandbox builds.

---

## 1. CI Infrastructure Overview

### Primary Workflows
| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `runtime_build_and_test.yml` | Push to main, PRs | Main CI - builds, tests, Flow checks |
| `shared_lint.yml` | Push to main, PRs | Prettier, ESLint, license checks |
| `compiler_typescript.yml` | PRs touching compiler | Compiler-specific tests |
| `devtools_regression_tests.yml` | PRs | DevTools testing |

### Test Matrix Scale
- **90 test shards** (18 configurations × 5 shards each)
- **50 build jobs** (25 workers × 2 release channels)
- **50 test-build shards** (5 configurations × 10 shards)
- Flow checks across multiple inline configs

---

## 2. Recent Main Branch Status

| Commit | Date | Description | Status |
|--------|------|-------------|--------|
| `bf1afad` | Dec 4, 2025 | [react-dom/server] Fix hanging on Deno | ✅ Merged |
| `0526c79` | Dec 3, 2025 | Update changelog with latest releases | ✅ Merged |
| `7dc903c` | Dec 3, 2025 | Patch FlightReplyServer (security fix) | ✅ Merged |
| `36df5e8` | Dec 2, 2025 | Allow building single release channel | ✅ Merged |

**Last 10 commits:** All successfully merged to main, indicating CI is passing.

---

## 3. Active PR CI Status

| PR | Title | CodeSandbox Status |
|----|-------|-------------------|
| #35267 | Fix spelling (behaviour → behavior) | 🟡 Pending (building) |
| #35238 | DevTools navigating commits hotkey | ✅ Success |
| #35287 | Compiler: Fix variable name issue | ✅ Success |
| #35278 | Add DevTools console suppress option | ✅ Success |
| #35226 | Fizz: Push stalled use() to ownerStack | ✅ Success |

---

## 4. Risk Assessment

### ✅ Positive Indicators
- **Main branch stable**: All recent commits merged successfully
- **No open CI failure issues**: Search returned zero CI-related open bugs
- **Active development**: Security patches and features landing regularly
- **PR builds passing**: Most open PRs show successful builds

### ⚠️ Areas to Monitor
- **Large test matrix**: 190+ parallel jobs mean potential for infrastructure flakiness
- **Playwright-based e2e tests**: Browser-based tests can be flaky (Flight fixtures, DevTools e2e)
- **Cache dependencies**: Multiple cache strategies (v6 keys) - cache misses could slow builds

### 📊 CI Complexity Metrics
- ~37KB workflow file for main CI (`runtime_build_and_test.yml`)
- Heavy parallelization with matrix strategies
- Multiple artifact upload/download operations

---

## 5. Recommended Actions

| Priority | Action | Rationale |
|----------|--------|-----------|
| **LOW** | Monitor PR #35267 | Currently building - verify completion |
| **LOW** | No immediate action required | Main branch healthy, PRs passing |
| **INFO** | Security patch merged Dec 3 | PR #35277 fixed critical security vuln in FlightReplyServer - verify downstream impact |

---

## 6. On-Call Notes

**TL;DR for On-Call Engineer:**
- 🟢 **CI is GREEN** - No action required
- Main branch is healthy with successful merges in last 24h
- All checked PRs showing green/passing status
- No open issues flagged for CI failures or flakiness
- Recent security patch (#35277) was successfully merged - monitor for any regressions

**If issues arise:**
1. Check GitHub Actions tab directly: `https://github.com/facebook/react/actions`
2. Key workflows to monitor: "(Runtime) Build and Test", "(Shared) Lint"
3. Caches use `v6` key prefix - if widespread failures, consider cache invalidation

---

*Analysis performed: December 4, 2025*
*Data sources: GitHub API (commits, PRs, status checks, workflow files)*



```python
reset_activity_context()
visualize_conversation(messages)
```




<style>
.conversation-timeline {
    font-family: ui-sans-serif, system-ui;
    max-width: 900px;
    margin: 1em 0;
}
.timeline-header {
    background: linear-gradient(135deg, #3b82f6, #9333ea);
    color: white;
    padding: 12px 16px;
    border-radius: 12px 12px 0 0;
    font-weight: 700;
    font-size: 14px;
}
.timeline-body {
    border: 1px solid #e5e7eb;
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 12px;
    background: #fafafa;
}
.msg-block {
    margin: 8px 0;
    padding: 10px 12px;
    border-radius: 8px;
    background: white;
    border-left: 3px solid #e5e7eb;
}
.msg-block.system { border-left-color: #6b7280; }
.msg-block.assistant { border-left-color: #3b82f6; }
.msg-block.tool { border-left-color: #10b981; background: #f0fdf4; }
.msg-block.subagent { border-left-color: #9333ea; background: #faf5ff; }
.msg-block.result { border-left-color: #f
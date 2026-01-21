# Building an Observability Agent with MCP Servers

## Introduction

In previous guides, we built a basic research agent and a Chief of Staff multi-agent framework. While powerful, these agents were limited to specific capabilities: web search and local filesystem interactions.

Real-world agents often need to interact with diverse systems like databases, APIs, and specialized services. The **Model Context Protocol (MCP)** is an open-source standard for AI-tool integrations that enables easy connections between agents and external systems. In this guide, you'll learn how to connect MCP servers to your agent, transforming it into a powerful observability tool.

> **Need more details on MCP?** For comprehensive setup instructions, configuration best practices, and troubleshooting tips, see the [Claude Code MCP documentation](https://docs.claude.com/en/docs/claude-code/mcp).

## Prerequisites

First, ensure you have the necessary libraries installed:

```python
import os
import shutil
import subprocess
from typing import Any

from dotenv import load_dotenv
from IPython.display import Markdown, display

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
```

## Part 1: Git MCP Server Integration

### Step 1: Configure the Git MCP Server

Let's give your agent the ability to understand and work with Git repositories. By adding the [Git MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/git), your agent gains access to 13 Git-specific tools for examining commit history, checking file changes, creating branches, and making commits.

```python
# Get the git repository root (mcp_server_git requires a valid git repo path)
# os.getcwd() may return a subdirectory, so we find the actual repo root
git_executable = shutil.which("git")
if git_executable is None:
    raise RuntimeError("Git executable not found in PATH")

git_repo_root = subprocess.run(
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

### Step 2: Initialize the Agent with Git Capabilities

Now, create an agent instance configured to use only the Git MCP tools:

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
        messages.append(msg)
```

### Step 3: Display the Results

```python
display(Markdown(f"\nResult:\n{messages[-1].result}"))
```

**Example Output:**
```
## Git Repository Summary

### Current Branch
You're on the **`upstream-contribution`** branch (up to date with origin), with `main` also available locally.

### Recent Commit Activity (Last ~5 Days)
| Date | Author | Summary |
|------|--------|---------|
| **Nov 27, 2025** | costiash | 3 commits enhancing the **Claude Agent SDK** - improved chief of staff agent, notebooks, observability agent, research agent, documentation, and utilities |
| **Nov 26, 2025** | Pedram Navid | Added GitHub issue templates, `/review-issue` command, `/add-registry` slash command, and new cookbook entries |
| ... | ... | ... |

### Working Directory Status
There are **uncommitted changes** in your working directory:
- **22 modified files** (mostly in `claude_agent_sdk/`)
- **4 deleted files** (documentation files in `docs/`)
- **6 untracked files** (new reports, plans, VS Code config)
```

## Part 2: GitHub MCP Server Integration

### Step 1: Set Up Your GitHub Token

Now let's level up from local Git operations to full GitHub platform integration. The [official GitHub MCP server](https://github.com/github/github-mcp-server/tree/main) provides access to over 100 tools for managing issues, pull requests, CI/CD workflows, and security alerts.

1. **Get a GitHub Personal Access Token:**
   - Visit [GitHub Token Settings](https://github.com/settings/personal-access-tokens/new)
   - Create a "Fine-grained" token with default options (public repos, no account permissions)
   - Add the token to your `.env` file: `GITHUB_TOKEN="<your_token>"`

2. **Install and Run Docker:**
   - Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Ensure Docker is running (check system tray icon)
   - Verify with `docker --version` in your terminal
   - **Troubleshooting:** If Docker won't start, check that virtualization is enabled in your BIOS

### Step 2: Define the GitHub MCP Server

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

### Step 3: Query GitHub Repository Information

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
        messages.append(msg)
```

### Step 4: Display GitHub Results

```python
display(Markdown(f"\nResult:\n{messages[-1].result}"))
```

**Example Output:**
```
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
```

## Part 3: Building an Observability Agent

Now, let's create a practical observability agent that can analyze CI/CD health for any GitHub repository.

### Step 1: Define the Analysis Prompt

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
```

### Step 2: Configure the Observability Agent

```python
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
        messages.append(msg)
```

### Step 3: Display the CI Health Analysis

```python
display(Markdown(f"\nResult:\n{messages[-1].result}"))
```

**Example Output Summary:**
```
# CI Health Analysis: facebook/react

## Executive Summary
**Overall Status: 🟢 HEALTHY**

The React repository's CI appears to be in good health. Recent commits to `main` have been successfully merged, and active PRs show passing CodeSandbox builds.

## 1. CI Infrastructure Overview
### Primary Workflows
| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `runtime_build_and_test.yml` | Push to main, PRs | Main CI - builds, tests, Flow checks |
| `shared_lint.yml` | Push to main, PRs | Prettier, ESLint, license checks |
| `compiler_typescript.yml` | PRs touching compiler | Compiler-specific tests |
| `devtools_regression_tests.yml` | PRs | DevTools testing |

## 5. Recommended Actions
| Priority | Action | Rationale |
|----------|--------|-----------|
| **LOW** | Monitor PR #35267 | Currently building - verify completion |
| **LOW** | No immediate action required | Main branch healthy, PRs passing |
| **INFO** | Security patch merged Dec 3 | PR #35277 fixed critical security vuln - verify downstream impact |

## 6. On-Call Notes
**TL;DR for On-Call Engineer:**
- 🟢 **CI is GREEN** - No action required
- Main branch is healthy with successful merges in last 24h
- All checked PRs showing green/passing status
- No open issues flagged for CI failures or flakiness
```

## Conclusion

You've successfully built an observability agent that can:

1. **Analyze local Git repositories** using the Git MCP server
2. **Query GitHub repositories** using the GitHub MCP server
3. **Perform CI/CD health analysis** on production repositories

This agent demonstrates how MCP servers transform your AI assistant from a passive observer into an active participant in your development workflow. With these foundations, you can extend this pattern to connect with databases, monitoring systems, cloud platforms, and any other service that offers an MCP server.

The key takeaways are:
- **MCP servers provide standardized interfaces** to external systems
- **Tool restrictions are crucial** for ensuring agents use the intended interfaces
- **Observability agents can automate complex analysis** that would require multiple manual steps

Experiment with different MCP servers from the [Model Context Protocol ecosystem](https://github.com/modelcontextprotocol) to build even more powerful agents for your specific use cases.
# Building a Coding Agent with GPT-5.1 and the OpenAI Agents SDK

GPT-5.1 is exceptionally strong at coding, and with the new code-editing and command-execution tools available in the [Responses API](https://platform.openai.com/docs/api-reference/responses), it’s now easier than ever to build coding agents that can work across full codebases and iterate quickly.

In this guide, you will use the [Agents SDK](https://openai.github.io/openai-agents-python/) to build a **coding agent that can scaffold a brand-new app from a prompt and refine it through user feedback**. Your agent will be equipped with the following tools:

- **apply_patch** — to edit files
- **shell** — to run shell commands
- **web_search** — to pull fresh information from the web
- **Context7 MCP** — to access up-to-date documentation

You’ll begin by focusing on the `shell` and `web_search` tools to generate a new project with web-sourced context. Then you’ll add `apply_patch` so the agent can iterate on the codebase, and connect it to the [Context7 MCP server](https://context7.com/) so it can write code informed by the most recent docs.

## Prerequisites

Before you start, ensure you have the necessary libraries installed and your OpenAI API key is available.

### 1. Install Required Packages

```bash
pip install openai-agents openai
```

### 2. Set Your OpenAI API Key

Make sure your OpenAI API key is defined in your environment.

```bash
export OPENAI_API_KEY="sk-..."
```

Alternatively, you can set it directly in your Python environment.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

## Step 1: Define a Working Environment and Shell Executor

For simplicity, you'll run shell commands locally and isolate them in a dedicated workspace directory. This ensures the agent only interacts with files inside that folder.

**Important Security Note:** In production, **always execute shell commands in a sandboxed environment**. Arbitrary command execution is inherently risky and must be tightly controlled.

### 1.1 Create an Isolated Workspace

First, create a directory to serve as your agent's isolated workspace.

```python
from pathlib import Path

workspace_dir = Path("coding-agent-workspace").resolve()
workspace_dir.mkdir(exist_ok=True)

print(f"Workspace directory: {workspace_dir}")
```

### 1.2 Implement a Shell Executor

You'll now define a `ShellExecutor` class that:

- Receives a `ShellCommandRequest` from the agent
- Optionally asks for approval before running commands
- Runs them using `asyncio.create_subprocess_shell`
- Returns a `ShellResult` with the outputs

All commands will run with `cwd=workspace_dir`, so they only affect files in that subfolder.

```python
import asyncio
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from agents import (
    ShellTool,
    ShellCommandRequest,
    ShellCommandOutput,
    ShellCallOutcome,
    ShellResult,
)


async def require_approval(commands: Sequence[str]) -> None:
    """
    Ask for confirmation before running shell commands.

    Set SHELL_AUTO_APPROVE=1 in your environment to skip this prompt
    (useful when you're iterating a lot or running in CI).
    """
    if os.environ.get("SHELL_AUTO_APPROVE") == "1":
        return

    print("Shell command approval required:")
    for entry in commands:
        print(" ", entry)
    response = input("Proceed? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        raise RuntimeError("Shell command execution rejected by user.")


class ShellExecutor:
    """
    Shell executor for the notebook cookbook.

    - Runs all commands inside `workspace_dir`
    - Captures stdout/stderr
    - Enforces an optional timeout from `action.timeout_ms`
    - Returns a ShellResult with ShellCommandOutput entries using ShellCallOutcome
    """

    def __init__(self, cwd: Path):
        self.cwd = cwd

    async def __call__(self, request: ShellCommandRequest) -> ShellResult:
        action = request.data.action
        await require_approval(action.commands)

        outputs: list[ShellCommandOutput] = []

        for command in action.commands:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self.cwd,
                env=os.environ.copy(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            timed_out = False
            try:
                timeout = (action.timeout_ms or 0) / 1000 or None
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                stdout_bytes, stderr_bytes = await proc.communicate()
                timed_out = True

            stdout = stdout_bytes.decode("utf-8", errors="ignore")
            stderr = stderr_bytes.decode("utf-8", errors="ignore")

            # Use ShellCallOutcome instead of exit_code/status fields directly
            outcome = ShellCallOutcome(
                type="timeout" if timed_out else "exit",
                exit_code=getattr(proc, "returncode", None),
            )

            outputs.append(
                ShellCommandOutput(
                    command=command,
                    stdout=stdout,
                    stderr=stderr,
                    outcome=outcome,
                )
            )

            if timed_out:
                # Stop running further commands if this one timed out
                break

        return ShellResult(
            output=outputs,
            provider_data={"working_directory": str(self.cwd)},
        )


shell_tool = ShellTool(executor=ShellExecutor(cwd=workspace_dir))
```

## Step 2: Define the Initial Coding Agent

With the Agents SDK, defining an agent is as simple as providing instructions and a list of tools. In this example, you want to use the newest `gpt-5.1` model for its state-of-the-art coding abilities.

You’ll start by enabling `web_search`, which gives the agent the ability to look up up-to-date information online, and `shell`, which lets the agent propose shell commands for tasks like scaffolding, installing dependencies, and running build steps.

### 2.1 Define the Agent's Instructions

```python
INSTRUCTIONS = '''
You are a coding assistant. The user will explain what they want to build, and your goal is to run commands to generate a new app.
You can search the web to find which command you should use based on the technical stack, and use commands to create code files.
You should also install necessary dependencies for the project to work.
'''
```

### 2.2 Instantiate the Agent

```python
from agents import Agent, Runner, ShellTool, WebSearchTool

coding_agent = Agent(
    name="Coding Agent",
    model="gpt-5.1",
    instructions=INSTRUCTIONS,
    tools=[
        WebSearchTool(),
        shell_tool
    ]
)
```

## Step 3: Start a New Project

Now, let's send a prompt to your coding agent and then inspect the files it creates in the `workspace_dir`. In this example, you'll create a NextJS dashboard using the [shadcn](https://ui.shadcn.com/) library.

**Note:** Sometimes you might run into a `MaxTurnsExceeded` error, or the project might have a dependency error. Simply run the agent loop again. In a production environment, you would implement an external loop or user input handling to iterate if the project creation fails.

### 3.1 Define the User Prompt

```python
prompt = "Create a new NextJS app that shows dashboard-01 from https://ui.shadcn.com/blocks on the home page"
```

### 3.2 Create a Helper Function to Run the Agent

This function runs the agent and streams logs to show what's happening.

```python
import asyncio
from agents import ItemHelpers, RunConfig

async def run_coding_agent_with_logs(prompt: str):
    """
    Run the coding agent and stream logs about what's happening
    """
    print("=== Run starting ===")
    print(f"[user] {prompt}\n")

    result = Runner.run_streamed(
        coding_agent,
        input=prompt
    )

    async for event in result.stream_events():

        # High-level items: messages, tool calls, tool outputs, MCP, etc.
        if event.type == "run_item_stream_event":
            item = event.item

            # 1) Tool calls (function tools, web_search, shell, MCP, etc.)
            if item.type == "tool_call_item":
                raw = item.raw_item
                raw_type_name = type(raw).__name__

                # Special-case the ones we care most about in this cookbook
                if raw_type_name == "ResponseFunctionWebSearch":
                    print("[tool] web_search_call – agent is calling web search")
                elif raw_type_name == "LocalShellCall":
                    # LocalShellCall.action.commands is where the commands live
                    commands = getattr(getattr(raw, "action", None), "commands", None)
                    if commands:
                        print(f"[tool] shell – running commands: {commands}")
                    else:
                        print("[tool] shell – running command")
                else:
                    # Generic fallback for other tools (MCP, function tools, etc.)
                    print(f"[tool] {raw_type_name} called")

            # 2) Tool call outputs
            elif item.type == "tool_call_output_item":
                # item.output is whatever your tool returned (could be structured)
                output_preview = str(item.output)
                if len(output_preview) > 400:
                    output_preview = output_preview[:400] + "…"
                print(f"[tool output] {output_preview}")

            # 3) Normal assistant messages
            elif item.type == "message_output_item":
                text = ItemHelpers.text_message_output(item)
                print(f"[assistant]\n{text}\n")

            # 4) Other event types (reasoning, MCP list tools, etc.) – ignore
            else:
                pass

    print("=== Run complete ===\n")

    # Once streaming is done, result.final_output contains the final answer
    print("Final answer:\n")
    print(result.final_output)
```

### 3.3 Execute the Agent

Run the agent with your prompt to create the project.

```python
await run_coding_agent_with_logs(prompt)
```

Once the agent is done creating the initial project (you should see a "=== Run complete ===" log followed by the final answer), you can check the output.

Navigate to the project directory and start the development server:

```bash
cd coding-agent-workspace/<name_of_the_project>
npm run dev
```

## Step 4: Iterate on the Project

Now that you have an initial version of the app, you can start iterating using the `apply_patch` tool. You also want to include calls to the OpenAI Responses API, and for that, the model should have access to the most up-to-date documentation. To make this possible, you’ll connect the agent to the [Context7 MCP server](https://context7.com/), which provides up-to-date docs.

### 4.1 Set Up the `apply_patch` Tool for In-Place Edits

**Note:** In production you’ll typically want to run these edits in a sandboxed project workspace (e.g., ephemeral containers), and work with IDEs.

```python
import hashlib
import os
from pathlib import Path

from agents import ApplyPatchTool
from agents.editor import ApplyPatchOperation, ApplyPatchResult


class ApprovalTracker:
    """Tracks which apply_patch operations have already been approved."""

    def __init__(self) -> None:
        self._approved: set[str] = set()

    def fingerprint(self, operation: ApplyPatchOperation, relative_path: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(operation.type.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update((operation.diff or "").encode("utf-8"))
        return hasher.hexdigest()

    def remember(self, fingerprint: str) -> None:
        self._approved.add(fingerprint)

    def is_approved(self, fingerprint: str) -> bool:
        return fingerprint in self._approved


class WorkspaceEditor:
    """
    Minimal editor for the apply_patch tool:
    - keeps all edits under `root`
    - optional manual approval (APPLY_PATCH_AUTO_APPROVE=1 to skip prompts)
    """

    def __init__(self, root: Path, approvals: ApprovalTracker, auto_approve: bool = False) -> None:
        self._root = root.resolve()
        self._approvals = approvals
        self._auto_approve = auto_approve or os.environ.get("APPLY_PATCH_AUTO_APPROVE") == "1"

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        relative = self._relative_path(operation.path)
        self._require_approval(operation, relative)
        target = self._resolve(operation.path, ensure_parent=True)
        diff = operation.diff or ""
        content = apply_unified_diff("", diff, create=True)
        target.write_text(content, encoding="utf-8")
        return ApplyPatchResult(output=f"Created {relative}")

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        relative = self._relative_path(operation.path)
        self._require_approval(operation, relative)
        target = self._resolve(operation.path)
        original = target.read_text(encoding="utf-8")
        diff = operation.diff or ""
        patched = apply_unified_diff(original, diff)
        target.write_text(patched, encoding="utf-8")
        return ApplyPatchResult(output=f"Updated {relative}")

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        relative = self._relative_path(operation.path)
        self._require_approval(operation, relative)
        target = self._resolve(operation.path)
        target.unlink(missing_ok=True)
        return ApplyPatchResult(output=f"Deleted {relative}")

    def _relative_path(self, value: str) -> str:
        resolved = self._resolve(value)
        return resolved.relative_to(self._root).as_posix()

    def _resolve(self, relative: str, ensure_parent: bool = False) -> Path:
        candidate = Path(relative)
        target = candidate if candidate.is_absolute() else (self._root / candidate)
        target = target.resolve()
        try:
            target.relative_to(self._root)
        except ValueError:
            raise RuntimeError(f"Operation outside workspace: {relative}") from None
        if ensure_parent:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _require_approval(self, operation: ApplyPatchOperation, display_path: str) -> None:
        fingerprint = self._approvals.fingerprint(operation, display_path)
        if self._auto_approve or self._approvals.is_approved(fingerprint):
            self._approvals.remember(fingerprint)
            return

        print("\n[apply_patch] approval required")
        print(f"- type: {operation.type}")
        print(f"- path: {display_path}")
        if operation.diff:
            preview = operation.diff if len(operation.diff) < 400 else f"{operation.diff[:400]}…"
            print("- diff preview:\n", preview)
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise RuntimeError("Apply patch operation rejected by user.")
        self._approvals.remember(fingerprint)


def apply_unified_diff(original: str, diff: str, create: bool = False) -> str:
    """
    Simple "diff" applier (adapt this based on your environment)

    - For create_file, the diff can be the full desired file contents,
      optionally with leading '+' on each line.
    - For update_file, we treat the diff as the new file contents:
      keep lines starting with ' ' or '+', drop '-' lines and diff headers.

    This avoids context/delete mismatch errors while still letting the model
    send familiar diff-like patches.
    """
    if not diff:
        return original

    lines = diff.splitlines()
    body: list[str] = []

    for line in lines:
        if not line:
            body.append("")
            continue

        # Skip typical unified diff headers / metadata
        if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
            continue

        prefix = line[0]
        content = line[1:]

        if prefix in ("+", " "):
            body.append(content)
        elif prefix in ("-", "\\"):
            # skip deletions and "\ No newline at end of file"
            continue
        else:
            # If it doesn't look like diff syntax, keep the full line
            body.append(line)

    text = "\n".join(body)
    if diff.endswith("\n"):
        text += "\n"
    return text


approvals = ApprovalTracker()
editor = WorkspaceEditor(root=workspace_dir, approvals=approvals, auto_approve=True)
apply_patch_tool = ApplyPatchTool(editor=editor)
```

### 4.2 Connect to the Context7 MCP Server

The Context7 MCP server provides up-to-date documentation. You can optionally set an API key for higher rate limits.

```python
# Optional: set CONTEXT7_API_KEY in your environment for higher rate limits
CONTEXT7_API_KEY = os.getenv("CONTEXT7_API_KEY")
```

```python
from agents import HostedMCPTool

context7_tool = HostedMCPTool(
    tool_config={
        "type": "mcp",
        "server_label": "context7",
        "server_url": "https://mcp.context7.com/mcp",
        # Basic usage works without auth; for higher rate limits, pass your key here.
        **(
            {"authorization": f"Bearer {CONTEXT7_API_KEY}"}
            if CONTEXT7_API_KEY
            else {}
        ),
        "require_approval": "never",
    },
)
```

### 4.
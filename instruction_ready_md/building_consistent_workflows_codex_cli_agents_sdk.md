# Building Consistent Workflows with Codex CLI & Agents SDK
## A Guide to Repeatable, Traceable, and Scalable Agentic Development

### Introduction
Developers strive for consistency in everything they do. With Codex CLI and the Agents SDK, that consistency can now scale like never before. Whether you’re refactoring a large codebase, rolling out new features, or introducing a new testing framework, Codex integrates seamlessly into CLI, IDE, and cloud workflows to automate and enforce repeatable development patterns.

In this guide, you will build both single and multi-agent systems using the Agents SDK, with Codex CLI exposed as an MCP (Model Context Protocol) Server. This enables:
- **Consistency and Repeatability** by providing each agent a scoped context.
- **Scalable Orchestration** to coordinate single and multi-agent systems.
- **Observability & Auditability** by reviewing the full agentic stack trace.

### What You'll Build
1.  **Initialize Codex CLI as an MCP Server**: Run Codex as a long-running MCP process.
2.  **Build a Single-Agent System**: Use Codex MCP for a scoped task to create a simple game.
3.  **Orchestrate a Multi-Agent Workflow**: Coordinate multiple specialized agents (Project Manager, Designer, Developer, Tester) to build a more complex application.

### Prerequisites & Setup
Before starting, ensure you have the following:
- Basic familiarity with Python.
- A development environment (like VS Code or Cursor).
- An OpenAI API key. You can create one in the [OpenAI Dashboard](https://platform.openai.com/api-keys).

### Step 1: Environment Setup
First, create a `.env` file in your project directory and add your OpenAI API key.

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Next, install the required Python packages.

```bash
pip install openai-agents openai python-dotenv
```

### Step 2: Initialize Codex CLI as an MCP Server
You will run Codex CLI as an MCP Server inside the Agents SDK. This exposes two tools on the server: `codex()` for creating a conversation and `codex-reply()` for continuing one.

Create a new Python file, `single_agent.py`, and add the following code. This sets up the MCP server with an extended timeout to allow Codex CLI enough time to execute tasks.

```python
import os
from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, set_default_openai_api
from agents.mcp import MCPServerStdio

# Load environment variables
load_dotenv(override=True)
set_default_openai_api(os.getenv("OPENAI_API_KEY"))

async def main() -> None:
    async with MCPServerStdio(
        name="Codex CLI",
        params={
            "command": "npx",
            "args": ["-y", "codex", "mcp-server"],
        },
        client_session_timeout_seconds=360000,
    ) as codex_mcp_server:
        print("Codex MCP server started.")
        # We will add agent definitions here in the next step
        return

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Build a Single-Agent System
Now, let's create a simple two-agent system within the `main()` function. A Designer agent will brainstorm a game idea, and a Developer agent will implement it.

Replace the comment `# We will add agent definitions here...` in the `main()` function with the following code:

```python
        # Define the Developer Agent
        developer_agent = Agent(
            name="Game Developer",
            instructions=(
                "You are an expert in building simple games using basic html + css + javascript with no dependencies. "
                "Save your work in a file called index.html in the current directory. "
                'Always call codex with {"approval-policy": "never"} and {"sandbox": "workspace-write"}.'
            ),
            mcp_servers=[codex_mcp_server],
        )

        # Define the Designer Agent
        designer_agent = Agent(
            name="Game Designer",
            instructions=(
                "You are an indie game connoisseur. Come up with an idea for a single page html + css + javascript game that a developer could build in about 50 lines of code. "
                "Format your request as a 3 sentence design brief for a game developer and call the Game Developer coder with your idea."
            ),
            model="gpt-5",
            handoffs=[developer_agent],
        )

        # Run the agentic workflow
        result = await Runner.run(designer_agent, "Implement a fun new game!")
        print("Workflow complete. Check for an index.html file.")
```

**How it works:**
1.  The `developer_agent` is configured to write files directly to the workspace without asking for user permission.
2.  The `designer_agent` uses the GPT-5 model and is instructed to hand off its idea to the `developer_agent`.
3.  The `Runner.run()` function starts the process with the designer's prompt.

**Run the script:**
```bash
python single_agent.py
```

After a few minutes, the script will complete. You should see a new `index.html` file in your directory. Open it in a browser to play the game your agents created!

### Step 4: Orchestrate a Multi-Agent Workflow
For more complex projects, you can orchestrate a team of specialized agents. You will build a system with five agents:
- **Project Manager**: Breaks down tasks and coordinates the team.
- **Designer**: Produces UI/UX specifications.
- **Frontend Developer**: Implements the user interface.
- **Backend Developer**: Implements APIs and logic.
- **Tester**: Validates outputs against acceptance criteria.

This structure enforces gating logic, where the Project Manager verifies deliverables exist before allowing the next agent to proceed.

Create a new file, `multi_agent.py`, and start with the same setup and MCP server initialization from Step 2.

```python
import os
from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, set_default_openai_api, ModelSettings, Reasoning
from agents.mcp import MCPServerStdio
from agents.tools import WebSearchTool

load_dotenv(override=True)
set_default_openai_api(os.getenv("OPENAI_API_KEY"))

# A recommended prefix to optimize agent handoffs
RECOMMENDED_PROMPT_PREFIX = "You are part of a multi-agent team. Communicate clearly and stick to your role.\n"

async def main() -> None:
    async with MCPServerStdio(
        name="Codex CLI",
        params={
            "command": "npx",
            "args": ["-y", "codex", "mcp-server"],
        },
        client_session_timeout_seconds=360000,
    ) as codex_mcp_server:
        print("Codex MCP server started.")
        # Agent definitions will go here
        return

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 5: Define the Specialized Agents
Inside the `async with` block of the `main()` function, define the four specialist agents. Notice they are given access to the Codex MCP server and specific instructions for their deliverables.

```python
        # Define Specialist Agents
        designer_agent = Agent(
            name="Designer",
            instructions=(
                f"""{RECOMMENDED_PROMPT_PREFIX}"""
                "You are the Designer.\n"
                "Your only source of truth is AGENT_TASKS.md and REQUIREMENTS.md from the Project Manager.\n"
                "Do not assume anything that is not written there.\n\n"
                "You may use the internet for additional guidance or research.\n"
                "Deliverables (write to /design):\n"
                "- design_spec.md – a single page describing the UI/UX layout, main screens, and key visual notes as requested in AGENT_TASKS.md.\n"
                "- wireframe.md – a simple text or ASCII wireframe if specified.\n\n"
                "Keep the output short and implementation-friendly.\n"
                "When complete, handoff to the Project Manager.\n"
                'When creating files, call Codex MCP with {"approval-policy":"never","sandbox":"workspace-write"}.'
            ),
            model="gpt-5",
            tools=[WebSearchTool()],
            mcp_servers=[codex_mcp_server],
            handoffs=[], # Will be set later
        )

        frontend_developer_agent = Agent(
            name="Frontend Developer",
            instructions=(
                f"""{RECOMMENDED_PROMPT_PREFIX}"""
                "You are the Frontend Developer.\n"
                "Read AGENT_TASKS.md and design_spec.md. Implement exactly what is described there.\n\n"
                "Deliverables (write to /frontend):\n"
                "- index.html – main page structure\n"
                "- styles.css or inline styles if specified\n"
                "- main.js or game.js if specified\n\n"
                "Follow the Designer’s DOM structure and any integration points given by the Project Manager.\n"
                "Do not add features or branding beyond the provided documents.\n\n"
                "When complete, handoff to the Project Manager.\n"
                'When creating files, call Codex MCP with {"approval-policy":"never","sandbox":"workspace-write"}.'
            ),
            model="gpt-5",
            mcp_servers=[codex_mcp_server],
            handoffs=[], # Will be set later
        )

        backend_developer_agent = Agent(
            name="Backend Developer",
            instructions=(
                f"""{RECOMMENDED_PROMPT_PREFIX}"""
                "You are the Backend Developer.\n"
                "Read AGENT_TASKS.md and REQUIREMENTS.md. Implement the backend endpoints described there.\n\n"
                "Deliverables (write to /backend):\n"
                "- package.json – include a start script if requested\n"
                "- server.js – implement the API endpoints and logic exactly as specified\n\n"
                "Keep the code as simple and readable as possible. No external database.\n\n"
                "When complete, handoff to the Project Manager.\n"
                'When creating files, call Codex MCP with {"approval-policy":"never","sandbox":"workspace-write"}.'
            ),
            model="gpt-5",
            mcp_servers=[codex_mcp_server],
            handoffs=[], # Will be set later
        )

        tester_agent = Agent(
            name="Tester",
            instructions=(
                f"""{RECOMMENDED_PROMPT_PREFIX}"""
                "You are the Tester.\n"
                "Read AGENT_TASKS.md and TEST.md. Verify that the outputs of the other roles meet the acceptance criteria.\n\n"
                "Deliverables (write to /tests):\n"
                "- TEST_PLAN.md – bullet list of manual checks or automated steps as requested\n"
                "- test.sh or a simple automated script if specified\n\n"
                "Keep it minimal and easy to run.\n\n"
                "When complete, handoff to the Project Manager.\n"
                'When creating files, call Codex MCP with {"approval-policy":"never","sandbox":"workspace-write"}.'
            ),
            model="gpt-5",
            mcp_servers=[codex_mcp_server],
            handoffs=[], # Will be set later
        )
```

### Step 6: Define the Project Manager Agent
The Project Manager is the orchestrator. It receives the initial prompt, creates planning documents, and manages the gated handoff process between specialists.

Add this definition after the specialist agents:

```python
        project_manager_agent = Agent(
            name="Project Manager",
            instructions=(
                f"""{RECOMMENDED_PROMPT_PREFIX}"""
                """
                You are the Project Manager.

                Objective:
                Convert the input task list into three project-root files the team will execute against.

                Deliverables (write in project root):
                - REQUIREMENTS.md: concise summary of product goals, target users, key features, and constraints.
                - TEST.md: tasks with [Owner] tags (Designer, Frontend, Backend, Tester) and clear acceptance criteria.
                - AGENT_TASKS.md: one section per role containing:
                    - Project name
                    - Required deliverables (exact file names and purpose)
                    - Key technical notes and constraints

                Process:
                - Resolve ambiguities with minimal, reasonable assumptions. Be specific so each role can act without guessing.
                - Create files using Codex MCP with {"approval-policy":"never","sandbox":"workspace-write"}.
                - Do not create folders. Only create REQUIREMENTS.md, TEST.md, AGENT_TASKS.md.

                Handoffs (gated by required files):
                1) After the three files above are created, hand off to the Designer and include REQUIREMENTS.md, and AGENT_TASKS.md.
                2) Wait for the Designer to produce /design/design_spec.md. Verify that file exists before proceeding.
                3) When design_spec.md exists, hand off in parallel to both:
                    - Frontend Developer (provide design_spec.md, REQUIREMENTS.md, AGENT_TASKS.md).
                    - Backend Developer (provide REQUIREMENTS.md, AGENT_TASKS.md).
                4) Wait for Frontend to produce /frontend/index.html and Backend to produce /backend/server.js. Verify both files exist.
                5) When both exist, hand off to the Tester and provide all prior artifacts and outputs.
                6) Do not advance to the next handoff until the required files for that step are present. If something is missing, request the owning agent to supply it and re-check.

                PM Responsibilities:
                - Coordinate all roles, track file completion, and enforce the above gating checks.
                - Do NOT respond with status updates. Just handoff to the next agent until the project is complete.
                """
            ),
            model="gpt-5",
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="medium")
            ),
            handoffs=[designer_agent, frontend_developer_agent, backend_developer_agent, tester_agent],
            mcp_servers=[codex_mcp_server],
        )
```

### Step 7: Configure Handoffs and Run the System
Finally, configure all specialist agents to hand off *back* to the Project Manager for validation. Then, define the task list and start the runner.

Add this code after the `project_manager_agent` definition:

```python
        # Configure all specialists to hand off back to the Project Manager
        designer_agent.handoffs = [project_manager_agent]
        frontend_developer_agent.handoffs = [project_manager_agent]
        backend_developer_agent.handoffs = [project_manager_agent]
        tester_agent.handoffs = [project_manager_agent]

        # Define the high-level task
        task_list = """
        Goal: Build a tiny browser game to showcase a multi-agent workflow.

        High-level requirements:
        - Single-screen game called "Bug Busters".
        - Player clicks a moving bug to earn points.
        - Game ends after 20 seconds and shows final score.
        - Optional: submit score to a simple backend and display a top-10 leaderboard.

        Roles:
        - Designer: create a one-page UI/UX spec and basic wireframe.
        - Frontend Developer: implement the page and game logic.
        - Backend Developer: implement a minimal API (GET /health, GET/POST /scores).
        - Tester: write a quick test plan and a simple script to verify core routes.

        Constraints:
        - No external database—memory storage is fine.
        - Keep everything readable for beginners; no frameworks required.
        - All outputs should be small files saved in clearly named folders.
        """

        # Run the multi-agent system
        print("Starting multi-agent workflow. This may take several minutes...")
        result = await Runner.run(project_manager_agent, task_list)
        print("Multi-agent workflow complete!")
```

**Run the script:**
```bash
python multi_agent.py
```

This orchestrated workflow will take approximately 10-15 minutes to complete. When finished, your project directory will be populated with files organized by role:

```
project_root/
├── AGENT_TASKS.md
├── REQUIREMENTS.md
├── TEST.md
├── backend/
│   ├── package.json
│   └── server.js
├── design/
│   ├── design_spec.md
│   └── wireframe.md
├── frontend/
│   ├── index.html
│   ├── game.js
│   └── styles.css
└── tests/
    └── TEST_PLAN.md
```

### Conclusion
You have successfully built both single and multi-agent systems using the Agents SDK with Codex CLI as an MCP Server. You've seen how to:
1.  Initialize a long-running Codex MCP server.
2.  Create agents with specialized roles and scoped contexts.
3.  Implement handoff logic for simple task passing.
4.  Orchestrate a complex, gated workflow with a Project Manager agent enforcing quality gates.

This pattern provides a foundation for building consistent, repeatable, and observable AI-powered development workflows that can scale to meet complex project demands.
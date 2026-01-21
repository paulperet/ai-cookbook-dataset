# Building an AI Chief of Staff Agent with Claude SDK

## Introduction

In this guide, you'll build a comprehensive AI Chief of Staff agent for a 50-person startup that just raised a $10M Series A. The CEO needs data-driven insights to balance aggressive growth with financial sustainability.

Your final Chief of Staff agent will:
- Coordinate specialized subagents for different domains
- Aggregate insights from multiple sources
- Provide executive summaries with actionable recommendations

## Prerequisites

First, set up your environment with the necessary imports and configuration:

```python
from dotenv import load_dotenv
from utils.agent_visualizer import (
    display_agent_response,
    print_activity,
    reset_activity_context,
    visualize_conversation,
)
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

load_dotenv()

# Define the model to use throughout this guide
# Using Opus 4.5 for its superior planning and reasoning capabilities
MODEL = "claude-opus-4-5"
print(f"📋 Notebook configured to use: {MODEL}")
```

## Feature 0: Persistent Memory with CLAUDE.md

### What is CLAUDE.md?
`CLAUDE.md` files serve as persistent memory and instructions for your agent. When present in the project directory, Claude Code automatically reads and incorporates this context when you initialize your agent.

### Why Use CLAUDE.md?
Instead of repeatedly providing project context, team preferences, or standards in each interaction, you can define them once in `CLAUDE.md`. This ensures consistent behavior and reduces token usage by avoiding redundant explanations.

### How to Implement

1. Create a `CLAUDE.md` file in your working directory (in our example: `chief_of_staff_agent/CLAUDE.md`)
2. Set the `cwd` argument of your `ClaudeSDKClient` to point to the directory containing your CLAUDE.md file
3. Use explicit prompts to guide the agent when you want it to prefer high-level context over detailed data files

**Important Behavior Note**: When both CLAUDE.md and detailed data files (like CSVs) are available, the agent may prefer to read the more granular data sources to provide precise answers. This is expected behavior - agents naturally seek authoritative data. To ensure the agent uses high-level CLAUDE.md context, use explicit prompt instructions.

### Implementation Example

```python
messages = []
async with ClaudeSDKClient(
    options=ClaudeAgentOptions(
        model=MODEL,
        cwd="chief_of_staff_agent",  # Points to subdirectory with our CLAUDE.md
        setting_sources=["project"],
    )
) as agent:
    await agent.query("What's our current runway?")
    async for msg in agent.receive_response():
        print_activity(msg)
        messages.append(msg)

# Display the response with HTML rendering
display_agent_response(messages)
```

**Expected Output**:
Based on the company information, **TechStart Inc's current runway is 20 months** (until September 2025).

Here are the key financial details:
- **Cash in Bank**: $10M
- **Monthly Burn Rate**: ~$500,000
- **Runway**: 20 months

### Understanding Agent Data Source Preferences

**What Just Happened:**
By adding context to our prompt, we guided the agent to rely on the CLAUDE.md context rather than seeking more granular data from CSV files.

**Key Insights:**
1. **CLAUDE.md as Context, Not Constraint**: When you set `cwd`, the CLAUDE.md file is loaded as background context. However, agents will naturally seek the most authoritative data sources available. If detailed CSV files exist, the agent may prefer them for precision.

2. **Prompt Engineering Matters**: The phrasing "high-level financial numbers from context" signals to the agent that you want the simplified executive summary from CLAUDE.md ($500K burn, 20 months runway) rather than the precise month-by-month data from financial_data/burn_rate.csv ($525K gross, $235K net burn).

3. **Architectural Design Choice**: This behavior is actually desirable in production systems - you want agents to find the best data source. CLAUDE.md should contain:
   - High-level context and strategy
   - Company information and standards
   - Pointers to where detailed data lives
   - Guidelines on when to use high-level vs. detailed numbers

4. **Real-World Pattern**: Think of CLAUDE.md as an "onboarding document" that orients the agent, while detailed files are "source systems" the agent can query when precision matters.

## Feature 1: Bash Tool for Python Script Execution

### What is the Bash Tool?
The Bash tool allows your agent to run Python scripts directly, enabling access to procedural knowledge, complex computations, data analysis and other integrations that go beyond the agent's native capabilities.

### Why Use the Bash Tool?
Your Chief of Staff might need to process data files, run financial models or generate visualizations based on this data. These are all good scenarios for using the Bash tool.

### How to Implement
Have your Python scripts set up in a place where your agent can reach them and add context on what they are and how they can be called. If the scripts are meant for your chief of staff agent, add this context to its CLAUDE.md file. If they are meant for subagents, add said context to their MD files.

For this tutorial, we added five example scripts to `chief_of_staff_agent/scripts`:
1. `hiring_impact.py`: Calculates how new engineering hires affect burn rate, runway, and cash position.
2. `talent_scorer.py`: Scores candidates on technical skills, experience, culture fit, and salary expectations.
3. `simple_calculation.py`: Performs quick financial calculations for runway, burn rate, and quarterly metrics.
4. `financial_forecast.py`: Models ARR growth scenarios (base/optimistic/pessimistic).
5. `decision_matrix.py`: Creates weighted decision matrices for strategic choices.

### Implementation Example

```python
messages = []
async with ClaudeSDKClient(
    options=ClaudeAgentOptions(
        model=MODEL,
        allowed_tools=["Bash", "Read"],
        cwd="chief_of_staff_agent",  # Points to subdirectory where our agent is defined
    )
) as agent:
    await agent.query(
        "Use your simple calculation script with a total runway of 2904829 and a monthly burn of 121938."
    )
    async for msg in agent.receive_response():
        print_activity(msg)
        messages.append(msg)

# Display the response with HTML rendering
display_agent_response(messages)
```

**Expected Output**:
Here are the financial metrics calculated using the simple calculation script:

| Metric | Value |
|---|---|
| **Total Runway** | $2,904,829.00 |
| **Monthly Burn** | $121,938.00 |
| **Runway Months** | ~23.82 months |
| **Quarterly Burn** | $365,814.00 |
| **Daily Burn Rate** | $4,064.60 |

Based on these calculations, with a total runway of $2,904,829 and a monthly burn rate of $121,938, you have approximately **23.8 months** (just under 2 years) of runway remaining.

## Feature 2: Output Styles

### What are Output Styles?
Output styles allow you to use different output formats for different audiences. Each style is defined in a markdown file.

### Why Use Output Styles?
Your agent might be used by people of different levels of expertise or with different priorities. Output styles help differentiate between these segments without having to create separate agents.

### How to Implement

1. Configure markdown files per style in `chief_of_staff_agent/.claude/output-styles/`
2. Each style file should include frontmatter with two fields: `name` and `description`
3. **Important**: The name in the frontmatter must match exactly the file's name (case sensitive)
4. For the SDK to load these files, you **must** include `setting_sources=["project"]` in your `ClaudeAgentOptions`

**SDK Configuration Note**: Output styles are stored on the filesystem in `.claude/output-styles/`. The `settings` parameter tells the SDK *which* style to use, but `setting_sources` is required to actually *load* the style definitions.

### Implementation Example

```python
messages_executive = []
async with ClaudeSDKClient(
    options=ClaudeAgentOptions(
        model=MODEL,
        cwd="chief_of_staff_agent",
        settings='{"outputStyle": "executive"}',
        # IMPORTANT: setting_sources must include "project" to load output styles
        setting_sources=["project"],
    )
) as agent:
    await agent.query("Tell me in two sentences about your writing output style.")
    async for msg in agent.receive_response():
        print_activity(msg)
        messages_executive.append(msg)

messages_technical = []
async with ClaudeSDKClient(
    options=ClaudeAgentOptions(
        model=MODEL,
        cwd="chief_of_staff_agent",
        settings='{"outputStyle": "technical"}',
        setting_sources=["project"],
    )
) as agent:
    await agent.query("Tell me in two sentences about your writing output style.")
    async for msg in agent.receive_response():
        print_activity(msg)
        messages_technical.append(msg)
```

**Executive Style Output**:
My writing style is direct, concise, and professional—I avoid unnecessary filler and get straight to actionable insights. I adapt my tone based on context: more formal for strategic recommendations and board-level communications, more conversational for quick operational questions.

**Technical Style Output**:
My writing style is direct, clear, and professional—I provide concise answers without unnecessary filler while ensuring the information is complete and actionable. I adapt my tone to the context, being more formal for business analysis and more conversational for general questions, and I use formatting (like bullet points, headers, or code blocks) when it helps organize complex information.

## Feature 3: Plan Mode - Strategic Planning Without Execution

### What is Plan Mode?
Plan mode instructs the agent to create a detailed execution plan without performing any actions. The agent analyzes requirements, proposes solutions, and outlines steps, but doesn't modify files, execute commands, or make changes.

### Why Use Plan Mode?
Complex tasks benefit from upfront planning to reduce errors, enable review and improve coordination. After the planning phase, the agent will have a clear roadmap to follow throughout its execution.

### How to Implement
Simply set `permission_mode="plan"` in your agent configuration.

**Plan Persistence**: Since plans are valuable artifacts for review and decision-making, we'll demonstrate how to capture and save them to persistent markdown files. This enables stakeholders to review plans before approving execution.

> **Note**: This feature shines in Claude Code but still needs to be fully adapted for headless applications with the SDK. The agent will try calling its `ExitPlanMode()` tool, which is only relevant in interactive mode. In this case, you can send a follow-up query with `continue_conversation=True` for the agent to execute its plan in context.

### Plan Mode Helper Functions

To handle the various ways an agent might output its plan, we need robust extraction from multiple sources:

```python
import glob as glob_module
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

def extract_plan_from_xml(text: str | None, min_length: int = 200) -> str | None:
    """
    Extract content between <plan> tags from text.
    """
    if not text:
        return None
    match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if len(extracted) > min_length:
            return extracted
    return None

def extract_plan_from_messages(
    plan_content: list[str], min_fallback_length: int = 500
) -> tuple[str | None, str | None]:
    """
    Try to extract plan from captured message stream content.
    """
    combined_text = "\n\n".join(plan_content)

    # First try: XML tags
    plan = extract_plan_from_xml(combined_text)
    if plan:
        return plan, "message stream"

    # Fallback: Use raw content if substantial
    if len(combined_text.strip()) > min_fallback_length:
        return combined_text.strip(), "full message content (fallback)"

    return None, None

def extract_plan_from_write_tool(
    write_contents: list[str], min_fallback_length: int = 500
) -> tuple[str | None, str | None]:
    """
    Try to extract plan from captured Write tool calls.
    """
    for content in write_contents:
        # Try XML extraction first
        plan = extract_plan_from_xml(content)
        if plan:
            return plan, "Write tool capture"

        # Fallback: substantial content without tags
        if content and len(content.strip()) > min_fallback_length:
            return content.strip(), "Write tool capture (no XML tags)"

    return None, None

def extract_plan_from_claude_dir(
    max_age_seconds: int = 300, min_fallback_length: int = 500
) -> tuple[str | None, str | None]:
    """
    Check Claude's internal plan directory for recently created plans.
    """
    claude_plans_dir = os.path.expanduser("~/.claude/plans")

    if not os.path.exists(claude_plans_dir):
        return None, None

    # Find most recent plan file
    plan_files = sorted(
        glob_module.glob(os.path.join(claude_plans_dir, "*.md")),
        key=os.path.getmtime,
        reverse=True,
    )

    if not plan_files:
        return None, None

    most_recent = plan_files[0]
    file_age = datetime.now().timestamp() - os.path.getmtime(most_recent)

    if file_age > max_age_seconds:
        return None, None

    with open(most_recent) as f:
        content = f.read()

    filename = os.path.basename(most_recent)

    # Try XML extraction first
    plan = extract_plan_from_xml(content)
    if plan:
        return plan, f"Claude plan file ({filename})"

    # Fallback: substantial content without tags
    if len(content.strip()) > min_fallback_length:
        return content.strip(), f"Claude plan file ({filename}, no XML tags)"

    return None, None

def save_plan_to_file(
    plan_content: str,
    plan_source: str,
    model_name: str,
    prompt_summary: str,
    output_dir: str = "chief_of_staff_agent/plans",
    title: str = "Agent Plan: Engineering Restructure for AI Focus",
) -> Path:
    """
    Save extracted plan to a timestamped markdown file.
    """
    plans_dir = Path(output_dir)
    plans_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = plans_dir / f"plan_{timestamp}.md"

    with open(plan_file, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: {model_name}\n")
        f.write(f"**Prompt**: {prompt_summary}\n")
        f.write(f"**Source**: {plan_source}\n\n")
        f.write("---\n\n")
        f.write(plan_content)

    return plan_file
```

These helper functions handle the various ways an agent might output its plan. Since agents can output plans via direct text, Write tool, or Claude's internal plan directory, we need robust extraction from multiple sources.

## Next Steps

You've now implemented the foundational features for your AI Chief of Staff agent:
1. **Persistent Memory** with CLAUDE.md for consistent context
2. **Script Execution** via the Bash tool for complex computations
3. **Adaptive Output Styles** for different audiences
4. **Strategic Planning** with Plan Mode for complex tasks

In the next sections, you'll build upon these foundations to create a fully coordinated multi-agent system that can handle complex business scenarios with specialized subagents working together under the Chief of Staff's coordination.
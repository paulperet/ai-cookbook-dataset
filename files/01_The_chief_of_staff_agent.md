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

# Define the model to use throughout this notebook
# Using Opus 4.5 for its superior planning and reasoning capabilities
MODEL = "claude-opus-4-5"
print(f"📋 Notebook configured to use: {MODEL}")
```

    📋 Notebook configured to use: claude-opus-4-5


# 01 - The Chief of Staff Agent

#### Introduction

In notebook 00, we built a simple research agent. In this notebook, we'll incrementally introduce key Claude Code SDK features for building comprehensive agents. For each introduced feature, we'll explain:
- **What**: what the feature is
- **Why**: what the feature can do and why you would want to use it
- **How**: a minimal implementation showing how to use it

If you are familiar with Claude Code, you'll notice how the SDK brings feature parity and enables you to leverage all of Claude Code's capabilities in a programmatic headless manner.

#### Scenario

Throughout this notebook, we'll build an **AI Chief of Staff** for a 50-person startup that just raised $10M Series A. The CEO needs data-driven insights to balance aggressive growth with financial sustainability.

Our final Chief of Staff agent will:
- **Coordinate specialized subagents** for different domains
- **Aggregate insights** from multiple sources
- **Provide executive summaries** with actionable recommendations

## Basic Features

### Feature 0: Memory with [CLAUDE.md](https://www.anthropic.com/engineering/claude-code-best-practices)

**What**: `CLAUDE.md` files serve as persistent memory and instructions for your agent. When present in the project directory, Claude Code automatically reads and incorporates this context when you initialize your agent.

**Why**: Instead of repeatedly providing project context, team preferences, or standards in each interaction, you can define them once in `CLAUDE.md`. This ensures consistent behavior and reduces token usage by avoiding redundant explanations.

**How**: 
- Have a `CLAUDE.md` file in the working directory - in our example: `chief_of_staff_agent/CLAUDE.md`
- Set the `cwd` argument of your ClaudeSDKClient to point to directory of your CLAUDE.md file
- Use explicit prompts to guide the agent when you want it to prefer high-level context over detailed data files

**Important Behavior Note**: When both CLAUDE.md and detailed data files (like CSVs) are available, the agent may prefer to read the more granular data sources to provide precise answers. This is expected behavior - agents naturally seek authoritative data. To ensure the agent uses high-level CLAUDE.md context, use explicit prompt instructions (see example below). This teaches an important lesson: CLAUDE.md provides *context and guidance*, not hard constraints on data sources.


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
# With this prompt, the agent should use CLAUDE.md values: ~$500K burn, 20 months runway
```

    🤖 Thinking...


Based on the company information, **TechStart Inc's current runway is 20 months** (until September 2025).

Here are the key financial details:
- **Cash in Bank**: $10M
- **Monthly Burn Rate**: ~$500,000
- **Runway**: 20 months

This gives you a solid runway to execute on Q2 2024 priorities, including hiring 10 engineers, launching the AI code review feature, European expansion, and beginning Series B conversations (targeting $30M).

#### Understanding Agent Data Source Preferences

**What Just Happened:**
By adding   to our prompt, we guided the agent to rely on the CLAUDE.md context rather than seeking more granular data from CSV files.

**Key Insights:**
1. **CLAUDE.md as Context, Not Constraint**: When you set `cwd`, the CLAUDE.md file is loaded as background context. However, agents will naturally seek the most authoritative data sources available. If detailed CSV files exist, the agent may prefer them for precision.

2. **Prompt Engineering Matters**: The phrasing "high-level financial numbers from context" signals to the agent that you want the simplified executive summary from CLAUDE.md ($500K burn, 20 months runway) rather than the precise month-by-month data from financial_data/burn_rate.csv ($525K gross, $235K net burn).

3. **Architectural Design Choice**: This behavior is actually desirable in production systems - you want agents to find the best data source. CLAUDE.md should contain:
   - High-level context and strategy
   - Company information and standards
   - Pointers to where detailed data lives
   - Guidelines on when to use high-level vs. detailed numbers

4. **Real-World Pattern**: Think of CLAUDE.md as an "onboarding document" that orients the agent, while detailed files are "source systems" the agent can query when precision matters.

### Feature 1: The Bash tool for Python Script Execution

**What**: The Bash tool allows your agent to (among other things) run Python scripts directly, enabling access to procedural knowledge, complex computations, data analysis and other integrations that go beyond the agent's native capabilities.

**Why**: Our Chief of Staff might need to process data files, run financial models or generate visualizations based on this data. These are all good scenarios for using the Bash tool.

**How**: Have your Python scripts set-up in a place where your agent can reach them and add some context on what they are and how they can be called. If the scripts are meant for your chief of staff agent, add this context to its CLAUDE.md file and if they are meant for one your subagents, add said context to their MD files (more details on this later). For this tutorial, we added five toy examples to `chief_of_staff_agent/scripts`:
1. `hiring_impact.py`: Calculates how new engineering hires affect burn rate, runway, and cash position. Essential for the `financial-analyst` subagent to model hiring scenarios against the $500K monthly burn and 20-month runway.
2. `talent_scorer.py`: Scores candidates on technical skills, experience, culture fit, and salary expectations using weighted criteria. Core tool for the `recruiter` subagent to rank engineering candidates against TechStart's $180-220K senior engineer benchmarks.
3. `simple_calculation.py`: Performs quick financial calculations for runway, burn rate, and quarterly metrics. Utility script for chief of staff to get instant metrics without complex modeling.
4. `financial_forecast.py`: Models ARR growth scenarios (base/optimistic/pessimistic) given the current $2.4M ARR growing at 15% MoM.Critical for `financial-analyst` to project Series B readiness and validate the $30M fundraising target.
5. `decision_matrix.py`: Creates weighted decision matrices for strategic choices like the SmartDev acquisition or office expansion. Helps chief of staff systematically evaluate complex decisions with multiple stakeholders and criteria.


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

    [🤖 Using: Glob(), ..., ✓ Tool completed]
    🤖 Thinking...


Here are the financial metrics calculated using the simple calculation script:

| Metric | Value |
|---|---|
| **Total Runway** | $2,904,829.00 |
| **Monthly Burn** | $121,938.00 |
| **Runway Months** | ~23.82 months |
| **Quarterly Burn** | $365,814.00 |
| **Daily Burn Rate** | $4,064.60 |

Based on these calculations, with a total runway of $2,904,829 and a monthly burn rate of $121,938, you have approximately **23.8 months** (just under 2 years) of runway remaining.

### Feature 2: Output Styles

**What**: Output styles allow you to use different output styles for different audiences. Each style is defined in a markdown file.

**Why**: Your agent might be used by people of different levels of expertise or they might have different priorities. Your output style can help differentiate between these segments without having to create a separate agent.

**How**:
- Configure a markdown file per style in `chief_of_staff_agent/.claude/output-styles/`. For example, check out the Executive Ouput style in `.claude/output-styles/executive.md`. Output styles are defined with a simple frontmatter including two fields: name and description. Note: Make sure the name in the frontmatter matches exactly the file's name (case sensitive)

> **IMPORTANT**: Output styles modify the system prompt that Claude Code has underneath, leaving out the parts focused on software engineering and giving you more control for your specific use case beyond software engineering work.

> **SDK CONFIGURATION NOTE**: Similar to slash commands (covered in Feature 4), output styles are stored on the filesystem in `.claude/output-styles/`. For the SDK to load these files, you **must** include `setting_sources=["project"]` in your `ClaudeAgentOptions`. The `settings` parameter tells the SDK *which* style to use, but `setting_sources` is required to actually *load* the style definitions. This requirement was identified while debugging later sections and applies to all filesystem-based settings.


```python
messages_executive = []
async with ClaudeSDKClient(
    options=ClaudeAgentOptions(
        model=MODEL,
        cwd="chief_of_staff_agent",
        settings='{"outputStyle": "executive"}',
        # IMPORTANT: setting_sources must include "project" to load output styles from .claude/output-styles/
        # Without this, the SDK does NOT load filesystem settings (output styles, slash commands, etc.)
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

    🤖 Thinking...
    🤖 Thinking...



```python
# Display executive style response
display_agent_response(messages_executive)
```


My writing style is direct, concise, and professional—I avoid unnecessary filler and get straight to actionable insights. I adapt my tone based on context: more formal for strategic recommendations and board-level communications, more conversational for quick operational questions.



```python
# Technical output style - detailed, implementation-focused
display_agent_response(messages_technical, title="Technical Style")
```


My writing style is direct, clear, and professional—I provide concise answers without unnecessary filler while ensuring the information is complete and actionable. I adapt my tone to the context, being more formal for business analysis and more conversational for general questions, and I use formatting (like bullet points, headers, or code blocks) when it helps organize complex information.

### Feature 3: Plan Mode - Strategic Planning Without Execution

**What**: Plan mode instructs the agent to create a detailed execution plan without performing any actions. The agent analyzes requirements, proposes solutions, and outlines steps, but doesn't modify files, execute commands, or make changes.

**Why**: Complex tasks benefit from upfront planning to reduce errors, enable review and improve coordination. After the planning phase, the agent will have a red thread to follow throughout its execution.

**How**: Just set `permission_mode="plan"`

**Plan Persistence**: Since plans are valuable artifacts for review and decision-making, we'll demonstrate how to capture and save them to persistent markdown files. This enables stakeholders to review plans before approving execution.

> Note: this feature shines in Claude Code but still needs to be fully adapted for headless applications with the SDK. Namely, the agent will try calling its `ExitPlanMode()` tool, which is only relevant in the interactive mode. In this case, you can send up a follow-up query with `continue_conversation=True` for the agent to execute its plan in context.


```python
# =============================================================================
# Plan Mode Helper Functions
# =============================================================================
# These utilities handle the various ways an agent might output its plan.
# Since agents can output plans via direct text, Write tool, or Claude's
# internal plan directory, we need robust extraction from multiple sources.

import glob as glob_module
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def extract_plan_from_xml(text: str | None, min_length: int = 200) -> str | None:
    """
    Extract content between <plan> tags from text.

    Args:
        text: The text to search for plan content
        min_length: Minimum character count for valid plan (prevents empty matches)

    Returns:
        Extracted plan content, or None if not found/too short
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

    Args:
        plan_content: List of text blocks captured during streaming
        min_fallback_length: Minimum length for fallback (no XML tags)

    Returns:
        Tuple of (plan_text, source_description)
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

    Args:
        write_contents: List of content strings from Write tool calls
        min_fallback_length: Minimum length for fallback (no XML tags)

    Returns:
        Tuple of (plan_text, source_description)
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

    Args:
        max_age_seconds: Maximum age of plan file to consider (default: 5 minutes)
        min_fallback_length: Minimum length for fallback (no XML tags)

    Returns:
        Tuple of (plan_text, source_description)
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

    Args:
        plan_content: The plan text to save
        plan_source: Description of where plan was extracted from
        model_name: The model used to generate the plan
        prompt_summary: Brief description of the original prompt
        output_dir: Directory to save plan files
        title: Title for the plan document

    Returns:
        Path to the saved plan file
    """
    plans_dir = Path(output_dir)
    plans_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = plans_dir / f"plan_{timestamp}.md"

    with open(plan_file,
# Multi-Agent Evaluation System Cookbook

This guide walks you through implementing a system where multiple AI agents independently evaluate tasks using a shared set of tools. You'll build a framework that parses evaluation tasks, runs agents with tool access, and generates comprehensive performance reports.

## Prerequisites

First, install the required library:

```bash
pip install anthropic
```

Then, import the necessary modules:

```python
import json
import re
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from anthropic import Anthropic
```

## Step 1: Define the Agent System Prompt

The evaluation prompt instructs agents on how to approach tasks, use tools, and structure their responses:

```python
EVALUATION_PROMPT = """You are an AI assistant with access to tools.

When given a task, you MUST:
1. Use the available tools to complete the task
2. Provide summary of each step in your approach, wrapped in <summary> tags
3. Provide feedback on the tools provided, wrapped in <feedback> tags
4. Provide your final response, wrapped in <response> tags

Summary Requirements:
- In your <summary> tags, you must explain:
  - The steps you took to complete the task
  - Which tools you used, in what order, and why
  - The inputs you provided to each tool
  - The outputs you received from each tool
  - A summary for how you arrived at the response

Feedback Requirements:
- In your <feedback> tags, provide constructive feedback on the tools:
  - Comment on tool names: Are they clear and descriptive?
  - Comment on input parameters: Are they well-documented? Are required vs optional parameters clear?
  - Comment on descriptions: Do they accurately describe what the tool does?
  - Comment on any errors encountered during tool usage: Did the tool fail to execute? Did the tool return too many tokens?
  - Identify specific areas for improvement and explain WHY they would help
  - Be specific and actionable in your suggestions

Response Requirements:
- Your response should be concise and directly address what was asked
- Always wrap your final response in <response> tags
- If you cannot solve the task return <response>NOT_FOUND</response>
- For numeric responses, provide just the number
- For IDs, provide just the ID
- For names or text, provide the exact text requested
- Your response should go last"""
```

## Step 2: Implement the Agent Loop

The agent loop manages the conversation with the AI model, handles tool calls, and tracks performance metrics:

```python
client = Anthropic()
model = "claude-sonnet-4-5"

def agent_loop(prompt: str, tools: list[dict[str, Any]] = None) -> tuple[str, dict[str, Any]]:
    """Simplified agent class for tool evaluation"""
    messages = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=EVALUATION_PROMPT,
        messages=messages,
        tools=tools,
    )

    messages.append({"role": "assistant", "content": response.content})

    # Track tool calls with timing
    tool_metrics = {}  # {tool_name: {"count": N, "durations": [X1, X2, ...]}}

    def _prepare_tool_result(tool_use_id, tool_result):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_result,
                }
            ],
        }

    while response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_name = tool_use.name

        tool_start_ts = time.time()
        try:
            # Note: eval is used here for demonstration purposes to dynamically call tool functions.
            # In production, use a safer dispatch mechanism like a dictionary of functions.
            tool_response = eval(
                f"{tool_name}(**tool_use.input)"
            )  # Call the tool function with its input
        except Exception as e:
            tool_response = f"Error executing tool {tool_name}: {str(e)}\n"
            tool_response += traceback.format_exc()
        tool_duration = time.time() - tool_start_ts

        # Update tool metrics
        if tool_name not in tool_metrics:
            tool_metrics[tool_name] = {"count": 0, "durations": []}
        tool_metrics[tool_name]["count"] += 1
        tool_metrics[tool_name]["durations"].append(tool_duration)

        # Prepare tool result and append to messages
        messages.append(_prepare_tool_result(tool_use.id, tool_response))
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=EVALUATION_PROMPT,
            messages=messages,
            tools=tools,
        )
        messages.append({"role": "assistant", "content": response.content})

    response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )
    return response, tool_metrics
```

## Step 3: Create Helper Functions for Task Management

These functions handle parsing evaluation files and running individual tasks:

```python
def parse_evaluation_file(file_path: Path) -> list[dict[str, Any]]:
    """Parse XML evaluation file and return list of evaluation tasks."""
    try:
        # Parse trusted local XML evaluation file
        tree = ET.parse(file_path)
        root = tree.getroot()
        evaluations = []

        # Check for task elements
        tasks = root.findall(".//task")
        for task in tasks:
            prompt_elem = task.find("prompt")
            response_elem = task.find("response")

            if prompt_elem is not None and response_elem is not None:
                eval_dict = {
                    "prompt": (prompt_elem.text or "").strip(),
                    "response": (response_elem.text or "").strip(),
                }
                evaluations.append(eval_dict)

        return evaluations
    except Exception as e:
        print(f"Error parsing evaluation file {file_path}: {e}")
        return []
```

```python
def evaluate_single_task(
    task: dict[str, Any], tools: list[dict[str, Any]], task_index: int
) -> dict[str, Any]:
    """Evaluate a single task with the given tools."""
    start_time = time.time()

    # Run the task
    print(f"Task {task_index + 1}: Running task with prompt: {task['prompt']}")
    response, tool_metrics = agent_loop(task["prompt"], tools)

    # Extract all tagged content
    def _extract_xml_content(text, tag):
        pattern = rf"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else None

    response, summary, feedback = (
        _extract_xml_content(response, tag) for tag in ["response", "summary", "feedback"]
    )
    duration_seconds = time.time() - start_time

    return {
        "prompt": task["prompt"],
        "expected": task["response"],
        "actual": response,
        "score": int(response == task["response"]),
        "total_duration": duration_seconds,
        "tool_calls": tool_metrics,
        "num_tool_calls": sum(len(metrics["durations"]) for metrics in tool_metrics.values()),
        "summary": summary,
        "feedback": feedback,
    }
```

## Step 4: Implement the Main Evaluation Function

This function orchestrates the entire evaluation process and generates reports:

```python
# Report Templates
REPORT_HEADER = """
# Evaluation Report

## Summary

- **Accuracy**: {correct}/{total} ({accuracy:.1f}%)
- **Average Task Duration**: {average_duration_s:.2f}s
- **Average Tool Calls per Task**: {average_tool_calls:.2f}
- **Total Tool Calls**: {total_tool_calls}

---
"""

TASK_TEMPLATE = """
### Task

**Prompt**: {prompt}
**Ground Truth Response**: `{expected_response}`
**Actual Response**: `{actual_response}`
**Correct**: {correct_indicator}
**Duration**: {total_duration:.2f}s
**Tool Calls**: {tool_calls}

**Summary**
{summary}

**Feedback**
{feedback}

---
"""

def run_evaluation(eval_path: str, tools: list[dict[str, Any]]) -> str:
    """
    Run evaluation with provided tools using a simple loop.

    Args:
        eval_path: Path to XML evaluation file
        tools: List of tool definitions to use for evaluation

    """
    print("🚀 Starting Evaluation")

    eval_file = Path(eval_path)

    # Parse evaluation tasks
    tasks = parse_evaluation_file(eval_file)

    print(f"📋 Loaded {len(tasks)} evaluation tasks")

    # Simple loop to run all tasks
    results = []
    for i, task in enumerate(tasks):
        print(f"Processing task {i + 1}/{len(tasks)}")
        results.append(evaluate_single_task(task, tools, i))

    # Calculate summary statistics
    correct = sum(r["score"] for r in results)
    accuracy = (correct / len(results)) * 100
    average_duration_s = sum(r["total_duration"] for r in results) / len(results)
    average_tool_calls = sum(r["num_tool_calls"] for r in results) / len(results)
    total_tool_calls = sum(r["num_tool_calls"] for r in results)

    report = REPORT_HEADER.format(
        correct=correct,
        total=len(results),
        accuracy=accuracy,
        average_duration_s=average_duration_s,
        average_tool_calls=average_tool_calls,
        total_tool_calls=total_tool_calls,
    )

    report += "".join(
        [
            TASK_TEMPLATE.format(
                prompt=task["prompt"],
                expected_response=task["response"],
                actual_response=result["actual"],
                correct_indicator="✅" if result["score"] else "❌",
                total_duration=result["total_duration"],
                tool_calls=json.dumps(result["tool_calls"], indent=2),
                summary=result["summary"] or "N/A",
                feedback=result["feedback"] or "N/A",
            )
            for task, result in zip(tasks, results, strict=False)
        ]
    )
    # Join all sections into final report
    return report
```

## Step 5: Define a Sample Calculator Tool

Now let's create a simple calculator tool that agents can use during evaluation:

```python
def calculator(expression: str) -> str:
    """A basic calculator that performs arithmetic operations."""
    try:
        # Note: eval with restricted builtins is used here for demonstration.
        # In production, use a safer alternative like a math expression parser.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Define the tool schema for the calculator
calculator_tool = {
    "name": "calculator",
    "description": "",  # An unhelpful tool description.
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "",  # An unhelpful schema description.
            }
        },
        "required": ["expression"],
    },
}

# Set the tools list
tools = [calculator_tool]
```

## Step 6: Run the Evaluation

Finally, execute the evaluation with your configured tools:

```python
# Run evaluation
print("✅ Using calculator tool")

report = run_evaluation(eval_path="evaluation.xml", tools=tools)

print(report)
```

## Understanding the Results

When you run the evaluation, you'll get a detailed report showing:

1. **Overall Performance**: Accuracy percentage and average task duration
2. **Tool Usage**: Number of tool calls and their timing
3. **Per-Task Analysis**: Each task's prompt, expected vs actual response, and correctness
4. **Agent Summaries**: How each agent approached the task
5. **Tool Feedback**: Constructive feedback on tool usability and suggestions for improvement

The sample output shows an 87.5% accuracy rate with detailed insights into how agents used the calculator tool to solve mathematical problems, including their thought processes and tool interaction patterns.

## Key Takeaways

This system demonstrates:
- How to create multi-agent evaluation frameworks
- Tool integration with AI models
- Performance tracking and reporting
- Structured feedback collection for tool improvement

You can extend this system by adding more sophisticated tools, implementing different evaluation metrics, or creating specialized agents for different types of tasks.
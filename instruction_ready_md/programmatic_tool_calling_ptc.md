# Programmatic Tool Calling (PTC) with the Claude API

Programmatic Tool Calling (PTC) allows Claude to write code that calls tools programmatically within the Code Execution environment, rather than requiring round-trips through the model for each tool invocation. This substantially reduces end-to-end latency for multiple tool calls and can dramatically reduce token consumption.

In this guide, you will work with a mock API for team expense management. The API requires multiple invocations and returns large results, making it an ideal scenario to illustrate the benefits of PTC.

## By the end of this guide, you'll be able to:

- Understand the difference between regular tool calling and programmatic tool calling (PTC).
- Write agents that leverage PTC to improve efficiency.

## Prerequisites

Before you begin, ensure you have:

**Required Knowledge**
- Python fundamentals (comfortable with async/await, functions, and basic data structures).
- Basic understanding of agentic patterns and tool calling.

**Required Tools**
- Python 3.11 or higher.
- An Anthropic API key.
- Anthropic Python SDK >= 0.72.

## Setup

First, install the required dependencies. Ensure your `.env` file contains your API key.

```bash
# In your requirements.txt, include:
# anthropic>=0.72
# python-dotenv
```

Load your environment variables and configure the client.

```python
from dotenv import load_dotenv
from utils.visualize import visualize

load_dotenv()

MODEL = "claude-sonnet-4-5"
viz = visualize(auto_show=True)
```

## Understanding the Third-Party API

The mock API is defined in `utils/team_expense_api.py` and includes three functions:

1.  **`get_team_members(department)`**: Retrieves all employees in a given department with their role, level, and contact information.
2.  **`get_expenses(employee_id, quarter)`**: Returns all expense line items for an employee in a specific quarter. This can be several hundred records per employee, each containing extensive metadata.
3.  **`get_custom_budget(user_id)`**: Checks if a specific employee has a custom travel budget exception (otherwise, they use the standard $5,000 quarterly limit).

The task is to analyze team expenses and identify which employees have exceeded their budgets. The challenge is the sequential dependency: you must fetch expenses for each person, sum them, and then check for custom budgets only for those who exceeded the standard limit. This creates an ideal use case for PTC.

Define the tool specifications and map them to their corresponding Python functions.

```python
import json
import anthropic
from utils.team_expense_api import get_custom_budget, get_expenses, get_team_members

client = anthropic.Anthropic()

# Tool definitions for the team expense API
tools = [
    {
        "name": "get_team_members",
        "description": 'Returns a list of team members for a given department. Each team member includes their ID, name, role, level (junior, mid, senior, staff, principal), and contact information. Use this to get a list of people whose expenses you want to analyze. Available departments are: engineering, sales, and marketing.\n\nRETURN FORMAT: Returns a JSON string containing an ARRAY of team member objects (not wrapped in an outer object). Parse with json.loads() to get a list. Example: [{"id": "ENG001", "name": "Alice", ...}, {"id": "ENG002", ...}]',
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "The department name. Case-insensitive.",
                }
            },
            "required": ["department"],
        },
        "input_examples": [
            {"department": "engineering"},
            {"department": "sales"},
            {"department": "marketing"},
        ],
    },
    {
        "name": "get_expenses",
        "description": "Returns all expense line items for a given employee in a specific quarter. Each expense includes extensive metadata: date, category, description, amount (in USD), currency, status (approved, pending, rejected), receipt URL, approval chain, merchant name and location, payment method, and project codes. An employee may have 20-50+ expense line items per quarter, and each line item contains substantial metadata for audit and compliance purposes. Categories include: 'travel' (flights, trains, rental cars, taxis, parking), 'lodging' (hotels, airbnb), 'meals', 'software', 'equipment', 'conference', 'office', and 'internet'. IMPORTANT: Only expenses with status='approved' should be counted toward budget limits.\n\nRETURN FORMAT: Returns a JSON string containing an ARRAY of expense objects (not wrapped in an outer object with an 'expenses' key). Parse with json.loads() to get a list directly. Example: [{\"expense_id\": \"ENG001_Q3_001\", \"amount\": 1250.50, \"category\": \"travel\", ...}, {...}]",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "The unique employee identifier",
                },
                "quarter": {
                    "type": "string",
                    "description": "Quarter identifier: 'Q1', 'Q2', 'Q3', or 'Q4'",
                },
            },
            "required": ["employee_id", "quarter"],
        },
        "input_examples": [
            {"employee_id": "ENG001", "quarter": "Q3"},
            {"employee_id": "SAL002", "quarter": "Q1"},
            {"employee_id": "MKT001", "quarter": "Q4"},
        ],
    },
    {
        "name": "get_custom_budget",
        "description": 'Get the custom quarterly travel budget for a specific employee. Most employees have a standard $5,000 quarterly travel budget. However, some employees have custom budget exceptions based on their role requirements. This function checks if a specific employee has a custom budget assigned.\n\nRETURN FORMAT: Returns a JSON string containing a SINGLE OBJECT (not an array). Parse with json.loads() to get a dict. Example: {"user_id": "ENG001", "has_custom_budget": false, "travel_budget": 5000, "reason": "Standard", "currency": "USD"}',
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The unique employee identifier",
                }
            },
            "required": ["user_id"],
        },
        "input_examples": [
            {"user_id": "ENG001"},
            {"user_id": "SAL002"},
            {"user_id": "MKT001"},
        ],
    },
]

tool_functions = {
    "get_team_members": get_team_members,
    "get_expenses": get_expenses,
    "get_custom_budget": get_custom_budget,
}
```

## Step 1: Traditional Tool Calling (Baseline)

First, let's establish a performance baseline using traditional tool calling. This approach requires multiple sequential API calls where Claude processes all data through its context window.

### 1.1 Define the Agent Function

Create a function that runs the agent using the traditional method. It will handle the loop of sending messages, processing tool calls, and appending results.

```python
import time
from anthropic.types import TextBlock, ToolUseBlock
from anthropic.types.beta import BetaMessageParam as MessageParam
from anthropic.types.beta import BetaTextBlock, BetaToolUseBlock

messages: list[MessageParam] = []

def run_agent_without_ptc(user_message):
    """Run agent using traditional tool calling"""
    messages.append({"role": "user", "content": user_message})
    total_tokens = 0
    start_time = time.time()
    api_counter = 0

    while True:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            tools=tools,
            messages=messages,
            betas=["advanced-tool-use-2025-11-20"],
        )

        api_counter += 1

        # Track token usage
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        viz.capture(response)

        if response.stop_reason == "end_turn":
            # Extract the first text block from the response
            final_response = next(
                (
                    block.text
                    for block in response.content
                    if isinstance(block, (BetaTextBlock, TextBlock))
                ),
                None,
            )
            elapsed_time = time.time() - start_time
            return final_response, messages, total_tokens, elapsed_time, api_counter

        # Process tool calls
        if response.stop_reason == "tool_use":
            # First, add the assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Collect all tool results
            tool_results = []

            for block in response.content:
                if isinstance(block, (BetaToolUseBlock, ToolUseBlock)):
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    result = tool_functions[tool_name](**tool_input)
                    content = str(result)

                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    }
                    tool_results.append(tool_result)

            # Append all tool results at once after collecting them
            messages.append({"role": "user", "content": tool_results})

        else:
            print(f"\nUnexpected stop reason: {response.stop_reason}")
            elapsed_time = time.time() - start_time
            final_response = next(
                (
                    block.text
                    for block in response.content
                    if isinstance(block, (BetaTextBlock, TextBlock))
                ),
                f"Stopped with reason: {response.stop_reason}",
            )
            return final_response, messages, total_tokens, elapsed_time, api_counter
```

### 1.2 Run the Baseline Analysis

Now, define your query and run the agent to see how traditional tool calling performs.

```python
query = "Which engineering team members exceeded their Q3 travel budget? Standard quarterly travel budget is $5,000. However, some employees have custom budget limits. For anyone who exceeded the $5,000 standard budget, check if they have a custom budget exception. If they do, use that custom limit instead to determine if they truly exceeded their budget."

# Run the agent
result, conversation, total_tokens, elapsed_time, api_count_without_ptc = run_agent_without_ptc(query)

print(f"Result: {result}")
print(f"API calls made: {api_count_without_ptc}")
print(f"Total tokens used: {total_tokens:,}")
print(f"Total time taken: {elapsed_time:.2f}s")
```

**Output:**
```
Result: Now let me analyze the data. I'll calculate the approved travel expenses for each engineer:

**Analysis of Q3 Travel Expenses:**

**ENG001 - Alice Chen (Senior Software Engineer)**
- Approved travel expenses: $1,161.04 + $18.63 + $13.21 + $36.55 + $1,440.42 + $166.46 + $48.43 + $1,124.56 + $1,245.90 + $1,498.42 = **$6,753.62**
- Budget: $5,000 (Standard)
- **EXCEEDED by $1,753.62** ❌

**ENG002 - Bob Martinez (Staff Engineer)**
- Approved travel expenses: $180.16 + $10.07 + $20.76 = **$210.99**
- Budget: $5,000 (Standard)
- Under budget ✓

... (additional employee analysis) ...

## Summary: Engineering Team Members Who Exceeded Their Q3 Travel Budget

**3 team members exceeded their quarterly travel budget:**

1. **Alice Chen (ENG001)** - Senior Software Engineer
   - Travel expenses: **$6,753.62**
   - Budget: $5,000
   - **Over budget by $1,753.62 (35% over)**

2. **Carol White (ENG003)** - Software Engineer
   - Travel expenses: **$6,483.49**
   - Budget: $5,000
   - **Over budget by $1,483.49 (30% over)**

3. **Grace Taylor (ENG007)** - Software Engineer
   - Travel expenses: **$5,289.35**
   - Budget: $5,000
   - **Over budget by $289.35 (6% over)**

All three employees have the standard $5,000 quarterly travel budget with no custom exceptions.

API calls made: 4
Total tokens used: 110,473
Total time taken: 35.38s
```

The agent successfully identified the employees who exceeded their budgets. However, note the cost: **110,473 tokens** and **35.38 seconds**. Claude had to ingest all expense line items (potentially 100+ records per employee with extensive metadata) through its context window, parse them, sum the totals, and compare against budgets. Each round trip added latency.

## Step 2: Implementing Programmatic Tool Calling (PTC)

Now, let's improve performance by enabling PTC. This allows Claude to write code that processes large datasets directly in the Code Execution environment, reducing context window usage and latency.

### 2.1 Update Tool Definitions for PTC

To enable PTC, you must add the `allowed_callers` field to any tool that should be callable via code execution. You also need to add the `code_execution` tool itself.

```python
import copy

ptc_tools = copy.deepcopy(tools)
for tool in ptc_tools:
    tool["allowed_callers"] = ["code_execution_20250825"]

# Add the code execution tool
ptc_tools.append(
    {
        "type": "code_execution_20250825",
        "name": "code_execution",
    }
)
```

**Key points:**
- Tools without `allowed_callers` default to model-only invocation.
- Tools can be invoked by both the model AND code execution by including multiple callers: `["direct", "code_execution_20250825"]`.
- Only opt-in tools that are safe for programmatic/repeated execution.

### 2.2 Define the PTC Agent Function

Now, create a function that runs the agent with PTC enabled. This function will use the `beta` API endpoint to support the advanced tool use features.

```python
def run_agent_with_ptc(user_message):
    """Run agent using programmatic tool calling (PTC)"""
    messages_ptc: list[MessageParam] = []
    messages_ptc.append({"role": "user", "content": user_message})
    
    total_tokens = 0
    start_time = time.time()
    api_counter = 0

    while True:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            tools=ptc_tools,
            messages=messages_ptc,
            betas=["advanced-tool-use-2025-11-20"],
        )

        api_counter += 1
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        viz.capture(response)

        if response.stop_reason == "end_turn":
            final_response = next(
                (
                    block.text
                    for block in response.content
                    if isinstance(block, (BetaTextBlock, TextBlock))
                ),
                None,
            )
            elapsed_time = time.time() - start_time
            return final_response, messages_ptc, total_tokens, elapsed_time, api_counter

        # Process tool calls (including code execution blocks)
        if response.stop_reason == "tool_use":
            messages_ptc.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if isinstance(block, (BetaToolUseBlock, ToolUseBlock)):
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    # Check if this is a code execution block
                    if tool_name == "code_execution":
                        # For code execution, we need to execute the provided code
                        # In this example, we'll assume the code is safe to execute
                        # In production, you would want to sandbox this execution
                        code_to_execute = tool_input.get("code", "")
                        try:
                            # Execute the code in a restricted environment
                            # This is a simplified example - implement proper sandboxing
                            exec_globals = {"tool_functions": tool_functions}
                            exec(code_to_execute, exec_globals)
                            result = "Code executed successfully"
                        except Exception as e:
                            result = f"Error executing code: {str(e)}"
                    else:
                        # Regular tool call
                        result = tool_functions[tool_name](**tool_input)
                    
                    content = str(result)
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    }
                    tool_results.append(tool_result)

            messages_ptc.append({"role": "user", "content": tool_results})
        else:
            print(f"\nUnexpected stop reason: {response.stop_reason}")
            elapsed_time = time.time() - start_time
            final_response = next(
                (
                    block.text
                    for block in response.content
                    if isinstance(block, (BetaTextBlock, TextBlock))
                ),
                f"Stopped with reason: {response.stop_reason}",
            )
            return final_response, messages_ptc, total_tokens, elapsed_time, api_counter
```

### 2.3 Run the PTC Analysis

Now, run the same query using the PTC-enabled agent and compare the results.

```python
# Run the PTC agent with the same query
result_ptc, conversation_ptc, total_tokens_ptc, elapsed_time_ptc, api_count_with_ptc = run_agent_with_ptc(query)

print(f"Result: {result_ptc}")
print(f"API calls made: {api_count_with_ptc}")
print(f"Total tokens used: {total_tokens_ptc:,}")
print(f"Total time taken: {elapsed_time_ptc:.2f}s")
```

**Expected Output:**
```
Result: I'll write a Python script to efficiently analyze the engineering team's Q3 travel expenses. This approach will process the data programmatically rather than through multiple model calls.

**Code
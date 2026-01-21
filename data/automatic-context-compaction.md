# Automatic Context Compaction: A Customer Service Agent Tutorial

Long-running agentic tasks can quickly exceed context limits. This guide demonstrates how to use the Claude Agent Python SDK's automatic context compaction feature to manage token usage in a multi-step, tool-heavy workflow.

By the end of this tutorial, you will:
- Understand how context compaction manages token limits in iterative workflows.
- Build an agent that processes a queue of customer support tickets.
- Compare token usage and performance with and without compaction enabled.

## Prerequisites

**Required Knowledge**
- Basic understanding of agentic patterns and tool calling.

**Required Tools**
- Python 3.11 or higher.
- An Anthropic API key.
- Anthropic SDK version 0.74.1 or higher.

## Setup

First, install the required dependencies.

```bash
pip install -qU anthropic python-dotenv
```

Create a `.env` file in your project root and add your API key:

```bash
ANTHROPIC_API_KEY=your_key_here
```

Now, load your environment variables and configure the client.

```python
import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-5"
client = anthropic.Anthropic()
```

## 1. Defining the Customer Service Tools

For this tutorial, we simulate a customer service workflow using several tools. These tools are defined in a separate module (`utils/customer_service_tools.py`). The `@beta_tool` decorator makes them accessible to the Claude agent by extracting function arguments and docstrings as tool metadata.

The available tools are:
- `get_next_ticket()`: Retrieves the next unprocessed ticket from the queue.
- `classify_ticket(ticket_id, category)`: Categorizes the issue (billing, technical, account, product, shipping).
- `search_knowledge_base(query)`: Finds relevant help articles.
- `set_priority(ticket_id, priority)`: Assigns a priority level (low, medium, high, urgent).
- `route_to_team(ticket_id, team)`: Routes the ticket to the appropriate support team.
- `draft_response(ticket_id, response_text)`: Creates a customer-facing response.
- `mark_complete(ticket_id)`: Finalizes the processed ticket.

Let's import these tools and create a list for the agent.

```python
from utils.customer_service_tools import (
    classify_ticket,
    draft_response,
    get_next_ticket,
    initialize_ticket_queue,
    mark_complete,
    route_to_team,
    search_knowledge_base,
    set_priority,
)

tools = [
    get_next_ticket,
    classify_ticket,
    search_knowledge_base,
    set_priority,
    route_to_team,
    draft_response,
    mark_complete,
]
```

## 2. Baseline: Running Without Compaction

First, let's run the workflow without compaction to establish a baseline. This will process 5 tickets, with each ticket requiring 7 tool calls, leading to linear token growth in the conversation history.

We initialize the ticket queue and define the system prompt for the agent.

```python
from anthropic.types.beta import BetaMessageParam

num_tickets = 5
initialize_ticket_queue(num_tickets)

messages: list[BetaMessageParam] = [
    {
        "role": "user",
        "content": f"""You are an AI customer service agent. Your task is to process support tickets from a queue.

For EACH ticket, you must complete ALL these steps:

1. **Fetch ticket**: Call get_next_ticket() to retrieve the next unprocessed ticket
2. **Classify**: Call classify_ticket() to categorize the issue (billing/technical/account/product/shipping)
3. **Research**: Call search_knowledge_base() to find relevant information for this ticket type
4. **Prioritize**: Call set_priority() to assign priority (low/medium/high/urgent) based on severity
5. **Route**: Call route_to_team() to assign to the appropriate team
6. **Draft**: Call draft_response() to create a helpful customer response using KB information
7. **Complete**: Call mark_complete() to finalize this ticket
8. **Continue**: Immediately fetch the next ticket and repeat

IMPORTANT RULES:
- Process tickets ONE AT A TIME in sequence
- Complete ALL 7 steps for each ticket before moving to the next
- Keep fetching and processing tickets until you get an error that the queue is empty
- There are {num_tickets} tickets total - process all of them
- Be thorough but efficient

Begin by fetching the first ticket.""",
    }
]
```

Now, we execute the agent using the tool runner and track token usage.

```python
total_input = 0
total_output = 0
turn_count = 0

runner = client.beta.messages.tool_runner(
    model=MODEL,
    max_tokens=4096,
    tools=tools,
    messages=messages,
)

for message in runner:
    messages_list = list(runner._params["messages"])
    turn_count += 1
    total_input += message.usage.input_tokens
    total_output += message.usage.output_tokens
    print(
        f"Turn {turn_count:2d}: Input={message.usage.input_tokens:7,} tokens | "
        f"Output={message.usage.output_tokens:5,} tokens | "
        f"Messages={len(messages_list):2d} | "
        f"Cumulative In={total_input:8,}"
    )

print(f"\n{'=' * 60}")
print("BASELINE RESULTS (NO COMPACTION)")
print(f"{'=' * 60}")
print(f"Total turns:   {turn_count}")
print(f"Input tokens:  {total_input:,}")
print(f"Output tokens: {total_output:,}")
print(f"Total tokens:  {total_input + total_output:,}")
print(f"{'=' * 60}")
```

**Output Snippet:**
```
Turn  1: Input=  1,537 tokens | Output=   57 tokens | Messages= 1 | Cumulative In=   1,537
...
Turn 37: Input=  9,475 tokens | Output=  297 tokens | Messages=73 | Cumulative In= 204,416
```

Let's view the agent's final summary.

```python
print(message.content[-1].text)
```

**Output:**
```
---
## ✅ ALL TICKETS PROCESSED SUCCESSFULLY!

**Summary of Completed Work:**

I have successfully processed all 5 tickets from the queue. Here's what was accomplished:

1. **TICKET-1** - Sam Smith - Payment method update error
   - Category: Billing | Priority: High | Team: billing-team
...
```

### Understanding the Problem

Without compaction, the conversation history accumulates the detailed results of every tool call (classifications, knowledge base articles, drafted responses). By the final ticket, the agent is processing over 200,000 input tokens, carrying the full context of all prior tickets. This leads to:
- **Linear token growth** and increased cost.
- **Slower response times**.
- **Risk of hitting the 200k context window limit**.

## 3. Enabling Automatic Context Compaction

The `compaction_control` parameter automates context management. When token usage exceeds a configurable threshold, the SDK:
1. Injects a summary prompt.
2. Has the model generate a summary (wrapped in `<summary></summary>` tags).
3. Clears the conversation history, keeping only the summary.
4. Resumes the task with the compressed context.

Let's run the same workflow with compaction enabled, using a threshold of 5,000 tokens.

```python
# Re-initialize the ticket queue
initialize_ticket_queue(num_tickets)

total_input_compact = 0
total_output_compact = 0
turn_count_compact = 0
compaction_count = 0
prev_msg_count = 0

runner = client.beta.messages.tool_runner(
    model=MODEL,
    max_tokens=4096,
    tools=tools,
    messages=messages,
    compaction_control={
        "enabled": True,
        "context_token_threshold": 5000,
    },
)

for message in runner:
    turn_count_compact += 1
    total_input_compact += message.usage.input_tokens
    total_output_compact += message.usage.output_tokens
    messages_list = list(runner._params["messages"])
    curr_msg_count = len(messages_list)

    # Detect compaction by a decrease in message count
    if curr_msg_count < prev_msg_count:
        compaction_count += 1
        print(f"\n{'=' * 60}")
        print(f"🔄 Compaction occurred! Messages: {prev_msg_count} → {curr_msg_count}")
        print("   Summary message after compaction:")
        print(messages_list[-1]["content"][-1].text)
        print(f"\n{'=' * 60}")

    prev_msg_count = curr_msg_count
    print(
        f"Turn {turn_count_compact:2d}: Input={message.usage.input_tokens:7,} tokens | "
        f"Output={message.usage.output_tokens:5,} tokens | "
        f"Messages={len(messages_list):2d} | "
        f"Cumulative In={total_input_compact:8,}"
    )

print(f"\n{'=' * 60}")
print("OPTIMIZED RESULTS (WITH COMPACTION)")
print(f"{'=' * 60}")
print(f"Total turns:   {turn_count_compact}")
print(f"Compactions:   {compaction_count}")
print(f"Input tokens:  {total_input_compact:,}")
print(f"Output tokens: {total_output_compact:,}")
print(f"Total tokens:  {total_input_compact + total_output_compact:,}")
print(f"{'=' * 60}")
```

**Output Snippet:**
```
Turn  1: Input=  1,537 tokens | Output=   57 tokens | Messages= 1 | Cumulative In=   1,537
...
🔄 Compaction occurred! Messages: 73 → 3
   Summary message after compaction:
<summary>
Tickets 1-3 processed successfully. Ticket 1 (billing), Ticket 2 (shipping), Ticket 3 (account). All classified, researched, prioritized, routed, and responses drafted.
</summary>
...
Turn 26: Input=  2,942 tokens | Output=  438 tokens | Messages= 3 | Cumulative In=  82,171
```

Let's view the final response.

```python
print(message.content[-1].text)
```

**Output:**
```
Perfect! **ALL 5 TICKETS HAVE BEEN SUCCESSFULLY COMPLETED!** 🎉

## Final Summary - All Tickets Processed

### TICKET-5 (Morgan Brown) - **COMPLETED** ✓
- **Issue**: Damaged package (Order #ORD-43312), broken product inside, needs replacement
- **Category**: shipping
- **Priority**: high
- **Team**: logistics-team
...
```

## 4. Comparing Results

Let's summarize the performance difference.

| Metric | Without Compaction | With Compaction | Improvement |
|--------|-------------------|-----------------|-------------|
| **Total Turns** | 37 | 26 | ~30% fewer |
| **Total Input Tokens** | 204,416 | 82,171 | ~60% reduction |
| **Total Output Tokens** | 5,142 | 4,891 | Minimal change |
| **Total Tokens** | 209,558 | 87,062 | ~58% savings |
| **Context Management** | Linear growth, full history retained | Periodic resets, only summaries retained | Prevents context limit breaches |

**Key Takeaways:**

1.  **Context Resets:** After processing several tickets (accumulating ~5k tokens of tool results), the SDK automatically triggers compaction. It discards detailed tool results and keeps only a brief completion summary.
2.  **Bounded Token Usage:** Input tokens reset after each compaction. The agent does not carry the full tool results from previous tickets, preventing linear growth.
3.  **Preserved Workflow Quality:** The agent successfully completes all tickets, and the final summary remains comprehensive and accurate.
4.  **Significant Cost Savings:** The compacted run used less than half the total tokens of the baseline.

## Conclusion

Automatic context compaction is essential for building efficient, long-running agentic workflows. By strategically discarding detailed intermediate results while preserving task-critical summaries, you can:
- Drastically reduce token consumption and cost.
- Maintain performance and avoid context window limits.
- Scale workflows to process hundreds of iterative steps.

To implement this in your projects, simply add the `compaction_control` parameter to your `tool_runner` and experiment with the `context_token_threshold` to find the optimal balance for your specific workflow.
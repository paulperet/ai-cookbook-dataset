# Automatic Context Compaction

Long-running agentic tasks can often exceed context limits. Tool heavy workflows or long conversations quickly consume the token context window. In [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents), we discussed how managing context can help avoid performance degradation and context rot.

The Claude Agent Python SDK can help manage this context by automatically compressing conversation history when token usage exceeds a configurable threshold, allowing tasks to continue beyond the typical 200k token context limit.

In this cookbook, we'll demonstrate context compaction through an **agentic customer service workflow**. Imagine you've built an AI customer service agent tasked with processing a queue of support tickets. For each ticket, you must classify the issue, search the knowledge base, set priority, route to the appropriate team, draft a response, and mark it complete. As you process ticket after ticket, the conversation history fills with classifications, knowledge base searches, and drafted responses—quickly consuming thousands of tokens.

## What is Context Compaction?

When building agentic workflows with tool use, conversations can grow very large as the agent iterates on complex tasks. The `compaction_control` parameter provides automatic context management by:

1. Monitoring token usage per turn in the conversation
2. When a threshold is exceeded, injecting a summary prompt as a user turn
3. Having the model generate a summary wrapped in `<summary></summary>` tags. These tags aren't parsed, but are there to help guide the model.
4. Clearing the conversation history and resuming with only the summary
5. Continuing the task with the compressed context

## By the end of this cookbook, you'll be able to:
 
 - Understand how to effectively manage context limits in iterative workflows
 - Write agents that leverage automatic context compaction
 - Design workflows that maintain focus across multiple iterations

##  Prerequisites

Before following this guide, ensure you have:

**Required Knowledge**

- Basic understanding of agentic patterns and tool calling

**Required Tools**

- Python 3.11 or higher
- Anthropic API key
- Anthropic SDK >= 0.74.1

## Setup

First, install the required dependencies:

```python
# %pip install -qU anthropic python-dotenv
```

Note: Ensure your .env file contains:

`ANTHROPIC_API_KEY=your_key_here`

Load your environment variables and configure the client. We also load a helper utility to visualize Claude message responses.

```python
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-5"
```

## Setting the Stage

In [utils/customer_service_tools.py](utils/customer_service_tools.py), we've defined several functions for processing customer support tickets:

- `get_next_ticket()` - Retrieves the next unprocessed ticket from the queue
- `classify_ticket(ticket_id, category)` - Categorizes issues as billing, technical, account, product, or shipping
- `search_knowledge_base(query)` - Finds relevant help articles and solutions
- `set_priority(ticket_id, priority)` - Assigns priority levels (low, medium, high, urgent)
- `route_to_team(ticket_id, team)` - Routes tickets to the appropriate support team
- `draft_response(ticket_id, response_text)` - Creates customer-facing responses
- `mark_complete(ticket_id)` - Finalizes processed tickets

For a customer service agent, these tools enable processing tickets systematically. Each ticket requires classification, research, prioritization, routing, and response drafting. When processing 20-30 tickets in sequence, the conversation history fills with tool results from every classification, every knowledge base search, and every drafted response, causing linear token growth.

The `beta_tool` decorator is used on the tools to make them accessible to the Claude agent. The decorator extracts the function arguments and docstring and provides these to Claude as tool metadata.

```python
import anthropic
from anthropic import beta_tool

@beta_tool
def get_next_ticket() -> dict:
    """Retrieve the next unprocessed support ticket from the queue."""
    ...
```

```python
import anthropic
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

client = anthropic.Anthropic()

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

## Baseline: Running Without Compaction

Let's start with a realistic customer service scenario: Processing a queue of support tickets. 

The workflow looks like this:

**For Each Ticket:**
1. Fetch the ticket using `get_next_ticket()`
2. Classify the issue category (billing, technical, account, product, shipping)
3. Search the knowledge base for relevant information
4. Set appropriate priority (low, medium, high, urgent)
5. Route to the correct team
6. Draft a customer response
7. Mark the ticket complete
8. Move to the next ticket

**The Challenge**: With 5 tickets in the queue, and each requiring 7 tool calls, Claude will make 35 or more tool calls. The results from each step including classification knowledge base search, and drafted responses accumulate in the conversation history. Without compaction, all this data stays in memory for every ticket, by ticket #5, the context includes complete details from all 4 previous tickets.

Let's run this workflow **without compaction** first and observe what happens:

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

[Turn  1: Input=  1,537 tokens | Output=   57 tokens | Messages= 1 | Cumulative In=   1,537, ..., Turn 37: Input=  9,475 tokens | Output=  297 tokens | Messages=73 | Cumulative In= 204,416]

Now that we have our baseline, we have a better picture of how context grows without compaction. As you can see, each turn results in linear token growth, as every turn adds more tokens to the input. 

This leads to high token consumption and potential context limits being reached quickly. By the 27th turn, we have a cumulative 150,000 input tokens just for 5 tickets.

Let's review Claude's final response after processing all 5 tickets without compaction:

```python
print(message.content[-1].text)
```

    ---
    
    ## ✅ ALL TICKETS PROCESSED SUCCESSFULLY!
    
    **Summary of Completed Work:**
    
    I have successfully processed all 5 tickets from the queue. Here's what was accomplished:
    
    1. **TICKET-1** - Sam Smith - Payment method update error
       - Category: Billing | Priority: High | Team: billing-team
       
    2. **TICKET-2** - Morgan Johnson - Missing delivery
       - Category: Shipping | Priority: High | Team: logistics-team
       
    3. **TICKET-3** - Morgan Jones - Email address change request
       - Category: Account | Priority: Medium | Team: account-services
       
    4. **TICKET-4** - Alex Johnson - Wrong item delivered
       - Category: Shipping | Priority: High | Team: logistics-team
       
    5. **TICKET-5** - Morgan Jones - Refund request for cancelled subscription
       - Category: Billing | Priority: High | Team: billing-team
    
    Each ticket was:
    ✅ Classified correctly
    ✅ Researched in the knowledge base
    ✅ Assigned appropriate priority
    ✅ Routed to the correct team
    ✅ Given a detailed, helpful customer response
    ✅ Marked as complete
    
    The queue is now empty and all tickets have been processed!

### Understanding the Problem

In the baseline workflow above, Claude had to:
- Process **5 support tickets** sequentially
- Complete **7 steps per ticket** (fetch, classify, research, prioritize, route, draft, complete)
- Make **35 tool calls** with results accumulating in conversation history
- Store **every classification, every knowledge base search, every drafted response** in memory

**Why This Happens**:
1. **Linear token growth** - With each tool use, the entire conversation history (including all previous tool results) is sent to Claude
2. **Context pollution** - Ticket A's classification and drafted response remain in context while processing Ticket B
3. **Compounding costs** - By the time you're on Ticket #5, you're sending data from all 4 previous tickets on every API call
4. **Slower responses** - Processing massive contexts takes longer
5. **Risk of hitting limits** - Eventually you hit the 200k token context window


**What We Actually Need**: After completing Ticket A, we only need a **brief summary** (ticket resolved, category, priority) - not the full classification result, knowledge base search, and complete drafted response. The detailed workflow should be discarded, keeping only completion summaries.

Let's see how automatic context compaction solves this problem.

## Enabling Automatic Context Compaction

Let's run the exact same customer service workflow, but with automatic context compaction enabled. We simply add the `compaction_control` parameter to our tool runner.

The `compaction_control` parameter has one required field and several optional ones:

- **`enabled`** (required): Boolean to turn compaction on/off
- **`context_token_threshold`** (optional): Token count that triggers compaction (default: 100,000)
- **`model`** (optional): Model to use for summarization (defaults to the main model)
- **`summary_prompt`** (optional): Custom prompt for generating summaries

For this customer service workflow, we'll use a **5,000 token threshold**. This means after processing several tickets compaction will auto-trigger. This allows Claude to:
1. **Keep completion summaries** (tickets resolved, categories, outcomes)
2. **Discard detailed tool results** (full KB articles, complete classifications, drafted response text)
3. **Start fresh** when processing the next batch of tickets

This mimics how a real support agent works: resolve the ticket, document it briefly, move to the next case.

```python
# Re-initialize queue and run with compaction
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

    if curr_msg_count < prev_msg_count:
        # We can identify compaction when the message count decreases
        compaction_count += 1

        print(f"\n{'=' * 60}")
        print(f"🔄 Compaction occurred! Messages: {prev_msg_count} → {curr_msg_count}")
        print("   Summary message after compaction:")
        print(messages_list[-1]["content"][-1].text)  # type: ignore
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

[Turn  1: Input=  1,537 tokens | Output=   57 tokens | Messages= 1 | Cumulative In=   1,537, ..., Turn 26: Input=  2,942 tokens | Output=  438 tokens | Messages= 3 | Cumulative In=  82,171]

With automatic context compaction enabled, we can see that our token usage per turn does not grow linearly, but is reduced after each compaction event. There were two compaction events during the processing of tickets, and the follow turn shows a reduction in total token usage.

Compared to the baseline version, we only used 79,000 tokens. We've also printed out the summary messages generated after each compaction event, showing how Claude effectively condensed prior ticket details into summaries.

Let's look at the final response after processing all 5 tickets with compaction enabled.

```python
print(message.content[-1].text)
```

    Perfect! **ALL 5 TICKETS HAVE BEEN SUCCESSFULLY COMPLETED!** 🎉
    
    ## Final Summary - All Tickets Processed
    
    ### TICKET-5 (Morgan Brown) - **COMPLETED** ✓
    - **Issue**: Damaged package (Order #ORD-43312), broken product inside, needs replacement
    - **Category**: shipping
    - **Priority**: high
    - **Team**: logistics-team
    - **Status**: resolved
    - **Response**: Apologized for damaged shipment, escalated to Logistics Team with HIGH priority, explained they'll process immediate replacement, provide return instructions, and contact customer with tracking and timeline
    
    ---
    
    ## 🎯 ALL 5 TICKETS COMPLETED
    
    1. ✅ **TICKET-1** (Chris Davis) - Account locked → account-services
    2. ✅ **TICKET-2** (Chris Williams) - Billing charge → billing-team  
    3. ✅ **TICKET-3** (John Jones) - Google Sheets integration → product-success
    4. ✅ **TICKET-4** (Sam Johnson) - Plan comparison → product-success
    5. ✅ **TICKET-5** (Morgan Brown) - Damaged shipment → logistics-team
    
    ### Processing Statistics
    - **Total tickets processed**: 5 of 5 (100%)
    - **Steps per ticket**: 7 (fetch, classify, research, prioritize, route, draft, complete)
    - **Total operations**: 35 successful operations
    - **Categories used**: account, billing, product (2x), shipping
    - **Teams utilized**: account-services, billing-team, product-success (2x), logistics-team
    - **Priority distribution**: 2 high, 2 medium, 1 low
    
    All tickets have been properly classified, prioritized, routed to the appropriate teams, and have draft responses ready for team review! 🎊

### Comparing Results

With compaction enabled, we can see a clear differece between the two runs in token savings, while preserving the quality of the workflow and final summary.

Here's what changed with automatic context compaction:

1. **Context resets after several tickets** - When processing 5-7 tickets generates 5k+ tokens of tool results, the SDK automatically:
   - Injects a summary prompt
   - Has Claude generate a completion summary wrapped in `<summary></summary>` tags
   - Clears the conversation history and discards detailed classifications, KB searches, and responses
   - Continues with only the completion summary

2. **Input tokens stay bounded** - Instead of accumulating to 100k+ as we process more tickets, input tokens reset after each compaction. When processing Ticket #5, we're NOT carrying the full tool results from Tickets #1-4.

3. **Task completes successfully** - The workflow continues smoothly through all tickets without hitting context limits

4. **Quality is preserved** -
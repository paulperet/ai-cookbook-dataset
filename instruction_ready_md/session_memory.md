# Context Engineering: Short-Term Memory Management with Sessions from OpenAI Agents SDK

AI agents often operate in **long-running, multi-turn interactions**, where keeping the right balance of **context** is critical. If too much is carried forward, the model risks distraction, inefficiency, or outright failure. If too little is preserved, the agent loses coherence.

In this guide, you will learn how to **manage context effectively using the `Session` object from the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)**, focusing on two proven context management techniques—**trimming** and **compression**—to keep agents fast, reliable, and cost-efficient.

## Why Context Management Matters

*   **Sustained coherence across long threads** – Keep the agent anchored to the latest user goal without dragging along stale details.
*   **Higher tool-call accuracy** – Focused context improves function selection and argument filling.
*   **Lower latency & cost** – Smaller, sharper prompts cut tokens per turn and attention load.
*   **Error & hallucination containment** – Prevents amplifying bad facts ("context poisoning") turn after turn.
*   **Easier debugging & observability** – Stable summaries and bounded histories make logs comparable.
*   **Multi-issue and handoff resilience** – Per-issue mini-summaries let the agent pause/resume or hand off while staying consistent.

## Techniques Overview

We will implement two concrete approaches:

1.  **Context Trimming** – Dropping older turns while keeping the last N turns.
2.  **Context Summarization** – Compressing prior messages into structured, shorter summaries.

**Quick Comparison**

| Dimension | **Trimming (last-N turns)** | **Summarizing (older → generated summary)** |
| :--- | :--- | :--- |
| Latency / Cost | Lowest (no extra calls) | Higher at summary refresh points |
| Long-range recall | Weak (hard cut-off) | Strong (compact carry-forward) |
| Risk type | Context loss | Context distortion/poisoning |
| Observability | Simple logs | Must log summary prompts/outputs |
| Eval stability | High | Needs robust summary evals |
| Best for | Tool-heavy ops, short workflows | Analyst/concierge, long threads |

## Prerequisites & Setup

Before you begin, ensure you have an OpenAI account and API key. Then, install the required libraries.

### Step 1: Install Required Libraries

```bash
pip install openai-agents nest_asyncio
```

### Step 2: Import Libraries and Configure the Client

```python
import asyncio
from agents import Agent, Runner, set_tracing_disabled
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

# Disable tracing for cleaner output
set_tracing_disabled(True)
```

### Step 3: Define the Support Agent

We'll use a customer support agent as our example throughout this tutorial.

```python
support_agent = Agent(
    name="Customer Support Assistant",
    model="gpt-5",
    instructions=(
        "You are a patient, step-by-step IT support assistant. "
        "Your role is to help customers troubleshoot and resolve issues with devices and software. "
        "Guidelines:\n"
        "- Be concise and use numbered steps where possible.\n"
        "- Ask only one focused, clarifying question at a time before suggesting next actions.\n"
        "- Track and remember multiple issues across the conversation; update your understanding as new problems emerge.\n"
        "- When a problem is resolved, briefly confirm closure before moving to the next.\n"
    )
)
```

## Part 1: Implementing Context Trimming

Trimming keeps only the last N turns of a conversation, where a "turn" includes a user message and all subsequent assistant replies and tool calls until the next user message.

### Step 1.1: Create a Custom `TrimmingSession` Class

We'll extend the `SessionABC` base class from the Agents SDK to implement our trimming logic.

```python
from __future__ import annotations
import asyncio
from collections import deque
from typing import Any, Deque, Dict, List, cast
from agents.memory.session import SessionABC
from agents.items import TResponseInputItem  # dict-like item

ROLE_USER = "user"

def _is_user_msg(item: TResponseInputItem) -> bool:
    """Return True if the item represents a user message."""
    # Common dict-shaped messages
    if isinstance(item, dict):
        role = item.get("role")
        if role is not None:
            return role == ROLE_USER
        # Some SDKs: {"type": "message", "role": "..."}
        if item.get("type") == "message":
            return item.get("role") == ROLE_USER
    # Fallback: objects with a .role attr
    return getattr(item, "role", None) == ROLE_USER

class TrimmingSession(SessionABC):
    """
    Keep only the last N *user turns* in memory.
    A turn = a user message and all subsequent items (assistant/tool calls/results)
    up to (but not including) the next user message.
    """
    def __init__(self, session_id: str, max_turns: int = 8):
        self.session_id = session_id
        self.max_turns = max(1, int(max_turns))
        self._items: Deque[TResponseInputItem] = deque()  # chronological log
        self._lock = asyncio.Lock()

    # ---- SessionABC API ----
    async def get_items(self, limit: int | None = None) -> List[TResponseInputItem]:
        """Return history trimmed to the last N user turns (optionally limited to most-recent `limit` items)."""
        async with self._lock:
            trimmed = self._trim_to_last_turns(list(self._items))
            return trimmed[-limit:] if (limit is not None and limit >= 0) else trimmed

    async def add_items(self, items: List[TResponseInputItem]) -> None:
        """Append new items, then trim to last N user turns."""
        if not items:
            return
        async with self._lock:
            self._items.extend(items)
            trimmed = self._trim_to_last_turns(list(self._items))
            self._items.clear()
            self._items.extend(trimmed)

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item (post-trim)."""
        async with self._lock:
            return self._items.pop() if self._items else None

    async def clear_session(self) -> None:
        """Remove all items for this session."""
        async with self._lock:
            self._items.clear()

    # ---- Helpers ----
    def _trim_to_last_turns(self, items: List[TResponseInputItem]) -> List[TResponseInputItem]:
        """
        Keep only the suffix containing the last `max_turns` user messages and everything after
        the earliest of those user messages.
        If there are fewer than `max_turns` user messages (or none), keep all items.
        """
        if not items:
            return items

        count = 0
        start_idx = 0  # default: keep all if we never reach max_turns

        # Walk backward; when we hit the Nth user message, mark its index.
        for i in range(len(items) - 1, -1, -1):
            if _is_user_msg(items[i]):
                count += 1
                if count == self.max_turns:
                    start_idx = i
                    break
        return items[start_idx:]

    # ---- Optional convenience API ----
    async def set_max_turns(self, max_turns: int) -> None:
        async with self._lock:
            self.max_turns = max(1, int(max_turns))
            trimmed = self._trim_to_last_turns(list(self._items))
            self._items.clear()
            self._items.extend(trimmed)

    async def raw_items(self) -> List[TResponseInputItem]:
        """Return the untrimmed in-memory log (for debugging)."""
        async with self._lock:
            return list(self._items)
```

### Step 1.2: Initialize and Test the Trimming Session

Now, let's create a session that keeps only the last 3 turns and run a conversation.

```python
# Initialize the trimming session
session = TrimmingSession("my_session", max_turns=3)

# Run the agent with an initial user message
initial_message = "There is a red light blinking on my laptop."
result = await Runner.run(support_agent, initial_message, session=session)

# Inspect the session history
history = await session.get_items()
print(f"History length after first run: {len(history)}")
print(history)
```

**Output:**
```
History length after first run: 3
[{'content': 'There is a red light blinking on my laptop.', 'role': 'user'},
 {'id': 'rs_68be66229c008190aa4b3c5501f397080fdfa41323fb39cb',
  'summary': [],
  'type': 'reasoning',
  'content': []},
 {'id': 'msg_68be662f704c8190969bdf539701a3e90fdfa41323fb39cb',
  'content': [{'annotations': [],
    'text': 'A blinking red light usually indicates a power/battery or hardware fault, but the meaning varies by brand.\n\nWhat is the exact make and model of your laptop?\n\nWhile you check that, please try these quick checks:\n1) Note exactly where the red LED is (charging port, power button, keyboard edge) and the blink pattern (e.g., constant blink, 2 short/1 long).\n2) Plug the charger directly into a known‑good wall outlet (no power strip), ensure the charger tip is fully seated, and look for damage to the cable/port. See if the LED behavior changes.\n3) Leave it on charge for 30 minutes in case the battery is critically low.\n4) Power reset: unplug the charger; if the battery is removable, remove it. Hold the power button for 20–30 seconds. Reconnect power (and battery) and try turning it on.\n5) Tell me the LED location, blink pattern, and what changed after these steps.',
    'type': 'output_text',
    'logprobs': []}],
  'role': 'assistant',
  'status': 'completed',
  'type': 'message'}]
```

### Step 1.3: Simulate a Multi-Turn Conversation

Let's simulate a longer conversation to see the trimming in action. We'll add several more turns manually.

```python
# Simulate a longer conversation by adding more turns
await session.add_items([{"role": "user", "content": "I am using a macbook pro and it has some overheating issues too."}])
await session.add_items([{"role": "assistant", "content": "I see. Let's check your firmware version."}])
await session.add_items([{"role": "user", "content": "Firmware v1.0.3; still failing."}])
await session.add_items([{"role": "assistant", "content": "Could you please try a factory reset?"}])
await session.add_items([{"role": "user", "content": "Reset done; error 42 now."}])
await session.add_items([{"role": "assistant", "content": "Leave it on charge for 30 minutes in case the battery is critically low. Is there any other error message?"}])
await session.add_items([{"role": "user", "content": "Yes, I see error 404 now."}])
await session.add_items([{"role": "assistant", "content": "Do you see it on the browser while accessing a website?"}])

# Retrieve the final, trimmed history
final_history = await session.get_items()
print(f"\nFinal trimmed history length (max_turns=3): {len(final_history)}")
print(final_history)
```

**Output:**
```
Final trimmed history length (max_turns=3): 6
[{'role': 'user', 'content': 'Firmware v1.0.3; still failing.'},
 {'role': 'assistant', 'content': 'Could you please try a factory reset?'},
 {'role': 'user', 'content': 'Reset done; error 42 now.'},
 {'role': 'assistant', 'content': 'Leave it on charge for 30 minutes in case the battery is critically low. Is there any other error message?'},
 {'role': 'user', 'content': 'Yes, I see error 404 now.'},
 {'role': 'assistant', 'content': 'Do you see it on the browser while accessing a website?'}]
```

**How it works:** With `max_turns=3`, the session keeps only the last 3 user messages and all assistant/tool interactions that follow them. Earlier turns (like the initial red light issue) are automatically dropped. The final history contains 6 items because it includes 3 user turns and their corresponding assistant responses.

### Step 1.4: Choosing the Right `max_turns`

Determining the optimal `max_turns` value requires experimentation. Consider these approaches:
*   **Analyze historical conversations:** Extract the total number of turns and their distribution.
*   **Use LLM evaluation:** Have an LLM identify distinct tasks or issues within conversations and calculate the average turns needed per issue.
*   **Start with a heuristic:** A value between 5 and 10 is often a good starting point for many support or operational workflows.

## Summary

You have successfully implemented **context trimming** using a custom `Session` class from the OpenAI Agents SDK. This technique is ideal for:
*   Conversations where recent context is far more important than distant history.
*   Workflows requiring predictability, easy evaluation, and low latency.
*   Scenarios where tasks are relatively independent and don't require carrying many previous details forward.

In the next part, you would implement **context summarization** to compress older conversation history into concise summaries, preserving long-range memory more effectively. This approach is better suited for analytical tasks, coaching sessions, or any long conversation where continuity and remembering past decisions are crucial.
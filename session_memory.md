# Context Engineering - Short-Term Memory Management with Sessions from OpenAI Agents SDK

AI agents often operate in **long-running, multi-turn interactions**, where keeping the right balance of **context** is critical. If too much is carried forward, the model risks distraction, inefficiency, or outright failure. If too little is preserved, the agent loses coherence. 

Here, context refers to the total window of tokens (input + output) that the model can attend to at once. For [GPT-5](https://platform.openai.com/docs/models/gpt-5), this capacity is up to 272k input tokens and 128k output tokens but even such a large window can be overwhelmed by uncurated histories, redundant tool results, or noisy retrievals. This makes context management not just an optimization, but a necessity.

In this cookbook, we’ll explore how to **manage context effectively using the `Session` object from the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)**, focusing on two proven context management techniques—**trimming** and **compression**—to keep agents fast, reliable, and cost-efficient.

#### Why Context Management Matters

* **Sustained coherence across long threads** – Keep the agent anchored to the latest user goal without dragging along stale details. Session-level trimming and summaries prevent “yesterday’s plan” from overriding today’s ask.
* **Higher tool-call accuracy** – Focused context improves function selection and argument filling, reducing retries, timeouts, and cascading failures during multi-tool runs.
* **Lower latency & cost** – Smaller, sharper prompts cut tokens per turn and attention load.
* **Error & hallucination containment** – Summaries act as “clean rooms” that correct or omit prior mistakes; trimming avoids amplifying bad facts (“context poisoning”) turn after turn.
* **Easier debugging & observability** – Stable summaries and bounded histories make logs comparable: you can diff summaries, attribute regressions, and reproduce failures reliably.
* **Multi-issue and handoff resilience** – In multi-problem chats, per-issue mini-summaries let the agent pause/resume, escalate to humans, or hand off to another agent while staying consistent.


The [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses/create#responses-create-previous_response_id) includes **basic memory support** through built-in state and message chaining with `previous_response_id`.

You can continue a conversation by passing the prior response’s `id` as `previous_response_id`, or you can manage context manually by collecting outputs into a list and resubmitting them as the `input` for the next response.

What you don’t get is **automatic memory management**. That’s where the **Agents SDK** comes in. It provides [session memory](https://openai.github.io/openai-agents-python/sessions/) on top of Responses, so you no longer need to manually append `response.output` or track IDs yourself. The session becomes the **memory object**: you simply call `session.run("...")` repeatedly, and the SDK handles context length, history, and continuity—making it far easier to build coherent, multi-turn agents.


#### Real-World Scenario

We’ll ground the techniques in a practical example for one of the common long-running tasks, such as:

* **Multi-turn Customer Service Conversations**
In extended conversations about tech products—spanning both hardware and software—customers often surface multiple issues over time. The agent must stay consistent and goal-focused while retaining only the essentials rather than hauling along every past detail.

#### Techniques Covered

To address these challenges, we introduce two separate concrete approaches using OpenAI Agents SDK:

- **Context Trimming** – dropping older turns while keeping the last N turns.
  - **Pros**

    * **Deterministic & simple:** No summarizer variability; easy to reason about state and to reproduce runs.
    * **Zero added latency:** No extra model calls to compress history.
    * **Fidelity for recent work:** Latest tool results, parameters, and edge cases stay verbatim—great for debugging.
    * **Lower risk of “summary drift”:** You never reinterpret or compress facts.

    **Cons**

    * **Forgets long-range context abruptly:** Important earlier constraints, IDs, or decisions can vanish once they scroll past N.
    * **User experience “amnesia”:** Agent can appear to “forget” promises or prior preferences midway through long sessions.
    * **Wasted signal:** Older turns may contain reusable knowledge (requirements, constraints) that gets dropped.
    * **Token spikes still possible:** If a recent turn includes huge tool payloads, your last-N can still blow up the context.

  - **Best when**

    - Your tasks in the conversation is indepentent from each other with non-overlapping context that does not reuqire carrying previous details further.
    - You need predictability, easy evals, and low latency (ops automations, CRM/API actions).
    - The conversation’s useful context is local (recent steps matter far more than distant history).

- **Context Summarization** – compressing prior messages(assistant, user, tools, etc.) into structured, shorter summaries injected into the conversation history.

  - **Pros**

    * **Retains long-range memory compactly:** Past requirements, decisions, and rationales persist beyond N.
    * **Smoother UX:** Agent “remembers” commitments and constraints across long sessions.
    * **Cost-controlled scale:** One concise summary can replace hundreds of turns.
    * **Searchable anchor:** A single synthetic assistant message becomes a stable “state of the world so far.”

    **Cons**

    * **Summarization loss & bias:** Details can be dropped or misweighted; subtle constraints may vanish.
    * **Latency & cost spikes:** Each refresh adds model work (and potentially tool-trim logic).
    * **Compounding errors:** If a bad fact enters the summary, it can **poison** future behavior (“context poisoning”).
    * **Observability complexity:** You must log summary prompts/outputs for auditability and evals.

  - **Best when**

    - You have use cases where your tasks needs context collected accross the flow such as  planning/coaching, RAG-heavy analysis, policy Q&A.
    - You need continuity over long horizons and carry the important details further to solve related tasks.
    - Sessions exceed N turns but must preserve decisions, IDs, and constraints reliably.
<br>

**Quick comparison**

| Dimension         | **Trimming (last-N turns)**         | **Summarizing (older → generated summary)** |
| ----------------- | ------------------------------- | ------------------------------------ |
| Latency / Cost    | Lowest (no extra calls)     | Higher at summary refresh points |
| Long-range recall | Weak (hard cut-off)         | Strong (compact carry-forward)   |
| Risk type         | Context loss                | Context distortion/poisoning     |
| Observability     | Simple logs                 | Must log summary prompts/outputs |
| Eval stability    | High                        | Needs robust summary evals       |
| Best for          | Tool-heavy ops, short workflows | Analyst/concierge, long threads      |


## Prerequisites

Before running this cookbook, you must set up the following accounts and complete a few setup actions. These prerequisites are essential to interact with the APIs used in this project.

#### Step0: OpenAI Account and `OPENAI_API_KEY`

- **Purpose:**  
  You need an OpenAI account to access language models and use the Agents SDK featured in this cookbook.

- **Action:**  
  [Sign up for an OpenAI account](https://openai.com) if you don’t already have one. Once you have an account, create an API key by visiting the [OpenAI API Keys page](https://platform.openai.com/api-keys).

**Before running the workflow, set your environment variables:**

```
# Your openai key
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
```

Alternatively, you can set your OpenAI API key for use by the agents via the `set_default_openai_key` function by importing agents library .

```
from agents import set_default_openai_key
set_default_openai_key("YOUR_API_KEY")
```

#### Step1: Install the Required Libraries

Below we install the `openai-agents` library ([OpenAI Agents SDK](https://github.com/openai/openai-agents-python))


```python
%pip install openai-agents nest_asyncio
```


```python
from openai import OpenAI

client = OpenAI()
```


```python
from agents import set_tracing_disabled
set_tracing_disabled(True)
```

Let's test the installed libraries by defining and running an agent.


```python
import asyncio
from agents import Agent, Runner


agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

result = await Runner.run(agent, "Tell me why it is important to evaluate AI agents.")
print(result.final_output)

```

    Evaluating AI agents ensures reliability, safety, ethical alignment, performance accuracy, and helps avoid biases, improving overall trust and effectiveness.


### Define Agents

We can start by defining the necessary components from Agents SDK Library. Instructions added based on the use case during agent creation.

#### Customer Service Agent


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

## Context Trimming

#### Implement Custom Session Object

We are using [Session](https://openai.github.io/openai-agents-python/sessions/) object from [OpenAI Agents Python SDK](https://openai.github.io/openai-agents-python/). Here’s a `TrimmingSession` implementation that **keeps only the last N turns** (a “turn” = one user message and everything until the next user message—including the assistant reply and any tool calls/results). It’s in-memory and trims automatically on every write and read.



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

Let's define the custom session object we implemented with max_turns=3.


```python
# Keep only the last 8 turns (user + assistant/tool interactions)
session = TrimmingSession("my_session", max_turns=3)
```

**How to choose the right `max_turns`?**

Determining this parameter usually requires experimentation with your conversation history. One approach is to extract the total number of turns across conversations and analyze their distribution. Another option is to use an LLM to evaluate conversations—identifying how many tasks or issues each one contains and calculating the average number of turns needed per issue.



```python
message = "There is a red light blinking on my laptop."
```


```python
result = await Runner.run(
    support_agent,
    message,
    session=session
)
```


```python
history = await session.get_items()

```


```python
history
```




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




```python
# Example flow
await session.add_items([{"role": "user", "content": "I am using a macbook pro and it has some overheating issues too."}])
await session.add_items([{"role": "assistant", "content": "I see. Let's check your firmware version."}])
await session.add_items([{"role": "user", "content": "Firmware v1.0.3; still failing."}])
await session.add_items([{"role": "assistant", "content": "Could you please try a factory reset?"}])
await session.add_items([{"role": "user", "content": "Reset done; error 42 now."}])
await session.add_items([{"role": "assistant", "content": "Leave it on charge for 30 minutes in case the battery is critically low. Is there any other error message?"}])
await session.add_items([{"role": "user", "content": "Yes, I see error 404 now."}])
await session.add_items([{"role": "assistant", "content": "Do you see it on the browser while accessing a website?"}])
# At this point, with max_turns=3, everything *before* the earliest of the last 3 user
# messages is summarized into a synthetic pair, and the last 3 turns remain verbatim.

history = await session.get_items()
# Pass `history` into your agent runner / responses call as the conversation context.

```


```python
len(history)
```




    6




```python
history
```




    [{'role': 'user', 'content': 'Firmware v1.0.3; still failing.'},
     {'role': 'assistant', 'content': 'Could you please try a factory reset?'},
     {'role': 'user', 'content': 'Reset done; error 42 now.'},
    
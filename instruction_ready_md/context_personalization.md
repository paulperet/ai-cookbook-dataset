# Context Engineering for Personalization: State Management with Long-Term Memory using OpenAI Agents SDK

## Introduction

Modern AI agents are evolving from reactive assistants into adaptive collaborators. The key to this transformation is **context engineering**—the practice of managing what an agent knows at any given moment. By controlling what is stored, recalled, and injected into the model's working memory, you can create agents that feel personal, consistent, and deeply context-aware.

This guide demonstrates a **state-based long-term memory** pattern using the OpenAI Agents SDK. You will build a travel concierge agent that:
* Starts each session with a structured user profile and curated memory notes.
* Captures new user preferences during interaction via a dedicated tool.
* Consolidates these preferences into long-term memory at the end of each run.
* Resolves conflicts using a clear precedence order: **latest user input → session overrides → global defaults**.

## Prerequisites

Before you begin, ensure you have the following:

### 1. OpenAI API Key
You need an active OpenAI account and an API key.
1. [Sign up for an OpenAI account](https://openai.com) if you don't have one.
2. Create an API key on the [OpenAI API Keys page](https://platform.openai.com/api-keys).
3. Set the key as an environment variable.

### 2. Install Required Libraries
Install the necessary Python packages.

```bash
pip install openai-agents nest_asyncio
```

### 3. Initial Setup and Test
Let's verify the installation and perform a quick test.

```python
import asyncio
from agents import Agent, Runner, set_tracing_disabled, set_default_openai_key

# Set your OpenAI API key
set_default_openai_key("YOUR_API_KEY_HERE")
set_tracing_disabled(True)

# Define a simple agent
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

# Quick test
async def test_agent():
    result = await Runner.run(agent, "Tell me why it is important to evaluate AI agents.")
    print(result.final_output)

# Run the test
asyncio.run(test_agent())
```

**Expected Output:**
```
Evaluating AI agents ensures they are accurate, safe, reliable, ethical, and effective for their intended tasks.
```

## Step 1: Define the State Object (Local-First Memory Store)

We start by defining a **local-first state object** that serves as the single source of truth for personalization and memory. This state is initialized at the beginning of each run and evolves over time.

The state includes:
* **`profile`**: Structured, predefined fields (often hydrated from internal systems or CRM).
* **`global_memory`**: Long-term, unstructured notes that persist across sessions.
* **`session_memory`**: Short-term, contextual notes relevant only to the current interaction.

Here is the initial state structure we will use:

```json
{
  "profile": {
    "global_customer_id": "crm_12345",
    "name": "John Doe",
    "age": 31,
    "home_city": "San Francisco",
    "currency": "USD",
    "passport_expiry_date": "2029-06-12",
    "loyalty_status": {"airline": "United Gold", "hotel": "Marriott Titanium"},
    "loyalty_ids": {"marriott": "MR998877", "hilton": "HH445566", "hyatt": "HY112233"},
    "seat_preference": "aisle",
    "tone": "concise and friendly",
    "active_visas": ["Schengen", "US"],
    "tight_connection_ok": false,
    "insurance_coverage_profile": {
      "car_rental": "primary_cdw_included",
      "travel_medical": "covered"
    }
  },
  "global_memory": {
    "notes": [
      {
        "text": "For trips shorter than a week, user generally prefers not to check bags.",
        "last_update_date": "2025-04-05",
        "keywords": ["baggage"]
      },
      {
        "text": "User usually prefers aisle seats.",
        "last_update_date": "2024-06-25",
        "keywords": ["seat_preference"]
      },
      {
        "text": "User generally likes staying in central, walkable city-center neighborhoods.",
        "last_update_date": "2024-02-11",
        "keywords": ["neighborhood"]
      },
      {
        "text": "User generally likes to compare options side-by-side.",
        "last_update_date": "2023-02-17",
        "keywords": ["pricing"]
      },
      {
        "text": "User prefers high floors.",
        "last_update_date": "2023-02-11",
        "keywords": ["room"]
      }
    ]
  }
}
```

In the next step, you will learn how to wrap this state using the `RunContextWrapper` to make it persistent across agent runs.
# Getting Started with the OpenAI Assistants API

## Overview

The new [Assistants API](https://platform.openai.com/docs/assistants/overview) is a stateful evolution of the Chat Completions API. It simplifies building assistant-like experiences by handling conversation state, tool use, and file context automatically. This guide walks you through the core concepts—Assistants, Threads, and Runs—with a practical math tutor example.

### Key Differences: Chat Completions vs. Assistants API

*   **Chat Completions API:** Uses stateless `Messages` and `Completions`. You manage conversation history, tools, and context manually.
*   **Assistants API:** Uses three stateful primitives:
    *   **Assistants:** Configure the model, instructions, tools, and knowledge.
    *   **Threads:** Represent the conversation state.
    *   **Runs:** Execute the Assistant on a Thread, enabling multi-step reasoning and tool use.

## Prerequisites

Ensure you have the latest OpenAI Python SDK installed.

```bash
pip install --upgrade openai
```

Verify the installation:

```bash
pip show openai | grep Version
```

## Step 1: Set Up Your Environment

First, import the necessary libraries and create a helper function for clean JSON output.

```python
import json
import os
import time
from openai import OpenAI

# Initialize the client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your-api-key>"))

def show_json(obj):
    """Helper to print JSON objects cleanly."""
    print(json.dumps(json.loads(obj.model_dump_json()), indent=2))
```

## Step 2: Create Your Assistant

An Assistant encapsulates the AI's personality, capabilities, and knowledge. Let's create a math tutor.

```python
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    model="gpt-4o",
)
show_json(assistant)
```

**Output:**
```json
{
  "id": "asst_qvXmYlZV8zhABI2RtPzDfV6z",
  "name": "Math Tutor",
  "instructions": "You are a personal math tutor. Answer questions briefly, in a sentence or less.",
  "model": "gpt-4o",
  "tools": []
}
```

> **Note:** Save the Assistant ID (`assistant.id`). You'll need it to reference this Assistant in future steps.

## Step 3: Create a Thread and Add a Message

A Thread holds the conversation state. Create a new Thread and add a user message to it.

```python
# Create a new Thread
thread = client.beta.threads.create()
show_json(thread)
```

**Output:**
```json
{
  "id": "thread_j4dc1TiHPfkviKUHNi4aAsA6",
  "object": "thread"
}
```

```python
# Add a user message to the Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
)
show_json(message)
```

**Output:**
```json
{
  "id": "msg_1q4Y7ZZ9gIcPoAKSx9UtrrKJ",
  "role": "user",
  "content": [{"text": {"value": "I need to solve the equation `3x + 11 = 14`. Can you help me?"}}]
}
```

## Step 4: Execute a Run

A Run tells your Assistant to process the messages in a Thread. Unlike Chat Completions, this is an asynchronous operation.

```python
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
show_json(run)
```

**Output:**
```json
{
  "id": "run_qVYsWok6OCjHxkajpIrdHuVP",
  "status": "queued",
  "assistant_id": "asst_qvXmYlZV8zhABI2RtPzDfV6z",
  "thread_id": "thread_j4dc1TiHPfkviKUHNi4aAsA6"
}
```

The Run starts in a `queued` state. You need to poll for its completion.

## Step 5: Wait for the Run to Complete

Define a helper function to poll the Run status, then use it.

```python
def wait_on_run(run, thread):
    """Poll the Run until it is no longer queued or in progress."""
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

run = wait_on_run(run, thread)
show_json(run)
```

**Output:**
```json
{
  "id": "run_qVYsWok6OCjHxkajpIrdHuVP",
  "status": "completed",
  "assistant_id": "asst_qvXmYlZV8zhABI2RtPzDfV6z"
}
```

## Step 6: Retrieve the Assistant's Response

Once the Run is `completed`, list the Thread's messages to see the Assistant's reply.

```python
messages = client.beta.threads.messages.list(thread_id=thread.id)
show_json(messages)
```

**Output:**
```json
{
  "data": [
    {
      "id": "msg_A5eAN6ZAJDmFBOYutEm5DFCy",
      "role": "assistant",
      "content": [{"text": {"value": "Sure! Subtract 11 from both sides to get \\(3x = 3\\), then divide by 3 to find \\(x = 1\\)."}}]
    },
    {
      "id": "msg_1q4Y7ZZ9gIcPoAKSx9UtrrKJ",
      "role": "user",
      "content": [{"text": {"value": "I need to solve the equation `3x + 11 = 14`. Can you help me?"}}]
    }
  ]
}
```

> **Note:** Messages are listed in reverse-chronological order by default (newest first). Use `order="asc"` to get chronological order.

## Step 7: Continue the Conversation

Add a follow-up message to the same Thread and execute another Run.

```python
# Add a new user message
message = client.beta.threads.messages.create(
    thread_id=thread.id, role="user", content="Could you explain this to me?"
)

# Start a new Run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# Wait for completion
wait_on_run(run, thread)

# Retrieve new messages after our last user message
messages = client.beta.threads.messages.list(
    thread_id=thread.id, order="asc", after=message.id
)
show_json(messages)
```

**Output:**
```json
{
  "data": [
    {
      "id": "msg_wSHHvaMnaWktZWsKs6gyoPUB",
      "role": "assistant",
      "content": [{"text": {"value": "Certainly! To isolate \\(x\\), first subtract 11 from both sides of the equation \\(3x + 11 = 14\\), resulting in \\(3x = 3\\). Then, divide both sides by 3 to solve for \\(x\\), giving you \\(x = 1\\)."}}]
    }
  ]
}
```

## Step 8: Build a Reusable Workflow

Let's encapsulate the common operations into helper functions for cleaner code.

```python
MATH_ASSISTANT_ID = assistant.id  # Use your saved Assistant ID

def submit_message(assistant_id, thread, user_message):
    """Adds a user message to a Thread and starts a Run."""
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

def get_response(thread):
    """Retrieves all messages from a Thread in chronological order."""
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

def create_thread_and_run(user_input):
    """Creates a new Thread and starts a Run with the user's input."""
    thread = client.beta.threads.create()
    run = submit_message(MATH_ASSISTANT_ID, thread, user_input)
    return thread, run
```

## Step 9: Handle Multiple Concurrent Conversations

The asynchronous nature of Runs allows you to handle multiple user requests efficiently.

```python
# Simulate multiple concurrent user requests
thread1, run1 = create_thread_and_run(
    "I need to solve the equation `3x + 11 = 14`. Can you help me?"
)
thread2, run2 = create_thread_and_run("Could you explain linear algebra to me?")
thread3, run3 = create_thread_and_run("I don't like math. What can I do?")

# All three Runs are now executing concurrently...
```

Wait for each Run to finish and print the responses.

```python
# Wait for all Runs to complete
run1 = wait_on_run(run1, thread1)
run2 = wait_on_run(run2, thread2)
run3 = wait_on_run(run3, thread3)

# Get and display responses
print("Thread 1 Response:")
show_json(get_response(thread1))
print("\nThread 2 Response:")
show_json(get_response(thread2))
print("\nThread 3 Response:")
show_json(get_response(thread3))
```

## Summary

You've successfully built a stateful AI assistant using the OpenAI Assistants API. The key steps are:

1.  **Create an Assistant** with specific instructions and a model.
2.  **Manage conversation state** using Threads.
3.  **Execute actions** with asynchronous Runs.
4.  **Retrieve and display** the Assistant's responses.

This foundation enables you to add powerful features like Code Interpreter and File Search without changing your core workflow. The stateful design handles complexity for you, making it easier to build sophisticated, multi-turn AI applications.
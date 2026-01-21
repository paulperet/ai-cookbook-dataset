# Guide: How to Format Inputs for ChatGPT Models

This guide explains how to structure your inputs when using OpenAI's chat models, such as `gpt-3.5-turbo` and `gpt-4`, via the API. You'll learn the required format, see practical examples, and discover tips for effective prompting.

## Prerequisites

First, ensure you have the OpenAI Python library installed and your API key ready.

```bash
pip install --upgrade openai
```

```python
import os
from openai import OpenAI

# Initialize the client with your API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

## Step 1: Understanding the Chat Completion Request

A chat completion request requires two main parameters:

1.  **`model`**: The identifier of the model you want to use (e.g., `gpt-3.5-turbo`, `gpt-4`).
2.  **`messages`**: A list of message objects that form the conversation history.

Each message object must have:
*   `role`: The sender's role. This can be `system`, `user`, `assistant`, or `tool`.
*   `content`: The actual text of the message.

Messages can also include an optional `name` field to identify different speakers (names cannot contain spaces).

### Optional Parameters
You can fine-tune the model's behavior with numerous optional parameters, such as `temperature` (for randomness), `max_tokens` (to limit response length), and `stream` (for real-time responses). For a full list, consult the [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat).

## Step 2: Making Your First API Call

Let's start with a simple conversational example. The model expects a sequence of messages. A typical pattern is a `system` message to set the assistant's behavior, followed by alternating `user` and `assistant` messages.

```python
MODEL = "gpt-3.5-turbo"

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ],
    temperature=0, # Setting temperature to 0 makes the output deterministic
)

print(response.choices[0].message.content)
```

The response is a rich object. To extract just the assistant's reply text, access `response.choices[0].message.content`.

## Step 3: Structuring Non-Conversational Tasks

You can use the chat format for standard instruction-following tasks by placing your instruction in the first `user` message. The `system` message is optional but useful for guiding the model's tone.

```python
# Example with a system message
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain asynchronous programming in the style of the pirate Blackbeard."},
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

```python
# The same task without a system message
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Explain asynchronous programming in the style of the pirate Blackbeard."},
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

## Step 4: Effective Prompting Techniques

### Using System Messages
The `system` message primes the assistant's behavior. You can use it to define personality, expertise, or response style.

```python
# A system message that creates a detailed teaching assistant
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a friendly and helpful teaching assistant. You explain concepts in great depth using simple terms, and you give examples to help people learn. At the end of each explanation, you ask a question to check for understanding"},
        {"role": "user", "content": "Can you explain how fractions work?"},
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

```python
# A system message that creates a laconic, brief assistant
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a laconic assistant. You reply with brief, to-the-point answers with no elaboration."},
        {"role": "user", "content": "Can you explain how fractions work?"},
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

### Few-Shot Prompting
Sometimes it's more effective to show the model what you want rather than tell it. You can do this by providing example interactions within the `messages` list.

```python
# Few-shot example: Teaching the model to translate business jargon
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful, pattern-following assistant."},
        {"role": "user", "content": "Help me translate the following corporate jargon into plain English."},
        {"role": "assistant", "content": "Sure, I'd be happy to!"},
        {"role": "user", "content": "New synergies will help drive top-line growth."},
        {"role": "assistant", "content": "Things working well together will increase revenue."},
        {"role": "user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
        {"role": "assistant", "content": "Let's talk later when we're less busy about how to do better."},
        {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

To make it clearer that the examples are not part of the ongoing conversation, you can assign `name` fields to the example messages.

```python
# Few-shot example using the 'name' field for example messages
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
        {"role": "system", "name":"example_user", "content": "New synergies will help drive top-line growth."},
        {"role": "system", "name": "example_assistant", "content": "Things working well together will increase revenue."},
        {"role": "system", "name":"example_user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
        {"role": "system", "name": "example_assistant", "content": "Let's talk later when we're less busy about how to do better."},
        {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

**Tip:** Prompt engineering often requires experimentation. If your first approach doesn't work, try restructuring your messages, adding encouraging feedback within the conversation, or moving key instructions. For more advanced techniques, see the guide on [techniques to improve reliability](../techniques_to_improve_reliability).

## Step 5: Counting Tokens

Understanding token usage is crucial as it impacts cost, response time, and whether your request hits the model's context limit (e.g., 4,096 tokens for `gpt-3.5-turbo`).

You can estimate the token count for a list of messages using the `tiktoken` library and the helper function below. Note that tokenization can change between model versions, so treat this as an estimate.

```python
import tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
```

You can verify this estimate against the API's own count.

```python
example_messages = [
    {"role": "system", "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
    {"role": "system", "name": "example_user", "content": "New synergies will help drive top-line growth."},
    {"role": "system", "name": "example_assistant", "content": "Things working well together will increase revenue."},
    {"role": "system", "name": "example_user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
    {"role": "system", "name": "example_assistant", "content": "Let's talk later when we're less busy about how to do better."},
    {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
]

for model in ["gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]:
    print(model)
    estimated_tokens = num_tokens_from_messages(example_messages, model)
    print(f"{estimated_tokens} prompt tokens counted by num_tokens_from_messages().")

    # Get the actual count from the API
    response = client.chat.completions.create(
        model=model,
        messages=example_messages,
        temperature=0,
        max_tokens=1  # We only need the token count, not a full completion
    )
    api_tokens = response.usage.prompt_tokens
    print(f'{api_tokens} prompt tokens counted by the OpenAI API.\n')
```

For a deeper dive into token counting, read the dedicated guide: [How to count tokens with tiktoken](How_to_count_tokens_with_tiktoken.ipynb).

## Next Steps

You now understand the core format for interacting with ChatGPT models via the API. To build more advanced applications, explore these related topics:
*   **[How to call functions with chat models](How_to_call_functions_with_chat_models.ipynb)**: Learn how to connect the model to external tools and APIs.
*   **The [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat)**: The complete reference for all parameters and features.
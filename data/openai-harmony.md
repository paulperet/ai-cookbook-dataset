# Guide to the OpenAI Harmony Response Format

## Overview

The OpenAI Harmony response format is a structured prompt and response format designed for `gpt-oss` models. It defines conversation structures, manages reasoning output (chain-of-thought), and handles function/tool calls. This guide explains the core concepts and provides practical examples for implementing the format, whether you use the official library or build your own renderer.

## Prerequisites

To follow the code examples, install the official Python library:

```bash
pip install openai-harmony
```

## Core Concepts

### Message Roles

Every message processed by the model has an associated role, which defines its purpose and authority. The roles follow a strict hierarchy: `system` > `developer` > `user` > `assistant` > `tool`.

| Role | Purpose |
| :--- | :--- |
| `system` | Specifies meta-information: reasoning effort, knowledge cutoff, built-in tools, and valid channels. |
| `developer` | Provides the core instructions (the "system prompt") and defines available function tools. |
| `user` | Represents the human or client input to the model. |
| `assistant` | The model's output, which can be a text response or a tool call. |
| `tool` | Represents the result returned from a called tool. The specific tool name is used as the role. |

### Assistant Channels

Assistant messages are output into specific "channels" to separate internal reasoning from user-facing responses.

| Channel | Purpose | Safety Note |
| :--- | :--- | :--- |
| `final` | Contains the final answer intended for the end-user. | Safe for users. |
| `analysis` | Used for the model's internal chain-of-thought (CoT) reasoning. | **Not safe for users.** Do not show these messages to end-users. |
| `commentary` | Typically used for function tool calls. May also contain preambles before multiple calls. | Use with caution. |

## Using the Official Harmony Renderer (Recommended)

The `openai-harmony` library handles tokenization, message rendering, and parsing automatically. Here’s how to construct a conversation.

### Step 1: Import and Initialize

First, import the necessary components and load the encoding.

```python
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)

# Load the correct encoding for gpt-oss models
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
```

### Step 2: Construct the System Message

The system message sets the model's identity, meta-dates, reasoning level, and channel rules.

```python
system_message = (
    SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_conversation_start_date("2025-06-28")
)
```

### Step 3: Construct the Developer Message

The developer message contains your instructions and defines any available function tools.

```python
developer_message = (
    DeveloperContent.new()
        .with_instructions("Always respond in riddles")
        .with_function_tools(
            [
                ToolDescription.new(
                    "get_current_weather",
                    "Gets the current weather in the provided location.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["location"],
                    },
                ),
            ]
        )
)
```

### Step 4: Build the Conversation History

Create a `Conversation` object from a list of messages, simulating a multi-turn interaction.

```python
convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, "What is the weather in Tokyo?"),
        # Model's internal analysis (CoT)
        Message.from_role_and_content(
            Role.ASSISTANT,
            'User asks: "What is the weather in Tokyo?" We need to use get_current_weather tool.',
        ).with_channel("analysis"),
        # Model's tool call on the commentary channel
        Message.from_role_and_content(Role.ASSISTANT, '{"location": "Tokyo"}')
        .with_channel("commentary")
        .with_recipient("functions.get_current_weather")
        .with_content_type("<|constrain|> json"),
        # Tool's response
        Message.from_author_and_content(
            Author.new(Role.TOOL, "functions.get_current_weather"),
            '{ "temperature": 20, "sunny": true }',
        ).with_channel("commentary"),
    ]
)
```

### Step 5: Render and Parse

Render the conversation into tokens for the model, and later parse the model's token response back into messages.

```python
# Render the conversation, specifying the next expected role is 'assistant'
tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

# ... Send `tokens` to your model for completion ...

# After receiving new tokens from the model, parse them.
# Do not include the stop token (e.g., <|return|>) in `new_tokens`.
parsed_response = encoding.parse_messages_from_completion_tokens(new_tokens, Role.ASSISTANT)
```

### Step 6: Stream Parsing (Optional)

For streaming inference, use the `StreamableParser` to decode incrementally.

```python
from openai_harmony import StreamableParser

stream = StreamableParser(encoding, role=Role.ASSISTANT)

# Example token stream (simplified)
tokens = [200005, 35644, 200008, 1844, 31064, 25, 392, 4827, 382, 220, 17, 659, 220, 17, 16842, 12295, 81645, 13, 51441, 6052, 13, 200007, 200006, 173781, 200005, 17196, 200008, 17, 659, 220, 17, 314, 220, 19, 13, 200002]

for token in tokens:
    stream.process(token)
    # Inspect the parser's state after each token
    print("current_role", stream.current_role)
    print("current_channel", stream.current_channel)
    print("last_content_delta", stream.last_content_delta)
    # ... etc.
```

## Implementing Your Own Renderer

If you choose to build a custom renderer, you must adhere to the token and format specifications.

### Special Tokens

The format uses special tokens (found in the `o200k_harmony` tiktoken encoding) to structure messages.

| Special Token | Purpose | Token ID |
| :--- | :--- | :--- |
| `<\|start\|>` | Begins a message header. | `200006` |
| `<\|end\|>` | Ends a complete message. | `200007` |
| `<\|message\|>` | Separates the header from the content. | `200008` |
| `<\|channel\|>` | Indicates the channel in the header. | `200005` |
| `<\|constrain\|>` | Indicates a data type definition (e.g., for tool calls). | `200003` |
| `<\|return\|>` | Stop token: model is done generating. | `200002` |
| `<\|call\|>` | Stop token: model wants to call a tool. | `200012` |

### Basic Message Structure

A complete message follows this pattern:
```
<|start|>{header}<|message|>{content}<|end|>
```
The `{header}` contains the role and, for assistant messages, the channel.

### Example: Simple Chat

**Input to the model:**
```
<|start|>user<|message|>What is 2 + 2?<|end|>
<|start|>assistant
```

**Potential model output:**
```
<|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
```

**Implementation Note:** `<|return|>` is a decode-time stop token. When storing the assistant's reply in conversation history for the next turn, replace the trailing `<|return|>` with `<|end|>` to maintain the proper `<|start|>...<|end|>` message structure.

### System Message Format

For best performance, structure your system message precisely. Include identity, dates, reasoning level, valid channels, and a note about tool channels if needed.

**Basic Example:**
```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
```

**With Function Tools:**
Append this line before `<|end|>`:
```
Calls to these tools must go to the commentary channel: 'functions'.
```

### Developer Message Format

This is your primary "system prompt". For instructions only:

```
<|start|>developer<|message|># Instructions

{your_instructions_here}<|end|>
```

For defining function tools, see the dedicated section below.

## Managing Reasoning (Chain-of-Thought)

### Controlling Effort
Set the reasoning level (`low`, `medium`, or `high`) in the system message:
```
Reasoning: high
```

### Handling CoT in Multi-Turn Conversations
After a turn where the model provides a `final` answer, you should **drop the previous `analysis` (CoT) messages** from the history for the next user input.

**Example History for Next Turn:**
```
<|start|>user<|message|>What is 2 + 2?<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|end|>
<|start|>user<|message|>What about 9 / 2?<|end|>
<|start|>assistant
```
**Exception:** If the assistant's previous turn involved a tool call (which happens in the `commentary` or `analysis` channels), you must keep those intermediate messages in the history.

## Defining and Calling Functions

### Defining Tools in the Developer Message
Define available functions within a `functions` namespace using a TypeScript-like syntax inside a `# Tools` section.

**Formatting Guidelines:**
- Define each function as `type {function_name} = (_: {args_type}) => any`.
- Place descriptions as comments on the line above.
- Use `any` as the return type.
- Maintain an empty line between function definitions.

**Example Developer Message with Tools:**
```
<|start|>developer<|message|># Instructions

Use a friendly tone.

# Tools

## functions

namespace functions {

// Gets the location of the user.
type get_location = () => any;

// Gets the current weather in the provided location.
type get_current_weather = (_: {
  // The city and state, e.g. San Francisco, CA
  location: string,
  // The temperature unit
  format?: "celsius" | "fahrenheit"
}) => any;

}<|end|>
```

### Model Tool Call Format
When the model decides to call a function, it will output a message on the `commentary` channel with a specific recipient and a constrained JSON content type.

**Example Tool Call from Model:**
```
<|start|>assistant<|channel|>commentary<|message|><|constrain|> json{"location": "Tokyo"}<|end|>
```
The `recipient` would be `"functions.get_current_weather"`.

### Providing Tool Results
Inject the tool's result as a message where the role is the tool name.

**Example Tool Response:**
```
<|start|>functions.get_current_weather<|channel|>commentary<|message|>{ "temperature": 20, "sunny": true }<|end|>
```

## Key Takeaways

1.  **Use the Library:** The `openai-harmony` library handles complexity and is the recommended path.
2.  **Structure is Key:** Precisely follow the system and developer message formats for optimal performance.
3.  **Channels Separate Concerns:** Use `analysis` for CoT, `commentary` for tool calls, and `final` for user answers.
4.  **Safety First:** Never expose `analysis` channel content to end-users.
5.  **Manage History:** Prune `analysis` CoT from history after a `final` answer, but keep it if tool calls were involved.

By adhering to this format, you ensure the `gpt-oss` models operate as intended, leveraging their full reasoning and tool-calling capabilities.
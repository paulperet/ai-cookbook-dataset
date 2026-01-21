# How to Count Tokens with tiktoken

## Introduction

When working with OpenAI's models, it's essential to understand how text is converted into tokens. Tokens are the fundamental units of text that models like GPT-4 process. Knowing the token count of your text helps you:
1.  Ensure your input fits within a model's context window.
2.  Estimate the cost of an API call, as usage is priced per token.

[`tiktoken`](https://github.com/openai/tiktoken) is OpenAI's fast, open-source tokenizer library for Python. This guide will walk you through its core functionalities.

## Prerequisites

First, ensure you have the necessary libraries installed.

```bash
pip install --upgrade tiktoken openai
```

## Step 1: Import tiktoken

Begin by importing the library.

```python
import tiktoken
```

## Step 2: Load an Encoding

Encodings define how text is split into tokens. Different models use different encodings. You can load an encoding by name or automatically for a specific model.

**Load by encoding name:**
```python
encoding = tiktoken.get_encoding("cl100k_base")
```
*Note: The first run requires an internet connection to download the encoding files.*

**Load for a specific model (recommended):**
```python
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
```

### Common OpenAI Encodings
| Encoding | Example Models |
| :--- | :--- |
| `o200k_base` | `gpt-4o`, `gpt-4o-mini` |
| `cl100k_base` | `gpt-4-turbo`, `gpt-3.5-turbo`, `text-embedding-ada-002` |
| `p50k_base` | Codex models, `text-davinci-002`, `text-davinci-003` |
| `r50k_base` (or `gpt2`) | GPT-3 models like `davinci` |

## Step 3: Convert Text to Tokens

Use the `.encode()` method to convert a string into a list of token integers.

```python
tokens = encoding.encode("tiktoken is great!")
print(tokens)
# Example output: [83, 1609, 5963, 374, 2294, 0]
```

To count the tokens in a string, simply check the length of this list.

```python
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

print(num_tokens_from_string("tiktoken is great!", "cl100k_base"))
```

## Step 4: Convert Tokens Back to Text

The `.decode()` method converts a list of token integers back into a string.

```python
decoded_text = encoding.decode([83, 1609, 5963, 374, 2294, 0])
print(decoded_text)
# Output: 'tiktoken is great!'
```

**Warning:** Using `.decode()` on a single token can be lossy. For safe, single-token conversion, use `.decode_single_token_bytes()` to see the raw bytes.

```python
token_bytes = [encoding.decode_single_token_bytes(token) for token in [83, 1609, 5963, 374, 2294, 0]]
print(token_bytes)
# Output: [b't', b'ik', b'token', b' is', b' great', b'!']
```

## Step 5: Compare Different Encodings

Encodings can tokenize the same text differently. The function below helps visualize these differences.

```python
def compare_encodings(example_string: str) -> None:
    """Prints a comparison of string encodings."""
    print(f'\nExample string: "{example_string}"')
    for encoding_name in ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"]:
        encoding = tiktoken.get_encoding(encoding_name)
        token_integers = encoding.encode(example_string)
        token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]
        print(f"\n{encoding_name}: {len(token_integers)} tokens")
        print(f"token integers: {token_integers}")
        print(f"token bytes: {token_bytes}")

# Compare on different strings
compare_encodings("antidisestablishmentarianism")
compare_encodings("2 + 2 = 4")
compare_encodings("お誕生日おめでとう")
```

## Step 6: Count Tokens for Chat Completions

For chat models (e.g., `gpt-4o`, `gpt-3.5-turbo`), token counting is more complex due to message formatting. The following function provides an estimate for common models.

```python
def num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")

    # Set tokens per message and name based on the model
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        # Handle generic model names by redirecting to a specific version
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )

    # Count tokens in each message
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
    return num_tokens
```

### Verify the Count with the OpenAI API

You can verify the accuracy of the function above by comparing it to the token count returned by the API itself.

```python
from openai import OpenAI
import os

# Initialize the client. Ensure your API key is set in the environment.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

example_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

for model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
    print(f"\n--- {model} ---")
    
    # Count using our function
    counted_tokens = num_tokens_from_messages(example_messages, model)
    print(f"Function count: {counted_tokens} tokens")
    
    # Count using the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=example_messages,
        temperature=0,
        max_tokens=1
    )
    print(f"API count: {response.usage.prompt_tokens} tokens")
```

## Step 7: Count Tokens for Chat Completions with Tool Calls

When your request includes function/tool definitions, you must account for the extra tokens they consume. The function below extends the previous logic to handle tools.

```python
def num_tokens_for_tools(functions, messages, model):
    """Return the total tokens for messages including tool/function definitions."""
    
    # Model-specific formatting constants
    if model in ["gpt-4o", "gpt-4o-mini"]:
        func_init = 7
        prop_init = 3
        prop_key = 3
        enum_init = -3
        enum_item = 3
        func_end = 12
    elif model in ["gpt-3.5-turbo", "gpt-4"]:
        func_init = 10
        prop_init = 3
        prop_key = 3
        enum_init = -3
        enum_item = 3
        func_end = 12
    else:
        raise NotImplementedError(f"num_tokens_for_tools() is not implemented for model {model}.")
    
    # Load encoding
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    
    # Count tokens for all function definitions
    func_token_count = 0
    if len(functions) > 0:
        for f in functions:
            func_token_count += func_init
            function = f["function"]
            
            # Name and description
            f_name = function["name"]
            f_desc = function["description"].rstrip('.')
            line = f_name + ":" + f_desc
            func_token_count += len(encoding.encode(line))
            
            # Parameters and properties
            if len(function["parameters"]["properties"]) > 0:
                func_token_count += prop_init
                for key, props in function["parameters"]["properties"].items():
                    func_token_count += prop_key
                    
                    # Property name, type, and description
                    p_name = key
                    p_type = props["type"]
                    p_desc = props["description"].rstrip('.')
                    line = f"{p_name}:{p_type}:{p_desc}"
                    func_token_count += len(encoding.encode(line))
                    
                    # Enum values if present
                    if "enum" in props:
                        func_token_count += enum_init
                        for item in props["enum"]:
                            func_token_count += enum_item
                            func_token_count += len(encoding.encode(item))
        func_token_count += func_end
    
    # Add tokens from the messages themselves
    messages_token_count = num_tokens_from_messages(messages, model)
    total_tokens = messages_token_count + func_token_count
    return total_tokens
```

### Example Usage with Tools

```python
tools = [
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {
            "type": "string",
            "description": "The unit of temperature to return",
            "enum": ["celsius", "fahrenheit"]
          },
        },
        "required": ["location"],
      },
    }
  }
]

example_messages = [
    {"role": "system", "content": "You are a helpful weather assistant."},
    {"role": "user", "content": "What's the weather like in London?"},
]

for model in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
    total = num_tokens_for_tools(tools, example_messages, model)
    print(f"{model}: {total} total tokens (messages + tools)")
```

## Summary

You've learned how to use `tiktoken` to:
1.  Load encodings for different OpenAI models.
2.  Convert text to tokens and back.
3.  Compare tokenization across encodings.
4.  Estimate token counts for standard chat completions.
5.  Accurately count tokens for requests involving tool/function calls.

These skills are crucial for managing costs and ensuring your prompts fit within model context limits. For the most precise counts, especially with newer models, always refer to the latest OpenAI documentation.
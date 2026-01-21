# Running GPT-OSS with vLLM: A Step-by-Step Guide

vLLM is an open-source, high-throughput inference engine optimized for serving large language models (LLMs). This guide walks you through deploying **gpt-oss-20b** or **gpt-oss-120b** on a server using vLLM, exposing it as an API, and integrating it with the OpenAI Agents SDK.

> **Note:** This guide is intended for server environments with dedicated GPUs (e.g., NVIDIA H100). For local inference on consumer hardware, refer to our Ollama or LM Studio guides.

## Prerequisites

Ensure you have a compatible GPU with sufficient VRAM:
*   **`openai/gpt-oss-20b`**: Requires ~16GB VRAM.
*   **`openai/gpt-oss-120b`**: Requires ≥60GB VRAM (single H100 or multi-GPU setup).

Both models are served with **MXFP4 quantization** by default.

## Step 1: Install vLLM

vLLM recommends using `uv` for environment management. This ensures the correct implementation for your system.

1.  Create and activate a new Python virtual environment:
    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    ```

2.  Install vLLM with the GPT-OSS-specific wheels:
    ```bash
    uv pip install --pre vllm==0.10.1+gptoss \
        --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
        --index-strategy unsafe-best-match
    ```

## Step 2: Start the Model Server

The `vllm serve` command downloads the model from HuggingFace and launches an OpenAI-compatible API server on `localhost:8000`.

Run the appropriate command for your chosen model in a terminal on your server:

```bash
# For the 20B model
vllm serve openai/gpt-oss-20b

# For the 120B model
vllm serve openai/gpt-oss-120b
```

The server will start and begin downloading the model. Once complete, the API is ready to accept requests.

## Step 3: Use the OpenAI-Compatible API

vLLM exposes both Chat Completions and Responses API endpoints, allowing you to use the standard OpenAI SDK with minimal changes.

### Basic Chat Completion Example

Create a Python script (`example.py`) with the following code:

```python
from openai import OpenAI

# Configure the client to point to your local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # API key is not required for local serving
)

# Use the Chat Completions API
result = client.chat.completions.create(
    model="openai/gpt-oss-20b",  # Specify your model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what MXFP4 quantization is."}
    ]
)

print("Chat Completions Response:")
print(result.choices[0].message.content)
print("\n---\n")

# Use the Responses API
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant.",
    input="Explain what MXFP4 quantization is."
)

print("Responses API Output:")
print(response.output_text)
```

Run the script:
```bash
python example.py
```

Your existing OpenAI SDK code should work seamlessly by simply changing the `base_url`.

## Step 4: Implement Function Calling

vLLM supports function calling (tool use) through both API formats. Here's how to use it with the Chat Completions API.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Define the available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            },
        },
    }
]

# Make a request that may trigger tool use
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "What's the weather in Berlin right now?"}],
    tools=tools
)

print(response.choices[0].message)
```

**Important:** The model performs tool calling as part of its chain-of-thought reasoning. You must handle the conversation loop by returning the model's reasoning and tool calls back in subsequent API requests until a final answer is reached.

## Step 5: Integrate with the OpenAI Agents SDK

You can use your vLLM-hosted model with the OpenAI Agents SDK by overriding the base client configuration.

1.  Install the Agents SDK:
    ```bash
    uv pip install openai-agents
    ```

2.  Create an agent script (`agent_example.py`):

```python
import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, OpenAIResponsesModel, set_tracing_disabled

# Disable tracing for cleaner output
set_tracing_disabled(True)

# Define a simple tool
@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."

async def main():
    # Create an agent that uses your vLLM server
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=OpenAIResponsesModel(
            model="openai/gpt-oss-120b",
            openai_client=AsyncOpenAI(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
            ),
        ),
        tools=[get_weather],
    )

    # Run the agent
    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

Run the agent:
```bash
python agent_example.py
```

## Step 6: Direct Sampling with vLLM (Advanced)

For more control, you can use vLLM's Python library directly instead of the API server. This requires careful prompt formatting using the Harmony response format.

1.  Install the Harmony SDK:
    ```bash
    uv pip install openai-harmony
    ```

2.  Create a direct sampling script (`direct_sampling.py`):

```python
import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
from vllm import LLM, SamplingParams

# --- 1) Prepare the prompt using Harmony ---
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Build a structured conversation
convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("Always respond in riddles"),
        ),
        Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
    ]
)

# Convert conversation to token IDs for the model
prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

# Get stop tokens to prevent incorrect generation
stop_token_ids = encoding.stop_tokens_for_assistant_actions()

# --- 2) Run inference with vLLM ---
llm = LLM(
    model="openai/gpt-oss-120b",
    trust_remote_code=True,
)

sampling = SamplingParams(
    max_tokens=128,
    temperature=1,
    stop_token_ids=stop_token_ids,
)

# Generate completion
outputs = llm.generate(
    prompts=[{"prompt_token_ids": prefill_ids}],  # Batch size of 1
    sampling_params=sampling,
)

# Extract results
gen = outputs[0].outputs[0]
text = gen.text
output_tokens = gen.token_ids  # Completion tokens (excluding prefill)

print("Generated Text:")
print(text)
print("\n---\n")

# --- 3) Parse the structured output ---
entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)

print("Structured Messages:")
for message in entries:
    print(json.dumps(message.to_dict(), indent=2))
```

Run the script:
```bash
python direct_sampling.py
```

This approach gives you direct access to the token-level output and structured message parsing, which is useful for advanced pipeline integration.

## Summary

You've now successfully:
1.  Installed vLLM with GPT-OSS support
2.  Launched a model server with either gpt-oss-20b or gpt-oss-120b
3.  Used the OpenAI-compatible API for chat and responses
4.  Implemented function calling
5.  Integrated with the OpenAI Agents SDK
6.  Explored direct sampling for advanced use cases

Your vLLM server is now ready to power applications with high-performance, self-hosted GPT-OSS inference.
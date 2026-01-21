# Local GPT-OSS Guide: Running Models with LM Studio

## Overview
LM Studio is a user-friendly desktop application that enables you to run large language models (LLMs) locally on your personal hardware. This guide walks you through setting up and running **GPT-OSS-20B** or **GPT-OSS-120B** models using LM Studio, covering chat interactions, MCP server integration, and API usage through Python and TypeScript.

**Note:** This guide is designed for consumer hardware (PCs and Macs). For server deployments with dedicated GPUs like NVIDIA H100s, refer to our vLLM guide.

## Prerequisites
- LM Studio installed on your system (Windows, macOS, or Linux)
- Sufficient hardware resources for your chosen model

## Step 1: Choose Your Model
LM Studio supports both GPT-OSS model sizes:

- **`openai/gpt-oss-20b`**
  - Smaller model requiring at least **16GB of VRAM**
  - Ideal for high-end consumer GPUs or Apple Silicon Macs

- **`openai/gpt-oss-120b`**
  - Full-sized model requiring **≥60GB VRAM**
  - Best for multi-GPU setups or powerful workstations

LM Studio includes both llama.cpp (for GGUF models) and Apple MLX engines for optimal performance across platforms.

## Step 2: Install and Configure LM Studio

### 2.1 Install LM Studio
Download and install LM Studio from the official website for your operating system.

### 2.2 Download Your Model
Use the LM Studio command-line interface to download your chosen model:

```bash
# For the 20B model
lms get openai/gpt-oss-20b

# For the 120B model
lms get openai/gpt-oss-120b
```

### 2.3 Load the Model
Load the downloaded model into LM Studio using either the graphical interface or command line:

```bash
# For the 20B model
lms load openai/gpt-oss-20b

# For the 120B model
lms load openai/gpt-oss-120b
```

Once loaded, the model is ready for interaction through LM Studio's chat interface or API.

## Step 3: Chat with GPT-OSS
You can interact with GPT-OSS directly through LM Studio's chat interface or via the terminal:

```bash
lms chat openai/gpt-oss-20b
```

**Note:** LM Studio uses OpenAI's Harmony library to format prompts for GPT-OSS models, ensuring proper input construction for both llama.cpp and MLX engines.

## Step 4: Use the Local API Endpoint
LM Studio exposes a Chat Completions-compatible API at `http://localhost:1234/v1`. This allows you to use the OpenAI SDK with minimal code changes.

### 4.1 Python Example
First, ensure you have the OpenAI Python package installed:

```bash
pip install openai
```

Then use the following code to interact with your local model:

```python
from openai import OpenAI

# Configure client for local LM Studio instance
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"  # No API key required for local use
)

# Create a chat completion
result = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what MXFP4 quantization is."}
    ]
)

# Print the response
print(result.choices[0].message.content)
```

This approach maintains compatibility with existing OpenAI SDK code—simply change the `base_url` to point to your local instance.

## Step 5: Configure MCP Servers
LM Studio functions as an MCP (Model Context Protocol) client, enabling you to connect external tools to GPT-OSS models.

The MCP configuration file is located at:
```bash
~/.lmstudio/mcp.json
```

Edit this file to define and connect MCP servers that provide additional capabilities to your local models.

## Step 6: Implement Local Tool Use
LM Studio provides SDKs for both Python and TypeScript that enable tool calling and local function execution with GPT-OSS models.

### 6.1 Python Implementation
Install the LM Studio Python SDK:

```bash
uv pip install lmstudio
```

Here's an example that provides a file creation tool to the model:

```python
import readline  # Enables input line editing
from pathlib import Path
import lmstudio as lms

# Define a tool function
def create_file(name: str, content: str):
    """Create a file with the given name and content."""
    dest_path = Path(name)
    if dest_path.exists():
        return "Error: File already exists."
    try:
        dest_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc!r}"
    return "File created."

# Callback for streaming responses
def print_fragment(fragment, round_index=0):
    print(fragment.content, end="", flush=True)

# Initialize model and chat
model = lms.llm("openai/gpt-oss-20b")
chat = lms.Chat("You are a helpful assistant running on the user's computer.")

# Interactive chat loop
while True:
    try:
        user_input = input("User (leave blank to exit): ")
    except EOFError:
        print()
        break
    
    if not user_input:
        break
    
    chat.add_user_message(user_input)
    print("Assistant: ", end="", flush=True)
    
    # Execute with tool availability
    model.act(
        chat,
        [create_file],  # Provide tools to the model
        on_message=chat.append,
        on_prediction_fragment=print_fragment,
    )
    print()
```

The `.act()` method enables the model to alternate between reasoning and tool execution until completing the task.

### 6.2 TypeScript Implementation
Install the LM Studio TypeScript SDK:

```bash
npm install @lmstudio/sdk zod
```

Here's the equivalent TypeScript implementation:

```typescript
import { Chat, LMStudioClient, tool } from "@lmstudio/sdk";
import { existsSync } from "fs";
import { writeFile } from "fs/promises";
import { createInterface } from "readline/promises";
import { z } from "zod";

const rl = createInterface({ input: process.stdin, output: process.stdout });
const client = new LMStudioClient();
const model = await client.llm.model("openai/gpt-oss-20b");
const chat = Chat.empty();

// Define a file creation tool
const createFileTool = tool({
  name: "createFile",
  description: "Create a file with the given name and content.",
  parameters: { 
    name: z.string(), 
    content: z.string() 
  },
  implementation: async ({ name, content }) => {
    if (existsSync(name)) {
      return "Error: File already exists.";
    }
    await writeFile(name, content, "utf-8");
    return "File created.";
  },
});

// Interactive chat loop
while (true) {
  const input = await rl.question("User: ");
  chat.append("user", input);

  process.stdout.write("Assistant: ");
  await model.act(chat, [createFileTool], {
    onMessage: (message) => chat.append(message),
    onPredictionFragment: ({ content }) => {
      process.stdout.write(content);
    },
  });
  process.stdout.write("\n");
}
```

## Next Steps
- Explore additional tools beyond file creation by defining custom functions
- Connect multiple MCP servers to expand model capabilities
- Experiment with different system prompts to tailor model behavior
- Monitor resource usage to optimize performance for your hardware

By following this guide, you now have a fully functional local GPT-OSS instance capable of chat interactions, API access, and tool-augmented reasoning—all running on your personal hardware.
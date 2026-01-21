# Build an Agent with Tool-Calling Superpowers Using Smolagents

## Introduction

This guide demonstrates how to use the [**smolagents**](https://huggingface.co/docs/smolagents/index) library to build powerful AI agents. Agents are systems powered by a Large Language Model (LLM) that can use specific *tools*—functions the LLM cannot perform well on its own—to solve complex problems. For example, a text-generation LLM can leverage tools for image generation, web search, or calculations.

In this tutorial, you will create a multimodal assistant capable of browsing the web and generating images.

## Prerequisites

Before you begin, ensure you have the necessary libraries installed. Run the following command:

```bash
pip install smolagents datasets langchain sentence-transformers faiss-cpu duckduckgo-search openai langchain-community --upgrade -q
```

You will also need to authenticate with the Hugging Face Hub to use the Inference API. Execute the following in your Python environment:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Follow the prompts to log in with your Hugging Face credentials.

## Step 1: Create a Multimodal Web-Browsing Assistant

Your goal is to build an agent that can search the web and generate images. This requires two tools:
1. An **image generation tool** that uses the Hugging Face Inference API.
2. A **web search tool** for retrieving information.

### 1.1 Import Required Modules and Load Tools

Start by importing the necessary components from `smolagents` and loading the tools.

```python
from smolagents import load_tool, CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

# Load the image generation tool from the Hugging Face Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

# Instantiate the built-in web search tool
search_tool = DuckDuckGoSearchTool()
```

The `load_tool` function fetches a remote tool definition. The `DuckDuckGoSearchTool` is a built-in tool that performs web searches.

### 1.2 Initialize the LLM and Agent

Next, select an LLM to power your agent and initialize the `CodeAgent` with your tools.

```python
# Specify the model to use via the Hugging Face Inference API
model = InferenceClientModel("Qwen/Qwen2.5-72B-Instruct")

# Create the agent, providing the list of available tools
agent = CodeAgent(
    tools=[image_generation_tool, search_tool],
    model=model
)
```

The `InferenceClientModel` connects to a model hosted on the Hugging Face Inference API. The `CodeAgent` is configured with this model and the tools you loaded.

### 1.3 Run the Agent

Now, task your agent with a request that requires both web search and image generation.

```python
# Execute the agent with a complex query
result = agent.run(
    "Generate me a photo of the car that James bond drove in the latest movie.",
)
print(result)
```

When you run this, the agent will:
1. **Plan its actions.** It recognizes it needs to find out which car James Bond drove in the latest movie.
2. **Execute the web search tool** to gather this information.
3. **Use the image generation tool** with the specific car details to create the requested photo.

The agent will output the final result, which includes the generated image.

## How It Works

The agent operates in a loop:
1. **Reasoning:** The LLM analyzes the user's query and decides which tool(s) to use and in what order.
2. **Tool Execution:** The agent calls the selected tool with the appropriate parameters (e.g., a search query).
3. **Observation:** The tool's output (e.g., search results) is fed back to the LLM.
4. **Iteration:** The LLM uses this new information to decide the next step, repeating until the task is complete or a final answer is ready.

In the provided example, the agent first searches for "latest James Bond movie," identifies "No Time to Die," then searches for the specific car model in that film, and finally generates a photorealistic image of that car.

## Conclusion

You have successfully built a multimodal AI agent using `smolagents` that can leverage external tools to perform tasks beyond the native capabilities of an LLM. This pattern can be extended by adding more tools—such as calculators, database connectors, or custom APIs—to create even more powerful and versatile assistants.

To explore further, check the [smolagents documentation](https://huggingface.co/docs/smolagents/index) for more tool examples and advanced configuration options.
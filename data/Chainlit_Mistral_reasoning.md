# Build a Self-Reflective AI Agent with Chainlit and Mistral AI

## Overview

This guide walks you through building a conversational AI application using Chainlit and Mistral AI. The application features a self-reflective agent capable of handling multi-step reasoning tasks, such as answering nested questions like "What is the weather in Napoleon's hometown?".

The agent will determine when to use available tools to gather information before providing a final answer.

## Prerequisites

Before starting, ensure you have:
- A Mistral AI API key
- Python 3.8 or higher installed

## Setup

### 1. Install Required Packages

Create a new Python environment and install the necessary dependencies:

```bash
pip install mistralai chainlit python-dotenv
```

### 2. Configure Environment Variables

Create a `.env` file in your project directory and add your Mistral AI API key:

```bash
MISTRAL_API_KEY=your_api_key_here
```

### 3. Optional: Enable Tracing with Literal AI

For enhanced debugging and visualization of your application's flow, you can set up tracing with Literal AI:

1. Get a `LITERAL_API_KEY` from [Literal AI](https://docs.getliteral.ai/get-started/installation#how-to-get-my-api-key)
2. Add it to your `.env` file: `LITERAL_API_KEY=your_literal_key_here`

## Building the Application

### Step 1: Define the Tools

First, we'll create the tools our agent can use to gather information. We'll define two tools: one to get a person's hometown and another to get current weather information.

Create a file named `app.py` and add the following code:

```python
import json
import asyncio
import os
import chainlit as cl
from mistralai.client import MistralClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Mistral client
mai_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])

# Optional: Enable Mistral AI instrumentation for tracing
chainlit.instrument_mistralai()

@cl.step(type="tool", name="get_current_weather")
async def get_current_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # In a real application, you would make an API call to a weather service
    # For this example, we'll return mock data
    return json.dumps({
        "location": location,
        "temperature": "29",
        "unit": "celsius",
        "forecast": ["sunny"],
    })

@cl.step(type="tool", name="get_home_town")
async def get_home_town(person: str) -> str:
    """Get the hometown of a person."""
    # In a real application, you might query a knowledge base or API
    # For this example, we'll return a hardcoded value
    return "Ajaccio, Corsica"

# Define tool schemas for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_home_town",
            "description": "Get the home town of a specific person",
            "parameters": {
                "type": "object",
                "properties": {
                    "person": {
                        "type": "string",
                        "description": "The name of a person (first and last names) to identify."
                    }
                },
                "required": ["person"]
            },
        },
    },
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
                },
                "required": ["location"],
            },
        },
    }
]

async def run_multiple(tool_calls):
    """
    Execute multiple tool calls asynchronously.
    
    Args:
        tool_calls: List of tool call objects from the LLM
        
    Returns:
        List of tool results formatted for the LLM
    """
    available_tools = {
        "get_current_weather": get_current_weather,
        "get_home_town": get_home_town
    }

    async def run_single(tool_call):
        function_name = tool_call.function.name
        function_to_call = available_tools[function_name]
        function_args = json.loads(tool_call.function.arguments)

        function_response = await function_to_call(**function_args)
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }

    # Run all tool calls in parallel
    tool_results = await asyncio.gather(
        *(run_single(tool_call) for tool_call in tool_calls)
    )
    return tool_results
```

### Step 2: Implement the Agent Logic

Now, let's create the core agent logic. This function will handle the conversation with the Mistral LLM, deciding when to use tools and when to provide a final answer.

Add the following code to your `app.py` file:

```python
@cl.step(type="run", tags=["to_score"])
async def run_agent(user_query: str):
    """
    Main agent function that handles user queries with tool usage.
    
    Args:
        user_query: The user's question
        
    Returns:
        The final answer from the agent
    """
    messages = [
        {
            "role": "system",
            "content": "If needed, leverage the tools at your disposal to answer the user question, otherwise provide the answer."
        },
        {
            "role": "user", 
            "content": user_query
        }
    ]

    number_iterations = 0
    answer_message_content = None

    # Allow up to 5 iterations for multi-step reasoning
    while number_iterations < 5:
        # Call Mistral AI with the current conversation context
        completion = mai_client.chat(
            model="mistral-large-latest",
            messages=messages,
            tool_choice="auto",  # Let the LLM decide when to use tools
            tools=tools,
        )
        
        message = completion.choices[0].message
        messages.append(message)
        answer_message_content = message.content

        # If no tool calls are needed, we have our final answer
        if not message.tool_calls:
            break

        # Execute the requested tool calls
        tool_results = await run_multiple(message.tool_calls)
        messages.extend(tool_results)

        number_iterations += 1

    return answer_message_content
```

### Step 3: Set Up the Message Handler

Chainlit uses decorators to handle incoming messages. We'll create a handler that processes each user message through our agent.

Add this code to your `app.py`:

```python
@cl.on_message
async def main(message: cl.Message):
    """
    Main message handler for incoming user messages.
    """
    answer_message = await run_agent(message.content)
    await cl.Message(content=answer_message).send()
```

### Step 4: Add Starter Questions (Optional)

To make your application more user-friendly, you can add starter questions that users can click to try common queries.

Add this function to your `app.py`:

```python
@cl.on_chat_start
async def set_starters():
    """
    Set up starter questions for the chat interface.
    """
    await cl.Message(
        content="Hello! I'm an AI assistant that can answer complex questions. Try one of the starter questions or ask your own!"
    ).send()
    
    starters = [
        cl.Starter(
            label="What's the weather in Napoleon's hometown?",
            message="What's the weather in Napoleon's hometown?",
            icon="/public/idea.svg",
        ),
        cl.Starter(
            label="What's the weather in Paris, TX?",
            message="What's the weather in Paris, TX?",
            icon="/public/learn.svg",
        ),
        cl.Starter(
            label="What's the weather in Michelangelo's hometown?",
            message="What's the weather in Michelangelo's hometown?",
            icon="/public/write.svg",
        ),
    ]
    
    # Note: Icons require proper setup in Chainlit's public directory
    # For simplicity, you can remove the icon parameter if needed
    return starters
```

## Running the Application

### Step 5: Launch Your Chainlit App

To run your application, use the following command in your terminal:

```bash
chainlit run app.py
```

This will start a local server, typically at `http://localhost:8000`. Open this URL in your browser to interact with your AI agent.

## Testing Your Application

Once your application is running, try these test questions:

1. **Simple weather query**: "What's the weather in Paris, France?"
2. **Nested query**: "What's the weather in Napoleon's hometown?"
3. **Person lookup**: "Where was Michelangelo born?"

The agent should demonstrate multi-step reasoning for nested questions, using the `get_home_town` tool first, then the `get_current_weather` tool with the result.

## How It Works

### The Agent's Reasoning Process

When you ask a question like "What's the weather in Napoleon's hometown?", the agent:

1. **Analyzes the question** and recognizes it needs Napoleon's hometown first
2. **Calls the `get_home_town` tool** with "Napoleon" as the parameter
3. **Receives "Ajaccio, Corsica"** as the response
4. **Calls the `get_current_weather` tool** with "Ajaccio, Corsica" as the location
5. **Receives weather data** and formulates a final answer
6. **Returns the complete answer** to the user

### Key Features

- **Self-reflection**: The agent decides when tool usage is necessary
- **Multi-step reasoning**: Handles complex, nested questions
- **Parallel tool execution**: Runs multiple tool calls simultaneously when needed
- **Tracing support**: Optional integration with Literal AI for debugging

## Next Steps

Now that you have a working AI agent, consider enhancing it with:

1. **Real API integrations**: Replace mock tool functions with actual API calls
2. **Additional tools**: Add more capabilities like web search, calculations, or database queries
3. **Custom UI elements**: Use Chainlit's components to create richer interfaces
4. **Authentication**: Add user authentication for production deployments
5. **Feedback collection**: Implement mechanisms to collect user feedback on responses

For more advanced features and customization options, visit the [Chainlit documentation](https://docs.chainlit.io/).

## Troubleshooting

- **Missing API key**: Ensure your `.env` file contains `MISTRAL_API_KEY=your_key_here`
- **Import errors**: Make sure all packages are installed: `pip install mistralai chainlit python-dotenv`
- **Port already in use**: Chainlit defaults to port 8000. Use `chainlit run app.py --port 8001` to specify a different port
- **Starter icons not showing**: Ensure icon files exist in your Chainlit public directory or remove the icon parameter

Congratulations! You've built a sophisticated AI agent capable of complex reasoning using Chainlit and Mistral AI.
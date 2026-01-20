# Build a Chainlit App with Mistral AI
The goal of this cookbook is to show how one can build a **Chainlit** application on top of **Mistral AI**'s APIs!

We will highlight the reasoning capabilities of Mistral's LLMs by letting a self-reflective agent assess whether it has gathered enough information to answer _nested_ user questions, such as **"What is the weather in Napoleon's hometown?"**

To answer such questions, our application should go through multiple-step reasoning: first get Napoleon's hometown, then fetch the weather for that location.

You can read through this notebook or simply go with `chainlit run app.py` since the whole code is in `app.py`. 
You will find here a split of the whole application code with explanations:

- [Setup](#setup)
- [Define available tools](#define-tools)
- [Agent logic](#agent-logic)
- [On message callback](#on-message)
- [Starter questions](#starter-questions)

<a id="setup"></a>
## Setup

### Requirements
We will install `mistralai`, `chainlit` and `python-dotenv`. 

Be sure to create a `.env` file with the line `MISTRAL_API_KEY=` followed by your Mistral AI API key.

```python
!pip install mistralai chainlit python-dotenv
```

### Optional - Tracing

You can get a `LITERAL_API_KEY` from [Literal AI](https://docs.getliteral.ai/get-started/installation#how-to-get-my-api-key) to setup tracing and visualize the flow of your application. 

Within the code, Chainlit offers the `@chainlit.step` decorators to trace your functions, along with an automatic instrumentation of Mistral's API via `chainlit.instrument_mistralai()`.

The trace for this notebook example is: https://cloud.getliteral.ai/thread/ea173d7d-a53f-4eaf-a451-82090b07e6ff.

<a id="define-tools"></a>
## Define available tools

In the next cell, we define the tools, and their JSON definitions, which we will provide to the agent. We have two tools:
- `get_current_weather` -> takes in a location
- `get_home_town` -> takes in a person's name

Optionally, you can decorate your tool definitions with `@cl.step()`, specifying a type and name to organize the traces you can visualize from [Literal AI](https://literalai.com).

```python
import json
import chainlit as cl

@cl.step(type="tool", name="get_current_weather")
async def get_current_weather(location):
    # Make an actual API call! To open-meteo.com for instance.
    return json.dumps({
        "location": location,
        "temperature": "29",
        "unit": "celsius",
        "forecast": ["sunny"],
    })

@cl.step(type="tool", name="get_home_town")
async def get_home_town(person: str) -> str:
    """Get the hometown of a person"""
    return "Ajaccio, Corsica"

"""
JSON tool definitions provided to the LLM.
"""
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

# This helper function runs multiple tool calls in parallel, asynchronously.
async def run_multiple(tool_calls):
    """
    Execute multiple tool calls asynchronously.
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

    # Run tool calls in parallel.
    tool_results = await asyncio.gather(
        *(run_single(tool_call) for tool_call in tool_calls)
    )
    return tool_results
```

<a id="agent-logic"></a>
## Agent logic

For the agent logic, we simply repeat the following pattern (max. 5 times):
- ask the user question to Mistral, making both tools available
- execute tools if Mistral asks for it, otherwise return message

You will notice that we added an optional `@cl.step` of type `run` and with optional tags to trace the call accordingly in [Literal AI](https://literalai.com). 

Visual trace: https://cloud.getliteral.ai/thread/ea173d7d-a53f-4eaf-a451-82090b07e6ff

```python
import os
import chainlit as cl

from mistralai.client import MistralClient

mai_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])

@cl.step(type="run", tags=["to_score"])
async def run_agent(user_query: str):
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

    while number_iterations < 5:
        completion = mai_client.chat(
            model="mistral-large-latest",
            messages=messages,
            tool_choice="auto", # use `any` to force a tool call
            tools=tools,
        )
        message = completion.choices[0].message
        messages.append(message)
        answer_message_content = message.content

        if not message.tool_calls:
            # The LLM deemed no tool calls necessary,
            # we break out of the loop and display the returned message
            break

        tool_results = await run_multiple(message.tool_calls)
        messages.extend(tool_results)

        number_iterations += 1

    return answer_message_content

```

<a id="on-message"></a>
## On message callback

The callback below, properly annotated with `@cl.on_message`, ensures our `run_agent` function is called upon every new user message.

```python
import chainlit as cl

@cl.on_message
async def main(message: cl.Message):
    """
    Main message handler for incoming user messages.
    """
    answer_message = await run_agent(message.content)
    await cl.Message(content=answer_message).send()

```

<a id="starter-questions"></a>
## Starter questions

You can define starter questions for your users to easily try out your application.

We have got many more Chainlit features in store (authentication, feedback, Slack/Discord integrations, etc.) to let you build custom LLM applications and really take advantage of Mistral's LLM capabilities.

Please visit the [Chainlit documentation](https://docs.chainlit.io/) to learn more!

```python
async def set_starters():
    return [
        cl.Starter(
            label="What's the weather in Napoleon's hometown",
            message="What's the weather in Napoleon's hometown?",
            icon="/images/idea.svg",
        ),
        cl.Starter(
            label="What's the weather in Paris, TX?",
            message="What's the weather in Paris, TX?",
            icon="/images/learn.svg",
        ),
        cl.Starter(
            label="What's the weather in Michel-Angelo's hometown?",
            message="What's the weather in Michel-Angelo's hometown?",
            icon="/images/write.svg",
        ),
    ]
```
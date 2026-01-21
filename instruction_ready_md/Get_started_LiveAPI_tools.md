# Guide: Using Tools with the Gemini Multimodal Live API via WebSockets

This guide demonstrates how to integrate and use tools (Google Search, Code Execution, and Function Calling) with the Gemini Multimodal Live API using a direct WebSocket connection. You will learn to set up a live conversation, handle tool calls, and process multimodal responses.

## Prerequisites

Before you begin, ensure you have:
1.  A Google AI API key.
2.  The API key stored in a Colab Secret named `GOOGLE_API_KEY`.

## Step 1: Environment Setup

First, install the required `websockets` package and import the necessary modules.

```bash
pip install -q websockets
```

```python
import asyncio
import base64
import contextlib
import json
import os
import wave
import logging
import time

from IPython import display
from websockets.asyncio.client import connect
from google.colab import userdata

# Retrieve your API key
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

## Step 2: Configure the Connection

Define the WebSocket URI and the target Gemini model.

```python
uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}"
model = "models/gemini-2.0-flash-live-001"
```

## Step 3: Create Helper Functions

You will need several helper functions to manage audio file creation, logging, and WebSocket communication.

### 3.1 Audio File Context Manager

This context manager creates a `.wav` file to store streamed audio data.

```python
@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
```

### 3.2 Configure Logging

Set up a custom logger to optionally display detailed request/response information.

```python
logger = logging.getLogger("Live")
logger.setLevel("INFO")  # Change to "DEBUG" for verbose logs
```

### 3.3 Core WebSocket Handlers

These functions manage the connection lifecycle: initial setup, sending prompts, and processing server responses.

```python
async def setup(ws, modality, tools):
    """Send initial configuration to the server."""
    setup_msg = {
        "setup": {
            "model": model,
            "tools": tools,
            "generation_config": {
                "response_modalities": [modality]
            }
        }
    }
    await ws.send(json.dumps(setup_msg))
    setup_response = json.loads(await ws.recv())
    logger.debug(setup_response)

async def send(ws, prompt):
    """Send a user prompt to the model."""
    msg = {
        "client_content": {
            "turns": [{"role": "user", "parts": [{"text": prompt}]}],
            "turn_complete": True,
        }
    }
    print(">>> ", msg)
    await ws.send(json.dumps(msg))

def handle_server_content(wf, server_content):
    """Process text and audio content from the server."""
    audio = False
    model_turn = server_content.pop("modelTurn", None)
    if model_turn:
        text = model_turn["parts"][0].pop("text", None)
        if text is not None:
            print(text)

        inline_data = model_turn['parts'][0].pop('inlineData', None)
        if inline_data is not None:
            print('.', end='')
            b64data = inline_data['data']
            pcm_data = base64.b64decode(b64data)
            wf.writeframes(pcm_data)
            audio = True

    turn_complete = server_content.pop('turnComplete', None)
    return turn_complete, audio

async def handle_tool_call(ws, tool_call):
    """Handle a tool call request from the model and send a mock response."""
    print("    ", tool_call)
    for fc in tool_call['functionCalls']:
        msg = {
            'tool_response': {
                'function_responses': [{
                    'id': fc['id'],
                    'name': fc['name'],
                    'response': {'result': {'string_value': 'ok'}}
                }]
            }
        }
        print('>>> ', msg)
        await ws.send(json.dumps(msg))
```

### 3.4 Main Execution Function

This orchestrator function manages the entire conversation flow for a single prompt.

```python
async def run(prompt, modality='TEXT', tools=None):
    """Execute a single conversational turn with the Live API."""
    if tools is None:
        tools = []

    async with connect(uri, additional_headers={"Content-Type": "application/json"}) as ws:
        await setup(ws, modality, tools)
        await send(ws, prompt)

        audio = False
        filename = 'audio.wav'
        with wave_file(filename) as wf:
            async for raw_response in ws:
                response = json.loads(raw_response.decode())
                logger.debug(str(response)[:150])

                server_content = response.pop("serverContent", None)
                if server_content is not None:
                    turn_complete, a = handle_server_content(wf, server_content)
                    audio = audio or a
                    if turn_complete:
                        print()
                        print('Turn complete')
                        break

                tool_call = response.pop('toolCall', None)
                if tool_call is not None:
                    await handle_tool_call(ws, tool_call)

        if audio:
            display.display(display.Audio(filename, autoplay=True))
```

## Step 4: Test the Connection

Run a simple test to verify your setup is working correctly.

```python
await run(prompt='Hello?')
```

**Expected Output:**
```
>>>  {'client_content': {'turns': [{'role': 'user', 'parts': [{'text': 'Hello?'}]}], 'turn_complete': True}}
Hello! How can I help you today?

Turn complete
```

## Step 5: Implement Function Calling

Define the schemas for your functions and pass them as tools to the model.

### 5.1 Define Function Schemas

```python
turn_on_the_lights_schema = {'name': 'turn_on_the_lights'}
turn_off_the_lights_schema = {'name': 'turn_off_the_lights'}
```

### 5.2 Call a Function with Text Output

```python
prompt = "Turn on the lights"
tools = [{'function_declarations': [turn_on_the_lights_schema, turn_off_the_lights_schema]}]

await run(prompt, tools=tools)
```

**Expected Output:**
```
>>>  {'client_content': {'turns': [{'role': 'user', 'parts': [{'text': 'Turn on the lights'}]}], 'turn_complete': True}}
     {'functionCalls': [{'name': 'turn_on_the_lights', 'args': {}, 'id': 'function-call-97628297814691086'}]}
>>>  {'tool_response': {'function_responses': [{'id': 'function-call-97628297814691086', 'name': 'turn_on_the_lights', 'response': {'result': {'string_value': 'ok'}}}]}}
OK, I've turned on the lights.

Turn complete
```

### 5.3 Call a Function with Audio Output

Change the response modality to `AUDIO` to receive a spoken response.

```python
await run(prompt, tools=tools, modality="AUDIO")
```

**Expected Output:** You will see the same tool call and response logs, followed by dots (`...`) indicating audio data is being streamed and saved. The audio file will play automatically upon completion.

## Step 6: Use the Code Execution Tool

The model can generate and execute Python code to solve problems. Enable this by adding the `code_execution` tool.

```python
prompt = "What is the largest prime palindrome under 100000"
tools = [{'code_execution': {}}]

await run(prompt, tools=tools, modality='AUDIO')
```

The model will generate code, execute it, and return the result (in this case, as audio).

## Step 7: Integrate Google Search

Enable real-time web searches by including the `google_search` tool.

```python
prompt = "Can you use google search to tell me about the largest earthquake in California the week of Dec 5 2024?"
tools = [{'google_search': {}}]

await run(prompt, tools=tools, modality='TEXT')
```

**Expected Output:** The model will perform a search and return a summarized text answer based on the results.

You can also request the answer as audio:

```python
await run(prompt, tools=tools, modality='AUDIO')
```

## Step 8: Explore Advanced Tool Use Cases

The model can use multiple tools in sophisticated ways.

### 8.1 Compositional Function Calling

The model can write and execute code that calls your defined functions. Here, you ask it to create a script that turns lights on and off with a delay.

```python
prompt = """
Hey, can you write and run some Python code to turn on the lights, wait 10 seconds, and then turn off the lights?
"""

tools = [
    {'code_execution': {}},
    {'function_declarations': [turn_on_the_lights_schema, turn_off_the_lights_schema]}
]

start = time.time()
await run(prompt, tools=tools, modality="AUDIO")
end = time.time()
print(f'Elapsed: {end-start}s')
```

The model will generate code that calls `turn_on_the_lights`, uses `time.sleep(10)`, and then calls `turn_off_the_lights`. The total elapsed time should be slightly over 10 seconds.

### 8.2 Multi-Tool Task Execution

You can ask the model to perform a sequence of tasks using different tools in a single turn.

```python
prompt = """
Hey, I need you to do three things for me.
1. Turn on the lights
2. Then compute the largest prime palindrome under 100000.
3. Then use google search to lookup information about the largest earthquake in California the week of Dec 5 2024?
Thanks!
"""

tools = [
    {'google_search': {}},
    {'code_execution': {}},
    {'function_declarations': [turn_on_the_lights_schema, turn_off_the_lights_schema]}
]

await run(prompt, tools=tools, modality="AUDIO")
```

The model will attempt to execute all three tasks in order, using the appropriate tool for each.

## Next Steps

This guide covered the fundamentals of using tools with the Gemini Live API via WebSockets. To continue learning:

*   **Experiment in Google AI Studio**: Try the [Live API in Google AI Studio](https://aistudio.google.com/app/live) for a no-code interface.
*   **Use the Official SDK**: For a simpler developer experience, use the [Python GenAI SDK](https://github.com/google-gemini/generative-ai-python) which handles WebSocket details for you.
*   **Explore More Examples**: Check the [Gemini Cookbook](https://github.com/google-gemini/cookbook/blob/main/gemini-2/) for additional tutorials and advanced patterns.
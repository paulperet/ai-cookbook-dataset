# Gemini 2.X Multi-Tool Guide: Using the Multimodal Live API

This guide demonstrates how to use the Gemini 2.0 Multimodal Live API with multiple tools, including Google Search, code execution, charting, and mapping. You will learn to establish a persistent WebSocket connection and orchestrate a multi-turn conversation where the model can call various tools to gather information, analyze data, and generate visualizations.

## Prerequisites & Setup

### 1. Install Required Libraries
You need the `websockets` and `altair` packages.

```bash
pip install -q 'websockets~=14.0' altair
```

### 2. Configure Your API Keys
You need two API keys:
1.  A **Gemini API Key** stored in a Colab Secret named `GOOGLE_API_KEY`.
2.  A **Google Maps Static API Key** (for the mapping example) stored in a Colab Secret named `MAPS_API_KEY`.

```python
import os
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
MAPS_API_KEY = userdata.get('MAPS_API_KEY')
```

### 3. Configure the API Endpoint and Model
Set the WebSocket URI and specify the Gemini 2.0 model. The `v1beta` client version is required.

```python
uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}"
model = "models/gemini-2.5-flash-native-audio-preview-09-2025"
```

## Helper Functions and Connection Management

Before interacting with the API, you need to set up several helper functions to manage the WebSocket connection, handle audio, and process tool calls.

### 1. Audio File Context Manager
This context manager wraps raw PCM audio data into a playable `.wav` file.

```python
import contextlib
import wave

@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    """Define a wave context manager using the audio parameters supplied."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
```

### 2. Connection and Message Handling Functions
These core functions establish the connection, send prompts, and handle the streamed responses and tool calls from the API.

```python
import asyncio
import base64
import json
import time
import logging
from websockets.asyncio.client import connect
from IPython import display

logger = logging.getLogger("Live")
logger.setLevel("INFO")  # Set to "DEBUG" to see detailed request/response logs

async def setup(ws, modality, tools):
    """Perform a setup handshake to configure the conversation."""
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
    await ws.recv()  # Wait for setup acknowledgment

async def send(ws, prompt):
    """Send a user content message (only text is supported)."""
    msg = {
        "client_content": {
            "turns": [{"role": "user", "parts": [{"text": prompt}]}],
            "turn_complete": True,
        }
    }
    await ws.send(json.dumps(msg))

def handle_server_content(wf, server_content):
    """Handle any server content messages, e.g., incoming audio or text."""
    audio = False
    model_turn = server_content.pop("modelTurn", None)
    if model_turn:
        text = model_turn["parts"][0].pop("text", None)
        if text:
            print(text, end='')

        inline_data = model_turn['parts'][0].pop('inlineData', None)
        if inline_data:
            print('.', end='')
            pcm_data = base64.b64decode(inline_data['data'])
            wf.writeframes(pcm_data)
            audio = True

    turn_complete = server_content.pop('turnComplete', None)
    return turn_complete, audio

async def handle_tool_call(ws, tool_call, responses):
    """Process an incoming tool call request, returning a response."""
    for fc in tool_call['functionCalls']:
        if fc['name'] in responses:
            result_entry = responses[fc['name']]
            if callable(result_entry):
                result = result_entry(**fc['args'])
        else:
            result = {'string_value': 'ok'}

        msg = {
            'tool_response': {
                'function_responses': [{
                    'id': fc['id'],
                    'name': fc['name'],
                    'response': {'result': result}
                }]
            }
        }
        await ws.send(json.dumps(msg))

@contextlib.asynccontextmanager
async def quick_connect(modality='TEXT', tools=()):
    """Establish a connection and keep it open while the context is active."""
    async with connect(uri, additional_headers={"Content-Type": "application/json"}) as ws:
        await setup(ws, modality, tools)
        yield ws

audio_lock = time.time()

async def run(ws, prompt, responses=()):
    """Send the provided prompt and handle the streamed response."""
    print('>', prompt)
    await send(ws, prompt)

    audio = False
    filename = 'audio.wav'
    with wave_file(filename) as wf:
        async for raw_response in ws:
            response = json.loads(raw_response.decode())

            server_content = response.pop("serverContent", None)
            if server_content:
                turn_complete, a = handle_server_content(wf, server_content)
                audio = audio or a
                if turn_complete:
                    print()
                    print('<Turn complete>')
                    break

            tool_call = response.pop('toolCall', None)
            if tool_call:
                await handle_tool_call(ws, tool_call, responses)

    if audio:
        global audio_lock
        if (delta := audio_lock - time.time()) > 0:
            print('Pausing for audio to complete...')
            await asyncio.sleep(delta + 1.0)

        display.display(display.Audio(filename, autoplay=True))
        audio_lock = time.time() + (wf.getnframes() / wf.getframerate())
```

## Tutorial: Building a Multi-Tool Conversation

Now you will use the helpers to create a conversational agent that can search the web, execute code, and generate charts.

### Step 1: A Simple One-Turn Example
First, test the connection with a basic request using Google Search and code execution tools.

```python
tools = [
    {'google_search': {}},
    {'code_execution': {}},
]

async def simple_example():
    async with quick_connect(tools=tools, modality="TEXT") as ws:
        await run(ws, "Please find the last 5 Denis Villeneuve movies and look up their runtimes and the year published.")

await simple_example()
```

**Expected Output:**
The model will use Google Search to find the movies and their details, then output a summary.

### Step 2: Define a Custom Charting Tool
To enable data visualization, you will define a tool that uses the Altair library. This involves creating a function schema and a corresponding Python function.

```python
import altair as alt
from google.api_core import retry

def apply_altair_theme(altair_json: str, theme: str) -> str:
    chart = alt.Chart.from_json(altair_json)
    with alt.themes.enable(theme):
        themed_altair_json = chart.to_json()
    return themed_altair_json

@retry.Retry()
def render_altair(altair_json: str, theme: str = "default"):
    themed_altair_json = apply_altair_theme(altair_json, theme)
    chart = alt.Chart.from_json(themed_altair_json)
    chart.display()
    return {'string_value': 'ok'}

altair_fns = [
    {
        'name': 'render_altair',
        'description': 'Displays an Altair chart in JSON format.',
        'parameters': {
            'type': 'OBJECT',
            'properties': {
                'altair_json': {
                    'type': 'STRING',
                    'description': 'JSON STRING representation of the Altair chart to render. Must be a string, not a json object',
                },
                'theme': {
                    'type': 'STRING',
                    'description': 'Altair theme. Choose from one of "dark", "ggplot2", "default", "opaque".',
                },
            },
        },
    },
]

tool_calls = {
    'render_altair': render_altair,
}
```

### Step 3: Execute a Multi-Turn Conversation
Now, orchestrate a conversation where the model sequentially uses different tools across multiple turns.

```python
tools = [
    {'google_search': {}},
    {'code_execution': {}},
    {'function_declarations': altair_fns},
]

async def multi_tool_conversation():
    async with quick_connect(tools=tools, modality="AUDIO") as ws:
        # Turn 1: Use Google Search
        await run(ws, "Please find the last 5 Denis Villeneuve movies and find their runtimes.")
        # Turn 2: Use Code Execution
        await run(ws, "Can you write some code to work out which has the longest and shortest runtimes?")
        # Turn 3: Use the Charting Tool
        await run(ws, "Now can you plot them in a line chart showing the year on the x-axis and runtime on the y-axis?", responses=tool_calls)
        # Turn 4: Interactive Chart Modification (User Input)
        user_request = input('Any requests? > ')
        await run(ws, user_request, responses=tool_calls)

await multi_tool_conversation()
```

**How it works:**
1.  The first turn uses **Google Search** to gather data.
2.  The second turn uses **code execution** to analyze the data (e.g., find min/max runtimes).
3.  The third turn uses the **custom `render_altair` tool** to generate a chart from the data.
4.  The fourth turn is interactive, allowing you to request modifications to the chart (e.g., "use a dark theme").

### Step 4: Add a Mapping Tool
Finally, extend the agent's capabilities with a tool that generates static maps using the Google Maps Static API.

First, define the tool's schema. The model needs to provide parameters like `center`, `zoom`, and `markers`.

```python
map_fns = [
    {
        'name': 'draw_map',
        'description': 'Render a Google Maps static map using the specified parameters.',
        'parameters': {
            'type': 'OBJECT',
            'properties': {
                'center': {
                    'type': 'STRING',
                    'description': 'Location to center the map. It can be a lat,lng pair (e.g., 40.714728,-73.998672), or a string address (e.g., Berkeley,CA).',
                },
                'zoom': {
                    'type': 'NUMBER',
                    'description': 'Zoom level for the map (0-21).',
                },
                'markers': {
                    'type': 'ARRAY',
                    'description': 'List of markers to place on the map.',
                    'items': {
                        'type': 'OBJECT',
                        'properties': {
                            'location': {'type': 'STRING'},
                            'label': {'type': 'STRING'},
                            'color': {'type': 'STRING'},
                        }
                    }
                },
                'size': {
                    'type': 'STRING',
                    'description': 'Map image size in pixels, e.g., "600x400".',
                },
            },
            'required': ['center', 'zoom'],
        },
    },
]
```

Next, implement the function that calls the Maps Static API.

```python
import requests

def draw_map(center: str, zoom: int, markers=None, size: str = "600x400"):
    """Calls the Google Maps Static API to generate and display a map image."""
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        'center': center,
        'zoom': zoom,
        'size': size,
        'key': MAPS_API_KEY
    }

    if markers:
        markers_param = []
        for marker in markers:
            marker_str = f"color:{marker.get('color', 'red')}|label:{marker.get('label', '').upper()}|{marker['location']}"
            markers_param.append(marker_str)
        params['markers'] = markers_param

    response = requests.get(base_url, params=params)
    # Display the image in the notebook
    from IPython.display import Image, display
    display(Image(response.content))
    return {'string_value': 'ok'}

# Update the tool responses mapping
tool_calls['draw_map'] = draw_map
```

Now, you can include this tool in a conversation. For example:

```python
tools_with_maps = tools + [{'function_declarations': map_fns}]

async def map_example():
    async with quick_connect(tools=tools_with_maps, modality="TEXT") as ws:
        await run(ws, "Show me a map of the Bay Area with markers for San Francisco, San Jose, and Oakland.", responses=tool_calls)

await map_example()
```

## Summary

You have successfully built a system that uses the Gemini 2.0 Multimodal Live API to conduct a multi-turn conversation where the model can:
1.  **Search the web** for real-time information.
2.  **Execute Python code** for data analysis.
3.  **Generate and modify charts** using a custom Altair tool.
4.  **Create maps** by integrating the Google Maps Static API.

The key pattern is using the `quick_connect` context manager to maintain a persistent WebSocket session and the `run` function to handle each conversational turn, processing streaming text/audio and executing tool calls as they are requested by the model.
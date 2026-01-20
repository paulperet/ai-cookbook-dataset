##### Copyright 2025 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini 2.X - Multi-tool with the Multimodal Live API

<a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/LiveAPI_plotting_and_mapping.ipynb"></a>

In this notebook you will learn how to use tools, including charting tools, Google Search and code execution in the [Gemini 2](https://ai.google.dev/gemini-api/docs/models/gemini-v2) Multimodal Live API. For an overview of new capabilities refer to the [Gemini 2 docs](https://ai.google.dev/gemini-api/docs/models/gemini-v2).

This notebook is written in Python and uses the secure Websockets protocol directly, it does *not* use the GenAI SDK.

If you aren't looking for code, and just want to try multimedia streaming use [Live API in Google AI Studio](https://aistudio.google.com/app/live).

## Get set up


```
%pip install -q 'websockets~=14.0' altair
```

    [?, ..., <Turn complete>]

### Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](../quickstarts/Authentication.ipynb) quickstart for an example.


```
import os
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

Multimodal Live API are a new capability introduced with the [Gemini 2.0](https://ai.google.dev/gemini-api/docs/models/gemini-v2) model. It won't work with previous generation models.

You also need to set the client version to `v1alpha`.


```
uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}"
model = "models/gemini-2.5-flash-native-audio-preview-09-2025"
```

### Set up some helpers

Before interacting with the API, define some helpers that you'll need in this codelab.

In this notebook, you'll be buffering the streamed PCM audio responses, so create a context manager to wrap the PCM audio data in a wave audio file with the relevant audio parameters. This way, you can play the audio back directly within Colab.


```
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

Use a custom logger so you can easily toggle the log level in order to see in-flight requests and responses from the API.


```
import logging
logger = logging.getLogger("Live")
# Switch to "DEBUG" to see the in-flight requests & responses
logger.setLevel("INFO")
```

### Define connection functions

This code defines some functions that will connect to (`quick_connect`), execute and handle prompts (`run`) and handle specific server responses (`handle_tool_call`, `handle_server_content`).

This code uses the [websockets](https://pypi.org/project/websockets) PyPI package, specifically the async interface available in 14.0 and will not work with significantly older packages.


```
import asyncio
import base64
import json
import time

from websockets.asyncio.client import connect
from IPython import display


async def setup(ws, modality, tools):
  """Perform a setup handshake to configure the conversation."""
  setup = {
      "setup": {
          "model": model,
          "tools": tools,
          "generation_config": {
              "response_modalities": [modality]
          }
      }
  }
  setup_json = json.dumps(setup)
  logger.debug(">>> " + setup_json)
  await ws.send(setup_json)

  setup_response = json.loads(await ws.recv())
  logger.debug("<<< " + json.dumps(setup_response))

async def send(ws, prompt):
  """Send a user content message (only text is supported)."""
  msg = {
    "client_content": {
      "turns": [{"role": "user", "parts": [{"text": prompt}]}],
      "turn_complete": True,
    }
  }
  json_msg = json.dumps(msg)
  logger.debug(">>> " + json_msg)
  await ws.send(json_msg)


def handle_server_content(wf, server_content):
  """Handle any server content messages, e.g. incoming audio or text."""
  audio = False
  model_turn = server_content.pop("modelTurn", None)
  if model_turn:
    text = model_turn["parts"][0].pop("text", None)
    if text:
      print(text, end='')

    inline_data = model_turn['parts'][0].pop('inlineData', None)
    if inline_data:
      print('.', end='')
      b64data = inline_data['data']
      pcm_data = base64.b64decode(b64data)
      wf.writeframes(pcm_data)
      audio = True

  turn_complete = server_content.pop('turnComplete', None)
  return turn_complete, audio


async def handle_tool_call(ws, tool_call, responses):
  """Process an incoming tool call request, returning a response."""
  logger.debug("<<< " + json.dumps(tool_call))
  for fc in tool_call['functionCalls']:

    if fc['name'] in responses:
      # Use a response from `responses` if provided.
      result_entry = responses[fc['name']]
      # If it's a function, actuall call it.
      if callable(result_entry):
        result = result_entry(**fc['args'])
    else:
      # Otherwise it's a stub, just say "OK"
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
    json_msg = json.dumps(msg)
    logger.debug(">>> " + json_msg)
    await ws.send(json_msg)


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
      logger.debug("<<< " + str(response)[:150])

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
    # Sleep before playing audio to make sure it doesn't play over an existing clip.
    if (delta := audio_lock - time.time()) > 0:
      print('Pausing for audio to complete...')
      await asyncio.sleep(delta + 1.0)  # include a buffer so there's a breather

    display.display(display.Audio(filename, autoplay=True))
    audio_lock = time.time() + (wf.getnframes() / wf.getframerate())
```

## Use the API

### One-turn example

Now, let's see how all the pieces you've defined fit together in a simple example. You'll send a single prompt to the API and observe the response.

This example uses the `quick_connect` context manager to create a connection to the API. As long as you're inside the async with block, the connection remains active and is accessible through the ws variable. Then use the run function to send our prompt and process the API's response.


Make a simple request to understand how the above code works. A connection is created through a context manager using `quick_connect`, and while the context is active, the web-socket connection is stored in `ws`, and passed to subsequent `run` calls that execute the prompts.

Note that you can change the modality from `AUDIO` to `TEXT`. and adjust the prompt.


```
tools = [
    {'google_search': {}},
    {'code_execution': {}},
]

async def go():
  async with quick_connect(tools=tools, modality="TEXT") as ws:
    await run(ws, "Please find the last 5 Denis Villeneuve movies and look up their runtimes and the year published.")

logger.setLevel('INFO')
await go()
```

    > Please find the last 5 Denis Villeneuve movies and look up their runtimes and the year published.
    Based on the search results, the last 5 Denis Villeneuve movies are:
    
    1.  *Dune: Part Two* (2024)
    2.  *Dune: Part One* (2021)
    3.  *Blade Runner 2049* (2017)
    4.  *Arrival* (2016)
    5.  *Sicario* (2015)
    
    Now, let's find the runtimes for these movies.
    Here's a summary of the last 5 Denis Villeneuve movies, their release year, and runtime:
    
    *   **Dune: Part Two** (2024): 166 minutes (2 hours 46 minutes)
    *   **Dune: Part One** (2021): 155 minutes (2 hours 35 minutes)
    *   **Blade Runner 2049** (2017): 163 minutes (2 hours 43 minutes)
    *   **Arrival** (2016): 116 minutes (1 hour 56 minutes)
    *   **Sicario** (2015): 121 minutes (2 hours 1 minute)
    <Turn complete>


### Complex multi-tool example

Now define additional tools. Add a tool for charting by defining a schema (in `altair_fns`), a function to execute (`render_altair`) and connect the two using the `tool_calls` mapping.

The charting tool used here is [Vega-Altair](https://pypi.org/project/altair/), a "declarative statistical visualization library for Python". Altair supports chart persistance using JSON, which you will expose as a tool so that the Gemini model can produce a chart.

The helper code defined earlier will run as soon as it can, but audio takes some time to play so you may see output from later turns displayed before the audio has played.


```
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

Now put that all together into a chat conversation. This code opens a streaming session (with `quick_connect`), and each `run` invocation will send the text prompt, read the streamed response (and buffer if it's audio), handle any server responses (such as tool calls) and finally return once the end-of-turn signal has been sent.

By sequencing multiple `run` calls within a `quick_connect` session, you are executing a multi-turn, streamed, conversation. Once the code reaches the end of the `quick_connect` block, the session is terminated.


```
tools = [
    {'google_search': {}},
    {'code_execution': {}},
    {'function_declarations': altair_fns},
]

async def go():
  async with quick_connect(tools=tools, modality="AUDIO") as ws:

    # Google Search
    await run(ws, "Please find the last 5 Denis Villeneuve movies and find their runtimes.")
    # Code execution
    await run(ws, "Can you write some code to work out which has the longest and shortest runtimes?")
    # Tool use
    await run(ws, "Now can you plot them in a line chart showing the year on the x-axis and runtime on the y-axis?", responses=tool_calls)
    # Tool use - this step takes user input, so you can ask the model to tweak the chart to your liking.
    # Try changing to dark mode, or lay out the data differently.
    await run(ws, input('Any requests? > '), responses=tool_calls)


logger.setLevel('INFO')
await go()
```

    > Please find the last 5 Denis Villeneuve movies and find their runtimes.
    [., ..., .]
    <Turn complete>

    > Can you write some code to work out which has the longest and shortest runtimes?
    [., ..., .]
    <Turn complete>
    Pausing for audio to complete...

    > Now can you plot them in a line chart showing the year on the x-axis and runtime on the y-axis?


    <ipython-input-10-14ecbacb69f1>:7: AltairDeprecationWarning: 
    Deprecated since `altair=5.5.0`. Use altair.theme instead.
    Most cases require only the following change:
    
        # Deprecated
        alt.themes.enable('quartz')
    
        # Updated
        alt.theme.enable('quartz')
    
    If your code registers a theme, make the following change:
    
        # Deprecated
        def custom_theme():
            return {'height': 400, 'width': 700}
        alt.themes.register('theme_name', custom_theme)
        alt.themes.enable('theme_name')
    
        # Updated
        @alt.theme.register('theme_name', enable=True)
        def custom_theme():
            return alt.theme.ThemeConfig(
                {'height': 400, 'width': 700}
            )
    
    See the updated User Guide for further details:
        https://altair-viz.github.io/user_guide/api.html#theme
        https://altair-viz.github.io/user_guide/customization.html#chart-themes
      with alt.themes.enable(theme):

    [., ..., .]
    <Turn complete>

    Any requests? > can you add the movie names into the dots as also make it in a dark theme?
    > can you add the movie names into the dots as also make it in a dark theme?

    [., ..., .]
    <Turn complete>

### Maps example

For this example you will use the [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static) to draw on a map during the conversation. You'll need to [make sure your API key is enabled for the Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/get-api-key). It can be the same API key as you used for the Gemini API, or a new one, as long as the Static Maps API is enabled.

Add the key in Colab Secrets, or add it in the code directly (`MAPS_API_KEY = 'AIza...'`).


```
from google.colab import userdata
MAPS_API_KEY = userdata.get('MAPS_API_KEY')
```

The following cell is hidden by default, but needs te be run. It comtains the function schema for the `draw_map` function, including some documentation on how to draw markers with the Google Maps API.

Note that the model needs to produce a fairly complex set of parameters in order to call `draw_map`, including defining a center-point for the map, an integer zoom level and custom marker styles and locations.


```
# @title Map tool schema (run this cell)

map_fns = [
  {
    'name': 'draw_map',
    'description': 'Render a Google Maps static map using the specified parameters. No information is returned.',
    'parameters': {
      'type': 'OBJECT',
      'properties': {
        'center': {
            'type': 'STRING',
            'description': 'Location to center the map. It can be a lat,lng pair (e.g. 40.714728,-73.998672), or a string address of a location (e.g. Berkeley,CA).',
        },
        'zoom': {
            'type': 'NUMBER',
            '
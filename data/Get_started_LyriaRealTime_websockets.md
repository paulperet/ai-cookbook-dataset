# Real-Time Music Generation with Lyria and WebSockets

## Introduction

Lyria RealTime provides access to a state-of-the-art, real-time, streaming music generation model. It enables developers to build applications where users can interactively create, continuously steer, and perform instrumental music using text prompts.

### Key Features
- **High-quality text-to-audio**: Generates instrumental music (no voice) using DeepMind's latest models
- **Continuous streaming**: Uses WebSockets to generate music in real-time without stopping
- **Creative blending**: Mix multiple prompts to describe musical ideas, genres, instruments, moods, or characteristics
- **Real-time control**: Adjust guidance, BPM, density, brightness, and scale during generation

> **Note**: This guide uses WebSockets for real-time interaction. Due to Colab limitations, you'll only get limited audio output. For the full experience, use the [Python script](../Get_started_LyriaRealTime.py) or AI Studio's [Prompt DJ](https://aistudio.google.com/apps/bundled/promptdj) and [MIDI DJ](https://aistudio.google.com/apps/bundled/promptdj-midi) apps.

Lyria RealTime is currently a preview feature with free usage (subject to quota limitations).

## Prerequisites

### Install Required Packages

```bash
pip install websockets
```

### Import Libraries

```python
import asyncio
import base64
import contextlib
import datetime
import os
import json
import wave
import itertools
import logging

from websockets.asyncio.client import connect
from IPython.display import display, Audio
```

### Set Up API Authentication

To use Lyria RealTime, you need a Google API key stored in your environment:

```python
from google.colab import userdata
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

> **Note**: If you don't have an API key or need help setting up Colab Secrets, refer to the [Authentication guide](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb).

### Configure API Endpoint

Lyria RealTime requires the `v1alpha` client version:

```python
MODEL = 'models/lyria-realtime-exp'
HOST = 'generativelanguage.googleapis.com'
URI = f'wss://{HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateMusic?key={os.environ["GOOGLE_API_KEY"]}'
```

## Helper Functions

### Configure Logging

For debugging purposes, enable logging to understand how Lyria RealTime works:

```python
logger = logging.getLogger('Bidi')
logger.setLevel('DEBUG')
```

### Wave File Writer

Create a context manager for writing audio to WAV files:

```python
@contextlib.contextmanager
def wave_file(filename, channels=2, rate=48000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
```

## Building the Audio Loop Class

The `AudioLoop` class manages the WebSocket connection and handles bidirectional communication with the Lyria RealTime API.

### Step 1: Initialize the Audio Loop

```python
class AudioLoop:
    def __init__(self, max_chunks=10, prompts=None, config=None):
        self.ws = None
        self.index = 0
        self.max_chunks = max_chunks
        self.prompts = prompts
        self.config = config
```

### Step 2: Implement the Main Run Loop

The `run` method establishes the WebSocket connection and coordinates sending and receiving:

```python
async def run(self):
    print("Type 'q' to quit")
    
    logger.debug('connect')
    async with connect(URI, additional_headers={'Content-Type': 'application/json'}) as ws:
        self.ws = ws
        await self.setup()
        
        # Ideally these would be separate tasks
        await self.send()
        await self.recv()
```

### Step 3: Set Up the Connection

The `setup` method sends the initial configuration to the API:

```python
async def setup(self):
    logger.debug("set_up")
    await self.ws.send(json.dumps({
        'setup': {
            "model": MODEL,
        }
    }))
    raw_response = await self.ws.recv(decode=False)
    setup_response = json.loads(raw_response.decode('ascii'))
    logger.debug(f'Connected: {setup_response}')
```

### Step 4: Send Prompts and Configuration

The `send` method transmits user prompts and music configuration to the model:

```python
async def send(self):
    logger.debug('send')
    if self.prompts is not None:
        # If provided, send the message to the model
        message = {
            "client_content": {
                "weighted_prompts": self.prompts
            }
        }
        await self.ws.send(json.dumps(message))
        logger.debug('sent client message')
    else:
        # Wait for user input
        text = await asyncio.to_thread(input, "prompt? > ")
        
        # Quit if user types 'q'
        if text.lower() == 'q':
            return False
        
        # Wrap text in a "client_content" message
        message = {
            "client_content": {
                "weighted_prompts": [{
                    "text": text,
                    "weight": 1.0,
                }]
            }
        }
        await self.ws.send(json.dumps(message))
        logger.debug('sent client prompt')
    
    # Send music configuration
    if self.config is not None:
        await self.ws.send(json.dumps(self.config))
        logger.debug('sent music_generation_config')
    else:
        # Default configuration
        await self.ws.send(json.dumps({'music_generation_config': {'bpm': '120'}}))
        logger.debug('sent default music_generation_config')
    
    # Start streaming music
    await self.ws.send(json.dumps({'playback_control': 'PLAY'}))
    logger.debug('play')
    
    return True
```

### Step 5: Receive and Process Audio

The `recv` method collects audio chunks and writes them to a WAV file:

```python
async def recv(self):
    # Start a new WAV file
    file_name = f"audio_{self.index}.wav"
    with wave_file(file_name) as wav:
        self.index += 1
        logger.debug('receive')
        
        # Read chunks from the socket
        n = 0
        async for raw_response in self.ws:
            n += 1
            if n > self.max_chunks:
                break
            
            response = json.loads(raw_response.decode())
            logger.debug(f'got chunk: {str(response)[:200]}')
            
            server_content = response.pop('serverContent', None)
            if server_content is None:
                logger.error(f'Unhandled server message! - {response}')
                break
            
            # Write audio chunk to WAV file
            audio_chunk = server_content.pop('audioChunks', None)
            if audio_chunk is not None:
                b64data = audio_chunk[0]
                pcm_data = base64.b64decode(b64data['data'])
                print('.', end='')
                logger.debug('Got pcm_data')
                wav.writeframes(pcm_data)
    
    # Display and play the audio
    display(Audio(file_name, autoplay=True))
    await asyncio.sleep(2)
```

## Generating Your First Music Track

### Simple Example

Start with a basic prompt to generate Eurodance music:

```python
await AudioLoop(prompts=[{"text": "eurodance", "weight": 1.0}]).run()
```

## Advanced Music Generation

Now let's create a more complex composition by mixing multiple prompts and adjusting musical parameters.

### Step 1: Define Your Prompts

Prompts should follow a specific format: a list of text descriptions with weights (any value except 0, including negative values):

```python
# Enter your prompts
prompt_1 = "Heavy Metal"
prompt_1_weight = 1.0

prompt_2 = "Harmonica"
prompt_2_weight = 2

prompt_3 = "Emotional"
prompt_3_weight = 1.0

prompt_4 = ""
prompt_4_weight = 1.0

prompt_5 = ""
prompt_5_weight = 1.0
```

> **Tip**: Keep prompts simple (e.g., "meditation", "eerie", "harp") rather than complex descriptions. The model responds better to concise musical concepts.

### Step 2: Configure Musical Parameters

Adjust these parameters to shape your composition:

```python
# Music configuration
BPM = 140
scale = "C_MAJOR_A_MINOR"
density = 0.2
brightness = 0.7
guidance = 4.0

# Duration in seconds
duration = 10
```

**Available Controls:**
- `bpm`: Beats per minute
- `guidance`: How strictly the model follows prompts
- `density`: Density of musical notes/sounds
- `brightness`: Tonal quality
- `scale`: Musical scale (key and mode)

> **Note**: Additional options like `mute_bass` are available. Check the [documentation](https://ai.google.dev/gemini-api/docs/music-generation#controls) for the full list.

### Step 3: Build the Prompt List

Combine your prompts into the required format:

```python
prompts = [{
    "text": prompt_1,
    "weight": prompt_1_weight,
}]

# Add additional prompts if provided
if prompt_2:
    prompts.append({
        "text": prompt_2,
        "weight": prompt_2_weight,
    })
if prompt_3:
    prompts.append({
        "text": prompt_3,
        "weight": prompt_3_weight,
    })
if prompt_4:
    prompts.append({
        "text": prompt_4,
        "weight": prompt_4_weight,
    })
if prompt_5:
    prompts.append({
        "text": prompt_5,
        "weight": prompt_5_weight,
    })
```

### Step 4: Create the Configuration Object

```python
config = {
    'music_generation_config': {
        'bpm': BPM,
        'scale': scale,
        'density': density,
        'brightness': brightness,
        'guidance': guidance
    }
}
```

### Step 5: Generate the Music

```python
await AudioLoop(max_chunks=duration/2, prompts=prompts, config=config).run()
```

## Next Steps

Now that you can generate music with Lyria RealTime, explore these related capabilities:

- **Real-time conversations**: Have interactive conversations with Gemini using the [Live API](./Get_started_LiveAPI.ipynb)
- **Text-to-speech**: Generate multi-speaker conversations using [TTS models](../Get_started_TTS.ipynb)
- **Visual generation**: Create [images](../Get_started_imagen.ipynb) or [videos](../Get_started_Veo.ipynb)
- **Audio understanding**: Learn how Gemini can [understand audio files](../Audio.ipynb)

## Troubleshooting

- **No audio output**: Ensure your API key is correctly set in the Colab Secret
- **Connection errors**: Check your internet connection and API quota
- **Poor quality results**: Simplify your prompts and adjust guidance values
- **Long wait times**: Reduce the duration parameter for quicker results

Remember that for the full real-time experience, you'll need to run this code outside of Colab using the provided Python script or AI Studio applications.
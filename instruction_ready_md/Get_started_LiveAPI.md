# Multimodal Live API - Quickstart Guide

**Note:** The Live API is currently in preview.

This guide demonstrates the core functionality of the Gemini Multimodal Live API, enabling low-latency, bidirectional voice and video interactions. You will build a simple turn-based chat where you send text messages and receive audio responses. For a comprehensive overview of the API's capabilities, refer to the official [Gemini Live API documentation](https://ai.google.dev/gemini-api/docs/live).

## Prerequisites & Setup

### 1. Install the SDK
The new Google Gen AI SDK provides access to Gemini models. Install it using pip.

```bash
pip install -U google-genai
```

### 2. Set Up Your API Key
Your API key must be available in an environment variable named `GOOGLE_API_KEY`. You can set it directly in your environment or within your script.

```python
import os
os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY_HERE'  # Replace with your actual key
```

### 3. Initialize the Client and Select a Model
Import the necessary modules and initialize the client. For the Live API, you'll use a model that supports live interactions.

```python
import asyncio
import base64
import contextlib
import datetime
import json
import wave
import logging

from IPython.display import display, Audio
from google import genai
from google.genai import types

# Initialize the client
client = genai.Client()

# Select a Live API compatible model
MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025'  # Or another Live model
```

### 4. Helper: WAV File Writer
To play audio in environments like Colab, you'll write audio data to `.wav` files. Here's a context manager to handle that.

```python
@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    """Context manager to write audio data to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
```

## Tutorial: Building a Live API Chat

### Step 1: Basic Text-to-Text Interaction
Let's start with the simplest use case: a text-based chat. This confirms your connection is working.

```python
config = {
    "response_modalities": ["TEXT"]
}

async with client.aio.live.connect(model=MODEL, config=config) as session:
    message = "Hello? Gemini are you there?"
    print(f"> {message}\n")
    await session.send_client_content(
        turns={"role": "user", "parts": [{"text": message}]},
        turn_complete=True
    )

    # Receive the model's text response
    turn = session.receive()
    async for chunk in turn:
        if chunk.text is not None:
            print(f'- {chunk.text}')
```

**Output:**
```
> Hello? Gemini are you there?

- Hello
-  there! I am indeed here. How can I help you today?
```

### Step 2: Simple Text-to-Audio Response
Now, let's get an audio response. Configure the session to return `AUDIO` and write the received data to a file.

```python
# Helper to enumerate async iterators
async def async_enumerate(aiterable):
    n = 0
    async for item in aiterable:
        yield n, item
        n += 1

config = {
    "response_modalities": ["AUDIO"]
}

async with client.aio.live.connect(model=MODEL, config=config) as session:
    file_name = 'audio.wav'
    with wave_file(file_name) as wav:
        message = "Hello? Gemini are you there?"
        print(f"> {message}\n")
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": message}]},
            turn_complete=True
        )

        # Receive and write audio chunks
        turn = session.receive()
        async for n, response in async_enumerate(turn):
            if response.data is not None:
                wav.writeframes(response.data)
                if n == 0:
                    # Print the MIME type of the first chunk
                    print(response.server_content.model_turn.parts[0].inline_data.mime_type)
                print('.', end='')

# Play the generated audio
display(Audio(file_name, autoplay=True))
```

### Step 3: Creating a Reusable Audio Chat Loop
To build a more interactive experience, encapsulate the send/receive logic into a class. This separates concerns and prepares for fully asynchronous operation.

```python
class AudioLoop:
    def __init__(self, turns=None, config=None):
        self.session = None
        self.index = 0
        self.turns = turns  # Pre-defined list of messages, or None for interactive input
        if config is None:
            config = {"response_modalities": ["AUDIO"]}
        self.config = config

    async def run(self):
        """Main loop to open a session and process turns."""
        async with client.aio.live.connect(model=MODEL, config=self.config) as session:
            self.session = session
            async for sent in self.send():
                # In a full async app, send and receive would be separate tasks.
                await self.recv()

    async def _iter(self):
        """Iterator yielding user input messages."""
        if self.turns:
            for text in self.turns:
                print(f"message > {text}")
                yield text
        else:
            print("Type 'q' to quit")
            while True:
                text = await asyncio.to_thread(input, "message > ")
                if text.lower() == 'q':
                    break
                yield text

    async def send(self):
        """Sends user text to the Live API session."""
        async for text in self._iter():
            await self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True
            )
            yield text

    async def recv(self):
        """Receives audio from the API, saves it to a file, and plays it."""
        file_name = f"audio_{self.index}.wav"
        with wave_file(file_name) as wav:
            self.index += 1

            turn = self.session.receive()
            async for n, response in async_enumerate(turn):
                if response.data is not None:
                    wav.writeframes(response.data)
                    if n == 0:
                        print(response.server_content.model_turn.parts[0].inline_data.mime_type)
                    print('.', end='')
            print('\n<Turn complete>')

        display(Audio(file_name, autoplay=True))
        await asyncio.sleep(2)  # Brief pause between turns
```

**Run the chat loop with predefined messages:**

```python
await AudioLoop(['Hello', "What's your name?"]).run()
```

### Step 4: Working with Resumable Sessions
The Live API supports session resumption, allowing you to pause and later resume a conversation within a 24-hour window. This is useful for maintaining context.

First, set up helper functions for managing resumable sessions.

```python
import traceback
from asyncio.exceptions import CancelledError

last_handle = None  # Global variable to store the session handle

async def async_enumerate(aiterable):
    n = 0
    async for item in aiterable:
        yield n, item
        n += 1

def show_response(response):
    """Process and display a response from the Live API."""
    new_handle = None
    if text := response.text:
        print(text, end="")
    else:
        print(response.model_dump_json(indent=2, exclude_none=True))
    if response.session_resumption_update:
        new_handle = response.session_resumption_update.new_handle
    return new_handle

async def recv(session):
    """Task to continuously receive responses from the session."""
    global last_handle
    try:
        while True:
            async for response in session.receive():
                new_handle = show_response(response)
                if new_handle:
                    last_handle = new_handle
    except asyncio.CancelledError:
        pass

async def send(session):
    """Task to handle user input and send messages."""
    while True:
        message = await asyncio.to_thread(input, "message > ")
        if message.lower() == "q":
            break
        await session.send_client_content(turns={
            'role': 'user',
            'parts': [{'text': message}]
        })

async def async_main(last_handle=None):
    """Main function to manage a resumable Live API session."""
    config = types.LiveConnectConfig.model_validate({
        "response_modalities": ["TEXT"],
        "session_resumption": {
            'handle': last_handle,
        }
    })
    try:
        async with (
            client.aio.live.connect(model=MODEL, config=config) as session,
            asyncio.TaskGroup() as tg
        ):
            recv_task = tg.create_task(recv(session))
            send_task = tg.create_task(send(session))
            await send_task
            raise asyncio.CancelledError()  # Signal cancellation to other tasks
    except asyncio.CancelledError:
        pass
    except ExceptionGroup as EG:
        traceback.print_exception(EG)
```

**Start a new session and have a conversation:**

```python
await async_main()
```
During this session, ask a question like *"What is the capital of Brazil?"*. The API will return a session handle upon completion.

**Resume a previous session:**
After your first session, a handle is stored in `last_handle`. You can start a new session with this handle to resume the conversation.

```python
print(f"Last session handle: {last_handle}")
await async_main(last_handle)
```
When you resume, you can ask *"What was my last question?"* and the model will recall the context from the previous session.

## Next Steps & Resources

You've built a foundational text-to-audio chat using the Gemini Live API. To explore further:

1.  **Full Asynchrony:** Implement truly concurrent `send` and `recv` tasks for real-time, interruptible conversations.
2.  **Streaming Playback:** Use a library like `PyAudio` to play audio chunks as they arrive, instead of waiting for the entire response.
3.  **Multimodal Input:** Explore sending audio or video streams directly to the model.
4.  **Native Audio:** Try the new native audio generation for more expressive and natural-sounding responses. See the example script: `Get_started_LiveAPI_NativeAudio.py`.
5.  **Official Documentation:**
    *   [Gemini Live API Docs](https://ai.google.dev/gemini-api/docs/live)
    *   [Google AI SDK Documentation](https://ai.google.dev/gemini-api/docs/sdks)
    *   [Try Live API in Google AI Studio](https://aistudio.google.com/app/live) for a no-code experience.
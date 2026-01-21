# Comparing Speech-to-Text Methods with the OpenAI API

## Overview
This guide provides a hands-on tutorial for getting started with Speech-to-Text (STT) using the OpenAI API. You will explore four practical transcription methods, understand their ideal use cases, and learn how to implement each one.

By the end, you will be able to select and apply the appropriate transcription method for your projects.

**Note:** This tutorial uses pre-recorded WAV/MP3 audio files for simplicity. It does not cover real-time microphone streaming (e.g., from a web app or direct microphone input).

### Method Comparison
| Mode | Latency to First Token | Best For | Advantages | Key Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **File Upload (Blocking)** | Seconds | Voicemail, meeting recordings | Simple setup, processes full context | No partial results, 25 MB file limit |
| **File Upload (Streaming)** | Subseconds | Voice memos, mobile apps | Provides a "live" feel via token streaming | Requires a completed file upfront |
| **Realtime WebSocket API** | Subseconds | Live captions, webinars | True real-time, continuous audio stream | Complex integration, 30-min session limit, specific audio format required |
| **Agents SDK VoicePipeline** | Subseconds | Help-desk assistants, agentic workflows | Minimal boilerplate, built-in agent integration | Python-only beta, less low-level control |

## Prerequisites

### 1. Install Required Packages
Run the following command in your terminal or notebook cell to install the necessary libraries.

```bash
pip install --upgrade openai openai-agents websockets sounddevice pyaudio nest_asyncio resampy httpx websocket-client soundfile
```

### 2. Set Up Authentication
You must have a valid OpenAI API key. Set it as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Import Libraries and Initialize Client
Now, let's import the necessary modules and verify our setup.

```python
import asyncio
import struct
import base64
import json
import os
import time
from pathlib import Path
from typing import List

import nest_asyncio
import numpy as np
from openai import OpenAI
import resampy
import soundfile as sf
import websockets
from agents import Agent
from agents.voice import (
    SingleAgentVoiceWorkflow,
    StreamedAudioInput,
    VoicePipeline,
    VoicePipelineConfig,
)

# Apply nest_asyncio for running async code in notebooks
nest_asyncio.apply()

# Initialize the OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
print("✅ OpenAI client ready")
```

## Method 1: Speech-to-Text with Audio File (Blocking)

### When to Use This Method
Use this approach when:
* You have a completed audio file (up to 25 MB).
* You are processing recordings in batch (e.g., podcasts, call-center logs).
* You do not need real-time feedback or partial results.

### How It Works
The API processes the entire file in a single, blocking HTTP request and returns the complete transcript.

**Benefits:** Simple to use, accurate due to full-context processing, and supports many file formats (MP3, WAV, M4A, etc.).

**Limitations:** No partial results, latency scales with file duration, and has a 25 MB file size limit.

### Step 1: Load and Preview Your Audio File
First, let's load a sample audio file. Ensure you have a file available at the specified path.

```python
AUDIO_PATH = Path('./data/sample_audio_files/lotsoftimes-78085.mp3')  # Update this path
MODEL_NAME = "gpt-4o-transcribe"

# Check if the file exists
if not AUDIO_PATH.exists():
    print('⚠️ Please provide a valid audio file path.')
```

### Step 2: Transcribe the Audio
Now, call the STT endpoint to transcribe the entire file.

```python
if AUDIO_PATH.exists():
    with AUDIO_PATH.open('rb') as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model=MODEL_NAME,
            response_format='text',
        )
    print('\n--- TRANSCRIPT ---\n')
    print(transcript)
```

**Output:**
```
--- TRANSCRIPT ---

And lots of times you need to give people more than one link at a time. A band could give their fans a couple new videos from a live concert, a behind-the-scenes photo gallery, an album to purchase, like these next few links.
```

## Method 2: Speech-to-Text with Audio File (Streaming)

### When to Use This Method
Choose this method when:
* You have a completed audio file.
* You want to provide users with immediate, incremental transcription results (e.g., for a progress bar).
* You need a "real-time feel" for uploaded voice memos.

**Benefits:** Users see transcription updates as they arrive, improving engagement.
**Limitations:** Still requires the full file upfront and adds implementation complexity for handling the stream.

### Step 1: Stream the Transcription
We use the same endpoint but with the `stream=True` parameter.

```python
if AUDIO_PATH.exists():
    with AUDIO_PATH.open('rb') as audio_file:
        stream = client.audio.transcriptions.create(
            file=audio_file,
            model=MODEL_NAME,
            response_format='text',
            stream=True
        )

    for event in stream:
        # Print incremental updates (deltas) as they arrive
        if getattr(event, "delta", None):
            print(event.delta, end="", flush=True)
            time.sleep(0.05)  # Simulate real-time pacing
        # Print the final, complete transcript
        elif getattr(event, "text", None):
            print("\n" + event.text)
```

**Output:**
```
And lots of times you need to give people more than one link at a time. A band could give their fans a couple new videos from a live concert, a behind-the-scenes photo gallery, an album to purchase, like these next few links.

And lots of times you need to give people more than one link at a time. A band could give their fans a couple new videos from a live concert, a behind-the-scenes photo gallery, an album to purchase, like these next few links.
```

## Method 3: Realtime Transcription API (WebSocket)

### When to Use This Method
Ideal for:
* True real-time scenarios like live meeting captions or webinar subtitles.
* Applications requiring built-in voice activity detection (VAD) and noise suppression.
* Developers comfortable managing WebSocket connections and event streams.

**Benefits:** Ultra-low latency (300–800 ms), dynamic partial updates, and advanced features like turn detection.
**Limitations:** Complex integration, 30-minute session limit, and requires audio in specific PCM16 format.

### Step 1: Define Helper Functions
We need functions to handle audio encoding, WebSocket communication, and session management.

```python
TARGET_SR = 24_000          # Target sample rate (Hz)
PCM_SCALE = 32_767          # Scale factor for 16-bit PCM
CHUNK_SAMPLES = 3_072       # ~128 ms of audio at 24 kHz
RT_URL = "wss://api.openai.com/v1/realtime?intent=transcription"

# WebSocket event types
EV_DELTA = "conversation.item.input_audio_transcription.delta"
EV_DONE = "conversation.item.input_audio_transcription.completed"

def float_to_16bit_pcm(float32_array):
    """Convert a float32 array to 16-bit PCM bytes."""
    clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
    pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
    return pcm16

def base64_encode_audio(float32_array):
    """Encode a PCM audio chunk to base64."""
    pcm_bytes = float_to_16bit_pcm(float32_array)
    encoded = base64.b64encode(pcm_bytes).decode('ascii')
    return encoded

def load_and_resample(path: str, sr: int = TARGET_SR) -> np.ndarray:
    """Load an audio file, convert to mono, and resample to target rate."""
    data, file_sr = sf.read(path, dtype="float32")
    if data.ndim > 1:  # Convert stereo to mono if necessary
        data = data.mean(axis=1)
    if file_sr != sr:  # Resample if needed
        data = resampy.resample(data, file_sr, sr)
    return data

async def _send_audio(ws, pcm: np.ndarray, chunk: int, sr: int) -> None:
    """Producer: Stream base64-encoded audio chunks in real-time."""
    dur = 0.025  # Pacing interval
    t_next = time.monotonic()

    for i in range(0, len(pcm), chunk):
        float_chunk = pcm[i:i + chunk]
        payload = {
            "type": "input_audio_buffer.append",
            "audio": base64_encode_audio(float_chunk),
        }
        await ws.send(json.dumps(payload))
        t_next += dur
        await asyncio.sleep(max(0, t_next - time.monotonic()))
    # Signal end of audio
    await ws.send(json.dumps({"type": "input_audio_buffer.end"}))

async def _recv_transcripts(ws, collected: List[str]) -> None:
    """
    Consumer: Collect transcription deltas and finalize sentences.
    """
    current: List[str] = []
    try:
        async for msg in ws:
            ev = json.loads(msg)
            typ = ev.get("type")
            if typ == EV_DELTA:
                delta = ev.get("delta")
                if delta:
                    current.append(delta)
                    print(delta, end="", flush=True)
            elif typ == EV_DONE:
                # Sentence finished, add to final collection
                collected.append("".join(current))
                current.clear()
    except websockets.ConnectionClosedOK:
        pass
    # Flush any remaining partial sentence
    if current:
        collected.append("".join(current))

def _session(model: str, vad: float = 0.5) -> dict:
    """Create a session configuration dictionary."""
    return {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "threshold": vad},
            "input_audio_transcription": {"model": model},
        },
    }

async def transcribe_audio_async(
    wav_path,
    api_key,
    *,
    model: str = MODEL_NAME,
    chunk: int = CHUNK_SAMPLES,
) -> str:
    """Main async function to perform real-time transcription."""
    pcm = load_and_resample(wav_path)
    headers = {"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}

    async with websockets.connect(RT_URL, additional_headers=headers, max_size=None) as ws:
        # Initialize the session
        await ws.send(json.dumps(_session(model)))
        transcripts: List[str] = []
        # Run producer and consumer concurrently
        await asyncio.gather(
            _send_audio(ws, pcm, chunk, TARGET_SR),
            _recv_transcripts(ws, transcripts),
        )
    return " ".join(transcripts)
```

### Step 2: Run the Real-Time Transcription
Now, execute the function with your audio file.

```python
transcript = await transcribe_audio_async(AUDIO_PATH, OPENAI_API_KEY)
print("\n--- FINAL TRANSCRIPT ---")
print(transcript)
```

**Output:**
```
And lots of times you need to give people more than one link at a time.A band could give their fans a couple new videos from a live concert, a behind-the-scenes photo galleryLike these next few linksAn album to purchase.

--- FINAL TRANSCRIPT ---
And lots of times you need to give people more than one link at a time. A band could give their fans a couple new videos from a live concert, a behind-the-scenes photo gallery Like these next few linksAn album to purchase.
```

## Method 4: Agents SDK Realtime Transcription

### When to Use This Method
Use this method when:
* You are building with the OpenAI Agents SDK and want minimal setup.
* You need to integrate transcription directly into an agentic workflow (e.g., a voice assistant).
* You prefer a high-level API that manages WebSockets, buffering, and voice activity detection for you.

**Benefits:** Drastically reduces boilerplate code and seamlessly integrates with GPT agents.
**Limitations:** Currently in beta for Python only, and offers less low-level control.

### Step 1: Define an Agent and a Custom Workflow
We'll create a simple agent that translates user speech into French.

```python
# 1. Create an agent that replies in French
fr_agent = Agent(
    name="Assistant-FR",
    instructions="Translate the user's words into French.",
    model="gpt-4o-mini",
)

# 2. Create a custom workflow that prints interactions
class PrintingWorkflow(SingleAgentVoiceWorkflow):
    """Subclass that prints every chunk it yields (the agent's reply)."""
    async def run(self, transcription: str):
        print()
        print("[User]:", transcription)
        print("[Assistant]: ", end="", flush=True)
        async for chunk in super().run(transcription):
            print(chunk, end="", flush=True)   # Print agent's French text
            yield chunk                        # Forward to TTS (if enabled)

# 3. Initialize the VoicePipeline
pipeline = VoicePipeline(
    workflow=PrintingWorkflow(fr_agent),
    stt_model=MODEL_NAME,
    config=VoicePipelineConfig(tracing_disabled=True),
)
```

### Step 2: Create an Audio Streaming Helper
We need a function to chunk our audio file into real-time sized pieces.

```python
def audio_chunks(path: str, target_sr: int = 24_000, chunk_ms: int = 40):
    """
    Generator that yields chunks of PCM16 audio from a file.
    """
    # Reuse the load_and_resample helper from earlier
    audio = load_and_resample(path, target_sr)
    # Convert float32 to int16
    pcm = (np.clip(audio, -1, 1) * 32_767).astype(np.int16)
    # Calculate hop size in samples
    hop = int(target_sr * chunk_ms / 1_000)
    for off in range(0, len(pcm), hop):
        yield pcm[off : off + hop]
```

### Step 3: Stream Audio Through the Pipeline
Finally, we stream the audio chunks through the pipeline.

```python
async def stream_audio(path: str):
    """Stream an audio file through the VoicePipeline."""
    sai = StreamedAudioInput()
    # Start the pipeline in the background
    run_task = asyncio.create_task(pipeline.run(sai))

    for chunk in audio_chunks(path):
        await sai.add_audio(chunk)
        # Pace the audio to simulate real-time
        await asyncio.sleep(len(chunk) / 24_000)

    # Wait for the pipeline to finish processing
    await run_task

# Run the streaming function
await stream_audio(AUDIO_PATH)
```

**Output:**
```
[User]: And lots of times you need to give people more than one link at a time.
[Assistant]: Et souvent, vous devez donner aux gens plusieurs liens à la fois.
[User]: A band could give their fans a couple new videos from a live concert, a behind-the-scenes photo gallery.
[Assistant]: Un groupe pourrait donner à ses fans quelques nouvelles vidéos d'un concert live, ainsi qu'une galerie de photos des coulisses.
[User]: An album to purchase.
[Assistant]: Un album à acheter.
[User]: like these next few links.
[Assistant]: comme ces quelques liens suivants.
```

## Conclusion
In this tutorial, you explored four distinct methods for speech-to-text transcription using the OpenAI API:

1.  **Blocking File Upload:** The simplest method, ideal for batch processing completed recordings.
2.  **Streaming File Upload:** Provides incremental results for better user experience during uploads.
3.  **Realtime WebSocket API:** Enables true low-latency transcription for live audio streams, with advanced features like VAD.
4.  **Agents SDK VoicePipeline:** Offers a high-level, integrated approach for building real-time voice agents with minimal code.

Your choice depends on your specific needs: latency requirements, whether the audio is pre-recorded or live, and the complexity you are willing to manage. With these tools, you can now implement professional-grade speech-to-text features in your applications.
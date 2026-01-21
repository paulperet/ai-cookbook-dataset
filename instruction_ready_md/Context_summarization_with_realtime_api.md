# Context Summarization with Realtime API: A Voice Bot Tutorial

## Overview
Build an end-to-end **voice bot** that listens to your microphone, speaks back in real time, and **summarizes long conversations** to maintain consistent performance.

### What You'll Learn
1. **Live microphone streaming** using OpenAI's Realtime API (voice-to-voice endpoint)
2. **Instant transcripts & speech playback** on every conversation turn
3. **Conversation state management** that stores every user/assistant message
4. **Automatic context trimming** – when the token window becomes large, older turns are compressed into a summary
5. **Extensible design** you can adapt for customer-support bots, kiosks, or multilingual assistants

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python ≥ 3.10** | Ensures compatibility with required libraries |
| **OpenAI API key** | Set `OPENAI_API_KEY` environment variable |
| Microphone + speakers | Grant OS permission if prompted |

**Need help setting up the key?**  
Follow the [official quick-start guide](https://platform.openai.com/docs/quickstart#step-2-set-your-api-key).

### Important Notes
1. The gpt-realtime model supports a 32k token context window, but performance may degrade as you add more tokens
2. Token window includes all tokens (words and audio tokens) the model keeps in memory for the session
3. The Realtime API GA includes a new `truncation` parameter that automatically optimizes context truncation

## Setup

First, install the required dependencies:

```bash
pip install --upgrade openai websockets sounddevice simpleaudio
```

Now, import the necessary libraries:

```python
# Standard library imports
import os
import sys
import json
import base64
from dataclasses import dataclass, field
from typing import List, Literal

# Third-party imports
import asyncio
import sounddevice as sd         # microphone capture
import simpleaudio               # speaker playback
import websockets                # WebSocket client
import openai                    # OpenAI Python SDK >= 1.14.0
```

Set up your API key:

```python
# Set your API key safely
openai.api_key = os.getenv("OPENAI_API_KEY", "")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found – please set env var or edit this cell.")
```

## Understanding Token Utilization: Text vs Voice

Large token windows are precious resources – every extra token costs latency and money. For **audio**, the input token window increases much faster than for plain text because amplitude, timing, and other acoustic details must be represented.

In practice, you'll often see **≈ 10× more tokens** for the same sentence in audio versus text.

Key considerations:
- gpt-realtime accepts up to **32k tokens**
- Every user/assistant turn consumes tokens, causing the window to **only grow**
- **Strategy**: Summarize older turns into a single assistant message, keep the last few verbatim turns, and continue

## Step 1: Define Conversation State Management

Unlike HTTP-based Chat Completions, the Realtime API maintains an open, **stateful** session with two key components:

| Component | Purpose |
|-----------|---------|
| **Session** | Controls global settings – model, voice, modalities, VAD, etc. |
| **Conversation** | Stores turn-by-turn messages between user and assistant – both audio and text |

We'll wrap these components inside a simple `ConversationState` object to keep your logic clean, track history, and manage summarization when context windows fill up.

```python
@dataclass
class Turn:
    """One utterance in the dialogue (user **or** assistant)."""
    role: Literal["user", "assistant"]
    item_id: str                    # Server-assigned identifier
    text: str | None = None         # Filled once transcript is ready

@dataclass
class ConversationState:
    """All mutable data the session needs — nothing more, nothing less."""
    history: List[Turn] = field(default_factory=list)         # Ordered log
    waiting: dict[str, asyncio.Future] = field(default_factory=dict)  # Pending transcript fetches
    summary_count: int = 0

    latest_tokens: int = 0          # Window size after last reply
    summarising: bool = False       # Guard so we don't run two summaries at once
```

Add a helper function to view the conversation history:

```python
def print_history(state) -> None:
    """Pretty-print the running transcript so far."""
    print("—— Conversation so far ———————————————")
    for turn in state.history:
        text_preview = (turn.text or "").strip().replace("\n", " ")
        print(f"[{turn.role:<9}] {text_preview}  ({turn.item_id})")
    print("——————————————————————————————————————————")
```

## Step 2: Configure Audio Settings

Set up the audio configuration parameters:

```python
# Audio/config parameters
SAMPLE_RATE_HZ    = 24_000   # Required by pcm16
CHUNK_DURATION_MS = 40       # Chunk size for audio capture
BYTES_PER_SAMPLE  = 2        # pcm16 = 2 bytes/sample
SUMMARY_TRIGGER   = 2_000    # Summarize when context ≥ this
KEEP_LAST_TURNS   = 2        # Keep these turns verbatim
SUMMARY_MODEL     = "gpt-4o-mini"  # Cheaper, fast summarizer
```

**Note**: This tutorial uses `SUMMARY_TRIGGER = 2000` and `KEEP_LAST_TURNS = 2` to make summarization easier to demo quickly. In production, you should tune these values based on your application's needs. A typical `SUMMARY_TRIGGER` falls between **20,000–32,000 tokens**, depending on how performance degrades with larger context for your use case.

## Step 3: Implement Audio Streaming

We'll stream raw PCM-16 microphone data straight into the Realtime API using the pipeline: mic → async.Queue → WebSocket → Realtime API.

### 3.1 Capture Microphone Input

Create a coroutine that:
- Opens the default mic at **24 kHz, mono, PCM-16** (one of the formats Realtime accepts)
- Slices the stream into **≈ 40 ms** blocks
- Dumps each block into an `asyncio.Queue` so another task can forward it to OpenAI

```python
async def mic_to_queue(pcm_queue: asyncio.Queue[bytes]) -> None:
    """
    Capture raw PCM-16 microphone audio and push ~CHUNK_DURATION_MS chunks
    to *pcm_queue* until the surrounding task is cancelled.
    """
    blocksize = int(SAMPLE_RATE_HZ * CHUNK_DURATION_MS / 1000)

    def _callback(indata, _frames, _time, status):
        if status:                               # XRuns, device changes, etc.
            print("⚠️", status, file=sys.stderr)
        try:
            pcm_queue.put_nowait(bytes(indata))  # 1-shot enqueue
        except asyncio.QueueFull:
            # Drop frame if upstream (WebSocket) can't keep up.
            pass

    # RawInputStream is synchronous; wrap in context manager to auto-close.
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE_HZ,
        blocksize=blocksize,
        dtype="int16",
        channels=1,
        callback=_callback,
    ):
        try:
            # Keep coroutine alive until cancelled by caller.
            await asyncio.Event().wait()
        finally:
            print("⏹️  Mic stream closed.")
```

### 3.2 Send Audio Chunks to the API

Now that our mic task is filling an `asyncio.Queue` with raw PCM-16 blocks, we need to pull chunks off that queue, **base-64 encode** them (the protocol requires JSON-safe text), and ship each block to the Realtime WebSocket as an `input_audio_buffer.append` event.

```python
# Helper function to encode audio chunks in base64
b64 = lambda blob: base64.b64encode(blob).decode()

async def queue_to_websocket(pcm_queue: asyncio.Queue[bytes], ws):
    """Read audio chunks from queue and send as JSON events."""
    try:
        while (chunk := await pcm_queue.get()) is not None:
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64(chunk),
            }))
    except websockets.ConnectionClosed:
        print("WebSocket closed – stopping uploader")
```

## Step 4: Handle Incoming Events

Once audio reaches the server, the Realtime API pushes a stream of JSON events back over the **same** WebSocket. Understanding these events is critical for:
- Printing live transcripts
- Playing incremental audio back to the user
- Keeping an accurate conversation state so context trimming works later

| Event type | When it arrives | Why it matters | Typical handler logic |
|------------|-----------------|---------------|-----------------------|
| **`session.created`** | Immediately after WebSocket handshake | Confirms session is open and provides `session.id` | Log the ID for traceability and verify connection |
| **`session.updated`** | After you send a `session.update` call | Acknowledges server applied new session settings | Inspect echoed settings and update local cache |
| **`conversation.item.created`** (user) | A few ms after user stops speaking (client VAD fires) | Reserves timeline slot; transcript may still be **`null`** | Insert placeholder user turn in `state.history` marked "pending transcript" |
| **`conversation.item.retrieved`** | ~100–300 ms later, once audio transcription is complete | Supplies final user transcript (with timing) | Replace placeholder with transcript and print if desired |
| **`response.audio.delta`** | Every 20–60 ms while assistant is speaking | Streams PCM-16 audio chunks (and optional incremental text) | Buffer each chunk and play it; optionally show partial text in console |
| **`response.done`** | After assistant's last token | Signals both audio & text are complete; includes usage stats | Finalize assistant turn, update `state.latest_tokens`, and log usage |
| **`conversation.item.deleted`** | Whenever you prune with `conversation.item.delete` | Confirms turn was removed, freeing tokens on server | Mirror deletion locally so context window matches server's |

## Step 5: Implement Context Summarization

The Realtime model keeps a **large 32k-token window**, but quality can drift long before that limit as you add more context. Our goal: **auto-summarize** once the running window nears a safe threshold, then prune the superseded turns both locally *and* server-side.

We monitor `latest_tokens` returned in `response.done`. When it exceeds `SUMMARY_TRIGGER` and we have more than `KEEP_LAST_TURNS`, we spin up a background summarization coroutine.

### 5.1 Create the Summary LLM Function

```python
async def run_summary_llm(text: str) -> str:
    """Call a lightweight model to summarise `text`."""
    resp = await asyncio.to_thread(lambda: openai.chat.completions.create(
        model=SUMMARY_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "Summarise in French the following conversation "
                            "in one concise paragraph so it can be used as "
                            "context for future dialogue."},
            {"role": "user", "content": text},
        ],
    ))
    return resp.choices[0].message.content.strip()
```

**Important implementation detail**: The summary is appended as a SYSTEM message rather than an ASSISTANT message. Testing revealed that during extended conversations, using ASSISTANT messages for summaries can cause the model to mistakenly switch from audio responses to text responses. By using SYSTEM messages for summaries, we clearly signal to the model that these are context-setting instructions, preventing it from incorrectly adopting the modality of the ongoing user-assistant interaction.

### 5.2 Implement Summarization and Pruning

```python
async def summarise_and_prune(ws, state):
    """Summarise old turns, delete them server-side, and prepend a single summary
    turn locally + remotely."""
    state.summarising = True
    print(
        f"⚠️  Token window ≈{state.latest_tokens} ≥ {SUMMARY_TRIGGER}. Summarising…",
    )
    old_turns, recent_turns = state.history[:-KEEP_LAST_TURNS], state.history[-KEEP_LAST_TURNS:]
    convo_text = "\n".join(f"{t.role}: {t.text}" for t in old_turns if t.text)
    
    if not convo_text:
        print("Nothing to summarise (transcripts still pending).")
        state.summarising = False

    summary_text = await run_summary_llm(convo_text) if convo_text else ""
    state.summary_count += 1
    summary_id = f"sum_{state.summary_count:03d}"
    state.history[:] = [Turn("assistant", summary_id, summary_text)] + recent_turns
    
    print_history(state)    

    # Create summary on server
    await ws.send(json.dumps({
        "type": "conversation.item.create",
        "previous_item_id": "root",
        "item": {
            "id": summary_id,
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": summary_text}],
        },
    }))

    # Delete old items
    for turn in old_turns:
        await ws.send(json.dumps({
            "type": "conversation.item.delete",
            "item_id": turn.item_id,
        }))

    print(f"✅ Summary inserted ({summary_id})")
    
    state.summarising = False
```

### 5.3 Add Transcript Fetching Helper

```python
async def fetch_full_item(
    ws, item_id: str, state: ConversationState, attempts: int = 1
):
    """
    Ask the server for a full conversation item; retry up to 5× if the
    transcript field is still null. Resolve the waiting future when done.
    """
    # If there is already a pending fetch, just await it
    if item_id in state.waiting:
        return await state.waiting[item_id]

    fut = asyncio.get_running_loop().create_future()
    state.waiting[item_id] = fut

    await ws.send(json.dumps({
        "type": "conversation.item.retrieve",
        "item_id": item_id,
    }))
    item = await fut

    # If transcript still missing retry (max 5×)
    if attempts < 5 and not item.get("content", [{}])[0].get("transcript"):
        await asyncio.sleep(0.4 * attempts)
        return await fetch_full_item(ws, item_id, state, attempts + 1)

    # Done – remove the marker
    state.waiting.pop(item_id, None)
    return item
```

## Step 6: Create the Main Realtime Session

Now let's implement the main coroutine that connects to the Realtime endpoint, spawns helper tasks, and processes incoming events.

```python
async def realtime_session(model="gpt-realtime", voice="shimmer", enable_playback=True):
    """
    Main coroutine: connects to the Realtime endpoint, spawns helper tasks,
    and processes incoming events in a big async-for loop.
    """
    state = ConversationState()  # Reset state for each run
    pcm_queue: asyncio.Queue[bytes] = asyncio.Queue()
    assistant_audio: List[bytes] = []

    # Open the WebSocket connection to the Realtime API
    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {openai.api_key}"}

    async with websockets.connect(url, extra_headers=headers, max_size=1 << 24) as ws:
        print(f"✅ Connected to {model} (voice: {voice})")
        
        # Start microphone capture
        mic_task = asyncio.create_task(mic_to_queue(pcm_queue))
        
        # Start audio uploader
        upload_task = asyncio.create_task(queue_to_websocket(pcm_queue, ws))
        
        # Configure session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "You are a helpful voice assistant.",
                "voice": voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad"},
            }
        }))
        
        # Process incoming events
        try:
            async for message in ws:
                event = json.loads(message)
                event_type = event.get("type")
                
                # Handle different event types
                if event_type == "session.created":
                    print(f"Session ID: {event['session']['id']}")
                    
                elif event_type == "conversation.item.created":
                    item = event["item"]
                    if item["role"] == "user":
                        # Add placeholder for user turn
                        state.history.append(Turn("user", item["id"]))
                        
                elif event_type == "conversation.item.retrieved":
                    item = event["item"]
                    # Update transcript in history
                    for turn in state.history:
                        if turn.item_id == item["id"]:
                            if item["content"] and item["content"][0].get("transcript"):
                                turn.text = item["content"][0]["transcript"]
                                print(f"User: {turn.text}")
                            break
                            
                elif event_type == "response.audio.delta":
                    # Buffer assistant audio for playback
                    audio_data = base64.b64decode(event["delta"])
                    assistant_audio.append(audio_data)
                    
                elif event_type == "response.done":
                    # Update token usage
                    state.latest_tokens = event.get("usage", {}).get("total_tokens", 0)
                    
                    # Add assistant turn to history
                    if event.get("output"):
                        state.history.append(Turn("assistant", event["item_id"], event["output"][0]["text"]))
                        print
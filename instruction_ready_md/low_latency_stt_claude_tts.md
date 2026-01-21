# Building a Low-Latency Voice Assistant with ElevenLabs and Claude

This guide walks you through building a low-latency voice assistant that uses ElevenLabs for speech-to-text and text-to-speech, combined with Claude for intelligent conversation. You'll learn how to optimize response times by streaming both the LLM and audio generation.

## Prerequisites

Before you begin, ensure you have the following:

1.  **API Keys:**
    *   **ElevenLabs:** Get your API key from the [ElevenLabs Developers Console](https://elevenlabs.io/app/developers/api-keys).
    *   **Anthropic:** Get your API key from the [Anthropic Console](https://console.anthropic.com/settings/keys).

2.  **Environment Setup:** Create a `.env` file in your project directory with your keys:
    ```bash
    ELEVENLABS_API_KEY=your_elevenlabs_key_here
    ANTHROPIC_API_KEY=your_anthropic_key_here
    ```

## Step 1: Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
anthropic
elevenlabs
python-dotenv
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

## Step 2: Import Libraries and Configure API Clients

Now, import the necessary libraries and initialize the clients for ElevenLabs and Anthropic.

```python
import io
import os
import time

import anthropic
import elevenlabs
from dotenv import load_dotenv

# Load API keys from the .env file
load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Validate that keys are present
assert ELEVENLABS_API_KEY is not None, (
    "ERROR: ELEVENLABS_API_KEY not found. Please add it to your .env file."
)
assert ANTHROPIC_API_KEY is not None, (
    "ERROR: ANTHROPIC_API_KEY not found. Please add it to your .env file."
)

# Initialize the API clients
elevenlabs_client = elevenlabs.ElevenLabs(
    api_key=ELEVENLABS_API_KEY, base_url="https://api.elevenlabs.io"
)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
```

## Step 3: Select a Voice

First, let's list the available voices and select one for the assistant to use.

```python
print("Available Voices:")
voices = elevenlabs_client.voices.search().voices
for voice in voices:
    print(f"{voice.name}: {voice.voice_id}")

# Select the first voice for this example
selected_voice = voices[0]
VOICE_ID = selected_voice.voice_id
print(f"\nSelected voice: {selected_voice.name} with ID: {VOICE_ID}")
```

## Step 4: Simulate User Input with Text-to-Speech

We'll start by simulating a user speaking. We'll generate an audio file from text using ElevenLabs' TTS.

```python
# Generate audio from a sample user query
audio = elevenlabs_client.text_to_speech.convert(
    voice_id=VOICE_ID,
    output_format="mp3_44100_128",
    model_id="eleven_v3",
    text="Hello, Claude. ",
)

# Store the audio in a buffer
audio_data = io.BytesIO()
for chunk in audio:
    audio_data.write(chunk)

print("Sample user audio generated.")
```

## Step 5: Transcribe Audio to Text

Next, we'll transcribe the generated audio back into text using ElevenLabs' speech-to-text model. This simulates the assistant "hearing" the user.

```python
# Reset the buffer pointer to the beginning
audio_data.seek(0)

start_time = time.time()
# Perform the transcription
transcription = elevenlabs_client.speech_to_text.convert(file=audio_data, model_id="scribe_v1")
end_time = time.time()

transcription_time = end_time - start_time
print(f"Transcribed text: {transcription.text}")
print(f"Transcription time: {transcription_time:.2f} seconds")
```

## Step 6: Get a Response from Claude (Baseline)

Now, let's send the transcribed text to Claude and get a response. We'll first do this without streaming to establish a baseline latency.

```python
start_time = time.time()

message = anthropic_client.messages.create(
    model="claude-haiku-4-5",  # A fast, capable model
    max_tokens=1000,
    temperature=0,
    messages=[{"role": "user", "content": transcription.text}],
)

end_time = time.time()
non_streaming_response_time = end_time - start_time

print(f"Claude's Response: {message.content[0].text}")
print(f"Non-streaming response time: {non_streaming_response_time:.2f} seconds")
```

## Step 7: Optimize Latency with Claude Streaming

To improve the user experience, we can use Claude's streaming API. This sends tokens as they are generated, allowing us to receive and process the beginning of the response much faster.

```python
start_time = time.time()
first_token_time = None
claude_full_response = ""

print("Streaming Response: ", end="", flush=True)

with anthropic_client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=1000,
    temperature=0,
    messages=[{"role": "user", "content": transcription.text}],
) as stream:
    for text in stream.text_stream:
        claude_full_response += text
        print(text, end="", flush=True)
        if first_token_time is None:
            first_token_time = time.time()

streaming_time_to_first_token = first_token_time - start_time
latency_reduction = ((non_streaming_response_time - streaming_time_to_first_token) /
                     non_streaming_response_time) * 100

print(f"\n\nStreaming time to first token: {streaming_time_to_first_token:.2f} seconds")
print(f"Latency reduced by: {latency_reduction:.2f}%")
```

## Step 8: Stream Text-to-Speech

Just as we streamed Claude's response, we can also stream the TTS process. Instead of waiting for the entire audio file to be generated, we can begin playback as soon as the first audio chunk is ready.

```python
start_time = time.time()
first_audio_chunk_time = None
audio_buffer = io.BytesIO()

print("Generating streaming TTS...")
audio_generator = elevenlabs_client.text_to_speech.stream(
    voice_id=VOICE_ID,
    output_format="mp3_44100_128",
    text=claude_full_response,
    model_id="eleven_turbo_v2_5",  # A faster model for streaming
)

for chunk in audio_generator:
    if first_audio_chunk_time is None:
        first_audio_chunk_time = time.time()
    audio_buffer.write(chunk)

streaming_tts_time_to_first_chunk = first_audio_chunk_time - start_time
print(f"Streaming TTS time to first audio chunk: {streaming_tts_time_to_first_chunk:.2f} seconds")
```

## Step 9: Advanced Optimization: Sentence-by-Sentence Streaming

We can achieve even lower perceived latency by combining the streams. As Claude generates text, we can detect complete sentences and immediately send them to the TTS engine.

**Note:** This method can cause unnatural pauses between sentences, as the TTS lacks cross-sentence context.

```python
import re

sentence_pattern = re.compile(r"[.!?]+")
sentence_buffer = ""
audio_chunks = []

start_time = time.time()
first_audio_time = None

print("Sentence-by-Sentence Streaming: ", end="", flush=True)

with anthropic_client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=1000,
    temperature=0,
    messages=[{"role": "user", "content": transcription.text}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
        sentence_buffer += text

        # Check if we have a complete sentence
        if sentence_pattern.search(sentence_buffer):
            sentences = sentence_pattern.split(sentence_buffer)

            # Process all complete sentences
            for i in range(len(sentences) - 1):
                complete_sentence = sentences[i].strip()
                if complete_sentence:
                    # Stream TTS for this sentence
                    audio_gen = elevenlabs_client.text_to_speech.stream(
                        voice_id=VOICE_ID,
                        output_format="mp3_44100_128",
                        text=complete_sentence,
                        model_id="eleven_turbo_v2_5",
                    )

                    sentence_audio = io.BytesIO()
                    for chunk in audio_gen:
                        if first_audio_time is None:
                            first_audio_time = time.time()
                        sentence_audio.write(chunk)

                    audio_chunks.append(sentence_audio.getvalue())

            # Keep the last, potentially incomplete sentence in the buffer
            sentence_buffer = sentences[-1]

# Process any remaining text in the buffer
if sentence_buffer.strip():
    audio_gen = elevenlabs_client.text_to_speech.stream(
        voice_id=VOICE_ID,
        output_format="mp3_44100_128",
        text=sentence_buffer.strip(),
        model_id="eleven_turbo_v2_5",
    )
    sentence_audio = io.BytesIO()
    for chunk in audio_gen:
        sentence_audio.write(chunk)
    audio_chunks.append(sentence_audio.getvalue())

sentence_streaming_time_to_first_audio = first_audio_time - start_time
print(f"\n\nTime to first audio (sentence streaming): {sentence_streaming_time_to_first_audio:.2f} seconds")
```

## Next Steps: Building a Production System

The techniques above form the core of a low-latency voice assistant. For a production-ready application, consider these additional components:

1.  **WebSocket API:** ElevenLabs offers a WebSocket API for TTS, which is ideal for true bidirectional streaming. It accepts text chunks as they arrive and maintains prosodic context across them, solving the disjointed audio issue of the sentence-by-sentence method.
2.  **Audio Playback Management:** Implement a robust audio queue to play chunks seamlessly without gaps or artifacts.
3.  **Microphone Integration:** Capture real-time user audio from a microphone input.
4.  **Conversation State:** Manage a history of messages to maintain context across multiple turns in a conversation.

You now have the foundational knowledge to build and optimize a voice assistant that feels responsive and natural to the user.
# Guide: Generating Speech with the Gemini Text-to-Speech (TTS) API

This guide demonstrates how to use the Gemini API to generate high-quality speech from text. You'll learn to control voice characteristics, language, style, and even create multi-speaker audio conversations.

## Prerequisites

### 1. Obtain an API Key
You'll need a Gemini API key. Store it securely as an environment variable or in your development environment.

### 2. Install Required Libraries
Install the Google Generative AI Python SDK:

```bash
pip install -U "google-genai>=1.16.0"
```

**Note:** Version 1.16.0 or higher is required for multi-speaker audio functionality.

### 3. Import Libraries and Initialize Client
```python
from google import genai
from google.genai import types

# Initialize the client with your API key
client = genai.Client(api_key="YOUR_API_KEY")
```

### 4. Select a TTS Model
Only specific Gemini models support audio output. Choose one of these:
- `gemini-2.5-flash-preview-tts`
- `gemini-2.5-pro-preview-tts`

```python
MODEL_ID = "gemini-2.5-flash-preview-tts"
```

### 5. Create Audio Playback Helper Functions
These functions will help you play generated audio directly in your environment:

```python
import contextlib
import wave
from IPython.display import Audio

file_index = 0

@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

def play_audio_blob(blob):
    global file_index
    file_index += 1
    
    fname = f'audio_{file_index}.wav'
    with wave_file(fname) as wav:
        wav.writeframes(blob.data)
    
    return Audio(fname, autoplay=True)

def play_audio(response):
    return play_audio_blob(response.candidates[0].content.parts[0].inline_data)
```

## Step 1: Generate Basic Speech

Let's start with a simple text-to-speech conversion. The TTS model requires explicit instructions to "say" or "read" content.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Say 'hello, my name is Gemini!'",
    config={"response_modalities": ['Audio']},
)

# The audio data is stored in the response
blob = response.candidates[0].content.parts[0].inline_data
print(f"Audio format: {blob.mime_type}")

# Play the generated audio
play_audio_blob(blob)
```

The output will be in PCM format at 24kHz sample rate.

## Step 2: Control Voice Characteristics

### Select a Prebuilt Voice
Gemini offers 30 different built-in voices. You can specify which voice to use:

```python
voice_name = "Sadaltager"  # Choose from available voices

response = client.models.generate_content(
    model=MODEL_ID,
    contents="""Say "I am a very knowledgeable model, especially when using grounding", wait 5 seconds then say "Don't you think?".""",
    config={
        "response_modalities": ['Audio'],
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": voice_name
                }
            }
        }
    },
)

play_audio(response)
```

### Change Language
Simply instruct the model to speak in your desired language. Gemini supports 24 languages including French, Spanish, German, and more.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="""
        Read this in French:
        
        Les chaussettes de l'archiduchesse sont-elles sèches ? Archi-sèches ?
        Un chasseur sachant chasser doit savoir chasser sans son chien.
    """,
    config={"response_modalities": ['Audio']},
)

play_audio(response)
```

## Step 3: Control Speaking Style with Prompts

You can use natural language prompts to control style, tone, accent, and pace.

### Create a Spooky Whisper
```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="""
        Say in a spooky whisper:
        "By the pricking of my thumbs...
        Something wicked this way comes!"
    """,
    config={"response_modalities": ['Audio']},
)

play_audio(response)
```

### Generate Fast-Paced Speech
```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="""
        Read this disclaimer in as fast a voice as possible while remaining intelligible:
        
        [The author] assumes no responsibility or liability for any errors or omissions in the content of this site.
        The information contained in this site is provided on an 'as is' basis with no guarantees of completeness, accuracy, usefulness or timeliness
    """,
    config={"response_modalities": ['Audio']},
)

play_audio(response)
```

## Step 4: Create Multi-Speaker Conversations

### Basic Multi-Speaker Audio
Create a conversation between two speakers with different emotional tones:

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="""
        Make Speaker1 sound tired and bored, and Speaker2 sound excited and happy:
        
        Speaker1: So... what's on the agenda today?
        Speaker2: You're never going to guess!
    """,
    config={"response_modalities": ['Audio']},
)

play_audio(response)
```

### Advanced Multi-Speaker with Custom Voices
For more control, you can assign specific voices to each speaker and generate a complete conversation.

First, generate a conversation transcript using a text model:

```python
# Generate a podcast transcript using a text model
transcript = client.models.generate_content(
    model='gemini-2.5-flash',
    contents="""
        Hi, please generate a short (like 100 words) transcript that reads like
        it was clipped from a podcast by excited herpetologists, Dr. Claire and
        her assistant, the young Aurora.
    """
).text

print(transcript)
```

Then, render the conversation with specific voices for each speaker:

```python
# Configure multi-speaker voices
config = types.GenerateContentConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
            speaker_voice_configs=[
                types.SpeakerVoiceConfig(
                    speaker='Dr. Claire',
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='sulafat',
                        )
                    )
                ),
                types.SpeakerVoiceConfig(
                    speaker='Aurora',
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Leda',
                        )
                    )
                ),
            ]
        )
    )
)

# Generate the audio conversation
response = client.models.generate_content(
    model=MODEL_ID,
    contents="TTS the following conversation between a very excited Dr. Claire and her assistant, the young Aurora: " + transcript,
    config=config,
)

play_audio(response)
```

## Key Considerations

1. **TTS-Only Models**: The TTS models (`gemini-2.5-flash-preview-tts` and `gemini-2.5-pro-preview-tts`) only perform text-to-speech conversion. They don't have reasoning capabilities like other Gemini models.

2. **Explicit Instructions**: Always use verbs like "say", "read", or "TTS" in your prompts to ensure the model generates audio.

3. **Voice Selection**: Experiment with different voices to find the best fit for your application. Each voice has unique characteristics documented in the [Gemini API documentation](https://ai.google.dev/gemini-api/docs/speech-generation#voices).

## Next Steps

Now that you can generate speech with Gemini, consider exploring:
- **Music Generation**: Use the Lyria RealTime API for musical conversations
- **Image & Video Generation**: Explore Gemini's image and video generation capabilities
- **Audio Understanding**: Learn how Gemini can analyze and understand audio files
- **Real-time Conversations**: Implement live conversations using the Live API

For more detailed information, refer to the [official Gemini API documentation](https://ai.google.dev/gemini-api/docs/audio-generation).
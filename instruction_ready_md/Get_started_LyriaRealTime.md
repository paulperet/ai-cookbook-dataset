# Interactive Music Generation with Lyria RealTime

## Introduction

Lyria RealTime provides access to a state-of-the-art, streaming music generation model that enables developers to build applications where users can interactively create, continuously steer, and perform instrumental music using text prompts.

**Key Features:**
- **High-quality text-to-audio**: Generates instrumental music (no voice) using DeepMind's latest models
- **Continuous streaming**: Uses WebSockets to generate music in real-time
- **Creative blending**: Mix prompts to describe musical ideas, genres, instruments, moods, or characteristics
- **Real-time control**: Adjust guidance, BPM, density, brightness, and scale during generation

> **Note**: Lyria RealTime is a preview feature with quota limitations. Due to Colab limitations, this tutorial demonstrates the API but cannot showcase real-time streaming capabilities. For the full experience, use the Python script or AI Studio's Prompt DJ and MIDI DJ apps.

## Prerequisites

### 1. Install the SDK

```bash
pip install -U -q "google-genai>=1.16.0"
```

### 2. Set Up API Authentication

Store your API key in a Colab Secret named `GOOGLE_API_KEY`:

```python
from google.colab import userdata
import os

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### 3. Initialize the Client

Lyria RealTime requires the experimental `v1alpha` API version:

```python
from google import genai
from google.genai import types

client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={'api_version': 'v1alpha'},  # Required for experimental features
)

MODEL_ID = 'models/lyria-realtime-exp'
```

## Helper Functions

### 1. Logging Setup

For debugging purposes, enable detailed logging:

```python
import logging

logger = logging.getLogger('Bidi')
logger.setLevel('DEBUG')
```

### 2. WAV File Writer

Create a context manager for writing audio files:

```python
import contextlib
import wave

@contextlib.contextmanager
def wave_file(filename, channels=2, rate=48000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
```

### 3. Prompt Parser

Parse text input into weighted prompts:

```python
def parse_input(input_text):
    if ":" in input_text:
        parsed_prompts = []
        segments = input_text.split(',')
        malformed_segment_exists = False

        for segment_str_raw in segments:
            segment_str = segment_str_raw.strip()
            if not segment_str:
                continue

            parts = segment_str.split(':', 1)

            if len(parts) == 2:
                text_p = parts[0].strip()
                weight_s = parts[1].strip()

                if not text_p:
                    print(f"Error: Empty prompt text in segment '{segment_str_raw}'. Skipping this segment.")
                    malformed_segment_exists = True
                    continue
                
                try:
                    weight_f = float(weight_s)
                    parsed_prompts.append(types.WeightedPrompt(text=text_p, weight=weight_f))
                except ValueError:
                    print(f"Error: Invalid weight '{weight_s}' in segment '{segment_str_raw}'. Must be a number. Skipping this segment.")
                    malformed_segment_exists = True
                    continue
            else:
                print(f"Error: Segment '{segment_str_raw}' is not in 'text:weight' format. Skipping this segment.")
                malformed_segment_exists = True
                continue

        if parsed_prompts:
            prompt_repr = [f"'{p.text}':{p.weight}" for p in parsed_prompts]
            if malformed_segment_exists:
                print(f"Partially sending {len(parsed_prompts)} valid weighted prompt(s) due to errors in other segments: {', '.join(prompt_repr)}")
            else:
                print(f"Sending multiple weighted prompts: {', '.join(prompt_repr)}")
            return parsed_prompts
        else:
            print("Error: Input contained ':' suggesting multi-prompt format, but no valid 'text:weight' segments were successfully parsed. No action taken.")
            return None
    else:
        print(f"Sending single text prompt: \"{input_text}\"")
        return types.WeightedPrompt(text=input_text, weight=1.0)
```

## Core Music Generation Loop

### Implementation Overview

The following class manages the WebSocket connection and audio streaming. This simplified version focuses on understanding the core workflow:

```python
import asyncio

file_index = 0

async def generate_music(prompts=None, max_chunks=10, config=None):
    async with client.aio.live.music.connect(model=MODEL_ID) as session:
        
        async def receive():
            global file_index
            # Start a new WAV file
            file_name = f"audio_{file_index}.wav"
            with wave_file(file_name) as wav:
                file_index += 1
                logger.debug('receive')

                # Read audio chunks from the WebSocket
                n = 0
                async for message in session.receive():
                    n += 1
                    if n > max_chunks:
                        break

                    # Write audio chunk to WAV file
                    audio_chunk = message.server_content.audio_chunks[0].data
                    if audio_chunk is not None:
                        logger.debug('Got audio_chunk')
                        wav.writeframes(audio_chunk)

                    await asyncio.sleep(10**-12)

        # Prompt collection (if not provided)
        while prompts is None:
            input_prompt = await asyncio.to_thread(input, "prompt > ")
            prompts = parse_input(input_prompt)

        # Send initial prompts
        await session.set_weighted_prompts(prompts=prompts)

        # Set initial configuration
        if config is not None:
            await session.set_music_generation_config(config=config)

        # Start music generation
        await session.play()

        # Begin receiving audio
        receive_task = asyncio.create_task(receive())
        await asyncio.gather(receive_task)
```

**Key Methods:**
- `generate_music()`: Main function that establishes the WebSocket connection, sends prompts/config, and starts generation
- `receive()`: Collects audio chunks from the API and writes them to a WAV file

> **Note**: For real-time interaction, you would need to implement a `send()` method to update prompts/config during generation. See the Python code sample for a complete implementation.

## Basic Example: Single Prompt

Let's start with a simple music generation using a single prompt:

```python
from IPython.display import display, Audio

# Generate music with a single prompt
await generate_music(prompts=[{"text":"piano", "weight":1.0}])

# Play the generated audio
display(Audio(f"audio_{file_index-1}.wav"))
```

## Interactive Music Generation

Now let's create a more sophisticated example with multiple prompts and configuration options.

### Understanding Prompts and Configuration

**Prompt Format:**
```python
{
    "text": "Text of the prompt",
    "weight": 1.0,  # Can be any value (including negative), except 0
}
```

**Music Configuration Options:**
- `bpm`: Beats per minute (40-180)
- `guidance`: How strictly the model follows prompts (0-6)
- `density`: Density of musical notes/sounds (0-1)
- `brightness`: Tonal quality (0-1)
- `scale`: Musical scale (key and mode)
- `music_generation_mode`: Quality, diversity, or vocalization

### Interactive Generation Script

```python
# @markdown ### Enter some prompts:
prompt_1 = "Indie Pop"  # @param ["Hard Rock","Latin Jazz","Polka","Baroque","Chiptune","Indie Pop","Bluegrass","Heavy Metal","Contemporary R&B","Reggaeton"] {"allow-input":true}
prompt_1_weight = 0.6  # @param {type:"slider", min:0, max:2, step:0.1}
prompt_2 = "Sitar"  # @param ["Piano","Guitar","Bagpipes","Harpsichord","808 Hip Hop Beat","Sitar","Harmonica","Didgeridoo","Woodwinds","Organ"] {"allow-input":true}
prompt_2_weight = 2  # @param {type:"slider", min:0, max:2, step:0.1}
prompt_3 = "Danceable"  # @param ["Chill","Emotional","Danceable","Psychedelic","Acoustic Instruments","Glitchy Effects","Ominous Drone","Upbeat"] {"allow-input":true}
prompt_3_weight = 1.4  # @param {type:"slider", min:0, max:2, step:0.1}
prompt_4 = ""  # @param {"type":"string","placeholder":"Fourth prompt (optional)"}
prompt_4_weight = 1.0  # @param {type:"slider", min:0, max:2, step:0.1}
prompt_5 = ""  # @param {"type":"string","placeholder":"Fifth prompt (optional)"}
prompt_5_weight = 1.0  # @param {type:"slider", min:0, max:2, step:0.1}

# @markdown ### Music configuration:
BPM = 140  # @param {type:"slider", min:40, max:180, step:1}
scale = "F_MAJOR_D_MINOR"  # @param ["SCALE_UNSPECIFIED","C_MAJOR_A_MINOR","D_FLAT_MAJOR_B_FLAT_MINOR","D_MAJOR_B_MINOR","E_FLAT_MAJOR_C_MINOR","E_MAJOR_D_FLAT_MINOR","F_MAJOR_D_MINOR","G_FLAT_MAJOR_E_FLAT_MINOR","G_MAJOR_E_MINOR","A_FLAT_MAJOR_F_MINOR","A_MAJOR_G_FLAT_MINOR","B_FLAT_MAJOR_G_MINOR","B_MAJOR_A_FLAT_MINOR"]
density = 0.2  # @param {type:"slider", min:0, max:1, step:0.1}
brightness = 0.7  # @param {type:"slider", min:0, max:1, step:0.1}
guidance = 4.0  # @param {type:"slider", min:0, max:6, step:0.1}
music_generation_mode = "QUALITY"  # @param ["QUALITY","DIVERSITY","VOCALIZATION"]

# @markdown ### Duration (in seconds):
duration = 20  # @param {type:"slider", min:2, max:60, step:2}

# Build prompts list
prompts = [{
    "text": prompt_1,
    "weight": prompt_1_weight,
}]

# Add optional prompts
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

# Build configuration
config = {
    'music_generation_config': {
        'bpm': BPM,
        'scale': scale,
        'density': density,
        'brightness': brightness,
        'guidance': guidance,
        'music_generation_mode': music_generation_mode,
    }
}

# Generate music
await generate_music(max_chunks=duration/2, prompts=prompts, config=config)

# Play the result
display(Audio(f"audio_{file_index-1}.wav"))
```

## Next Steps

Now that you understand Lyria RealTime basics, explore these related capabilities:

1. **Multi-speaker conversations** using Text-to-Speech models
2. **Image and video generation** with other Google AI models
3. **Audio understanding** with Gemini's audio processing capabilities
4. **Real-time conversations** using the Live API

For production applications, refer to the complete Python code sample that includes proper thread management, error handling, and real-time interaction capabilities.
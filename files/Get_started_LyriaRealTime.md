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

# Get started with Music generation using Lyria RealTime

Lyria RealTime, provides access to a state-of-the-art, real-time, streaming music generation model. It allows developers to build applications where users can interactively create, continuously steer, and perform instrumental music using text prompts.

Lyria RealTime main characteristics are:
* **Highest quality text-to-audio model**: Lyria RealTime generates high-quality instrumental music (no voice) using the latest models produced by DeepMind.
* **Non-stopping music**: Using websockets, Lyria RealTime continuously generates music in real time.
* **Mix and match influences**: Prompt the model to describe musical idea, genre, instrument, mood, or characteristic. The prompts can be mixed to blend influences and create unique compositions.
* **Creative control**: Set the `guidance`, the `bpm`, the `density` of musical notes/sounds, the `brightness` and the `scale` in real time. The model will smoothly transition based on the new input.

Check Lyria RealTime's documentation for more details.

Lyria RealTime is a preview feature. It is free to use for now with quota limitations, but is subject to change.

**Also note that due to Colab limitation, you won't be able to experience the real time capabilities of Lyria RealTime but only limited audio output. Use the Python script or the AI studio's apps, Prompt DJ and MIDI DJ to fully experience Lyria RealTime**

# Setup

## Install the SDK
Even if this notebook won't use the SDK, it will still use python and colab function to manage the websockets and the audio output.


```
%pip install -U -q "google-genai>=1.16.0" # 1.16 is needed for the Lyria RealTime support
```

[..., ...]

## API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see Authentication for an example.


```
from google.colab import userdata
import os

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

## Selecting the model and initializing the SDK client

Lyria RealTime API is a new capability introduced with the Lyria RealTime model so only works with the `lyria-realtime-exp` model.

As it's an experimental feature, you also need to use the  `v1alpha` client version.



```
from google import genai
from google.genai import types

client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={'api_version': 'v1alpha'}, # v1alpha since Lyria RealTime is only experimental
)

MODEL_ID = 'models/lyria-realtime-exp'
```

## Helpers


```
# @title Logging
# For the sake of understanding how Lyria RealTime works, all logs are going to
# be displayed, but feel free to comment those lines if that's too much for you.

import logging

logger = logging.getLogger('Bidi')
logger.setLevel('DEBUG')
```


```
# @title Wave file writer

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


```
# @title Text to prompt parser

def parse_input(input_text):
  if ":" in input_text:
    parsed_prompts = []
    segments = input_text.split(',')
    malformed_segment_exists = False # Tracks if any segment had a parsing error

    for segment_str_raw in segments:
        segment_str = segment_str_raw.strip()
        if not segment_str: # Skip empty segments (e.g., from "text1:1, , text2:2")
            continue

        # Split on the first colon only, in case prompt text itself contains colons
        parts = segment_str.split(':', 1)

        if len(parts) == 2:
            text_p = parts[0].strip()
            weight_s = parts[1].strip()

            if not text_p: # Prompt text should not be empty
                print(f"Error: Empty prompt text in segment '{segment_str_raw}'. Skipping this segment.")
                malformed_segment_exists = True
                continue # Skip this malformed segment
            try:
                weight_f = float(weight_s) # Weights are floats
                parsed_prompts.append(types.WeightedPrompt(text=text_p, weight=weight_f))
            except ValueError:
                print(f"Error: Invalid weight '{weight_s}' in segment '{segment_str_raw}'. Must be a number. Skipping this segment.")
                malformed_segment_exists = True
                continue # Skip this malformed segment
        else:
            # This segment is not in "text:weight" format.
            print(f"Error: Segment '{segment_str_raw}' is not in 'text:weight' format. Skipping this segment.")
            malformed_segment_exists = True
            continue # Skip this malformed segment

    if parsed_prompts: # If at least one prompt is successfully parsed
        prompt_repr = [f"'{p.text}':{p.weight}" for p in parsed_prompts]
        if malformed_segment_exists:
            print(f"Partially sending {len(parsed_prompts)} valid weighted prompt(s) due to errors in other segments: {', '.join(prompt_repr)}")
        else:
            print(f"Sending multiple weighted prompts: {', '.join(prompt_repr)}")
        return parsed_prompts
    else: # No valid prompts were parsed from the input string that contained ":"
        print("Error: Input contained ':' suggesting multi-prompt format, but no valid 'text:weight' segments were successfully parsed. No action taken.")
        return None
  else:
    print(f"Sending single text prompt: \"{input_text}\"")
    return types.WeightedPrompt(text=input_text, weight=1.0)

```

# Main audio loop

The class below implements the interaction with the Lyria RealTime API.

This is a basic implementation that could be improved but was kept as simple as possible to keep it easy to understand.

The python script is a more complete example with better thread and error handling and most of all, real-time interractions.

There are 2 methods worth describing here:

`generate_music` - The main function

This method:

- Opens a `websocket` connecting to the real time API
- Sends the initial prompt to the model using `session.set_weighted_prompts`. If none was provided it asked for a prompt and parse it using the `parse_input` helper.
- If provided, it then send the music generation configuration using `session.set_music_generation_config`
- Finally it starts the music generation with `session.play()`

`receive` - Collects audio from the API and plays it

The `receive` method listen to the model ouputs and collects the audio chunks in a loop and writes them to a `.wav` file using the `wave_file` helper. It stops after a certain number of chunks (10 by default).

Ideally if you want to interact in real-time with Lyria RealTime you should also implement a `send` method to send the new prompts/config to the model. Check the python code sample for such an example.


```
import asyncio

file_index = 0

async def generate_music(prompts=None, max_chunks=10, config=None):
    async with client.aio.live.music.connect(model=MODEL_ID) as session:
        async def receive():
          global file_index
          # Start a new `.wav` file.
          file_name = f"audio_{file_index}.wav"
          with wave_file(file_name) as wav:
            file_index += 1

            logger.debug('receive')

            # Read chunks from the socket.
            n = 0
            async for message in session.receive():
              n+=1
              if n > max_chunks:
                break

              # Write audio the chunk to the `.wav` file.
              audio_chunk = message.server_content.audio_chunks[0].data
              if audio_chunk is not None:
                logger.debug('Got audio_chunk')
                wav.writeframes(audio_chunk)

              await asyncio.sleep(10**-12)

        # This code example doesn't have a way to receive requests because of colab
        # limitations, check the python code sample for a more complete example

        while prompts is None:
          input_prompt = await asyncio.to_thread(input, "prompt > ")
          prompts = parse_input(input_prompt)

        # Sending the provided prompts
        await session.set_weighted_prompts(
            prompts=prompts
        )

        # Set initial configuration
        if config is not None:
          await session.set_music_generation_config(config=config)

        # Start music generation
        await session.play()

        receive_task = asyncio.create_task(receive())

        # Don't quit the loop until tasks are done
        await asyncio.gather(receive_task)
```

# Try Lyria RealTime

Because of Colab limitation you won't be able to experience the "real time" part of Lyria RealTime, so all those examples are going to be one-offs prompt to get an audio file.

One thing to note is that the audio will only be played at the end of the session when all would have been written in the wav file. When using the API for real you'll be able to start plyaing as soon as the first chunk arrives. So the longer the duration (using the dedicated parameter) you set, the longer you'll have to wait until you hear something.

## Simple Lyria RealTime example
Here's first a simple example:


```
from IPython.display import display, Audio

await generate_music(prompts=[{"text":"piano", "weight":1.0}])
display(Audio(f"audio_{file_index-1}.wav"))
```

## Try Lyria RealTime by yourself

Now you can try mixing multiple prompts, and tinkering with the music configuration.

The prompts needs to follow their specific format which is a list of prompts with weights (which can be any values, including negative, except 0) like this:
```
{
    "text": "Text of the prompt",
    "weight": 1.0,
}
```

You should try to stay simple (unlike when you're using image-out) as the model will better understand things like "meditation", "eerie", "harp" than "An eerie and relaxing music illustrating the verdoyant forests of Scotland using string instruments".

The music configuration options available to you are:
* `bpm`: beats per minute
* `guidance`: how strictly the model follows the prompts
* `density`: density of musical notes/sounds
* `brightness`: tonal quality
* `scale`: musical scale (key and mode)
* `music_generation_mode`: quality (default), diversity, or allow vocalization (you'll need to add related prompts).

Other options are available (`mute_bass` for ex.). Check the documentation for the full list.

Select one of the sample prompts (genres,	instruments and	mood), or write your owns. Check the documentation for more details and prompt examples.


```
# prompt: I made a mistake, I need to append to message["client_content"] if prompt_2/3/4/5 are not empty

# @markdown ### Enter some prompts:
prompt_1 = "Indie Pop" # @param ["Hard Rock","Latin Jazz","Polka","Baroque","Chiptune","Indie Pop","Bluegrass","Heavy Metal","Contemporary R&B","Reggaeton"] {"allow-input":true}
prompt_1_weight = 0.6 # @param {type:"slider", min:0, max:2, step:0.1}
prompt_2 = "Sitar" # @param ["Piano","Guitar","Bagpipes","Harpsichord","808 Hip Hop Beat","Sitar","Harmonica","Didgeridoo","Woodwinds","Organ"] {"allow-input":true}
prompt_2_weight = 2 # @param {type:"slider", min:0, max:2, step:0.1}
prompt_3 = "Danceable" # @param ["Chill","Emotional","Danceable","Psychedelic","Acoustic Instruments","Glitchy Effects","Ominous Drone","Upbeat"] {"allow-input":true}
prompt_3_weight = 1.4 # @param {type:"slider", min:0, max:2, step:0.1}
prompt_4 = "" # @param {"type":"string","placeholder":"Fourth prompt (optional)"}
prompt_4_weight = 1.0 # @param {type:"slider", min:0, max:2, step:0.1}
prompt_5 = "" # @param {"type":"string","placeholder":"Fifth prompt (optional)"}
prompt_5_weight = 1.0 # @param {type:"slider", min:0, max:2, step:0.1}


# @markdown ### Music configuration:
BPM = 140 # @param {type:"slider", min:40, max:180, step:1}
scale = "F_MAJOR_D_MINOR" # @param ["SCALE_UNSPECIFIED","C_MAJOR_A_MINOR","D_FLAT_MAJOR_B_FLAT_MINOR","D_MAJOR_B_MINOR","E_FLAT_MAJOR_C_MINOR","E_MAJOR_D_FLAT_MINOR","F_MAJOR_D_MINOR","G_FLAT_MAJOR_E_FLAT_MINOR","G_MAJOR_E_MINOR","A_FLAT_MAJOR_F_MINOR","A_MAJOR_G_FLAT_MINOR","B_FLAT_MAJOR_G_MINOR","B_MAJOR_A_FLAT_MINOR"]
density = 0.2 # @param {type:"slider", min:0, max:1, step:0.1}
brightness = 0.7 # @param {type:"slider", min:0, max:1, step:0.1}
guidance = 4.0 # @param {type:"slider", min:0, max:6, step:0.1}
music_generation_mode = "QUALITY" # @param ["QUALITY","DIVERSITY","VOCALIZATION"]

# @markdown ### Duration (in seconds):
duration = 20 # @param {type:"slider", min:2, max:60, step:2}

# @markdown Now press the play button on the top right corner of this cell to run it and let Lyria RealTime generate your music

prompts = [{
    "text": prompt_1,
    "weight": prompt_1_weight,
}]

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

await generate_music(max_chunks=duration/2, prompts=prompts, config=config)
display(Audio(f"audio_{file_index-1}.wav"))
```

# What's next?

Now that you know how to generate music, here are other cool things to try:
*   Instead of music, learn how to generate multi-speakers conversation using the TTS models,
*   Discover how to generate images or videos,
*   Instead of generation music or audio, find out how to Gemini can understand Audio files,
*   Have a real-time conversation with Gemini using the Live API.
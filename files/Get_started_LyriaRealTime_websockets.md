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

# Get started with Music generation using Lyria RealTime and websockets

Lyria RealTime,
provides access to a state-of-the-art, real-time, streaming music
generation model. It allows developers to build applications where users
can interactively create, continuously steer, and perform instrumental
music using text prompts.

Lyria RealTime main characteristics are:
* **Highest quality text-to-audio model**: Lyria RealTime generates high-quality instrumental music (no voice) using the latest models produced by DeepMind.
* **Non-stopping music**: Using websockets, Lyria RealTime continuously generates music in real time.
* **Mix and match influences**: Prompt the model to describe musical idea, genre, instrument, mood, or characteristic. The prompts can be mixed to blend
influences and create unique compositions.
* **Creative control**: Set the `guidance`, the `bpm`, the `density` of musical notes/sounds, the `brightness` and the `scale` in real time. The model will smoothly transition based on the new input.

Check Lyria RealTime's [documentation](https://ai.google.dev/gemini-api/docs/music-generation) for more details.

Note: This notebook is the **Websocket version** of the get started with Lyria, please find the python version [here](../Get_started_LyriaRealTime.ipynb).

**Also note that due to Colab limitation, you won't be able to experience the real time capabilities of Lyria RealTime but only limited audio output. Use the [Python script](../Get_started_LyriaRealTime.py) or the AI studio's apps, [Prompt DJ](https://aistudio.google.com/apps/bundled/promptdj) and
[MIDI DJ](https://aistudio.google.com/apps/bundled/promptdj-midi) to fully experience Lyria RealTime**

Lyria RealTime is a preview feature. It is free to use for now with quota limitations, but is subject to change.

# Setup

## Install and import
Even if this notebook won't use the SDK, it will still use python and colab function to manage the websockets and the audio output.


```
%pip install -q websockets
```


```
import asyncio
import base64
import contextlib
import datetime
import os
import json
import wave
import itertools

from websockets.asyncio.client import connect
from IPython.display import display, Audio
```

## API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## API host
Lyria RealTime API is a new capability introduced with the Lyria RealTime model so only works with this model. You need to use the  `v1alpha` client version.



```
MODEL = 'models/lyria-realtime-exp'

HOST='generativelanguage.googleapis.com'

URI = f'wss://{HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateMusic?key={os.environ["GOOGLE_API_KEY"]}'
```

## Helpers

### Logging

For the sake of understanding how Lyria RealTime works, all logs are going to be displayed, but feel free to comment those lines if that's too much for you.


```
import logging

logger = logging.getLogger('Bidi')
logger.setLevel('DEBUG')
```

### Wave file writer


```
@contextlib.contextmanager
def wave_file(filename, channels=2, rate=48000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
```

# Main audio loop

The class below implements the interaction with the Lyria RealTime API.

This is a basic implementation that could be improved by using different threads or play the audio as soon as you start to receive it using `PyAudio`.

There are 4 methods worth describing here:

### `run` - The main loop

This method:

- Opens a `websocket` connecting to the real time API
- Calls the initial `setup` method
- Then enters the main loop where it alternates between `send` and `recv` until send returns `False`.

### `setup` - Initial setup

The `setup` method sends the `setup` message, and awaits the response. You shouldn't try to `send` or `recv` anything else from the model until you've gotten the model's `setup_complete` response.

The `setup` message (a `BidiGenerateMusicSetup` object) is where you can set the `model`.

### `send` - Sends input text to the api

The `send` method collects input text from the user, wraps it in a `client_content` message (an instance of `BidiGenerateContentMusicContent`), and sends it to the model.

Note that the prompts needs to follow a certain format (a list of prompts and their weights). Refer to the [documentation](https://ai.google.dev/gemini-api/docs/eap/lyria/music-generation) for more details.

### `recv` - Collects audio from the API and plays it

The `recv` method collects audio chunks in a loop and writes them to a `.wav` file.


```
class AudioLoop:
  def __init__(self, max_chunks=10, prompts=None, config=None):
    self.ws = None
    self.index = 0
    self.max_chunks = max_chunks
    self.prompts = prompts
    self.config = config

  async def run(self):
    print("Type 'q' to quit")

    logger.debug('connect')
    async with connect(URI, additional_headers={'Content-Type': 'application/json'}) as ws:
      self.ws = ws
      await self.setup()

      # Ideally these would be separate tasks.
      await self.send()
      await self.recv()

  async def setup(self):
      logger.debug("set_up")
      await self.ws.send(json.dumps({
          'setup' : {
               "model": MODEL,
          }
      }))
      raw_response = await self.ws.recv(decode=False)
      setup_response = json.loads(raw_response.decode('ascii'))
      logger.debug(f'Connected: {setup_response}')

  async def send(self):
    logger.debug('send')
    if self.prompts is not None:
      # If provided, send the message to the model.
      message = {
          "client_content": {
              "weighted_prompts": self.prompts
          }
        }
      await self.ws.send(json.dumps(message))
      logger.debug('sent client message')
    else:
      # Wait for the user to provide a prompt
      # `asyncio.to_thread` is important here, without it all other tasks are blocked.
      text = await asyncio.to_thread(input, "prompt? > ")

      # If the input returns 'q' quit.
      if text.lower() == 'q':
        return False

      # Wrap the text into a "client_content" message.
      message = {
          "client_content": {
              "weighted_prompts": [{
                  "text": text,
                  "weight": 1.0,
              }]
          }
        }

      # Send the message to the model.
      await self.ws.send(json.dumps(message))
      logger.debug('sent client prompt')

    if self.config is not None:
      # If provided, send the config to the model
      await self.ws.send(json.dumps(self.config))
      logger.debug('sent music_generation_config')
    else:
      # Send default config
      await self.ws.send(json.dumps({'music_generation_config': {'bpm': '120'}}))
      logger.debug('sent default music_generation_config')

    # Start streaming music
    await self.ws.send(json.dumps({'playback_control': 'PLAY'}))
    logger.debug('play')

    return True

  async def recv(self):
    # Start a new `.wav` file.
    file_name = f"audio_{self.index}.wav"
    with wave_file(file_name) as wav:
      self.index += 1

      logger.debug('receive')

      # Read chunks from the socket.
      n = 0
      async for raw_response in self.ws:
        n+=1
        if n > self.max_chunks:
          break
        response = json.loads(raw_response.decode())
        logger.debug(f'got chunk: {str(response)[:200]}')
        print(response)

        server_content = response.pop('serverContent', None)
        if server_content is None:
          logger.error(f'Unhandled server message! - {response}')
          break

        # Write audio the chunk to the `.wav` file.
        audio_chunk = server_content.pop('audioChunks', None)
        if audio_chunk is not None:
          b64data = audio_chunk[0]
          pcm_data = base64.b64decode(b64data['data'])
          print('.', end='')
          logger.debug('Got pcm_data')
          wav.writeframes(pcm_data)

    display(Audio(file_name, autoplay=True))
    await asyncio.sleep(2)
```

# Try Lyria RealTime

Because of Colab limitation you won't be able to experience the "real time" part of Lyria RealTime, so all those examples are going to be one-offs prompt to get an audio file.

One thing to note is that the audio will only be played at the end of the session when all would have been written in the wav file. When using the API for real you'll be able to start plyaing as soon as the first chunk arrives. So the longer the duration (using the dedicated parameter) you set, the longer you'll have to wait until you hear something.

## Simple Lyria RealTime example
Here's first a simple example:


```
await AudioLoop(prompts=[{"text":"eurodance", "weight":1.0}]).run()
```



    Type 'q' to quit

[., ..., .]

## Try Lyria RealTime by yourself

Now you can try mixing multiple prompts, and tinkering with the music configuration.

The prompts needs to follow their specific format which is a list of prompts with weights (which can be any values, including negative, except 0) like this:
```
{
    "text": "Text of the prompt",
    "weight": 1.0,
}
```

You should try to stay simple (unlike when you're using [image-out](../Get_Started_Nano_Banana.ipynb)) as the model will better understand things like "meditation", "eerie", "harp" than "An eerie and relaxing music illustrating the verdoyant forests of Scotland using string instruments".

The music configuration options available to you are:
* `bpm`: beats per minute
* `guidance`: how strictly the model follows the prompts
* `density`: density of musical notes/sounds
* `brightness`: tonal quality
* `scale`: musical scale (key and mode)

Other options are available (`mute_bass` for ex.). Check the [documentation](https://ai.google.dev/gemini-api/docs/music-generation#controls) for the full list.

Select one of the sample prompts (genres,	instruments and	mood), or write your owns. Check the [documentation](https://ai.google.dev/gemini-api/docs/music-generation#prompt-guide-lyria) for more details and prompt examples.


```
# prompt: I made a mistake, I need to append to message["client_content"] if prompt_2/3/4/5 are not empty

# Enter some prompts:
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


# Music configuration:
BPM = 140 
scale = "C_MAJOR_A_MINOR" 
density = 0.2 
brightness = 0.7 
guidance = 4.0 

# Duration (in seconds):
duration = 10 

# Now press the play button on the top right corner of this cell to run it and let Lyria RealTime generate your music

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
        'guidance': guidance
    }
}

await AudioLoop(max_chunks=duration/2, prompts=prompts, config=config).run()
```



    Type 'q' to quit

[., ..., .]

# What's next?

Now that you know how to generate music, here are other cool things to try:
*   Have a real-time conversation with Gemini over wesockets using the [Live API](./Get_started_LiveAPI.ipynb),
*   Instead of music, learn how to generate multi-speakers conversation using the [TTS models](../Get_started_TTS.ipynb),
*   Discover how to generate [images](../Get_started_imagen.ipynb) or [videos](../Get_started_Veo.ipynb),
*   Instead of generation music or audio, find out how to Gemini can [understand Audio files](../Audio.ipynb).
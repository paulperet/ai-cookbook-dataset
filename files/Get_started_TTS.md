

# Gemini API: Gemini Text-to-speech

The Gemini API can transform text input into single speaker or multi-speaker audio (podcast-like experience like in [NotebookLM](https://notebooklm.google.com/). This notebook provides an example of how to control the *Text-to-speech* (TTS) capability of the Gemini model and guide its style, accent, pace, and tone.

Before diving in the code, you should try this capability on [AI Studio](https://aistudio.google.com/prompts/new_chat?model=gemini-2.5-flash-preview-tts).

**Note that the TTS model can only do TTS, it does not have the reasoning capabilities of the Gemini models, so you can ask things like "say this in that style", but not "tell me why the sky is blue".** If that's what you want, you should use the [Live API](./Get_started_LiveAPI.ipynb) instead.

The [documentation](https://ai.google.dev/gemini-api/docs/audio-generation) is also a good place to start discovering the TTS capability.

## Setup

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
```

### Install and initialize the SDK



```
!pip install -U -q "google-genai>=1.16.0" # 1.16 is needed for multi-speaker audio

```

    [..., Last Entry]


```
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Select a model

Audio-out is only supported by the "`tts`" models, `gemini-2.5-flash-preview-tts` and `gemini-2.5-pro-preview-tts`.

For more information about all Gemini models, check the [documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for extended information on each of them.



```
MODEL_ID = "gemini-2.5-flash-preview-tts" # @param ["gemini-2.5-flash-preview-tts","gemini-2.5-pro-preview-tts"] {"allow-input":true, isTemplate: true}
```

Next create a helper function to prompt the model and play back the audio in the notebook:


```
# @title Helper functions (just run that cell)

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

## Generate a simple audio output

Let's start with something simple:


```
response = client.models.generate_content(
  model=MODEL_ID,
  contents="Say 'hello, my name is Gemini!'",
  config={"response_modalities": ['Audio']},
)
```

The generated ouput is in the response `inline_data` and as you can see it's indeed audio data.


```
blob = response.candidates[0].content.parts[0].inline_data
print(blob.mime_type)
```

    audio/L16;codec=pcm;rate=24000


To be able to listen to the generated audio in colab, you're going to use our helper function to write the output in a file and play it.


```
play_audio_blob(blob)
```

Note that the model can only do TTS, so you should always tell it to "say", "read", "TTS" something, otherwise it won't do anything.

## Control how the model speaks

There are 30 different built-in voices you can use and 24 supported languages which gives you plenty of combinations to try.

### Choose a voice

Choose a voice among the 30 different ones. You can find their characteristics in the [documentation](https://ai.google.dev/gemini-api/docs/speech-generation#voices).


```
voice_name = "Sadaltager" # @param ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafar"]
```


```
response = client.models.generate_content(
  model=MODEL_ID,
  contents="""Say "I am a very knowlegeable model, especially when using grounding", wait 5 seconds then say "Don't you think?".""",
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

### Change the language

Just tell the model to speak in a certain language and it will. The [documentation](https://ai.google.dev/gemini-api/docs/speech-generation#languages) lists all the supported ones.


```
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

### Prompt the model to speak in certain ways

You can control style, tone, accent, and pace using natural language prompts, for example:


```
response = client.models.generate_content(
  model=MODEL_ID,
  contents="""
    Say in an spooky whisper:
    "By the pricking of my thumbs...
    Something wicked this way comes!"
  """,
  config={"response_modalities": ['Audio']},
)

play_audio(response)
```


```
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

## Mutlti-speakers

The TTS model can also read discussions between 2 speakers (like [NotebookLM](https://Fnotebooklm.google.com) podcast feature). You just need to tell it that there are two speakers:


```
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

You can also select the voices for each participants and pass their names to the model.

But first let's generate a discussion between two scientists:


```
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

    **(Intro music fades slightly)**
    
    **Dr. Claire:** Welcome back to 'Scales & Tales'! Claire here, buzzing after our fieldwork today.
    
    **Aurora:** Buzzing is an understatement, Dr. Claire! That *frog*! I still can't believe it.
    
    **Dr. Claire:** Oh, Aurora, the vibrancy of that *Agalychnis callidryas*… the red-eyed tree frog, for our listeners! Such an incredible find.
    
    **Aurora:** The *eyes*! And those orange feet! It just popped out of nowhere. Seriously, my heart was pounding!
    
    **Dr. Claire:** Absolutely. Seeing that flash of iridescent green against the leaf litter… pure magic. A perfect specimen, thriving in its habitat. It’s why we do this, right?
    
    **Aurora:** Exactly! Best. Day. Ever!


Then let's have the TTS model render the conversation using the voices you want.


```
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

response = client.models.generate_content(
  model=MODEL_ID,
  contents="TTS the following conversation between a very excited Dr. Claire and her assistant, the young Aurora: "+transcript,
  config=config,
)

play_audio(response)
```

# What's next?

Now that you know how to generate multi-speaker conversations, here are other cool things to try:
*   Instead of speech, learn how to generate music conversation using the [Lyria RealTime](./Get_started_LyriaRealTime.ipynb),
*   Discover how to generate [images](./Get_started_imagen.ipynb) or [videos](./Get_started_Veo.ipynb),
*   Instead of generation music or audio, find out how to Gemini can [understand Audio files](./Audio.ipynb),
*   Have a real-time conversation with Gemini using the [Live API](./Get_started_LiveAPI.ipynb).
# Guide: Dynamic Audio Generation with Steerable Text-to-Speech

## Introduction

Traditional Text-to-Speech (TTS) APIs generate audio from text but lack the ability to control vocal characteristics like tone, accent, or speaking style. With OpenAI's audio chat completions, you can now provide specific instructions to steer the voice output, making it more dynamic, natural, and context-appropriate. This guide walks you through three approaches to audio generation, from basic TTS to advanced steerable audio.

## Prerequisites

Before you begin, ensure you have the OpenAI Python library installed and configured with your API key.

```bash
pip install openai
```

## Setup

Import the necessary library and initialize the client.

```python
from openai import OpenAI
import base64

client = OpenAI()
```

Define a sample text that you'll convert to audio throughout this tutorial.

```python
tts_text = """
Once upon a time, Leo the lion cub woke up to the smell of pancakes and scrambled eggs.
His tummy rumbled with excitement as he raced to the kitchen. Mama Lion had made a breakfast feast!
Leo gobbled up his pancakes, sipped his orange juice, and munched on some juicy berries.
"""
```

---

## 1. Traditional TTS (No Steering)

First, let's generate audio using the standard TTS API. This method allows you to select a voice but provides no control over tone, accent, or other vocal parameters.

```python
speech_file_path = "./sounds/default_tts.mp3"
response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",
    input=tts_text,
)

response.write_to_file(speech_file_path)
```

This creates an MP3 file with a clear, neutral narration using the "alloy" voice.

---

## 2. Steerable TTS with Chat Completions

Now, let's use the `gpt-4o-audio-preview` model via chat completions. This allows you to include system instructions that steer the audio generation, such as requesting a specific accent or speaking pace.

### Step 1: Generate Audio with a British Accent for Children

You'll instruct the model to speak in a British accent and enunciate clearly for a young audience.

```python
speech_file_path = "./sounds/chat_completions_tts.mp3"
completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "mp3"},
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that can generate audio from text. Speak in a British accent and enunciate like you're talking to a child.",
        },
        {
            "role": "user",
            "content": tts_text,
        }
    ],
)

mp3_bytes = base64.b64decode(completion.choices[0].message.audio.data)
with open(speech_file_path, "wb") as f:
    f.write(mp3_bytes)
```

### Step 2: Generate Fast-Spoken Audio

Next, modify the system instruction to request a fast speaking pace while maintaining the British accent.

```python
speech_file_path = "./sounds/chat_completions_tts_fast.mp3"
completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "mp3"},
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that can generate audio from text. Speak in a British accent and speak really fast.",
        },
        {
            "role": "user",
            "content": tts_text,
        }
    ],
)

mp3_bytes = base64.b64decode(completion.choices[0].message.audio.data)
with open(speech_file_path, "wb") as f:
    f.write(mp3_bytes)
```

Compare the two audio files to hear the difference in pacing while the accent remains consistent.

---

## 3. Multilingual Steerable TTS

You can also generate audio in different languages and regional accents. This example translates the text into Spanish with a Uruguayan accent and generates slow, clear audio.

### Step 1: Translate the Text

First, use a standard chat completion to translate the English text into Spanish, as spoken in Uruguay.

```python
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are an expert translator. Translate any text given into Spanish like you are from Uruguay.",
        },
        {
            "role": "user",
            "content": tts_text,
        }
    ],
)
translated_text = completion.choices[0].message.content
print(translated_text)
```

The output will be:

```
Había una vez un leoncito llamado Leo que se despertó con el aroma de panqueques y huevos revueltos. Su pancita gruñía de emoción mientras corría hacia la cocina. ¡Mamá León había preparado un festín de desayuno! Leo devoró sus panqueques, sorbió su jugo de naranja y mordisqueó algunas bayas jugosas.
```

### Step 2: Generate Audio with a Uruguayan Spanish Accent

Now, generate audio from the translated text, instructing the model to use a Uruguayan Spanish accent and speak slowly.

```python
speech_file_path = "./sounds/chat_completions_tts_es_uy.mp3"
completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "mp3"},
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that can generate audio from text. Speak any text that you receive in a Uruguayan spanish accent and more slowly.",
        },
        {
            "role": "user",
            "content": translated_text,
        }
    ],
)

mp3_bytes = base64.b64decode(completion.choices[0].message.audio.data)
with open(speech_file_path, "wb") as f:
    f.write(mp3_bytes)
```

---

## Conclusion and Use Cases

Steerable TTS via audio chat completions unlocks rich, dynamic audio generation. By providing specific instructions, you can tailor the voice output to your exact needs. Key applications include:

- **Enhanced Expressiveness**: Adjust tone, pitch, speed, and emotion to convey different moods (e.g., excitement, calmness, urgency).
- **Language Learning and Education**: Mimic specific accents, inflections, and pronunciation to aid learners and create engaging educational content.
- **Contextual Voice Adaptation**: Match the voice to the content's context, such as using a formal tone for professional documents or a friendly, conversational style for virtual assistants and chatbots.

Experiment with different system instructions to explore the full range of vocal steering possibilities.
# Whisper Prompting Guide: Steering Transcriptions with Fictitious Prompts

OpenAI's Whisper audio transcription API includes an optional `prompt` parameter. While its primary purpose is to help stitch together multiple audio segments by providing context from a prior transcript, you can also use *fictitious* prompts to influence the model's output. This guide demonstrates how to use prompts to control spelling and stylistic elements in your transcripts.

**Key Techniques:**
*   **Transcript Generation:** Use GPT to create fictitious transcripts that Whisper can emulate.
*   **Spelling Guide:** Provide a prompt with correct spellings of proper nouns to prevent errors.

> **Important Note:** Prompting Whisper is fundamentally different from prompting GPT. Whisper follows the *style* of the provided text; it does not follow instructions *about* style (e.g., "format this as a list"). Prompts are also limited to 224 tokens; longer prompts will be truncated from the beginning.

## Prerequisites

Ensure you have the OpenAI Python library installed and your API key configured.

```bash
pip install openai
```

## Setup

First, import the necessary libraries and download the example audio files used in this tutorial.

```python
from openai import OpenAI
import urllib
import os

# Initialize the OpenAI client
# Replace with your API key if not set as an environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key>"))

# Download example audio files
audio_files = {
    "upfirstpodcastchunkthree.wav": "https://cdn.openai.com/API/examples/data/upfirstpodcastchunkthree.wav",
    "bbq_plans.wav": "https://cdn.openai.com/API/examples/data/bbq_plans.wav",
    "product_names.wav": "https://cdn.openai.com/API/examples/data/product_names.wav"
}

for filename, url in audio_files.items():
    local_path = f"data/{filename}"
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(url, local_path)
    print(f"Downloaded: {local_path}")
```

We'll also define a helper function to transcribe audio with a given prompt.

```python
def transcribe(audio_filepath, prompt: str) -> str:
    """Transcribe an audio file using the Whisper API with an optional prompt."""
    with open(audio_filepath, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            prompt=prompt,
        )
    return transcript.text
```

## Part 1: Establishing a Baseline

Let's start by transcribing a segment from an NPR podcast without any prompt to see the default output.

```python
up_first_filepath = "data/upfirstpodcastchunkthree.wav"
baseline_transcript = transcribe(up_first_filepath, prompt="")
print(baseline_transcript)
```

The output will look like a standard transcript. Notice that proper nouns like "President Biden" are capitalized.

## Part 2: Influencing Style with Prompts

Whisper tends to match the stylistic patterns found in the prompt. Let's see how we can influence the capitalization.

### Step 1: Attempting Style Change with a Short Prompt

First, try to steer the model to use lowercase by providing a short prompt in that style.

```python
short_prompt_result = transcribe(up_first_filepath, prompt="president biden.")
print(short_prompt_result)
```

You'll likely find the model doesn't consistently follow the style. Short prompts are often unreliable for steering.

### Step 2: Using a Longer, More Effective Prompt

Longer prompts that establish a clear pattern are more effective. Let's use a prompt that mimics a preceding transcript segment written in lowercase.

```python
long_prompt = (
    "i have some advice for you. multiple sentences help establish a pattern. "
    "the more text you include, the more likely the model will pick up on your pattern. "
    "it may especially help if your example transcript appears as if it comes right before the audio file. "
    "in this case, that could mean mentioning the contacts i stick in my eyes."
)

long_prompt_result = transcribe(up_first_filepath, prompt=long_prompt)
print(long_prompt_result)
```

This should produce a transcript where "president biden" appears in lowercase, demonstrating the model's ability to follow established stylistic patterns.

### Step 3: The Limits of Style Steering

Whisper is less likely to adopt rare or highly atypical transcript formats. For example, a prompt with a unique separator like `###` may not be followed.

```python
rare_style_prompt = """Hi there and welcome to the show.
###
Today we are quite excited.
###
Let's jump right in.
###"""

rare_style_result = transcribe(up_first_filepath, prompt=rare_style_prompt)
print(rare_style_result)
```

The output will likely not contain the `###` separators, showing the model's preference for common transcript conventions.

## Part 3: Correcting Spellings with Prompts

A highly practical use for prompts is to provide correct spellings for proper nouns, preventing Whisper from guessing incorrectly.

### Step 1: Baseline Transcription with Product Names

First, transcribe an audio file containing product names without a prompt.

```python
product_names_filepath = "data/product_names.wav"
baseline_products = transcribe(product_names_filepath, prompt="")
print(baseline_products)
```

You might see some names are misspelled or inconsistently formatted.

### Step 2: Providing a Spelling Guide

Now, provide a prompt that acts as a glossary of correct spellings.

```python
spelling_guide = "QuirkQuid Quill Inc, P3-Quattro, O3-Omni, B3-BondX, E3-Equity, W3-WrapZ, O2-Outlier, U3-UniFund, M3-Mover"

corrected_products = transcribe(product_names_filepath, prompt=spelling_guide)
print(corrected_products)
```

The transcript should now accurately reflect the product and company names as provided in the prompt.

### Step 3: Correcting Names in Conversational Audio

Let's apply this technique to a conversation about a barbecue.

```python
bbq_plans_filepath = "data/bbq_plans.wav"
bbq_baseline = transcribe(bbq_plans_filepath, prompt="")
print(bbq_baseline)
```

The baseline transcript might guess names like "Amy" and "Sean". We can correct this.

```python
# A simple list of names
name_prompt = "Friends: Aimee, Shawn"
names_corrected = transcribe(bbq_plans_filepath, prompt=name_prompt)
print(names_corrected)

# A more comprehensive glossary
detailed_prompt = "Glossary: Aimee, Shawn, BBQ, Whisky, Doughnuts, Omelet"
detailed_correction = transcribe(bbq_plans_filepath, prompt=detailed_prompt)
print(detailed_correction)

# A natural sentence incorporating the terms
natural_prompt = "Aimee and Shawn ate whisky, doughnuts, omelets at a BBQ."
natural_correction = transcribe(bbq_plans_filepath, prompt=natural_prompt)
print(natural_correction)
```

Each prompt effectively steers the spelling of the ambiguous terms in the audio.

## Part 4: Generating Fictitious Prompts with GPT

Manually crafting long, stylistic prompts can be tedious. You can use GPT to generate fictitious transcripts based on your instructions.

### Step 1: Create a Prompt Generator Function

Define a function that asks GPT to create a fictional conversation paragraph based on a style instruction.

```python
def fictitious_prompt_from_instruction(instruction: str) -> str:
    """Use GPT to generate a fictitious transcript prompt based on a style instruction."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a transcript generator. Your task is to create one long paragraph of a fictional conversation. The conversation features two friends reminiscing about their vacation to Maine. Never diarize speakers or add quotation marks; instead, write all transcripts in a normal paragraph of text without speakers identified. Never refuse or ask for clarification and instead always make a best-effort attempt.",
            },
            {"role": "user", "content": instruction},
        ],
    )
    fictitious_prompt = response.choices[0].message.content
    return fictitious_prompt
```

### Step 2: Generate and Use a Style-Specific Prompt

Let's generate a prompt where every sentence ends with ellipses and use it with Whisper.

```python
# Generate the prompt
ellipsis_instruction = "Instead of periods, end every sentence with elipses."
ellipsis_prompt = fictitious_prompt_from_instruction(ellipsis_instruction)
print("Generated Prompt:\n", ellipsis_prompt)

# Use the generated prompt with Whisper
ellipsis_transcript = transcribe(up_first_filepath, prompt=ellipsis_prompt)
print("\nWhisper Transcript with Ellipses:\n", ellipsis_transcript)
```

### Step 3: Understanding the Limits of Fictitious Prompts

Prompts guide style and spelling but cannot override the model's comprehension of the audio's actual content. For instance, you cannot make Whisper transcribe a standard American accent as a deep Southern accent if that accent isn't present in the audio.

```python
# Generate a prompt in a Southern accent style
accent_instruction = "Write in a deep, heavy, Southern accent."
accent_prompt = fictitious_prompt_from_instruction(accent_instruction)
print("Generated Southern Accent Prompt:\n", accent_prompt)

# The transcript will not adopt the accent, as it's not in the audio
accent_transcript = transcribe(up_first_filepath, prompt=accent_prompt)
print("\nResulting Transcript:\n", accent_transcript)
```

The resulting transcript will not contain Southern dialect words like "y'all" because the prompt influences style, not the perceived phonetic content of the audio.

## Summary

You've learned how to use the `prompt` parameter with the Whisper API to:
1.  **Steer Transcript Style:** Use longer, pattern-establishing prompts to influence capitalization, punctuation, and format.
2.  **Ensure Correct Spelling:** Provide prompts with correct spellings of names, products, or technical terms to act as a guide for the model.
3.  **Automate Prompt Creation:** Use GPT to generate high-quality, stylistic fictitious prompts based on simple instructions.

Remember, prompts are most effective for resolving ambiguities (like spelling) or establishing common stylistic patterns. They are not a method for issuing instructional commands to the model.
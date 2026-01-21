# Translate a Public Domain Book with the Gemini API

In this guide, you will use the Gemini API to translate a public domain book from Polish to English. You'll learn how to prepare the text, manage token limits, create effective prompts, and save the translated output.

## Prerequisites

Before you begin, ensure you have the following:

1.  A Google AI API Key. If you don't have one, you can [get one here](https://makersuite.google.com/app/apikey).
2.  The API key stored in an environment variable named `GOOGLE_API_KEY`.

## Setup

First, install the required Python libraries.

```bash
pip install -U "google-genai>=1.0.0" tqdm
```

Now, import the necessary modules.

```python
from tqdm import tqdm
from google import genai
from google.genai import types
```

## Step 1: Configure the Gemini Client

Initialize the Gemini client using your API key.

```python
import os

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 2: Prepare the Source Text

You will translate *The Hound of the Baskervilles* by Arthur Conan Doyle from its Polish translation. Download the text from Project Gutenberg.

```python
import urllib.request

# Download the book
url = "https://www.gutenberg.org/cache/epub/34079/pg34079.txt"
urllib.request.urlretrieve(url, "sherlock.txt")

# Load the book into memory
with open("sherlock.txt", "r", encoding="utf-8") as f:
    book = f.read()
```

## Step 3: Configure Safety Settings

Books may contain content that triggers the model's default safety filters. To ensure smooth translation, adjust the safety settings to be more permissive for this task.

```python
safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]
```

## Step 4: Initialize the Model and Helper Function

Select a Gemini model and create a helper function to generate content. We'll use the efficient `gemini-2.5-flash` model for this task.

```python
MODEL_ID = "gemini-2.5-flash"

def generate_output(prompt):
    """Sends a prompt to the Gemini model and returns the text response."""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(safety_settings=safety_settings),
    )
    return response.text
```

## Step 5: Understand Token Limits

Large Language Models (LLMs) have context windows limited by tokens. For the Gemini 2.5 Flash model:
*   **Input Limit:** 1,048,576 tokens
*   **Output Limit:** 8,192 tokens

Since the translated output might be longer than the input, we must split the book into manageable chunks. A rough estimate is that **1 token ≈ 4 characters**. We'll split the text into chunks of approximately 5,000 characters to stay well within limits.

## Step 6: Split the Text into Chunks

First, remove the Project Gutenberg footer from the text. Then, split the content by double newlines (`\n\n`), which often corresponds to paragraph breaks, creating natural chunk boundaries.

```python
# Find the end of the actual book content (before the Gutenberg footer)
end_marker = "END OF THE PROJECT GUTENBERG EBOOK TAJEMNICA BASKERVILLE'ÓW: DZIWNE PRZYGODY SHERLOCKA HOLMES"
book_content = book[: book.find(end_marker)]

# Split the content into chunks based on paragraph breaks
chunks = book_content.split("\n\n")

# Filter out any empty chunks
chunks = [chunk for chunk in chunks if chunk.strip()]

print(f"Number of text chunks: {len(chunks)}")
```

## Step 7: Estimate Token Counts

Let's verify our chunks are a suitable size by estimating their token counts.

```python
estimated_token_counts = []

for chunk in chunks:
    # Use the 1 token ≈ 4 characters heuristic
    estimated_token_count = len(chunk) / 4
    estimated_token_counts.append(estimated_token_count)

# Check the size of a few sample chunks
print(f"Sample estimated tokens per chunk: {estimated_token_counts[:5]}")
print(f"Maximum estimated tokens in a chunk: {max(estimated_token_counts)}")
```

If the maximum estimated token count is significantly below 8,000, your chunks are a safe size for translation.

## Step 8: Translate the Chunks

Now, iterate through each chunk, send it to the Gemini model with a translation prompt, and collect the results. We'll use `tqdm` to add a progress bar.

```python
translated_chunks = []

for chunk in tqdm(chunks, desc="Translating chunks"):
    # Construct the prompt. You can adjust the instructions as needed.
    prompt = f"""Translate the following Polish text to English. Preserve the original formatting, paragraph breaks, and tone. Do not add any explanations or notes.

    Polish Text:
    {chunk}

    English Translation:
    """

    try:
        translated_text = generate_output(prompt)
        translated_chunks.append(translated_text)
    except Exception as e:
        print(f"Error translating a chunk: {e}")
        # Append a placeholder or the original chunk in case of error
        translated_chunks.append(f"[Translation Error for this chunk: {e}]")
```

## Step 9: Save the Translated Book

Combine all the translated chunks and write them to a new file.

```python
# Combine all translated chunks
full_translation = "\n\n".join(translated_chunks)

# Save the translation to a file
output_filename = "the_hound_of_the_baskervilles_translated.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(full_translation)

print(f"Translation complete! Saved to '{output_filename}'.")
```

## Step 10: Review the Output (Optional)

Finally, you can print a sample of the translated text to verify the quality.

```python
# Print the first 500 characters as a preview
print("Preview of the translation:")
print(full_translation[:500])
```

## Summary

You have successfully translated a full book using the Gemini API. The key steps were:
1.  Preparing the source text and splitting it into context-sized chunks.
2.  Configuring the API client and adjusting safety settings.
3.  Iteratively sending each chunk to the model with a clear translation prompt.
4.  Reassembling and saving the results.

This pipeline can be adapted for other languages, longer documents, or different text processing tasks like summarization or style transfer. Remember to always respect copyright and usage rights for the source material.
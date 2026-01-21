# Guide: Translating a LaTeX Book from Slovenian to English Using GPT-4

This guide demonstrates how to translate a LaTeX-formatted book from Slovenian into English using OpenAI's GPT-4 model. The process preserves all LaTeX commands and formatting while translating only the natural language text. We'll use the book *Euclidean Plane Geometry* by Milan Mitrović as an example.

## Prerequisites

Ensure you have the following installed and set up:

- Python 3.7+
- An OpenAI API key
- Required Python packages: `openai`, `tiktoken`

Install the necessary packages:

```bash
pip install openai tiktoken
```

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Step 1: Load the LaTeX Source File

First, read the LaTeX source file into a Python string.

```python
from openai import OpenAI
import tiktoken

# Initialize the OpenAI client
client = OpenAI()

# Load the tokenizer for counting tokens
tokenizer = tiktoken.get_encoding("o200k_base")

# Read the LaTeX file
with open("data/geometry_slovenian.tex", "r") as f:
    text = f.read()
```

## Step 2: Split the Text into Manageable Chunks

We'll split the text by double newlines (`\n\n`), which preserves logical paragraph breaks without disrupting LaTeX commands.

```python
# Split the text into chunks
chunks = text.split('\n\n')

# Count tokens in each chunk
ntokens = []
for chunk in chunks:
    ntokens.append(len(tokenizer.encode(chunk)))

print(f"Size of the largest chunk: {max(ntokens)} tokens")
print(f"Number of chunks: {len(chunks)}")
```

**Output:**
```
Size of the largest chunk: 1211 tokens
Number of chunks: 5877
```

Since the largest chunk is well under the GPT-4 token limit (16,384), we can proceed. However, translating each small chunk individually may break coherence. We'll group them into larger, page-like segments.

## Step 3: Group Chunks into Coherent Batches

Define a function to combine small chunks into batches of approximately 15,000 tokens.

```python
def group_chunks(chunks, ntokens, max_len=15000, hard_max_len=16000):
    """
    Group very short chunks to form approximately page-long segments.
    """
    batches = []
    cur_batch = ""
    cur_tokens = 0

    for chunk, ntoken in zip(chunks, ntokens):
        # Skip any chunk that exceeds the hard limit
        if ntoken > hard_max_len:
            print(f"Warning: Chunk discarded for being too long ({ntoken} tokens > {hard_max_len} token limit). Preview: '{chunk[:50]}...'")
            continue

        # If there's room in the current batch, add the chunk
        if cur_tokens + 1 + ntoken <= max_len:
            cur_batch += "\n\n" + chunk
            cur_tokens += 1 + ntoken  # Add 1 token for the two newlines
        else:
            # Otherwise, save the current batch and start a new one
            batches.append(cur_batch)
            cur_batch = chunk
            cur_tokens = ntoken

    # Append the final batch if it's not empty
    if cur_batch:
        batches.append(cur_batch)

    return batches

# Apply the grouping function
batches = group_chunks(chunks, ntokens)
print(f"Number of batches after grouping: {len(batches)}")
```

**Output:**
```
Number of batches after grouping: 39
```

We've reduced the number of translation calls from 5,877 to 39, improving efficiency and text continuity.

## Step 4: Define the Translation Function

Create a function that sends a chunk to the GPT-4 API with clear instructions to translate only the text, leaving LaTeX commands intact. Providing a translation example helps guide the model.

```python
def translate_chunk(chunk, model='gpt-4o',
                    dest_language='English',
                    sample_translation=(
                        r"\poglavje{Osnove Geometrije} \label{osn9Geom}",
                        r"\chapter{The basics of Geometry} \label{osn9Geom}")):
    """
    Translates a LaTeX chunk while preserving all commands.
    """
    prompt = f'''Translate only the text from the following LaTeX document into {dest_language}. Leave all LaTeX commands unchanged

"""
{sample_translation[0]}
{chunk}"""

{sample_translation[1]}
'''
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0,
        top_p=1,
        max_tokens=15000,
    )
    result = response.choices[0].message.content.strip()
    result = result.replace('"""', '')  # Remove the triple quotes used in the prompt
    return result
```

### Understanding the Prompt Structure

The prompt consists of four parts:
1. **High-level instruction**: Directs the model to translate only the text, not LaTeX commands.
2. **Sample untranslated command**: Shows a LaTeX command with Slovenian text (e.g., `\poglavje{Osnove Geometrije}`).
3. **The chunk to translate**: The actual LaTeX content.
4. **Sample translated command**: Demonstrates the expected output for the sample (e.g., `\chapter{The basics of Geometry}`).

This approach ensures the model understands the task and maintains formatting consistency.

## Step 5: Test the Translation on a Single Batch

Before processing all batches, test the function on a single batch to verify the output quality.

```python
# Translate the third batch (index 2) as a test
test_translation = translate_chunk(batches[2], model='gpt-4o', dest_language='English')
print(test_translation[:500])  # Print the first 500 characters as a preview
```

**Output Preview:**
```
\chapter{The basics of Geometry} \label{osn9Geom}
Let us mention that the group structure also requires the property of associativity...
```

The output shows that LaTeX commands like `\chapter{}` and `\label{}` remain unchanged, while the Slovenian text inside `\poglavje{}` has been correctly translated to English.

## Step 6: Translate All Batches

Now, iterate through all batches and translate each one. It's good practice to save each translated batch immediately to avoid data loss.

```python
translated_batches = []

for i, batch in enumerate(batches):
    print(f"Translating batch {i+1}/{len(batches)}...")
    translated = translate_chunk(batch, model='gpt-4o', dest_language='English')
    translated_batches.append(translated)
    # Optional: save each batch to a file
    with open(f"translated_batch_{i}.tex", "w") as f:
        f.write(translated)

print("Translation complete!")
```

## Step 7: Reassemble the Translated Book

Combine all translated batches into a single document.

```python
# Join all translated batches
translated_book = "\n\n".join(translated_batches)

# Save the final translated book
with open("geometry_english.tex", "w") as f:
    f.write(translated_book)

print("Translated book saved as 'geometry_english.tex'")
```

## Summary

You have successfully translated a LaTeX book from Slovenian to English using GPT-4. The key steps were:

1. **Load and chunk** the LaTeX source.
2. **Group chunks** into coherent batches for better translation continuity.
3. **Craft a precise prompt** that instructs the model to translate only text, preserving LaTeX commands.
4. **Iterate through batches**, calling the GPT-4 API for each.
5. **Reassemble** the translated batches into a complete document.

This method ensures that the book's structure, equations, labels, and references remain intact while the content is accurately translated. You can adapt this workflow for other languages or document formats by adjusting the chunking strategy and prompt examples.
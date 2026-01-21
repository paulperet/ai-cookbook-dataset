# Guide: Summarizing Long Documents with Controllable Detail

This guide demonstrates how to create summaries of long documents (e.g., 10k+ tokens) where you can precisely control the level of detail in the output. A common issue with large-language models is that they tend to produce disproportionately short summaries for long texts. By splitting the document into manageable chunks and summarizing each piece, we can reconstruct a full summary whose length and detail scale with the original document.

## Prerequisites

Ensure you have the required Python packages installed:

```bash
pip install openai tiktoken tqdm
```

You will also need an OpenAI API key set in your environment variables.

## 1. Setup and Initial Data Loading

First, import the necessary libraries and load your document.

```python
import os
from typing import List, Tuple, Optional
from openai import OpenAI
import tiktoken
from tqdm import tqdm

# Load your long document
with open("data/artificial_intelligence_wikipedia.txt", "r") as file:
    artificial_intelligence_wikipedia_text = file.read()

# Check the token length of the document
encoding = tiktoken.encoding_for_model('gpt-4-turbo')
document_token_count = len(encoding.encode(artificial_intelligence_wikipedia_text))
print(f"Document token count: {document_token_count}")
```

## 2. Define the OpenAI Client Helper

Create a simple function to handle calls to the OpenAI Chat Completions API.

```python
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_chat_completion(messages, model='gpt-4-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content
```

## 3. Implement Text Chunking Utilities

To process long documents, we need to split them into smaller pieces. The following functions handle tokenization and intelligent chunking based on a delimiter.

```python
def tokenize(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model('gpt-4-turbo')
    return encoding.encode(text)

def chunk_on_delimiter(input_string: str,
                       max_tokens: int, delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True
    )
    if dropped_chunk_count > 0:
        print(f"warning: {dropped_chunk_count} chunks were dropped due to overflow")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks

def combine_chunks_with_no_minimum(
        chunks: List[str],
        max_tokens: int,
        chunk_delimiter="\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int]]:
    dropped_chunk_count = 0
    output = []
    output_indices = []
    candidate = [] if header is None else [header]
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]
        if len(tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            print(f"warning: chunk overflow")
            if (
                    add_ellipsis_for_overflow
                    and len(tokenize(chunk_delimiter.join(candidate + ["..."]))) <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue
        extended_candidate_token_count = len(tokenize(chunk_delimiter.join(candidate + [chunk])))
        if extended_candidate_token_count > max_tokens:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header
            candidate_indices = [chunk_i]
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)
    if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count
```

## 4. Build the Core Summarization Function

The `summarize` function is the heart of this guide. It accepts a `detail` parameter (from 0 to 1) that controls how many chunks the document is split into, thereby controlling the final summary's length and detail.

```python
def summarize(text: str,
              detail: float = 0,
              model: str = 'gpt-4-turbo',
              additional_instructions: Optional[str] = None,
              minimum_chunk_size: Optional[int] = 500,
              chunk_delimiter: str = ".",
              summarize_recursively=False,
              verbose=False):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually.
    The level of detail in the summary can be adjusted, and the process can optionally be made recursive.

    Parameters:
    - text (str): The text to be summarized.
    - detail (float, optional): A value between 0 and 1 indicating the desired level of detail in the summary.
      0 leads to a higher level summary, and 1 results in a more detailed summary. Defaults to 0.
    - model (str, optional): The model to use for generating summaries. Defaults to 'gpt-4-turbo'.
    - additional_instructions (Optional[str], optional): Additional instructions to provide to the model for customizing summaries.
    - minimum_chunk_size (Optional[int], optional): The minimum size for text chunks. Defaults to 500.
    - chunk_delimiter (str, optional): The delimiter used to split the text into chunks. Defaults to ".".
    - summarize_recursively (bool, optional): If True, summaries are generated recursively, using previous summaries for context.
    - verbose (bool, optional): If True, prints detailed information about the chunking process.

    Returns:
    - str: The final compiled summary of the text.
    """
    assert 0 <= detail <= 1

    # Determine the number of chunks based on the detail parameter
    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # Calculate the target chunk size
    document_length = len(tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk token lengths: {[len(tokenize(x)) for x in text_chunks]}")

    # Prepare the system message
    system_message_content = "Rewrite this text in summarized form."
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    # Summarize each chunk
    accumulated_summaries = []
    for chunk in tqdm(text_chunks):
        if summarize_recursively and accumulated_summaries:
            accumulated_summaries_string = '\n\n'.join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            user_message_content = chunk

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]

        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    # Compile the final summary
    final_summary = '\n\n'.join(accumulated_summaries)
    return final_summary
```

## 5. Generate Summaries with Different Detail Levels

Now, let's use the function to create four summaries with increasing levels of detail. We'll set `verbose=True` to see how the document is split.

```python
# Generate a high-level summary (detail = 0)
summary_with_detail_0 = summarize(artificial_intelligence_wikipedia_text, detail=0, verbose=True)

# Generate a moderately detailed summary (detail = 0.25)
summary_with_detail_pt25 = summarize(artificial_intelligence_wikipedia_text, detail=0.25, verbose=True)

# Generate a more detailed summary (detail = 0.5)
summary_with_detail_pt5 = summarize(artificial_intelligence_wikipedia_text, detail=0.5, verbose=True)

# Generate a very detailed summary (detail = 1)
summary_with_detail_1 = summarize(artificial_intelligence_wikipedia_text, detail=1, verbose=True)
```

When you run this, you'll see output similar to:

```
Splitting the text into 1 chunks to be summarized.
Chunk token lengths: [14631]
100%|██████████| 1/1 [00:09<00:00,  9.68s/it]

Splitting the text into 9 chunks to be summarized.
Chunk token lengths: [1817, 1807, 1823, 1810, 1806, 1827, 1814, 1829, 103]
100%|██████████| 9/9 [01:33<00:00, 10.39s/it]

Splitting the text into 17 chunks to be summarized.
Chunk token lengths: [897, 890, 914, 876, 893, 906, 893, 902, 909, 907, 905, 889, 902, 890, 901, 880, 287]
100%|██████████| 17/17 [02:26<00:00,  8.64s/it]

Splitting the text into 31 chunks to be summarized.
Chunk token lengths: [492, 427, 485, 490, 496, 478, 473, 497, 496, 501, 499, 497, 493, 470, 472, 494, 489, 492, 481, 485, 471, 500, 486, 498, 478, 469, 498, 468, 493, 478, 103]
100%|██████████| 31/31 [04:08<00:00,  8.02s/it]
```

## 6. Compare Summary Lengths

Let's quantify the difference in detail by checking the token length of each summary.

```python
summary_lengths = [len(tokenize(x)) for x in
                   [summary_with_detail_0, summary_with_detail_pt25, summary_with_detail_pt5, summary_with_detail_1]]
print("Summary token lengths:", summary_lengths)
```

Output:
```
Summary token lengths: [235, 2529, 4336, 6742]
```

The original document is ~14,630 tokens. Notice how the most detailed summary (`detail=1`) is nearly 25 times longer than the highest-level summary (`detail=0`).

## 7. Examine the Summaries

Finally, let's look at excerpts from the summaries to see how the content changes with the `detail` parameter.

**High-Level Summary (`detail=0`):**
```python
print(summary_with_detail_0)
```
This summary provides a concise overview of AI, covering its definition, history, cycles of hype, recent advancements, and societal impacts in a few paragraphs.

**Very Detailed Summary (`detail=1`):**
```python
print(summary_with_detail_1)
```
This summary is much longer and includes specific subsections on knowledge representation, agents, planning, machine learning techniques, natural language processing, and social intelligence. It delves into technical concepts and research challenges.

## Key Takeaways

- **Controllable Detail:** The `detail` parameter (0 to 1) lets you interpolate between a brief overview and an extensive, section-by-section summary.
- **Chunk-Based Processing:** By splitting the document into more (and smaller) chunks, each chunk receives a dedicated summary, preserving more information from the original text.
- **Practical Use:** This method is ideal for creating executive summaries, detailed reports, or anything in between, all from the same source document.

You can further customize the summarization by providing `additional_instructions` to the model or enabling `summarize_recursively` to build context across chunks.
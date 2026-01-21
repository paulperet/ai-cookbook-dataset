# Guide: Extracting Key Information from Long Documents with GPT-4

This guide demonstrates how to extract specific information from a long PDF document that exceeds the typical context window of a language model. We'll use a chunking strategy to process the document in sections, apply a structured prompt to each chunk, and then consolidate the results.

## Prerequisites

Ensure you have the necessary Python packages installed and your OpenAI API key configured.

```bash
pip install textract tiktoken openai
```

```python
import textract
import os
import openai
import tiktoken

# Initialize the OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

## Step 1: Load and Prepare the Document

First, load the PDF and extract the raw text. We'll perform basic cleaning to format the text for processing.

```python
# Extract text from the PDF
text = textract.process(
    'data/fia_f1_power_unit_financial_regulations_issue_1_-_2022-08-16.pdf',
    method='pdfminer'
).decode('utf-8')

# Clean the text: replace double spaces and newlines
clean_text = text.replace("  ", " ").replace("\n", "; ").replace(';',' ')
```

## Step 2: Define the Chunking Strategy

To process a long document, we need to split it into manageable chunks. The function below creates chunks of approximately `n` tokens, trying to break at the end of a sentence for better coherence.

```python
def create_chunks(text, n, tokenizer):
    """
    Split text into chunks of approximately n tokens, preferring to end at a sentence boundary.
    """
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        # Define a search window for a sentence end
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no sentence end found, fall back to a simple n-token chunk
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j
```

## Step 3: Simple Entity Extraction

We'll start by extracting straightforward facts from the document. First, define a prompt template that instructs the model on what to extract and the desired output format.

```python
template_prompt = '''
Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output "Not specified".
When you extract a key piece of information, include the closest page number.
Use the following format:
0. Who is the author
1. What is the amount of the "Power Unit Cost Cap" in USD, GBP and EUR
2. What is the value of External Manufacturing Costs in USD
3. What is the Capital Expenditure Limit in USD

Document: """<document>"""

0. Who is the author: Tom Anderson (Page 1)
1.
'''
```

Now, create a function that takes a text chunk and the prompt template, sends it to GPT-4, and returns the extracted information.

```python
def extract_chunk(document, template_prompt):
    """
    Send a document chunk to GPT-4 with the extraction prompt.
    """
    prompt = template_prompt.replace('<document>', document)
    messages = [
        {"role": "system", "content": "You help extract information from documents."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model='gpt-4',
        messages=messages,
        temperature=0,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Prepend "1." to align with the prompt's numbering
    return "1." + response.choices[0].message.content
```

Initialize the tokenizer and process each chunk of the document.

```python
# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Create chunks of ~1000 tokens each
chunks = create_chunks(clean_text, 1000, tokenizer)
text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

# Extract information from each chunk
results = []
for chunk in text_chunks:
    extracted = extract_chunk(chunk, template_prompt)
    results.append(extracted)
    print(extracted)  # View each chunk's output
```

## Step 4: Consolidate Simple Results

After processing all chunks, we need to filter and combine the results. We'll remove any entries that are unspecified or contain placeholder underscores.

```python
# Split each result string into a list of lines
groups = [r.split('\n') for r in results]

# Transpose the list of lists and flatten
zipped = list(zip(*groups))
zipped = [x for y in zipped for x in y if "Not specified" not in x and "__" not in x]

# View the consolidated, filtered answers
print(zipped)
```

## Step 5: Complex Entity Extraction

Now, let's tackle more complex questions that require deeper reasoning. We'll adjust the prompt template accordingly.

```python
complex_template_prompt = '''
Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output "Not specified".
When you extract a key piece of information, include the closest page number.
Use the following format:
0. Who is the author
1. How is a Minor Overspend Breach calculated
2. How is a Major Overspend Breach calculated
3. Which years do these financial regulations apply to

Document: """<document>"""

0. Who is the author: Tom Anderson (Page 1)
1.
'''
```

Run the extraction again using the new prompt.

```python
complex_results = []
for chunk in text_chunks:
    extracted = extract_chunk(chunk, complex_template_prompt)
    complex_results.append(extracted)

# Consolidate the complex results
complex_groups = [r.split('\n') for r in complex_results]
complex_zipped = list(zip(*complex_groups))
complex_zipped = [x for y in complex_zipped for x in y if "Not specified" not in x and "__" not in x]

print(complex_zipped)
```

## Summary and Next Steps

You have successfully built a pipeline to extract both simple and complex information from a lengthy PDF document. The key steps were:
1.  Loading and cleaning the document text.
2.  Intelligently chunking the text into processable segments.
3.  Defining clear, structured prompts for extraction.
4.  Using GPT-4 to extract information from each chunk.
5.  Filtering and consolidating the results.

To improve this pipeline, consider:
*   **Prompt Engineering:** Make your prompts more descriptive or specific to reduce ambiguity.
*   **Chunking Strategy:** Experiment with chunk sizes, overlaps, or logical section breaks (e.g., by headings) to prevent information from being split across chunks.
*   **Model Fine-tuning:** If you have sufficient labeled data, fine-tuning a model can yield highly precise extractions for specific document types.

This reusable approach allows you to query any long document for specific entities, providing a scalable solution for document analysis.
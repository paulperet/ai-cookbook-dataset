# Long Document Content Extraction

GPT-3 can help us extract key figures, dates or other bits of important content from documents that are too big to fit into the context window. One approach for solving this is to chunk the document up and process each chunk separately, before combining into one list of answers. 

In this notebook we'll run through this approach:
- Load in a long PDF and pull the text out
- Create a prompt to be used to extract key bits of information
- Chunk up our document and process each chunk to pull any answers out
- Combine them at the end
- This simple approach will then be extended to three more difficult questions

## Approach

- **Setup**: Take a PDF, a Formula 1 Financial Regulation document on Power Units, and extract the text from it for entity extraction. We'll use this to try to extract answers that are buried in the content.
- **Simple Entity Extraction**: Extract key bits of information from chunks of a document by:
    - Creating a template prompt with our questions and an example of the format it expects
    - Create a function to take a chunk of text as input, combine with the prompt and get a response
    - Run a script to chunk the text, extract answers and output them for parsing
- **Complex Entity Extraction**: Ask some more difficult questions which require tougher reasoning to work out

## Setup

```python
!pip install textract
!pip install tiktoken
```

```python
import textract
import os
import openai
import tiktoken

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

# Extract the raw text from each PDF using textract
text = textract.process('data/fia_f1_power_unit_financial_regulations_issue_1_-_2022-08-16.pdf', method='pdfminer').decode('utf-8')
clean_text = text.replace("  ", " ").replace("\n", "; ").replace(';',' ')
```

## Simple Entity Extraction

```python
# Example prompt - 
document = '<document>'
template_prompt=f'''Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output \"Not specified\".
When you extract a key piece of information, include the closest page number.
Use the following format:\n0. Who is the author\n1. What is the amount of the "Power Unit Cost Cap" in USD, GBP and EUR\n2. What is the value of External Manufacturing Costs in USD\n3. What is the Capital Expenditure Limit in USD\n\nDocument: \"\"\"<document>\"\"\"\n\n0. Who is the author: Tom Anderson (Page 1)\n1.'''
print(template_prompt)
```

```python
# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def extract_chunk(document,template_prompt):
    prompt = template_prompt.replace('<document>',document)

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
    return "1." + response.choices[0].message.content
```

```python
# Initialise tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

results = []
    
chunks = create_chunks(clean_text,1000,tokenizer)
text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

for chunk in text_chunks:
    results.append(extract_chunk(chunk,template_prompt))
    #print(chunk)
    print(results[-1])
```

[First Entry, ..., Last Entry]

```python
groups = [r.split('\n') for r in results]

# zip the groups together
zipped = list(zip(*groups))
zipped = [x for y in zipped for x in y if "Not specified" not in x and "__" not in x]
zipped
```

## Complex Entity Extraction

```python
# Example prompt - 
template_prompt=f'''Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output \"Not specified\".
When you extract a key piece of information, include the closest page number.
Use the following format:\n0. Who is the author\n1. How is a Minor Overspend Breach calculated\n2. How is a Major Overspend Breach calculated\n3. Which years do these financial regulations apply to\n\nDocument: \"\"\"<document>\"\"\"\n\n0. Who is the author: Tom Anderson (Page 1)\n1.'''
print(template_prompt)
```

```python
results = []

for chunk in text_chunks:
    results.append(extract_chunk(chunk,template_prompt))
    
groups = [r.split('\n') for r in results]

# zip the groups together
zipped = list(zip(*groups))
zipped = [x for y in zipped for x in y if "Not specified" not in x and "__" not in x]
zipped
```

## Consolidation

We've been able to extract the first two answers safely, while the third was confounded by the date that appeared on every page, though the correct answer is in there as well.

To tune this further you can consider experimenting with:
- A more descriptive or specific prompt
- If you have sufficient training data, fine-tuning a model to find a set of outputs very well
- The way you chunk your data - we have gone for 1000 tokens with no overlap, but more intelligent chunking that breaks info into sections, cuts by tokens or similar may get better results

However, with minimal tuning we have now answered 6 questions of varying difficulty using the contents of a long document, and have a reusable approach that we can apply to any long document requiring entity extraction. Look forward to seeing what you can do with this!
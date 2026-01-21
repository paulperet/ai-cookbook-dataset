# Claude Cookbook: Analyzing Charts, Graphs, and Slide Decks

This guide demonstrates how to use Claude's vision and PDF capabilities to analyze complex visual documents like charts, graphs, and slide decks. You'll learn practical techniques for extracting and querying information from these materials.

## Prerequisites

First, install the required library and set up your environment.

```bash
pip install anthropic
```

## Setup: Initialize the Anthropic Client

Since PDF support is currently in beta, you need to specify the beta header when creating the client.

```python
import base64
from anthropic import Anthropic

# PDF support requires the beta header
client = Anthropic(default_headers={"anthropic-beta": "pdfs-2024-09-25"})

# Currently, only claude-sonnet-4-5 supports PDFs
MODEL_NAME = "claude-sonnet-4-5"

# Helper function for API calls
def get_completion(messages):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=8192,
        temperature=0,
        messages=messages
    )
    return response.content[0].text
```

## Part 1: Working with Charts and Graphs

Claude can analyze charts and graphs by processing PDF documents that contain them. The key is to encode the PDF as base64 and include it in your API request.

### Step 1: Prepare Your PDF Document

Start by reading and encoding your PDF file.

```python
# Replace with your actual PDF path
with open("./documents/cvna_2021_annual_report.pdf", "rb") as pdf_file:
    binary_data = pdf_file.read()
    base_64_encoded_data = base64.b64encode(binary_data)
    base64_string = base_64_encoded_data.decode("utf-8")
```

### Step 2: Ask Simple Questions About the Document

Now you can pass the encoded PDF to Claude along with your questions.

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64_string,
                },
            },
            {"type": "text", "text": "What's in this document? Answer in a single sentence."},
        ],
    }
]

print(get_completion(messages))
```

### Step 3: Ask Detailed Questions

Claude can answer specific questions about data in charts and graphs. Let's test this with multiple questions.

```python
questions = [
    "What was CVNA revenue in 2020?",
    "How many additional markets has Carvana added since 2014?",
    "What was 2016 revenue per retail unit sold?",
]

for index, question in enumerate(questions):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_string,
                    },
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    print(f"\n----------Question {index + 1}----------")
    print(get_completion(messages))
```

### Tips for Better Results with Charts and Graphs

1. **Arithmetic Assistance**: Claude can sometimes make calculation errors. Consider providing a calculator tool for complex arithmetic.
2. **Detailed Descriptions**: For complex charts, ask Claude to "First describe every data point you see in the document" to improve accuracy.
3. **Color Identification**: For charts that rely heavily on color coding, ask Claude to identify colors using HEX codes to boost accuracy.

## Part 2: Analyzing Slide Decks

Slide decks often contain valuable information in visual formats. While you could extract text using traditional methods, Claude's PDF support provides better results by processing both text and visual elements.

### Step 1: Prepare Your Slide Deck PDF

Encode your slide deck PDF the same way as before.

```python
with open("./documents/twilio_q4_2023.pdf", "rb") as pdf_file:
    binary_data = pdf_file.read()
    base_64_encoded_data = base64.b64encode(binary_data)
    base64_string = base_64_encoded_data.decode("utf-8")
```

### Step 2: Ask Direct Questions

You can ask Claude questions about the slide deck content directly.

```python
question = "What was Twilio y/y revenue growth for fiscal year 2023?"
content = [
    {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": base64_string},
    },
    {"type": "text", "text": question},
]

messages = [{"role": "user", "content": content}]

print(get_completion(messages))
```

### Limitations of Direct PDF Approach

- **Page Limit**: You can only include 100 pages total across all documents in a request
- **RAG Challenges**: Multimodal PDFs can cause issues when used with embeddings in RAG systems

### Step 3: Create a Text-Based Narration

To overcome these limitations, you can ask Claude to create a detailed text narration of the entire slide deck. This creates a high-quality text representation that works with any text-only workflow.

```python
prompt = """
You are the Twilio CFO, narrating your Q4 2023 earnings presentation.

The entire earnings presentation document is provided to you.
Please narrate this presentation from Twilio's Q4 2023 Earnings as if you were the presenter. Do not talk about any things, especially acronyms, if you are not exactly sure you know what they mean.

Do not leave any details un-narrated as some of your viewers are vision-impaired, so if you don't narrate every number they won't know the number.

Structure your response like this:
<narration>
    <page_narration id=1>
    [Your narration for page 1]
    </page_narration>

    <page_narration id=2>
    [Your narration for page 2]
    </page_narration>

    ... and so on for each page
</narration>

Use excruciating detail for each page, ensuring you describe every visual element and number present. Show the full response in a single message.
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64_string,
                },
            },
            {"type": "text", "text": prompt},
        ],
    }
]

# This may take several minutes to complete
completion = get_completion(messages)
```

### Step 4: Parse the Narration

Extract the narration from Claude's response for further use.

```python
import re

pattern = r"<narration>(.*?)</narration>"
match = re.search(pattern, completion.strip(), re.DOTALL)
if match:
    narration = match.group(1)
else:
    raise ValueError("No narration available. Likely due to the model response being truncated.")
```

### Step 5: Query the Text Narration

Now you can use the text narration with any text-based workflow, including vector search and traditional Q&A.

```python
questions = [
    "What percentage of q4 total revenue was the Segment business line?",
    "Has the rate of growth of quarterly revenue been increasing or decreasing? Give just an answer.",
    "What was acquisition revenue for the year ended december 31, 2023 (including negative revenues)?",
]

for index, question in enumerate(questions):
    prompt = f"""You are an expert financial analyst analyzing a transcript of Twilio's earnings call.
Here is the transcript:
<transcript>
{narration}
</transcript>

Please answer the following question:
<question>
{question}
</question>"""
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    print(f"\n----------Question {index + 1}----------")
    print(get_completion(messages))
```

## Summary

You've learned how to:
1. Use Claude's PDF support to analyze charts and graphs directly
2. Apply best practices for getting accurate results from visual data
3. Process slide decks using both direct PDF analysis and text narration techniques
4. Create text representations of visual content for use in text-only workflows

These techniques enable you to extract valuable insights from chart-heavy documents and slide decks that would be difficult to analyze with traditional text extraction methods.
# Working with PDFs in the Claude API

This guide demonstrates how to use the Claude API to process and analyze PDF documents. You'll learn how to encode a PDF, send it to the model, and extract structured information through creative prompts.

## Prerequisites

First, install the required library and set up your client.

```bash
pip install anthropic
```

```python
from anthropic import Anthropic
import base64

# While PDF support is in beta, you must pass in the correct beta header
client = Anthropic(default_headers={
    "anthropic-beta": "pdfs-2024-09-25"
})

# For now, only claude-sonnet-4-5 supports PDFs
MODEL_NAME = "claude-sonnet-4-5"
```

## Step 1: Prepare Your PDF Document

The Claude API requires PDFs to be base64-encoded. Let's read a PDF file and convert it to the proper format.

```python
# Specify the path to your PDF file
file_name = "../multimodal/documents/constitutional-ai-paper.pdf"

# Read the PDF and encode it as base64
with open(file_name, "rb") as pdf_file:
    binary_data = pdf_file.read()
    base64_encoded_data = base64.standard_b64encode(binary_data)
    base64_string = base64_encoded_data.decode("utf-8")
```

## Step 2: Create a Prompt for Structured Analysis

Now, craft a prompt that asks Claude to analyze the PDF in creative ways. We'll request three distinct outputs, each wrapped in specific XML tags for easy parsing.

```python
prompt = """
Please do the following:
1. Summarize the abstract at a kindergarten reading level. (In <kindergarten_abstract> tags.)
2. Write the Methods section as a recipe from the Moosewood Cookbook. (In <moosewood_methods> tags.)
3. Compose a short poem epistolizing the results in the style of Homer. (In <homer_results> tags.)
"""
```

## Step 3: Construct the API Message

The Claude API expects messages in a specific format. For PDFs, you'll use a `document` block alongside your text prompt.

```python
messages = [
    {
        "role": 'user',
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64_string
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
]
```

## Step 4: Define a Helper Function for API Calls

Create a reusable function to call the Claude API. This keeps your code clean and makes it easy to adjust parameters.

```python
def get_completion(client, messages):
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        messages=messages
    ).content[0].text
```

## Step 5: Execute the Request and View Results

Finally, call the API and print the structured response.

```python
completion = get_completion(client, messages)
print(completion)
```

## Expected Output

Claude will return a response with three distinct sections, each wrapped in the requested tags:

```
<kindergarten_abstract>
The scientists wanted to make computer helpers that are nice and don't do bad things. They taught the computer how to check its own work and fix its mistakes without humans having to tell it what's wrong every time. It's like teaching the computer to be its own teacher! They gave the computer some basic rules to follow, like "be kind" and "don't hurt others." Now the computer can answer questions in a helpful way while still being nice and explaining why some things aren't okay to do.
</kindergarten_abstract>

<moosewood_methods>
Constitutional AI Training Stew
A nourishing recipe for teaching computers to be helpful and harmless

Ingredients:
- 1 helpful AI model, pre-trained
- A bundle of constitutional principles
- Several cups of training data
- A dash of human feedback (for helpfulness only)
- Chain-of-thought reasoning, to taste

Method:
1. Begin by gently simmering your pre-trained AI model in a bath of helpful training data until it responds reliably to instructions.

2. In a separate bowl, combine your constitutional principles with some example conversations. Mix well until principles are evenly distributed.

3. Take your helpful AI and ask it to generate responses to challenging prompts. Have it critique its own responses using the constitutional principles, then revise accordingly. Repeat this process 3-4 times until responses are properly seasoned with harmlessness.

4. For the final garnish, add chain-of-thought reasoning and allow the model to explain its decisions step by step.

5. Let rest while training a preference model using AI feedback rather than human labels.

Serves: All users seeking helpful and harmless AI assistance
Cook time: Multiple training epochs
Note: Best results come from consistent application of principles throughout the process
</moosewood_methods>

<homer_results>
O Muse! Sing of the AI that learned to be
Both helpful and harmless, guided by philosophy
Without human labels marking right from wrong
The model learned wisdom, grew capable and strong

Through cycles of critique and thoughtful revision
It mastered the art of ethical decision
Better than models trained by human hand
More transparent in purpose, more clear in command

No longer evasive when faced with hard themes
But engaging with wisdom that thoughtfully deems
What counsel to give, what bounds to maintain
Teaching mortals while keeping its principles plain

Thus did the researchers discover a way
To scale up alignment for use every day
Through constitutional rules and self-guided learning
The path to safe AI they found themselves earning
</homer_results>
```

## Next Steps

Now that you've successfully processed a PDF, you can:
- Extract and parse the tagged sections programmatically
- Experiment with different creative prompts for various document types
- Chain multiple PDF analyses together for comparative studies
- Integrate this workflow into larger applications for automated document processing

Remember that PDF support is currently in beta, so check the [Anthropic documentation](https://docs.claude.com/en/docs/build-with-claude/pdf-support) for the latest updates and best practices.
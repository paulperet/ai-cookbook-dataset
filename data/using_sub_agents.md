# Analyzing Financial Reports with Claude Haiku Sub-Agents

This guide demonstrates how to analyze Apple's 2023 financial earnings reports using Claude 3 Haiku as sub-agents to extract information from PDF documents. We'll then use Claude 3 Opus to synthesize the findings and generate a visualization.

## Prerequisites

First, install the required Python libraries:

```bash
pip install anthropic IPython PyMuPDF matplotlib requests pillow
```

## Step 1: Import Libraries and Initialize the Client

Begin by importing the necessary modules and setting up the Anthropic client.

```python
import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor

import fitz
import requests
from anthropic import Anthropic
from PIL import Image

# Initialize the Anthropic client
client = Anthropic()
```

## Step 2: Define the Documents and Question

Specify the URLs for Apple's 2023 quarterly financial statements and define the question you want answered.

```python
# List of Apple's earnings release PDF URLs for FY 2023
pdf_urls = [
    "https://www.apple.com/newsroom/pdfs/fy2023-q4/FY23_Q4_Consolidated_Financial_Statements.pdf",
    "https://www.apple.com/newsroom/pdfs/fy2023-q3/FY23_Q3_Consolidated_Financial_Statements.pdf",
    "https://www.apple.com/newsroom/pdfs/FY23_Q2_Consolidated_Financial_Statements.pdf",
    "https://www.apple.com/newsroom/pdfs/FY23_Q1_Consolidated_Financial_Statements.pdf",
]

# Define the analytical question
QUESTION = "How did Apple's net sales change quarter to quarter in the 2023 financial year and what were the key contributors to the changes?"
```

## Step 3: Create Helper Functions for PDF Processing

Since financial PDFs contain complex tables, we'll convert them to images for easier analysis by the vision-capable Haiku model.

### Download PDFs

```python
def download_pdf(url, folder):
    """Download a PDF from a URL and save it to the specified folder."""
    response = requests.get(url, timeout=60)
    if response.status_code == 200:
        file_name = os.path.join(folder, url.split("/")[-1])
        with open(file_name, "wb") as file:
            file.write(response.content)
        return file_name
    else:
        print(f"Failed to download PDF from {url}")
        return None
```

### Convert PDF Pages to Base64-Encoded Images

```python
def pdf_to_base64_pngs(pdf_path, quality=75, max_size=(1024, 1024)):
    """Convert a PDF file to a list of base64-encoded PNG images."""
    doc = fitz.open(pdf_path)
    base64_encoded_pngs = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        
        # Convert to PIL Image
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Resize if necessary
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        image_data = io.BytesIO()
        image.save(image_data, format="PNG", optimize=True, quality=quality)
        image_data.seek(0)
        base64_encoded = base64.b64encode(image_data.getvalue()).decode("utf-8")
        base64_encoded_pngs.append(base64_encoded)

    doc.close()
    return base64_encoded_pngs
```

## Step 4: Download and Process the PDFs

Create a directory for the downloaded files and process them concurrently for efficiency.

```python
# Create directory for downloaded PDFs
folder = "../images/using_sub_agents"
os.makedirs(folder, exist_ok=True)

# Download PDFs concurrently
with ThreadPoolExecutor() as executor:
    pdf_paths = list(executor.map(download_pdf, pdf_urls, [folder] * len(pdf_urls)))

# Remove any failed downloads
pdf_paths = [path for path in pdf_paths if path is not None]
```

## Step 5: Generate a Specialized Prompt for the Sub-Agents

Use Claude 3 Opus to create a targeted prompt that each Haiku sub-agent will use to extract relevant information from a single quarterly report.

```python
def generate_haiku_prompt(question):
    """Generate a specialized prompt for Haiku sub-agents using Claude 3 Opus."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Based on the following question, please generate a specific prompt for an LLM sub-agent to extract relevant information from an earning's report PDF. Each sub-agent only has access to a single quarter's earnings report. Output only the prompt and nothing else.\n\nQuestion: {question}",
                }
            ],
        }
    ]

    response = client.messages.create(
        model="claude-3-opus-20240229", 
        max_tokens=2048, 
        messages=messages
    )

    return response.content[0].text

# Generate the specialized prompt
haiku_prompt = generate_haiku_prompt(QUESTION)
print("Generated Haiku Prompt:")
print(haiku_prompt)
```

The generated prompt will look like this:

```
Extract the following information from the Apple earnings report PDF for the quarter:
1. Apple's net sales for the quarter
2. Quarter-over-quarter change in net sales
3. Key product categories, services, or regions that contributed significantly to the change in net sales
4. Any explanations provided for the changes in net sales

Organize the extracted information in a clear, concise format focusing on the key data points and insights related to the change in net sales for the quarter.
```

## Step 6: Extract Information Using Haiku Sub-Agents

Now, process each PDF concurrently using Haiku models. Each sub-agent analyzes one quarterly report.

```python
def extract_info(pdf_path, haiku_prompt):
    """Extract information from a PDF using Claude 3 Haiku."""
    base64_encoded_pngs = pdf_to_base64_pngs(pdf_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                *[
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_encoded_png,
                        },
                    }
                    for base64_encoded_png in base64_encoded_pngs
                ],
                {"type": "text", "text": haiku_prompt},
            ],
        }
    ]

    response = client.messages.create(
        model="claude-3-haiku-20240307", 
        max_tokens=2048, 
        messages=messages
    )

    return response.content[0].text, pdf_path

def process_pdf(pdf_path):
    """Wrapper function for concurrent processing."""
    return extract_info(pdf_path, haiku_prompt)

# Process all PDFs concurrently
with ThreadPoolExecutor() as executor:
    extracted_info_list = list(executor.map(process_pdf, pdf_paths))

# Format the extracted information with XML tags
extracted_info = ""
for info, pdf_path in extracted_info_list:
    quarter = pdf_path.split("/")[-1].split("_")[1]
    extracted_info += f'<info quarter="{quarter}">\n{info}\n</info>\n'

print("Extracted Information:")
print(extracted_info)
```

## Step 7: Synthesize Findings with Claude 3 Opus

Pass all extracted information to Claude 3 Opus to generate a comprehensive answer and create visualization code.

```python
# Prepare the message for Claude 3 Opus
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Based on the following extracted information from Apple's earnings releases, please provide a response to the question: {QUESTION}\n\nAlso, please generate Python code using the matplotlib library to accompany your response. Enclose the code within <code> tags.\n\nExtracted Information:\n{extracted_info}",
            }
        ],
    }
]

# Generate the response with visualization code
response = client.messages.create(
    model="claude-3-opus-20240229", 
    max_tokens=4096, 
    messages=messages
)

generated_response = response.content[0].text
print("Generated Response:")
print(generated_response)
```

## Step 8: Extract and Execute the Visualization Code

Parse the response to separate the textual analysis from the generated matplotlib code, then execute the code to create the visualization.

```python
def extract_code_and_response(response):
    """Extract code blocks and non-code text from the response."""
    start_tag = "<code>"
    end_tag = "</code>"
    start_index = response.find(start_tag)
    end_index = response.find(end_tag)
    
    if start_index != -1 and end_index != -1:
        code = response[start_index + len(start_tag):end_index].strip()
        non_code_response = response[:start_index].strip()
        return code, non_code_response
    else:
        return None, response.strip()

# Extract components from the generated response
matplotlib_code, analysis_text = extract_code_and_response(generated_response)

print("Analysis:")
print(analysis_text)

if matplotlib_code:
    print("\nExecuting visualization code...")
    # Note: In production, execute untrusted code in a sandboxed environment
    exec(matplotlib_code)
else:
    print("No visualization code found in the response.")
```

## Summary

This workflow demonstrates a powerful pattern for document analysis:

1. **Parallel Processing**: Use sub-agents (Claude Haiku) to analyze individual documents concurrently
2. **Specialized Prompts**: Generate targeted prompts for specific extraction tasks
3. **Synthesis**: Combine extracted information using a more powerful model (Claude Opus)
4. **Visualization**: Automatically generate code to visualize the findings

The approach is particularly effective for financial documents with complex tables that are challenging for traditional PDF parsers. By converting pages to images and using vision-capable models, you can extract structured information efficiently.

**Security Note**: Always execute model-generated code in a sandboxed environment in production scenarios to prevent potential security risks.
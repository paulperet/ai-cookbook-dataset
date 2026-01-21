# Structured Generation from Images or Documents Using Vision Language Models

This guide demonstrates how to extract structured information from images or documents using a Vision Language Model (VLM). You will use the **SmolVLM-Instruct** model from Hugging Face, combined with the **Outlines** library, to generate JSON-formatted outputs like descriptions, questions, and quality tags from visual inputs. This technique is useful for creating structured datasets from images or document pages (e.g., PDFs converted to images).

## Prerequisites

Ensure you have the necessary libraries installed.

```bash
pip install accelerate outlines transformers torch flash-attn datasets sentencepiece
```

## Step 1: Import Required Libraries

Begin by importing the necessary modules.

```python
import outlines
import torch

from datasets import load_dataset
from outlines.models.transformers_vision import transformers_vision
from transformers import AutoModelForImageTextToText, AutoProcessor
from pydantic import BaseModel
```

## Step 2: Initialize the Model

You will load the SmolVLM-Instruct model and its processor. The function below retrieves the appropriate model and processor classes for use with Outlines.

```python
model_name = "HuggingFaceTB/SmolVLM-Instruct"

def get_model_and_processor_class(model_name: str):
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    classes = model.__class__, processor.__class__
    del model, processor
    return classes

model_class, processor_class = get_model_and_processor_class(model_name)

# Set device for computation
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Initialize the model with Outlines
model = transformers_vision(
    model_name,
    model_class=model_class,
    device=device,
    model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
    processor_kwargs={"device": device},
    processor_class=processor_class,
)
```

## Step 3: Define the Output Structure

Use Pydantic to define a schema for the structured output. The model will generate a JSON object containing a quality tag, a description, and a question for each image.

```python
class ImageData(BaseModel):
    quality: str
    description: str
    question: str

# Create a structured JSON generator
structured_generator = outlines.generate.json(model, ImageData)
```

## Step 4: Create the Extraction Prompt

Craft a prompt that instructs the model to analyze the image and return the structured data.

```python
prompt = """
You are an image analysis assistant.

Provide a quality tag, a description and a question.

The quality can either be "good", "okay" or "bad".
The question should be concise and objective.

Return your response as a valid JSON object.
""".strip()
```

## Step 5: Load the Image Dataset

For this example, you'll use a subset of the `openbmb/RLAIF-V-Dataset` from Hugging Face Datasets.

```python
dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:10]")
dataset
```

## Step 6: Extract Structured Information

Define a function that formats the prompt with the image, passes it to the model, and parses the structured output. Then, apply this function to each row in the dataset.

```python
def extract(row):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        },
    ]

    # Format the prompt using the processor's chat template
    formatted_prompt = model.processor.apply_chat_template(
        messages, add_generation_prompt=True
    )

    # Generate structured output
    result = structured_generator(formatted_prompt, [row["image"]])

    # Add the results to the dataset row
    row['synthetic_question'] = result.question
    row['synthetic_description'] = result.description
    row['synthetic_quality'] = result.quality
    return row

# Apply extraction to the dataset
dataset = dataset.map(lambda x: extract(x))
dataset
```

## Step 7: Save the Results

Push the newly enriched dataset to the Hugging Face Hub to save your work.

```python
dataset.push_to_hub("davidberenstein1957/structured-generation-information-extraction-vlms-openbmb-RLAIF-V-Dataset", split="train")
```

## Step 8: Extending to Documents

You can apply the same technique to documents by first converting PDF pages to images. Use a library like `pdf2image` for the conversion.

```python
# Example for processing a PDF
from pdf2image import convert_from_path

pdf_path = "path/to/your/pdf/file.pdf"
pages = convert_from_path(pdf_path)

for page in pages:
    # Use the same extraction function defined earlier
    extract_objects = extract(page, prompt)
```

## Conclusion

You have successfully used a Vision Language Model to extract structured JSON data from images. This method can be adapted for document analysis by converting pages to images. The results provide a foundation for creating structured datasets, which can be used for further analysis or model fine-tuning.

## Next Steps

- Explore the [Outlines library documentation](https://github.com/outlines-ai/outlines) to learn about additional generation methods and parameters.
- Experiment with different models and prompts tailored to your specific use case.
- Investigate alternative methods for structured information extraction from documents.
# Structured Generation from Images or Documents Using Vision Language Models

We will be using the SmolVLM-Instruct model from HuggingFaceTB to extract structured information from documents. We will run the VLM using the Hugging Face Transformers library and the [Outlines library](https://github.com/dottxt-ai/outlines), which facilitates structured generation based on limiting token sampling probabilities. 

> This approach is based on a [Outlines tutorial](https://dottxt-ai.github.io/outlines/latest/cookbook/atomic_caption/).

## Dependencies and imports

First, let's install the necessary libraries.

```python
%pip install accelerate outlines transformers torch flash-attn datasets sentencepiece
```

Let's continue with importing the necessary libraries.

```python
import outlines
import torch

from datasets import load_dataset
from outlines.models.transformers_vision import transformers_vision
from transformers import AutoModelForImageTextToText, AutoProcessor
from pydantic import BaseModel
```

## Initialising our model

We will start by initialising our model from [HuggingFaceTB/SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct). Outlines expects us to pass in a model class and processor class, so we will make this example a bit more generic by creating a function that returns those. Alternatively, you could look at the model and tokenizer config within the [Hub repo files](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/tree/main), and import those classes directly.

```python
model_name = "HuggingFaceTB/SmolVLM-Instruct"


def get_model_and_processor_class(model_name: str):
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    classes = model.__class__, processor.__class__
    del model, processor
    return classes


model_class, processor_class = get_model_and_processor_class(model_name)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = transformers_vision(
    model_name,
    model_class=model_class,
    device=device,
    model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
    processor_kwargs={"device": device},
    processor_class=processor_class,
)
```

## Structured Generation

Now, we are going to define a function that will define how the output of our model will be structured. We will be using the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset), which contains a set of images along with questions and their chosen and rejected reponses. This is an okay dataset but we want to create additional text-image-to-text data on top of the images to get our own structured dataset, and potentially fine-tune our model on it. We will use the model to generate a caption, a question and a simple quality tag for the image. 

```python
class ImageData(BaseModel):
    quality: str
    description: str
    question: str

structured_generator = outlines.generate.json(model, ImageData)
```

Now, let's come up with an extraction prompt.

```python
prompt = """
You are an image analysis assisant.

Provide a quality tag, a description and a question.

The quality can either be "good", "okay" or "bad".
The question should be concise and objective.

Return your response as a valid JSON object.
""".strip()
```

Let's load our image dataset.

```python
dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train[:10]")
dataset
```

Now, let's define a function that will extract the structured information from the image. We will format the prompt using the `apply_chat_template` method and pass it to the model along with the image after that.

```python
def extract(row):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        },
    ]

    formatted_prompt = model.processor.apply_chat_template(
        messages, add_generation_prompt=True
    )

    result = structured_generator(formatted_prompt, [row["image"]])
    row['synthetic_question'] = result.question
    row['synthetic_description'] = result.description
    row['synthetic_quality'] = result.quality
    return row


dataset = dataset.map(lambda x: extract(x))
dataset
```

Let's now push our new dataset to the Hub.

```python
dataset.push_to_hub("davidberenstein1957/structured-generation-information-extraction-vlms-openbmb-RLAIF-V-Dataset", split="train")
```

The results are not perfect, but they are a good starting point to continue exploring with different models and prompts!

## Conclusion

We've seen how to extract structured information from documents using a vision language model. We can use similar extractive methods to extract structured information from documents, using somehting like `pdf2image` to convert the document to images and running information extraction on each image pdf of the page.

```python
pdf_path = "path/to/your/pdf/file.pdf"
pages = convert_from_path(pdf_path)
for page in pages:
    extract_objects = extract_objects(page, prompt)
```

## Next Steps

- Take a look at the [Outlines](https://github.com/outlines-ai/outlines) library for more information on how to use it. Explore the different methods and parameters.
- Explore extraction on your own usecase with your own model.
- Use a different method of extracting structured information from documents.
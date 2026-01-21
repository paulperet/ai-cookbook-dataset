# Smol Multimodal RAG: Building with ColSmolVLM and SmolVLM on Colab's Free-Tier GPU

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

In this guide, you will build a **Multimodal Retrieval-Augmented Generation (RAG)** system by integrating [**ColSmolVLM**](https://huggingface.co/vidore/colsmolvlm-alpha) for document retrieval and [**SmolVLM**](https://huggingface.co/blog/smolvlm/) as the vision-language model (VLM). These lightweight models enable a fully functional multimodal RAG system to run on consumer GPUs and even on the Google Colab free-tier.

This guide is the third installment in the **Multimodal RAG Recipes** series. If you're new to the topic or want to explore more, check out these previous recipes:

- [Multimodal Retrieval-Augmented Generation (RAG) with Document Retrieval (ColPali) and Vision Language Models (VLMs)](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms)
- [Multimodal RAG with ColQwen2, Reranker, and Quantized VLMs on Consumer GPUs](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_reranker_and_vlms)

Let's dive in and build a powerful yet compact RAG system! 🚀

## Prerequisites

This tutorial requires a GPU runtime (like Google Colab's free-tier T4). Ensure you have the necessary Python libraries installed.

## Step 1: Install Dependencies

Let's start by installing the essential libraries. For this notebook, you'll need to download a **work-in-progress PR** of [byaldi](https://github.com/sergiopaniego/byaldi). Once the [PR](https://github.com/AnswerDotAI/byaldi/pull/69) is merged, these installation steps can be updated accordingly.

```bash
pip install -q git+https://github.com/sergiopaniego/byaldi.git@colsmolvlm-support
```

## Step 2: Load and Prepare the Dataset

You'll use charts and maps from [Our World in Data](https://ourworldindata.org/), focusing on the [life expectancy data](https://ourworldindata.org/life-expectancy). A curated subset is available as a [dataset on Hugging Face](https://huggingface.co/datasets/sergiopaniego/ourworldindata_example).

**Citation:**
```
Saloni Dattani, Lucas Rodés-Guirao, Hannah Ritchie, Esteban Ortiz-Ospina and Max Roser (2023) - “Life Expectancy” Published online at OurWorldinData.org. Retrieved from: 'https://ourworldindata.org/life-expectancy' [Online Resource]
```

First, load the dataset.

```python
from datasets import load_dataset

dataset = load_dataset("sergiopaniego/ourworldindata_example", split='train')
```

### Save Images Locally

To prepare the data for the RAG system, save the images locally. This enables the document retrieval model to efficiently index and process the visual content.

```python
import os
from PIL import Image

def save_images_to_local(dataset, output_folder="data/"):
    os.makedirs(output_folder, exist_ok=True)

    for image_id, image_data in enumerate(dataset):
        image = image_data['image']

        if isinstance(image, str):
            image = Image.open(image)

        output_path = os.path.join(output_folder, f"image_{image_id}.png")
        image.save(output_path, format='PNG')
        print(f"Image saved in: {output_path}")

save_images_to_local(dataset)
```

### Load and Inspect the Images

Load the saved images to explore the dataset.

```python
import os
from PIL import Image

def load_png_images(image_folder):
    png_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    all_images = {}

    for image_id, png_file in enumerate(png_files):
        image_path = os.path.join(image_folder, png_file)
        image = Image.open(image_path)
        all_images[image_id] = image

    return all_images

all_images = load_png_images("/content/data/")
```

Let's visualize a few samples to understand the data structure.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(20, 15))

for i, ax in enumerate(axes.flat):
    img = all_images[i]
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.show()
```

## Step 3: Initialize the ColSmolVLM Document Retrieval Model

Now, initialize the **Document Retrieval Model** which will extract relevant information from the images. You'll use **[Byaldi](https://github.com/AnswerDotAI/byaldi)**, a library designed for multimodal RAG pipelines.

First, load the model from the checkpoint.

```python
from byaldi import RAGMultiModalModel

docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colsmolvlm-alpha")
```

### Index the Documents

Index the images so the model can efficiently retrieve them based on queries.

```python
docs_retrieval_model.index(
    input_path="data/",
    index_name="image_index",
    store_collection_with_index=False,
    overwrite=True
)
```

## Step 4: Test Document Retrieval

Test the retrieval model by submitting a query. The model will return the most relevant documents.

```python
text_query = 'What is the overall trend in life expectancy across different countries and regions?'

results = docs_retrieval_model.search(text_query, k=1)
print(results)
```

Output:
```
[{'doc_id': 5, 'page_num': 1, 'score': 22.0, 'metadata': {}, 'base64': None}]
```

Let's view the retrieved image to verify the match.

```python
result_image = all_images[results[0]['doc_id']]
result_image
```

## Step 5: Initialize the SmolVLM Visual Language Model

Next, initialize the **Visual Language Model (VLM)** for question answering using **[SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)**.

Load the model and transfer it to the GPU.

```python
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
import torch

model_id = "HuggingFaceTB/SmolVLM-Instruct"
vl_model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager",
)
vl_model.eval()
```

Initialize the processor.

```python
vl_model_processor = AutoProcessor.from_pretrained(model_id)
```

## Step 6: Assemble and Test the Full RAG Pipeline

With all components loaded, you can now assemble the system. First, set up the chat structure with the retrieved image and the user's query.

```python
chat_template = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": text_query
            },
        ],
    }
]
```

Prepare the inputs for the VLM.

```python
text = vl_model_processor.apply_chat_template(
    chat_template, add_generation_prompt=True
)

inputs = vl_model_processor(
    text=text,
    images=[result_image],
    return_tensors="pt",
)
inputs = inputs.to("cuda")
```

Generate the answer.

```python
generated_ids = vl_model.generate(**inputs, max_new_tokens=500)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = vl_model_processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0])
```

Output:
```
The overall trend in life expectancy across different countries and regions is an increase over time.
```

Great! The **SmolVLM** answers the query correctly. 🎉

### Check Memory Consumption

Let's see the resource usage of the **SmolVLM**.

```python
print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

Output:
```
GPU allocated memory: 8.32 GB
GPU reserved memory: 10.38 GB
```

## Step 7: Create a Reusable Pipeline Function

Now, create a method that encapsulates the entire pipeline for easy reuse.

```python
def answer_with_multimodal_rag(vl_model, docs_retrieval_model, vl_model_processor, all_images, text_query, retrival_top_k, max_new_tokens):
    results = docs_retrieval_model.search(text_query, k=retrival_top_k)
    result_image = all_images[results[0]['doc_id']]

    chat_template = [
    {
      "role": "user",
      "content": [
          {"type": "image"},
          {"type": "text", "text": text_query}
        ],
      }
    ]

    # Prepare the inputs
    text = vl_model_processor.apply_chat_template(chat_template, add_generation_prompt=True)
    inputs = vl_model_processor(
        text=text,
        images=[result_image],
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate text from the vl_model
    generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text
```

Test the complete RAG system.

```python
output_text = answer_with_multimodal_rag(
    vl_model=vl_model,
    docs_retrieval_model=docs_retrieval_model,
    vl_model_processor=vl_model_processor,
    all_images=all_images,
    text_query='What is the overall trend in life expectancy across different countries and regions?',
    retrival_top_k=1,
    max_new_tokens=500
)
print(output_text[0])
```

Output:
```
The overall trend in life expectancy across different countries and regions is an increase over time.
```

🏆 You now have a fully operational **smol RAG pipeline** that integrates both a **smol Document Retrieval Model** and a **smol Visual Language Model**, optimized to run on a single consumer GPU!

## Step 8: Go Even Smoler with Quantization

Can we go **smoler**? Yes! Use a quantized version of **SmolVLM** to further reduce resource requirements. For a clear comparison, reinitialize the system and run all cells except the VLM instantiation steps.

First, install [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index).

```bash
pip install -q -U bitsandbytes
```

### Configure 4-bit Quantization

Create a `BitsAndBytesConfig` to load the model in a quantized **int-4** configuration.

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

Load the quantized model.

```python
from transformers import Idefics3ForConditionalGeneration, AutoProcessor

model_id = "HuggingFaceTB/SmolVLM-Instruct"
vl_model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    _attn_implementation="eager",
    device_map="auto"
)

vl_model_processor = AutoProcessor.from_pretrained(model_id)
```

### Test the Quantized Model

Test the capabilities of the quantized model.

```python
output_text = answer_with_multimodal_rag(
    vl_model=vl_model,
    docs_retrieval_model=docs_retrieval_model,
    vl_model_processor=vl_model_processor,
    all_images=all_images,
    text_query='What is the overall trend in life expectancy across different countries and regions?',
    retrival_top_k=1,
    max_new_tokens=500
)
print(output_text[0])
```

Output:
```
The overall trend in life expectancy across different countries and regions is an increase over time.
```

The model works correctly! 🎉 Now, check the memory consumption.

```python
print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

Output:
```
GPU allocated memory: 5.44 GB
GPU reserved memory: 7.86 GB
```

You've successfully reduced memory usage even further! 🚀

## Memory Consumption Comparison

Below is a table comparing the memory consumption of two other multimodal RAG notebooks in the [Cookbook](https://huggingface.co/learn/cookbook/) alongside the two versions described here. As you can see, these systems are an order of magnitude smaller in terms of resource requirements.

| Notebook | GPU Allocated Memory (GB) | GPU Reserved Memory (GB) |
|----------|---------------------------|--------------------------|
| Smol Multimodal RAG with Quantization | 5.44 GB | 7.86 GB |
| Smol Multimodal RAG | 8.32 GB | 10.38 GB |
| [Multimodal RAG with ColQwen2, Reranker, and Quantized VLMs on Consumer GPUs](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_reranker_and_vlms) | 13.93 GB | 14.59 GB |
| [Multimodal RAG with Document Retrieval (ColPali) and Vision Language Models (VLMs)](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval) | (Data from source) | (Data from source) |

## Conclusion

You have successfully built a **smol multimodal RAG system** using **ColSmolVLM** for retrieval and **SmolVLM** for answer generation. You also implemented a quantized version to further reduce GPU memory consumption, making it ideal for running on free-tier resources like Google Colab.

This pipeline can be extended to larger datasets and customized for various multimodal tasks. Happy building!
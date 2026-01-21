# Multimodal RAG with ColQwen2, Reranker, and Quantized VLMs on Consumer GPUs

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

This guide demonstrates how to build a **Multimodal Retrieval-Augmented Generation (RAG)** system. We will integrate:
1.  **[ColQwen2](https://huggingface.co/vidore/colqwen2-v1.0)** for document retrieval.
2.  **[MonoQwen2-VL-v0.1](https://huggingface.co/lightonai/MonoQwen2-VL-v0.1)** for reranking.
3.  A quantized version of **[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/)** as the Vision Language Model (VLM).

This pipeline enhances query responses by combining text-based queries with visual data from documents. Crucially, it's optimized to run on a single consumer-grade GPU (tested on an L4) by using a quantized VLM and efficient document retrieval, avoiding complex OCR pipelines.

> **Prerequisite:** This guide builds upon concepts from the [Multimodal RAG using Document Retrieval and VLMs](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms) cookbook. Reviewing it first is recommended.

## 1. Setup and Installation

Begin by installing the required libraries.

```bash
pip install -U -q byaldi pdf2image qwen-vl-utils transformers bitsandbytes peft
pip install -U -q rerankers[monovlm]
```

## 2. Load and Prepare the Dataset

We'll use a curated subset of charts and maps from [Our World in Data](https://ourworldindata.org/life-expectancy), focusing on life expectancy data.

**Citation:**
```
Saloni Dattani, Lucas Rodés-Guirao, Hannah Ritchie, Esteban Ortiz-Ospina and Max Roser (2023) - “Life Expectancy” Published online at OurWorldinData.org. Retrieved from: 'https://ourworldindata.org/life-expectancy' [Online Resource]
```

### 2.1 Download the Dataset
Load the dataset from the Hugging Face Hub.

```python
from datasets import load_dataset

dataset = load_dataset("sergiopaniego/ourworldindata_example", split='train')
```

### 2.2 Save and Resize Images Locally
The document retriever needs local files to index. We'll save the images and resize them to **448x448** to optimize memory usage and processing speed.

```python
import os
from PIL import Image

def save_images_to_local(dataset, output_folder="data/"):
    os.makedirs(output_folder, exist_ok=True)

    for image_id, image_data in enumerate(dataset):
        image = image_data['image']
        if isinstance(image, str):
            image = Image.open(image)
        image = image.resize((448, 448))
        output_path = os.path.join(output_folder, f"image_{image_id}.png")
        image.save(output_path, format='PNG')
        print(f"Image saved: {output_path}")

save_images_to_local(dataset)
```

### 2.3 Load Images for Inspection
Load the saved images into a dictionary for easy access.

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

all_images = load_png_images("data/")
```

## 3. Initialize the Document Retriever (ColQwen2)

We'll use **[Byaldi](https://github.com/AnswerDotAI/byaldi)**, a wrapper for late-interaction multimodal models like ColQwen2, to handle document retrieval.

### 3.1 Load the Model
Load the pre-trained ColQwen2 model.

```python
from byaldi import RAGMultiModalModel

docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0")
```

### 3.2 Index the Documents
Create a searchable index from the folder containing your images.

```python
docs_retrieval_model.index(
    input_path="data/",
    index_name="image_index",
    store_collection_with_index=False,
    overwrite=True
)
```

## 4. Retrieve and Rerank Documents

### 4.1 Perform an Initial Search
Test the retriever with a sample query. The model returns the top `k` most relevant documents.

```python
text_query = 'How does the life expectancy change over time in France and South Africa?'
results = docs_retrieval_model.search(text_query, k=3)
```

### 4.2 Examine Retrieved Images
Let's see which images were retrieved.

```python
def get_grouped_images(results, all_images):
    grouped_images = []
    for result in results:
        doc_id = result['doc_id']
        grouped_images.append(all_images[doc_id])
    return grouped_images

grouped_images = get_grouped_images(results, all_images)
```

### 4.3 Initialize the Reranker
To improve retrieval quality, we'll use a reranker model from the `rerankers` library.

```python
from rerankers import Reranker

ranker = Reranker("monovlm", device='cuda')
```

### 4.4 Convert Images for Reranking
The reranker expects images in base64 format.

```python
import base64
from io import BytesIO

def images_to_base64(images):
    base64_images = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(img_base64)
    return base64_images

base64_list = images_to_base64(grouped_images)
```

### 4.5 Rerank the Results
Pass the query and base64 images to the reranker. It will assign scores and reorder the results.

```python
results = ranker.rank(text_query, base64_list)
```

### 4.6 Process Reranker Results
Extract the top document after reranking.

```python
def process_ranker_results(results, grouped_images, top_k=1, log=False):
    new_grouped_images = []
    for i, doc in enumerate(results.top_k(top_k)):
        if log:
            print(f"Rank {i}: Score={doc.score}")
        new_grouped_images.append(grouped_images[doc.doc_id])
    return new_grouped_images

new_grouped_images = process_ranker_results(results, grouped_images, top_k=1, log=True)
```

## 5. Initialize the Vision Language Model (VLM)

We'll use a **quantized** version of **Qwen2-VL-7B-Instruct** to run efficiently on a consumer GPU.

### 5.1 Load the Quantized Model
Configure 4-bit quantization using `BitsAndBytesConfig` to reduce memory footprint.

```python
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
vl_model.eval()
```

### 5.2 Initialize the Processor
The processor handles tokenization and image preprocessing. We set pixel size constraints to manage GPU memory.

```python
min_pixels = 224*224
max_pixels = 448*448
vl_model_processor = Qwen2VLProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)
```

## 6. Generate the Final Answer

Now, we'll feed the reranked image and the user query to the VLM to generate an answer.

### 6.1 Prepare the Chat Template
Structure the input for the model, combining the image and the text query.

```python
chat_template = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": new_grouped_images[0]},
            {"type": "text", "text": text_query}
        ],
    }
]
```

### 6.2 Process Inputs
Apply the chat template and prepare the tensors for the model.

```python
text = vl_model_processor.apply_chat_template(
    chat_template, tokenize=False, add_generation_prompt=True
)
image_inputs, _ = process_vision_info(chat_template)
inputs = vl_model_processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
```

### 6.3 Generate and Decode the Response
Run inference and decode the generated tokens into text.

```python
generated_ids = vl_model.generate(**inputs, max_new_tokens=500)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = vl_model_processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

## 7. Assemble the Complete Pipeline

Let's wrap the entire process into a reusable function.

```python
def answer_with_multimodal_rag(vl_model, docs_retrieval_model, vl_model_processor, all_images, text_query, retrival_top_k, reranker_top_k, max_new_tokens):
    # 1. Retrieve documents
    results = docs_retrieval_model.search(text_query, k=retrival_top_k)
    grouped_images = get_grouped_images(results, all_images)

    # 2. Rerank documents
    base64_list = images_to_base64(grouped_images)
    results = ranker.rank(text_query, base64_list)
    grouped_images = process_ranker_results(results, grouped_images, top_k=reranker_top_k)

    # 3. Prepare VLM input
    chat_template = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image} for image in grouped_images
            ] + [
                {"type": "text", "text": text_query}
            ],
        }
    ]
    text = vl_model_processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(chat_template)
    inputs = vl_model_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 4. Generate answer
    generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text
```

### 7.1 Test the Complete System
Run the full pipeline with a new query.

```python
output_text = answer_with_multimodal_rag(
    vl_model=vl_model,
    docs_retrieval_model=docs_retrieval_model,
    vl_model_processor=vl_model_processor,
    all_images=all_images,
    text_query='What is the overall trend in life expectancy across different countries and regions?',
    retrival_top_k=3,
    reranker_top_k=1,
    max_new_tokens=500
)
print(output_text[0])
```

## Conclusion

You have successfully built a multimodal RAG system that:
1.  **Retrieves** relevant visual documents using ColQwen2.
2.  **Refines** the results using a MonoQwen2-VL reranker.
3.  **Generates** insightful answers using a quantized Qwen2-VL model.

This pipeline is efficient and capable of running on consumer hardware, providing a powerful tool for querying and understanding visual data. You can extend this system by using larger datasets, experimenting with different retrieval or reranking models, or adjusting the VLM's prompt template for specific use cases.
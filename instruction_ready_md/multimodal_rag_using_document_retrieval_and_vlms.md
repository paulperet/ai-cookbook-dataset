# Multimodal RAG with Document Retrieval (ColPali) and Vision Language Models (VLMs)

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

**Note**: This guide is resource-intensive and requires substantial computational power (e.g., an A100 GPU).

In this tutorial, you will build a **Multimodal Retrieval-Augmented Generation (RAG)** system. It combines the **ColPali** retriever for document retrieval with the **Qwen2-VL** Vision Language Model (VLM) to answer queries using both text-based documents and visual data. Instead of a complex OCR pipeline, we use a Document Retrieval Model to efficiently fetch relevant documents based on a user query.

## Prerequisites

Ensure you have the necessary libraries installed.

```bash
pip install -U -q byaldi pdf2image qwen-vl-utils transformers
# Tested with byaldi==0.0.4, pdf2image==1.17.0, qwen-vl-utils==0.0.8, transformers==4.45.0
```

Install `poppler-utils` for PDF manipulation.

```bash
sudo apt-get install -y poppler-utils
```

## Step 1: Load and Prepare the Dataset

We'll use IKEA assembly instructions (PDFs) as our dataset. You'll download a few examples and convert them to images for processing.

First, import the required modules and define the PDF URLs.

```python
import requests
import os

pdfs = {
    "MALM": "https://www.ikea.com/us/en/assembly_instructions/malm-4-drawer-chest-white__AA-2398381-2-100.pdf",
    "BILLY": "https://www.ikea.com/us/en/assembly_instructions/billy-bookcase-white__AA-1844854-6-2.pdf",
    "BOAXEL": "https://www.ikea.com/us/en/assembly_instructions/boaxel-wall-upright-white__AA-2341341-2-100.pdf",
    "ADILS": "https://www.ikea.com/us/en/assembly_instructions/adils-leg-white__AA-844478-6-2.pdf",
    "MICKE": "https://www.ikea.com/us/en/assembly_instructions/micke-desk-white__AA-476626-10-100.pdf"
}

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

for name, url in pdfs.items():
    response = requests.get(url)
    pdf_path = os.path.join(output_dir, f"{name}.pdf")

    with open(pdf_path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded {name} to {pdf_path}")

print("Downloaded files:", os.listdir(output_dir))
```

Now, convert the PDFs to images using `pdf2image`. This step is crucial because the document retrieval model processes visual content.

```python
from pdf2image import convert_from_path

def convert_pdfs_to_images(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    all_images = {}

    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path)
        all_images[doc_id] = images

    return all_images

all_images = convert_pdfs_to_images("/content/data/")
```

## Step 2: Initialize the ColPali Document Retrieval Model

We'll use **Byaldi**, a wrapper around ColPali, to load and index our documents.

```python
from byaldi import RAGMultiModalModel

docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
```

Index the documents from the `data/` folder. This allows the model to process and organize them for efficient retrieval.

```python
docs_retrieval_model.index(
    input_path="data/",
    index_name="image_index",
    store_collection_with_index=False,
    overwrite=True
)
```

## Step 3: Test Document Retrieval

Now, test the retriever with a sample query. The model will return the most relevant documents (images) based on the query.

```python
text_query = "How many people are needed to assemble the Malm?"

results = docs_retrieval_model.search(text_query, k=3)
results
```

The output will be a list of dictionaries containing metadata about the retrieved documents, including `doc_id` and `page_num`.

Next, define a helper function to extract the actual images from the results.

```python
def get_grouped_images(results, all_images):
    grouped_images = []

    for result in results:
        doc_id = result['doc_id']
        page_num = result['page_num']
        # page_num is 1-indexed, doc_id is 0-indexed
        grouped_images.append(all_images[doc_id][page_num - 1])

    return grouped_images

grouped_images = get_grouped_images(results, all_images)
```

## Step 4: Initialize the Vision Language Model (VLM)

We'll use **Qwen2-VL-7B-Instruct** as our VLM for question answering. Load the model and move it to the GPU.

```python
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import torch

vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
)
vl_model.cuda().eval()
```

Initialize the processor with optimized image resolution settings to fit more images into GPU memory.

```python
min_pixels = 224*224
max_pixels = 1024*1024
vl_model_processor = Qwen2VLProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)
```

## Step 5: Assemble and Test the Full Pipeline

First, create a chat template that includes the retrieved images and the user query. This structure tells the VLM what to process.

```python
chat_template = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": grouped_images[0],
            },
            {
                "type": "image",
                "image": grouped_images[1],
            },
            {
                "type": "image",
                "image": grouped_images[2],
            },
            {
                "type": "text",
                "text": text_query
            },
        ],
    }
]
```

Apply the chat template and process the inputs for the VLM.

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

The model should output an answer based on the retrieved images and the query.

## Step 6: Create a Reusable Pipeline Function

Encapsulate the entire process into a single function for easy reuse.

```python
def answer_with_multimodal_rag(vl_model, docs_retrieval_model, vl_model_processor, all_images, text_query, top_k, max_new_tokens):
    # Retrieve relevant documents
    results = docs_retrieval_model.search(text_query, k=top_k)
    grouped_images = get_grouped_images(results, all_images)

    # Build chat template
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

    # Prepare inputs
    text = vl_model_processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(chat_template)
    inputs = vl_model_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate answer
    generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode output
    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text
```

Test the complete pipeline with a new query.

```python
output_text = answer_with_multimodal_rag(
    vl_model=vl_model,
    docs_retrieval_model=docs_retrieval_model,
    vl_model_processor=vl_model_processor,
    all_images=all_images,
    text_query="How do I assemble the Micke desk?",
    top_k=3,
    max_new_tokens=500
)
print(output_text[0])
```

## Conclusion

You have successfully built a Multimodal RAG pipeline that combines a Document Retrieval Model (ColPali) with a Vision Language Model (Qwen2-VL). This system can answer complex queries by retrieving relevant visual documents and generating informed responses.

## Further Exploration

- **ColPali Resources**:
  - [Document Similarity Search with ColPali](https://huggingface.co/blog/fsommers/document-similarity-colpali)
  - [ColPali Cookbooks](https://github.com/tonywu71/colpali-cookbooks/tree/main)
  - [ColPali Fine-Tuning Query Generator](https://huggingface.co/spaces/davanstrien/ColPali-Query-Generator)
- **Additional Reads**:
  - [Beyond Text: The Rise of Vision-Driven Document Retrieval for RAG](https://blog.vespa.ai/the-rise-of-vision-driven-document-retrieval-for-rag/)
  - [Scaling ColPali to Billions of PDFs with Vespa](https://blog.vespa.ai/scaling-colpali-to-billions/)
- **Collections**:
  - [Multimodal RAG Collection](https://huggingface.co/collections/merve/multimodal-rag-66d97602e781122aae0a5139)
- **Original Paper and Code**:
  - [ColPali: Efficient Document Retrieval with Vision Language Models (Paper)](https://arxiv.org/pdf/2407.01449)
  - [ColPali: Efficient Document Retrieval with Vision Language Models (Repo)](https://github.com/illuin-tech/colpali)
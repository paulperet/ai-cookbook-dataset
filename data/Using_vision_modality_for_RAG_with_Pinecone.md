# Optimizing Retrieval-Augmented Generation with GPT-4o Vision

## Introduction
Implementing Retrieval-Augmented Generation (RAG) presents unique challenges when working with documents rich in images, graphics, and tables. Traditional RAG models excel with textual data but often falter when visual elements play a crucial role in conveying information. This guide bridges that gap by leveraging GPT-4o's vision modality to extract and interpret visual content, ensuring generated responses are both informative and accurate.

Our approach involves parsing documents into images and using metadata tagging to identify pages containing visual elements. When a semantic search retrieves such a page, we pass the page image to a vision model instead of relying solely on text. This method enhances the model's ability to understand and answer queries pertaining to visual data.

In this tutorial, you will learn to:
1. Set up a vector store with Pinecone
2. Parse PDFs and extract visual information using GPT-4o Vision
3. Generate and store embeddings with metadata flags for visual content
4. Perform semantic search and enhance responses with visual context

**Note:** Using the Vision Modality is resource-intensive, leading to increased latency and cost. It's advisable to use this approach only when performance with plain text extraction is unsatisfactory.

## Prerequisites
Before starting, ensure you have:
- An OpenAI API key
- A Pinecone API key (sign up at [pinecone.io](https://pinecone.io))
- Python 3.8+ installed

## Step 1: Environment Setup

First, install the required packages:

```bash
pip install pinecone-client python-dotenv PyPDF2 pdf2image openai pandas tqdm requests
```

Create a `.env` file in your project directory with your API keys:

```bash
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

## Step 2: Initialize Pinecone Vector Store

We'll set up a Pinecone index to store our document embeddings. We'll use the `text-embedding-3-large` model which produces 3072-dimensional embeddings.

```python
import os
import time
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key)

# Create a serverless index
index_name = "vision-rag-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,  # Matches text-embedding-3-large
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

print(f"Index '{index_name}' is ready!")
```

## Step 3: Parse PDF and Extract Visual Information

We'll use the World Bank's "A Better Bank for a Better World: Annual Report 2024" as our example document, which contains a mix of text, images, and tables.

```python
import base64
import requests
import os
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_bytes
from io import BytesIO
from openai import OpenAI
from tqdm import tqdm

# Initialize OpenAI client
oai_client = OpenAI()

# Document URL
document_url = "https://documents1.worldbank.org/curated/en/099101824180532047/pdf/BOSIB13bdde89d07f1b3711dd8e86adb477.pdf"
```

### 3.1 Chunk the PDF into Individual Pages

```python
def chunk_document(document_url):
    """Download PDF and split into individual page chunks."""
    response = requests.get(document_url)
    pdf_data = response.content
    pdf_reader = PdfReader(BytesIO(pdf_data))
    page_chunks = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        pdf_writer = PdfWriter()
        pdf_writer.add_page(page)
        pdf_bytes_io = BytesIO()
        pdf_writer.write(pdf_bytes_io)
        pdf_bytes_io.seek(0)
        pdf_bytes = pdf_bytes_io.read()
        
        page_chunks.append({
            'pageNumber': page_number,
            'pdfBytes': pdf_bytes
        })

    return page_chunks
```

### 3.2 Convert PDF Pages to Images

```python
def convert_page_to_image(pdf_bytes, page_number):
    """Convert a PDF page to an image file."""
    images = convert_from_bytes(pdf_bytes)
    image = images[0]  # Single page
    
    # Create images directory
    images_dir = 'images'
    os.makedirs(images_dir, exist_ok=True)
    
    # Save image
    image_file_name = f"page_{page_number}.png"
    image_file_path = os.path.join(images_dir, image_file_name)
    image.save(image_file_path, 'PNG')
    
    return image_file_path
```

### 3.3 Extract Text Using GPT-4o Vision

```python
def encode_image(image_path):
    """Encode image to base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_vision_response(prompt, image_path):
    """Send image to GPT-4o Vision for analysis."""
    base64_image = encode_image(image_path)
    
    response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content
```

### 3.4 Process the Entire Document

```python
def process_document(document_url):
    """Orchestrate the complete document processing pipeline."""
    print("Document processing started")
    
    # Get per-page chunks
    page_chunks = chunk_document(document_url)
    page_data_list = []
    
    # Process each page with progress bar
    for page_chunk in tqdm(page_chunks, desc='Processing Pages'):
        page_number = page_chunk['pageNumber']
        pdf_bytes = page_chunk['pdfBytes']
        
        # Convert page to image
        image_path = convert_page_to_image(pdf_bytes, page_number)
        
        # Prepare vision prompt
        system_prompt = (
            "The user will provide you an image of a document file. Perform the following actions: "
            "1. Transcribe the text on the page. **TRANSCRIPTION OF THE TEXT:**"
            "2. If there is a chart, describe the image and include the text **DESCRIPTION OF THE IMAGE OR CHART**"
            "3. If there is a table, transcribe the table and include the text **TRANSCRIPTION OF THE TABLE**"
        )
        
        # Get vision API response
        text = get_vision_response(system_prompt, image_path)
        
        # Collect page data
        page_data_list.append({
            'PageNumber': page_number,
            'ImagePath': image_path,
            'PageText': text
        })
    
    # Create DataFrame
    pdf_df = pd.DataFrame(page_data_list)
    print("Document processing completed.")
    
    return pdf_df

# Process the document
df = process_document(document_url)
```

## Step 4: Generate Embeddings and Flag Visual Content

Now we'll generate embeddings for each page and flag pages containing visual content.

```python
# Add flag for visual content
df['Visual_Input_Processed'] = df['PageText'].apply(
    lambda x: 'Y' if 'DESCRIPTION OF THE IMAGE OR CHART' in x or 'TRANSCRIPTION OF THE TABLE' in x else 'N'
)

def get_embedding(text_input):
    """Generate embeddings using OpenAI's embedding model."""
    response = oai_client.embeddings.create(
        input=text_input,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# Generate embeddings with progress bar
embeddings = []
for text in tqdm(df['PageText'], desc='Generating Embeddings'):
    embedding = get_embedding(text)
    embeddings.append(embedding)

# Add embeddings to DataFrame
df['Embeddings'] = embeddings

# Verify visual content flagging
print(f"Pages with visual content: {len(df[df['Visual_Input_Processed'] == 'Y'])}")
```

## Step 5: Upload Embeddings to Pinecone

We'll now upload our embeddings to Pinecone with associated metadata.

```python
# Prepare metadata for each page
def prepare_metadata(row):
    """Prepare metadata dictionary for Pinecone upload."""
    return {
        'pageNumber': str(row['PageNumber']),
        'text': row['PageText'][:1000],  # First 1000 chars
        'imagePath': row['ImagePath'],
        'graphicIncluded': row['Visual_Input_Processed']
    }

# Connect to our index
index = pc.Index(index_name)

# Upload embeddings in batches
batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    
    vectors = []
    for idx, row in batch.iterrows():
        vectors.append({
            'id': f"page_{row['PageNumber']}",
            'values': row['Embeddings'],
            'metadata': prepare_metadata(row)
        })
    
    # Upsert to Pinecone
    index.upsert(vectors=vectors)
    print(f"Uploaded batch {i//batch_size + 1}")

print("All embeddings uploaded to Pinecone!")
```

## Step 6: Perform Semantic Search with Visual Enhancement

Now we can query our document using semantic search. When relevant pages contain visual content, we can optionally pass the image to GPT-4o Vision for enhanced understanding.

```python
def semantic_search(query, top_k=3):
    """Perform semantic search and return relevant pages."""
    # Generate query embedding
    query_embedding = get_embedding(query)
    
    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results.matches

def answer_with_visual_context(query, search_results):
    """Generate answer using text and optional visual context."""
    context_parts = []
    
    for match in search_results:
        metadata = match.metadata
        context_text = f"Page {metadata['pageNumber']}: {metadata['text']}"
        context_parts.append(context_text)
        
        # If page has graphics, note this for potential visual enhancement
        if metadata['graphicIncluded'] == 'Y':
            context_parts.append(f"[Page {metadata['pageNumber']} contains visual content]")
    
    # Combine context
    full_context = "\n\n".join(context_parts)
    
    # Generate response
    response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering questions based on the provided document context."},
            {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {query}\n\nAnswer:"}
        ]
    )
    
    return response.choices[0].message.content

# Example query
query = "What sectors received funding in the Middle East and North Africa region?"
search_results = semantic_search(query)
answer = answer_with_visual_context(query, search_results)
print(f"Question: {query}")
print(f"Answer: {answer}")
```

## Step 7: Optional Visual Enhancement for Complex Queries

For queries specifically about visual content, we can pass the actual images to GPT-4o Vision:

```python
def answer_with_vision(query, page_number):
    """Answer query using the actual page image."""
    # Get image path from DataFrame
    image_path = df[df['PageNumber'] == page_number]['ImagePath'].iloc[0]
    
    # Create vision prompt
    vision_prompt = f"""
    Based on this document page image, answer the following question:
    {query}
    
    Please describe any relevant charts, tables, or images in your answer.
    """
    
    # Get vision response
    response = get_vision_response(vision_prompt, image_path)
    return response

# Example: Query about a specific visual element
visual_query = "Describe the pie chart on page 21"
vision_answer = answer_with_vision(visual_query, 21)
print(f"Visual-enhanced answer: {vision_answer}")
```

## Conclusion

You've successfully implemented a RAG system enhanced with GPT-4o Vision capabilities. This approach allows you to:

1. Process documents with mixed text and visual content
2. Flag pages containing images, charts, or tables
3. Perform semantic search across both textual and visual information
4. Enhance responses with visual context when needed

**Key Considerations:**
- Vision modality increases latency and cost—use selectively
- Complex visuals like engineering drawings may lose detail in translation
- Always evaluate whether visual enhancement improves your specific use case

This system provides a robust foundation for building AI solutions that can understand and reason about documents with complex visual elements, delivering richer and more accurate information to users.
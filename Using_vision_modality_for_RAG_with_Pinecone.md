## Optimizing Retrieval-Augmented Generation using GPT-4o Vision Modality

Implementing Retrieval-Augmented Generation (RAG) presents unique challenges when working with documents rich in images, graphics and tables. Traditional RAG models excel with textual data but often falter when visual elements play a crucial role in conveying information. In this cookbook, we bridge that gap by leveraging the vision modality to extract and interpret visual content, ensuring that the generated responses are as informative and accurate as possible.

Our approach involves parsing documents into images and utilizing metadata tagging to identify pages containing images, graphics and tables. When a semantic search retrieves such a page, we pass the page image to a vision model instead of relying solely on text. This method enhances the model's ability to understand and answer user queries that pertain to visual data.

In this cookbook, we will explore and demonstrate the following key concepts:

##### 1. Setting Up a Vector Store with Pinecone:
- Learn how to initialize and configure Pinecone to store vector embeddings efficiently.

##### 2. Parsing PDFs and Extracting Visual Information:
- Discover techniques for converting PDF pages into images.
- Use GPT-4o vision modality to extract textual information from pages with images, graphics or tables.  

##### 3. Generating Embeddings: 
- Utilize embedding models to create vector representations of textual data. 
- Flag the pages that have visual content so that we set a metadata flag on vector store, and retrieve images to pass on the GPT-4o using vision modality. 

##### 4. Uploading Embeddings to Pinecone: 
- Upload these embeddings to Pinecone for storage and retrieval. 

##### 5. Performing Semantic Search for Relevant Pages:
- Implement semantic search on page text to find pages that best match the user's query. 
- Provide the matching page text to GPT-4o as context to answer user's query.  

##### 6. Handling Pages with Visual Content (Optional Step):
- Learn how to pass the image using GPT-4o vision modality for question answering with additional context. 
- Understand how this process improves the accuracy of responses involving visual data.
 
By the end of this cookbook, you will have a robust understanding of how to implement RAG systems capable of processing and interpreting documents with complex visual elements. This knowledge will empower you to build AI solutions that deliver richer, more accurate information, enhancing user satisfaction and engagement.

We will use the World Bank report - [A Better Bank for a Better World: Annual Report 2024](https://documents1.worldbank.org/curated/en/099101824180532047/pdf/BOSIB13bdde89d07f1b3711dd8e86adb477.pdf) to illustrate the concepts as this document has a mix of images, tables and graphics data. 

Keep in mind that using the Vision Modality is resource-intensive, leading to increased latency and cost. It is advisable to use Vision Modality only for cases where performance on evaluation benchmarks is unsatisfactory with plain text extraction methods. With this context, let's dive in.

### Step 1: Setting up a Vector Store with Pinecone 
In this section, we'll set up a vector store using Pinecone to store and manage our embeddings efficiently. Pinecone is a vector database optimized for handling high-dimensional vector data, which is essential for tasks like semantic search and similarity matching.

**Prerequisites** 
1. Sign-up for Pinecone and obtain an API key by following the instructions here [Pinecone Database Quickstart](https://docs.pinecone.io/guides/get-started/quickstart)  
2. Install the Pinecone SDK using `pip install "pinecone[grpc]"`. gRPC (gRPC Remote Procedure Call) is a high-performance, open-source universal RPC framework that uses HTTP/2 for transport, Protocol Buffers (protobuf) as the interface definition language, and enables client-server communication in a distributed system. It is designed to make inter-service communication more efficient and suitable for microservices architectures.   

**Store the API Key Securely**  
1. Store the API key in an .env file for security purposes in you project directory as follows:  
 `PINECONE_API_KEY=your-api-key-here`.   
 2. Install `pip install python-dotenv` to read the API Key from the .env file. 

**Create the Pinecone Index**   
We'll use the `create_index` function to initialize our embeddings database on Pinecone. There are two crucial parameters to consider:

1. Dimension: This must match the dimensionality of the embeddings produced by your chosen model. For example, OpenAI's text-embedding-ada-002 model produces embeddings with 1536 dimensions, while text-embedding-3-large produces embeddings with 3072 dimensions. We'll use the text-embedding-3-large model, so we'll set the dimension to 3072.

2. Metric: The distance metric determines how similarity is calculated between vectors. Pinecone supports several metrics, including cosine, dotproduct, and euclidean. For this cookbook, we'll use the cosine similarity metric. You can learn more about distance metrics in the [Pinecone Distance Metrics documentation](https://docs.pinecone.io/guides/indexes/understanding-indexes#distance-metrics).


```python
import os
import time
# Import the Pinecone library
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key)

# Create a serverless index
index_name = "my-test-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)
```

Navigate to Indexes list on [Pinecone](https://app.pinecone.io/) and you should be able to view `my-test-index` in the list of indexes. 

### Step 2: Parsing PDFs and Extracting Visual Information:

In this section, we will parse our PDF document the World Bank report - [A Better Bank for a Better World: Annual Report 2024](https://documents1.worldbank.org/curated/en/099101824180532047/pdf/BOSIB13bdde89d07f1b3711dd8e86adb477.pdf) and extract textual and visual information, such as describing images, graphics, and tables. The process involves three main steps:

1. **Parse the PDF into individual pages:** We split the PDF into separate pages for easier processing.
2. **Convert PDF pages to images:** This enabled vision GPT-4o vision capability to analyze the page as an image.  
3. **Process images and tables:** Provide instructions to GPT-4o to extract text, and also describe the images, graphics or tables in the document. 

**Prerequisites**

Before proceeding, make sure you have the following packages installed. Also ensure your OpenAI API key is set up as an environment variable. You may also need to install Poppler for PDF rendering.  

`pip install PyPDF2 pdf2image pytesseract pandas tqdm`
 
**Step Breakdown:**

**1. Downloading and Chunking the PDF:**  
- The `chunk_document` function downloads the PDF from the provided URL and splits it into individual pages using PyPDF2.
- Each page is stored as a separate PDF byte stream in a list.

**2. Converting PDF Pages to Images:** 
- The `convert_page_to_image` function takes the PDF bytes of a single page and converts it into an image using pdf2image.
- The image is saved locally in an 'images' directory for further processing.

**3. Extracting Text Using GPT-4o vision modality:**
- The `extract_text_from_image` function uses GPT-4o vision capability to extract text from the image of the page.
- This method can extract textual information even from scanned documents.
- Note that this modality is resource intensive thus has higher latency and cost associated with it. 

**4. Processing the Entire Document:** 
- The process_document function orchestrates the processing of each page.
- It uses a progress bar (tqdm) to show the processing status.
- The extracted information from each page is collected into a list and then converted into a Pandas DataFrame.


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

# Link to the document we will use as the example 
document_to_parse = "https://documents1.worldbank.org/curated/en/099101824180532047/pdf/BOSIB13bdde89d07f1b3711dd8e86adb477.pdf"

# OpenAI client 
oai_client = OpenAI()


# Chunk the PDF document into single page chunks 
def chunk_document(document_url):
    # Download the PDF document
    response = requests.get(document_url)
    pdf_data = response.content

    # Read the PDF data using PyPDF2
    pdf_reader = PdfReader(BytesIO(pdf_data))
    page_chunks = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        pdf_writer = PdfWriter()
        pdf_writer.add_page(page)
        pdf_bytes_io = BytesIO()
        pdf_writer.write(pdf_bytes_io)
        pdf_bytes_io.seek(0)
        pdf_bytes = pdf_bytes_io.read()
        page_chunk = {
            'pageNumber': page_number,
            'pdfBytes': pdf_bytes
        }
        page_chunks.append(page_chunk)

    return page_chunks


# Function to encode the image
def encode_image(local_image_path):
    with open(local_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Function to convert page to image     
def convert_page_to_image(pdf_bytes, page_number):
    # Convert the PDF page to an image
    images = convert_from_bytes(pdf_bytes)
    image = images[0]  # There should be only one page

    # Define the directory to save images (relative to your script)
    images_dir = 'images'  # Use relative path here

    # Ensure the directory exists
    os.makedirs(images_dir, exist_ok=True)

    # Save the image to the images directory
    image_file_name = f"page_{page_number}.png"
    image_file_path = os.path.join(images_dir, image_file_name)
    image.save(image_file_path, 'PNG')

    # Return the relative image path
    return image_file_path


# Pass the image to the LLM for interpretation  
def get_vision_response(prompt, image_path):
    # Getting the base64 string
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
    return response


# Process document function that brings it all together 
def process_document(document_url):
    try:
        # Update document status to 'Processing'
        print("Document processing started")

        # Get per-page chunks
        page_chunks = chunk_document(document_url)
        total_pages = len(page_chunks)

        # Prepare a list to collect page data
        page_data_list = []

        # Add progress bar here
        for page_chunk in tqdm(page_chunks, total=total_pages, desc='Processing Pages'):
            page_number = page_chunk['pageNumber']
            pdf_bytes = page_chunk['pdfBytes']

            # Convert page to image
            image_path = convert_page_to_image(pdf_bytes, page_number)

            # Prepare question for vision API
            system_prompt = (
                "The user will provide you an image of a document file. Perform the following actions: "
                "1. Transcribe the text on the page. **TRANSCRIPTION OF THE TEXT:**"
                "2. If there is a chart, describe the image and include the text **DESCRIPTION OF THE IMAGE OR CHART**"
                "3. If there is a table, transcribe the table and include the text **TRANSCRIPTION OF THE TABLE**"
            )

            # Get vision API response
            vision_response = get_vision_response(system_prompt, image_path)

            # Extract text from vision response
            text = vision_response.choices[0].message.content

            # Collect page data
            page_data = {
                'PageNumber': page_number,
                'ImagePath': image_path,
                'PageText': text
            }
            page_data_list.append(page_data)

        # Create DataFrame from page data
        pdf_df = pd.DataFrame(page_data_list)
        print("Document processing completed.")
        print("DataFrame created with page data.")

        # Return the DataFrame
        return pdf_df

    except Exception as err:
        print(f"Error processing document: {err}")
        # Update document status to 'Error'


df = process_document(document_to_parse)
```

    Document processing started
    Processing Pages: 100%|██████████| 49/49 [18:54<00:00, 23.14s/it]
    Document processing completed.
    DataFrame created with page data.

Let's examine the DataFrame to ensure that the pages have been processed correctly. For brevity, we will retrieve and display only the first five rows. Additionally, you should be able to see the page images generated in the 'images' directory.


```python
from IPython.display import display, HTML

# Convert the DataFrame to an HTML table and display top 5 rows 
display(HTML(df.head().to_html()))
```

Let's take a look at a sample page, such as page 21, which contains embedded graphics and text. We can observe that the vision modality effectively extracted and described the visual information. For instance, the pie chart on this page is accurately described as:

`"FIGURE 6: MIDDLE EAST AND NORTH AFRICA IBRD AND IDA LENDING BY SECTOR - FISCAL 2024 SHARE OF TOTAL OF $4.6 BILLION" is a circular chart, resembling a pie chart, illustrating the percentage distribution of funds among different sectors. The sectors include:`


```python
# Filter and print rows where pageNumber is 21
filtered_rows = df[df['PageNumber'] == 21]
for text in filtered_rows.PageText:
    print(text)
```

### Step 3: Generating Embeddings: 

In this section, we focus on transforming the textual content extracted from each page of the document into vector embeddings. These embeddings capture the semantic meaning of the text, enabling efficient similarity searches and various Natural Language Processing (NLP) tasks. We also identify pages containing visual elements, such as images, graphics, or tables, and flag them for special handling.

**Step Breakdown:**

**1. Adding a flag for visual content**   
  
To process pages containing visual information, in Step 2 we used the vision modality to extract content from charts, tables, and images. By including specific instructions in our prompt, we ensure that the model adds markers such as `DESCRIPTION OF THE IMAGE OR CHART` or `TRANSCRIPTION OF THE TABLE` when describing visual content. In this step, if such a marker is detected, we set the Visual_Input_Processed flag to 'Y'; otherwise, it remains 'N'.

While the vision modality captures most visual information effectively, some details—particularly in complex visuals like engineering drawings—may be lost in translation. In Step 6, we will use this flag to determine when to pass the image of the page to GPT-4 Vision as additional context. This is an optional enhancement that can significantly improve the effectiveness of a RAG solution.  

**2. Generating Embeddings with OpenAI's Embedding Model**  

We use OpenAI's embedding model, `text-embedding-3-large`, to generate high-dimensional embeddings that represent the semantic content of each page. 

Note: It is crucial to ensure that the dimensions of the embedding model you use are consistent with the configuration of your Pinecone vector store. In our case, we set up the Pinecone database with 3072 dimensions to match the default dimensions of `text-embedding-3-large`.  



```python
# Add a column to flag pages with visual content
df['Visual_Input_Processed'] = df['PageText'].apply(
    lambda x: 'Y' if 'DESCRIPTION OF THE IMAGE OR CHART' in x or 'TRANSCRIPTION OF THE TABLE' in x else 'N'
)


# Function to get embeddings
def get_embedding(text_input):
    response = oai_client.embeddings.create(
        input=text_input,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding


# Generate embeddings with a progress bar
embeddings = []
for text in tqdm(df['PageText'], desc='Generating Embeddings'):
    embedding = get_embedding(text)
    embeddings.append(embedding)

# Add the embeddings to the DataFrame
df['Embeddings'] = embeddings
```

    Generating Embeddings: 100%|██████████| 49/49 [00:18<00:00,  2.61it/s]

We can verify that our logic correctly flagged pages requiring visual input. For instance, page 21, which we previously examined, has the Visual_Input_Needed flag set to "Y".


```python
# Display the flag for page 21 
filtered_rows = df[df['PageNumber'] == 21]
print(filtered_rows.Visual_Input_Processed)
```

#### Step 4: Uploading embeddings to Pinecone: 

In this section, we will upload the embeddings we've generated for each page of our document to Pinecone. Along with the embeddings, we'll include relevant metadata tags that describe each page, such as the page number, text content, image paths, and whether the page includes graphics. 

**Step Breakdown:**  

**1. Create Metadata Fields:**  
Metadata enhances our ability to perform more granular searches, find the text or image associated with the vector, and enables filtering within the vector database.
* pageId: Combines the document_id and pageNumber to create a unique identifier for each page. We will use this as a unique identifier for our embeddings. 
* pageNumber: The numerical page number within the document.
* text: The extracted text content from the page.
* ImagePath: The file path to the image associated with the page.
* GraphicIncluded
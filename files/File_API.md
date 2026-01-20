

# Gemini API: File API Quickstart

The Gemini API supports prompting with text, image, and audio data, also known as *multimodal* prompting. You can include text, image,
and audio in your prompts. For small images, you can point the Gemini model
directly to a local file when providing a prompt. For larger text files, images, videos, and audio, upload the files with the [File
API](https://ai.google.dev/api/rest/v1beta/files) before including them in
prompts.

The File API lets you store up to 20GB of files per project, with each file not
exceeding 2GB in size. Files are stored for 48 hours and can be accessed with
your API key for generation within that time period. It is available at no cost in all regions where the [Gemini API is
available](https://ai.google.dev/available_regions).

For information on valid file formats (MIME types) and supported models, see the documentation on
[supported file formats](https://ai.google.dev/tutorials/prompting_with_media#supported_file_formats)
and view the text examples at the end of this guide.

This guide shows how to use the File API to upload a media file and include it in a `GenerateContent` call to the Gemini API. For more information, see the [code
samples](../quickstarts/file-api).


### Install dependencies


```
%pip install -q -U "google-genai>=1.57.0"
```

### Authentication

**Important:** The File API uses API keys for authentication and access. Uploaded files are associated with the API key's cloud project. Unlike other Gemini APIs that use API keys, your API key also grants access data you've uploaded to the File API, so take extra care in keeping your API key secure. For best practices on securing API keys, refer to Google's [documentation](https://support.google.com/googleapi/answer/6310037).

#### Set up your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google import genai
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Upload file

The File API lets you upload a variety of multimodal MIME types, including images and audio formats. The File API handles inputs that can be used to generate content with [`model.generateContent`](https://ai.google.dev/api/rest/v1/models/generateContent) or [`model.streamGenerateContent`](https://ai.google.dev/api/rest/v1/models/streamGenerateContent).

The File API accepts files under 2GB in size and can store up to 20GB of files per project. Files last for 2 days and cannot be downloaded from the API.

First, you will prepare a sample image to upload to the API.

Note: You can also [upload your own files](../examples/Upload_files_to_Colab.ipynb) to use.


```
from IPython.display import Image

!curl -o image.jpg "https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg"
Image(filename="image.jpg")
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  349k  100  349k    0     0  1071k      0 --:--:-- --:--:-- --:--:-- 1071k



Next, you will upload that file to the File API.


```
sample_file = client.files.upload(file="image.jpg")

print(f"Uploaded file '{sample_file.name}' as: {sample_file.uri}")
```

    Uploaded file 'files/5djqlsrlsmw7' as: https://generativelanguage.googleapis.com/v1beta/files/5djqlsrlsmw7


The `response` shows that the File API stored the specified `display_name` for the uploaded file and a `uri` to reference the file in Gemini API calls. Use `response` to track how uploaded files are mapped to URIs.

Depending on your use cases, you could store the URIs in structures such as a `dict` or a database.

## Get file

After uploading the file, you can verify the API has successfully received the files by calling `files.get`.

It lets you get the file metadata that have been uploaded to the File API that are associated with the Cloud project your API key belongs to. Only the `name` (and by extension, the `uri`) are unique. Only use the `displayName` to identify files if you manage uniqueness yourself.


```
file = client.files.get(name=sample_file.name)
print(f"Retrieved file '{file.name}' as: {sample_file.uri}")
```

    Retrieved file 'files/5djqlsrlsmw7' as: https://generativelanguage.googleapis.com/v1beta/files/5djqlsrlsmw7


## Generate content

After uploading the file, you can make `GenerateContent` requests that reference the file by providing the URI. In the Python SDK you can pass the returned object directly.

Here you create a prompt that starts with text and includes the uploaded image.


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=["Describe the image with a creative description.", sample_file]
)

print(response.text)
```

    On the humble canvas of lined notebook paper, rendered in a confident blue ink, lies the nascent blueprint of tomorrow's commute: the "JETPACK BACKPACK."
    
    This isn't just a carry-all; it's a doodle of defiance, a charmingly analog vision of a future where mundane burdens meet sky-high ambitions. The backpack itself appears sleek and unassuming, looking like a "normal backpack" and promising to be "lightweight," yet engineered for comfort with "padded strap support." Practicality isn't overlooked, as it boasts the capacity to fit a hefty "18" laptop," ready for the urban professional or an adventurous student.
    
    But below its practical form, the real magic ignites. Twin "retractable boosters" extend, exhaling elegant, swirling plumes of "steam-powered" thrust, a visual promise of "green/clean" liberation. It’s a whimsical dance of vapor against gravity, implying a gentle, eco-conscious lift. While the concept embraces modern convenience with "USB-C charging," the inventor's honest hand reveals a humorous reality: a "15-min battery life," perfect for those urgent skyward dashes across town, or perhaps just a very ambitious coffee run.
    
    This image is more than a sketch; it's a delightful blend of sci-fi aspiration and everyday functionality, hand-drawn with the earnest enthusiasm of an idea taking flight, one ink stroke at a time.


## Multiple files

You'll often want to upload multiple files at once. Here's a quick example how you can do it:


```
!git clone -q --depth 1 https://github.com/googleapis/python-genai
```


```
import pathlib

files = []
for p in pathlib.Path("python-genai").rglob('*.py'):
  if 'test' in str(p):
    continue
  f = client.files.upload(file=p, config={'display_name': str(p)})
  # The API doesn't see the file name, so add those to the list of parts.
  files.append(f"<<<File: {str(p)}>>>")
  files.append(f)
  print('.', end='')
```

    [., ..., .]


```
client.models.generate_content(
    model=MODEL_ID,
    contents=["Hi, could you give me a summary of this code base?"] + files
)
```




    GenerateContentResponse(
      automatic_function_calling_history=[],
      candidates=[
        Candidate(
          content=Content(
            parts=[
              Part(
                text="""This Python codebase provides the **Google Generative AI SDK**, designed to allow developers to interact with both the **Google Gemini API** and **Google Cloud's Vertex AI platform**. It offers a comprehensive set of tools for building generative AI applications, abstracting away the complexities of the underlying APIs.
    
    Here's a summary of its core components and functionalities:
    
    1.  **Client Entry Point (`client.py`)**:
        *   The `Client` class is the main entry point for synchronous interactions, and `AsyncClient` (accessed via `client.aio`) for asynchronous operations.
        *   Handles authentication (API keys, Google Cloud Application Default Credentials) and configuration (project ID, location, HTTP options), supporting both Gemini API and Vertex AI.
        *   Integrates a `ReplayApiClient` for testing, allowing API interactions to be recorded and replayed.
    
    2.  **Core AI Model Interactions (`models.py`)**:
        *   Provides methods for generating text and multimodal content (images, videos), including streaming responses.
        *   Supports generating embeddings for text.
        *   Offers functionalities for image generation from text, image editing (inpaint, outpaint, style, upscale, recontextualize), and video generation (text-to-video, image-to-video, video extension).
        *   Includes model management capabilities to list, get, update, and delete models.
        *   Features **Automatic Function Calling (AFC)**, allowing models to invoke user-defined Python functions or external tools.
    
    3.  **Conversational AI (`chats.py`, `live.py`, `live_music.py`)**:
        *   **Chat Sessions (`chats.py`)**: Manages multi-turn conversations by maintaining a history of valid (curated) and all (comprehensive) turns, facilitating natural chat flows.
        *   **Live API (`live.py`)**: (Preview) Enables real-time, low-latency interactions with models via WebSockets, supporting various input types (text, audio, video) and tool responses.
        *   **Live Music (`live_music.py`)**: (Experimental) Offers real-time music generation capabilities through WebSockets, with controls for prompts, configuration, and playback.
    
    4.  **Data and Resource Management (`files.py`, `file_search_stores.py`, `documents.py`, `caches.py`, `batches.py`)**:
        *   **File Management (`files.py`)**: Provides tools to upload, retrieve, list, and delete files from the GenAI file service (Gemini API only), including resumable uploads and downloading generated files.
        *   **File Search Stores (`file_search_stores.py`, `documents.py`)**: Manages collections of documents (`FileSearchStore`) and individual documents (`Document`) for Retrieval-Augmented Generation (RAG) purposes (Gemini API only).
        *   **Cached Content (`caches.py`)**: Manages `CachedContent` resources for LLM queries, allowing users to explicitly cache prompts and configurations to improve performance.
        *   **Batch Processing (`batches.py`)**: Handles asynchronous batch jobs for content generation or embeddings, supporting input/output from cloud storage (GCS, BigQuery) and inlined requests.
    
    5.  **Model Customization and Operations (`tunings.py`, `operations.py`)**:
        *   **Model Tuning (`tunings.py`)**: (Experimental) Manages fine-tuning jobs for generative models, supporting different tuning methods (supervised fine-tuning, preference optimization), dataset configurations, and evaluation settings.
        *   **Long-Running Operations (`operations.py`)**: Provides an interface to monitor and retrieve the status of asynchronous long-running operations initiated by certain API calls.
    
    6.  **Underlying Utilities**:
        *   **Types (`types.py`)**: Defines all Pydantic models (data structures), enums, and utility classes that structure API requests and responses, ensuring type safety.
        *   **Transformers (`_transformers.py`, `_operations_converters.py`, `_live_converters.py`, `_tokens_converters.py`)**: A comprehensive set of functions to convert data between SDK Python objects and the specific JSON formats required by the underlying APIs, handling differences between Gemini API and Vertex AI.
        *   **Error Handling (`errors.py`)**: Defines custom exception classes for various API errors, along with utilities to parse HTTP responses and raise appropriate exceptions.
        *   **Pagination (`pagers.py`)**: Provides utility classes for iterating through paginated results from list-style API calls.
        *   **Local Tokenization (`local_tokenizer.py`, `_local_tokenizer_loader.py`)**: (Experimental) Offers client-side text-only token counting for supported models using SentencePiece tokenizers.
        *   **Tool Adapters (`_adapters.py`, `_mcp_utils.py`)**: Facilitates integration with external tools, such as MCP (Multimodal Conversational Platform) tools, by adapting their definitions to the SDK's function calling mechanism.
    
    In essence, this SDK provides a robust and flexible Python interface for interacting with Google's generative AI models and services, supporting a wide range of use cases from simple content generation to complex real-time applications and model customization."""
              ),
            ],
            role='model'
          ),
          finish_reason=<FinishReason.STOP: 'STOP'>,
          index=0
        ),
      ],
      model_version='gemini-2.5-flash',
      response_id='z9oaaevcE-ugqtsPncnYmAg',
      sdk_http_response=HttpResponse(
        headers=<dict len=10>
      ),
      usage_metadata=GenerateContentResponseUsageMetadata(
        candidates_token_count=1131,
        prompt_token_count=439459,
        prompt_tokens_details=[
          ModalityTokenCount(
            modality=<MediaModality.TEXT: 'TEXT'>,
            token_count=439459
          ),
        ],
        thoughts_token_count=6028,
        total_token_count=446618
      )
    )



## Use files from Google Cloud Storage

The Gemini API supports accessing objects stored in Google Cloud Storage (GCS). As GCS objects have a different security model to the Gemini API, you first need to register files before you can use them.

To register a GCS object for use in the File API, you need to authenticate with an identity that has **Storage Object Viewer** permissions, and with the appropriate OAuth scope enabled. The mechanism varies depending on the environment, for example you can [download service account credentials](https://docs.cloud.google.com/iam/docs/keys-create-delete) to use on your own infrastructure, or run on a Compute Engine instance that has [a configured service account](https://docs.cloud.google.com/compute/docs/access/create-enable-service-accounts-for-instances).

For this notebook, you will configure your user credentials via `gcloud`.


```
# Use the same project ID you're using for the Gemini API project
PROJECT_ID = "your-project-id"  # @param {type: "string"}

!gcloud config set project {PROJECT_ID}
!gcloud auth application-default login --no-launch-browser --scopes="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.read_only"
```

Now retrieve your credentials and register your object(s).

Note: If you are registering files separately from your generation code, you will need to create a `genai.Client` object initialised with an API key. The API key is unused by this method, but is still required to ensure the client is not connected to the Vertex AI backend.

Upload files to a bucket in the project you're using: https://docs.cloud.google.com/storage/docs/uploading-objects. Then get the URIs and register the files:


```
import google.auth

credentials, project_id = google.auth.default()

registered_gcs_files = client.files.register_files(
    auth=credentials,
    uris=["gs://your-bucket/some-file.pdf"]
)
```

The `register_files` endpoint returns the files associated with the supplied URIs. You can then use these directly in your prompt with `generate_content`, or store the `name`s to use at a later time.


```
response = client.models.generate_content(
    model=MODEL_ID,
    contents=
        ["What are these documents about?"] + 
        registered_gcs_files.files
)

print(response.text)
```

## Delete files

Files are automatically deleted after 2 days or you can manually delete them using `files.delete()`.


```
client.files.delete(name=sample_file.name)
print(f"Deleted {sample_file.name}.")
```

    Deleted files/5djqlsrlsmw7.


## Supported text types

As well as supporting media uploads, the File API can be used to embed text files, such as Python code, or Markdown files, into your prompts.

This example shows you how to load a markdown file into a prompt using the File API.


```
# Download a markdown file and ask a question.
from google.genai import types
from IPython.display import Markdown

!curl -so contrib.md https://raw.githubusercontent.com/google-gemini/cookbook/main/CONTRIBUTING.md

md_file = client.files.upload(
    file="contrib.md",
    config={
        "display_name": "CONTRIBUTING.md",
        "mime_type": "text/markdown"
    }
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "What should I do before I start writing, when following these guidelines?",
        md_file,
    ]
)

display(Markdown(response.text))
```


Based on the "Contributing to the Gemini API Cookbook" guidelines, here's what you should do before you start writing:

1.  **Sign the Contributor License Agreement (CLA):** All contributions require you (or your employer) to sign a CLA. Visit [https://cla.developers.google.com/](https://cla.developers.google.com/) to check your status or sign a new one. This gives Google permission to use your contributions.
2.  **File an Issue:** **
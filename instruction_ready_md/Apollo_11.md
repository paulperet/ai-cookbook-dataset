# Prompting Gemini with a Text File: Analyzing the Apollo 11 Transcript

This guide demonstrates how to use the Gemini API to analyze content from a text file. You will upload a transcript from the Apollo 11 mission and prompt the model to extract specific information from it.

## Prerequisites

Before you begin, ensure you have the following:

1.  A Google AI API key.
2.  The API key stored in a Colab Secret named `GOOGLE_API_KEY`. If you need help setting this up, refer to the [Authentication guide](https://ai.google.dev/gemini-api/docs/auth).

## Step 1: Install and Import Required Libraries

Start by installing the Google Generative AI Python SDK and importing the necessary modules.

```bash
pip install -U -q "google-genai>=1.0.0"
```

```python
from google.colab import userdata
from google import genai
```

## Step 2: Configure the API Client

Initialize the Gemini client using your API key.

```python
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 3: Download the Transcript File

You will use a 400-page transcript from the Apollo 11 mission. Download it to your working directory.

```bash
wget https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

## Step 4: Upload the File via the File API

To easily reference the file in your API calls, upload it using the Gemini File API. This provides a URI you can pass directly to the model.

```python
text_file_name = "a11.txt"
print(f"Uploading file...")
text_file = client.files.upload(file=text_file_name)
print(f"Completed upload: {text_file.uri}")
```

## Step 5: Generate Content with a Prompt

Now you can ask the model to analyze the content of the uploaded file. In this example, you will prompt Gemini to find lighthearted moments in the transcript.

First, define your prompt and choose a model.

```python
prompt = "Find four lighthearted moments in this text file."
MODEL_ID = "gemini-3-flash-preview"  # You can change this to another supported model
```

Next, make the API call. Pass both the prompt and the file object in the `contents` parameter. The `config` dictionary is used to set a longer timeout for processing large files.

```python
response = client.models.generate_content(
    model=f"models/{MODEL_ID}",
    contents=[prompt, text_file],
    config={"httpOptions": {"timeout": 600}}
)

print(response.text)
```

The model will process the transcript and return a text response listing the requested lighthearted moments.

## Step 6: Clean Up (Optional)

Files uploaded via the File API are automatically deleted after two days. You can also delete them manually immediately after use.

```python
client.files.delete(name=text_file.name)
```

## Summary

You have successfully:
1.  Installed the Gemini SDK and configured the client.
2.  Downloaded and uploaded a text file using the File API.
3.  Sent a prompt to the Gemini model to analyze the file's content.
4.  (Optionally) cleaned up the uploaded file.

## Further Reading

The File API supports files under 2GB in size, with a storage limit of 20GB per project. For more details on capabilities and best practices, see the [official File API documentation](https://ai.google.dev/gemini-api/docs/file-api).
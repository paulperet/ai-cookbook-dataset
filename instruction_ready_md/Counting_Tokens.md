# Understanding Tokens in the Gemini API: A Practical Guide

Tokens are the fundamental unit of processing for Large Language Models (LLMs). This guide provides a hands-on introduction to what tokens are and how they're used in the Gemini API, including how to count them and understand their impact on your usage.

## What Are Tokens?

LLMs process text at a granularity smaller than words but larger than individual characters. These units are called **tokens**. A token can be:
- A single character (like `z`)
- A whole word (like `the`)
- Part of a longer word (which may be broken into multiple tokens)

For Gemini models, one token is approximately equivalent to 4 characters, or about 60-80 English words per 100 tokens. When billing is enabled, you're charged based on the number of input and output tokens consumed, making token counting an essential skill.

## Prerequisites

### Install the SDK

First, install the Google Generative AI Python SDK:

```bash
pip install -q -U "google-genai>=1.0.0"
```

### Set Up Your API Key

Store your API key in an environment variable or configuration. For Google Colab users, you can store it as a secret:

```python
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### Initialize the Client

Initialize the Gemini client with your API key:

```python
from google import genai

client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 1: Understanding Context Windows

Gemini models have defined context windows measured in tokens. These windows determine:
- How much input you can provide (input token limit)
- How much output the model can generate (output token limit)

You can check a model's context window directly through the API:

```python
MODEL_ID = "gemini-2.5-flash"  # You can change this to any Gemini model

model_info = client.models.get(model=MODEL_ID)
print("Context window:", model_info.input_token_limit, "tokens")
print("Max output window:", model_info.output_token_limit, "tokens")
```

Example output:
```
Context window: 1048576 tokens
Max output window: 65536 tokens
```

Different models have different context windows. For example, Gemini Pro models offer up to 2 million tokens.

## Step 2: Counting Text Tokens

The Gemini API provides a dedicated endpoint for counting tokens in a request before sending it.

### Count Tokens in a Prompt

Use `client.models.count_tokens()` to count tokens in your input:

```python
response = client.models.count_tokens(
    model=MODEL_ID,
    contents="What's the highest mountain in Africa?",
)
print("Prompt tokens:", response.total_tokens)
```

Example output:
```
Prompt tokens: 10
```

### Get Token Counts After Generation

When you generate content, the response includes detailed token usage metadata:

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="The quick brown fox jumps over the lazy dog."
)
print(response.text)
```

Example output:
```
That's a classic!

It's a famous **pangram**, meaning it uses every letter of the alphabet at least once. It's often used for testing typewriters, keyboards, and fonts because it demonstrates all the characters.
```

Now examine the token usage:

```python
print("Prompt tokens:\t ", response.usage_metadata.prompt_token_count)
print("Thinking tokens:", response.usage_metadata.thoughts_token_count)
print("Output tokens:\t ", response.usage_metadata.candidates_token_count)
print("--------------")
print("Total tokens:\t", response.usage_metadata.total_token_count)
```

Example output:
```
Prompt tokens:	  11
Thinking tokens: 751
Output tokens:	  49
--------------
Total tokens:	 811
```

**Note:** When using thinking models (like Gemini 2.5), you'll see separate counts for thinking tokens, which represent the model's internal reasoning process.

## Step 3: Counting Multi-Modal Tokens

All input modalities—including images, audio, and video—are converted to tokens.

### Image Tokens

Images consume a fixed number of tokens regardless of their display or file size. Let's test this with an example image:

```python
import PIL
from IPython.display import display, Image

# Download and display an example image
!curl -L https://goo.gle/instrument-img -o organ.jpg
organ = PIL.Image.open('organ.jpg')
```

Count tokens for the image:

```python
response = client.models.count_tokens(
    model=MODEL_ID,
    contents=[organ]
)
print("Prompt with image tokens:", response.total_tokens)
```

Example output:
```
Prompt with image tokens: 259
```

**Important:** Different images will typically consume the same number of tokens, as the token count is based on the model's processing requirements rather than image properties.

### Using the Files API

You can also upload files through the Files API and count their tokens:

```python
organ_upload = client.files.upload(file='organ.jpg')

response = client.models.count_tokens(
    model=MODEL_ID,
    contents=organ_upload,
)
print("Prompt with image tokens:", response.total_tokens)
```

Example output:
```
Prompt with image tokens: 259
```

### Audio and Video Tokens

Audio and video files are converted to tokens at a fixed rate per second. Let's examine an audio example:

```python
# Download a sample audio file
!curl -q -o sample.mp3 "https://storage.googleapis.com/generativeai-downloads/data/State_of_the_Union_Address_30_January_1961.mp3"

# Check the audio duration
!ffprobe -v error -show_entries format=duration sample.mp3
```

Example output showing duration:
```
[FORMAT]
duration=2610.128938
[/FORMAT]
```

Now count the tokens:

```python
audio_sample = client.files.upload(file='sample.mp3')

response = client.models.count_tokens(
    model=MODEL_ID,
    contents=audio_sample
)
print("Prompt with audio tokens:", response.total_tokens)
print("Tokens per second: ", response.total_tokens/2610)
```

Example output:
```
Prompt with audio tokens: 83528
Tokens per second:  32.003065134099614
```

This shows that audio is processed at approximately 32 tokens per second. Video follows a similar fixed-rate conversion.

## Step 4: Current Limitations

The unified SDK's `count_tokens` method currently doesn't support:
- Chat conversations
- Tool usage
- Cached content

For these scenarios, you can still check token usage after the fact using the `usage_metadata` from the response object. When using caching, cached tokens will appear in `response.usage_metadata.cached_content_token_count`.

## Key Takeaways

1. **Tokens are the billing unit**: Both input and output tokens contribute to your usage costs.
2. **Different modalities have different token rates**: Text, images, audio, and video all convert to tokens at different rates.
3. **Context windows matter**: Choose your model based on the context window you need for your application.
4. **Count before you generate**: Use `count_tokens()` to estimate costs before sending large requests.
5. **Check usage metadata**: After generation, examine `usage_metadata` for detailed token breakdowns.

## Further Reading

For more detailed information on token counting, consult:
- [Gemini API Tokens Documentation](https://ai.google.dev/gemini-api/docs/tokens?lang=python#multimodal-tokens)
- [`countTokens` REST API Reference](https://ai.google.dev/api/rest/v1/models/countTokens)
- [`count_tokens` Python API Reference](https://googleapis.github.io/python-genai/#count-tokens-and-compute-tokens)

Understanding tokens is crucial for optimizing your Gemini API usage and managing costs effectively. By mastering token counting, you can make informed decisions about model selection, prompt design, and resource allocation for your AI applications.
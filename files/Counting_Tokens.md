

# Gemini API: All about tokens

An understanding of tokens is central to using the Gemini API. This guide will provide a interactive introduction to what tokens are and how they are used in the Gemini API.

## About tokens

LLMs break up their input and produce their output at a granularity that is smaller than a word, but larger than a single character or code-point.

These **tokens** can be single characters, like `z`, or whole words, like `the`. Long words may be broken up into several tokens. The set of all tokens used by the model is called the vocabulary, and the process of breaking down text into tokens is called tokenization.

For Gemini models, a token is equivalent to about 4 characters. **100 tokens are about 60-80 English words**.

When billing is enabled, the price of a paid request is controlled by the [number of input and output tokens](https://ai.google.dev/pricing), so knowing how to count your tokens is important.


## Setup

### Install SDK

Install the SDK from [PyPI](https://github.com/googleapis/python-genai).


```
%pip install -q -U "google-genai>=1.0.0"
```

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
```

### Initialize SDK client

With the new SDK you now only need to initialize a client with you API key (or OAuth if using [Vertex AI](https://cloud.google.com/vertex-ai)). The model is now set in each call.


```
from google import genai

client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Tokens in the Gemini API

### Context windows

The models available through the Gemini API have context windows that are measured in tokens. These define how much input you can provide, and how much output the model can generate, and combined are referred to as the "context window". This information is available directly through [the API](https://ai.google.dev/api/rest/v1/models/get) and in the [models](https://ai.google.dev/models/gemini) documentation.

In this example you can see the `gemini-2.5-flash` model has an 1M tokens context window. If you need more, Pro models have an even bigger 2M tokens context window.


```
model_info = client.models.get(model=MODEL_ID)

print("Context window:",model_info.input_token_limit, "tokens")
print("Max output window:",model_info.output_token_limit, "tokens")
```

    Context window: 1048576 tokens
    Max output window: 65536 tokens


## Counting tokens

The API provides an endpoint for counting the number of tokens in a request: [`client.models.count_tokens`](https://googleapis.github.io/python-genai/#count-tokens-and-compute-tokens). You pass the same arguments as you would to [`client.models.generate_content`](https://googleapis.github.io/python-genai/#generate-content) and the service will return the number of tokens in that request.

### Choose a model

Now select the model you want to use in this guide, either by selecting one in the list or writing it down. Keep in mind that some models, like the 2.5 ones are thinking models and thus take slightly more time to respond (cf. [thinking notebook](./Get_started_thinking.ipynb) for more details and in particular learn how to switch the thiking off).

The tokenization should be more or less the same for each of the Gemini models, but you can still switch between the different ones to double-check.

For more information about all Gemini models, check the [documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for extended information on each of them.


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

### Text tokens


```
response = client.models.count_tokens(
    model=MODEL_ID,
    contents="What's the highest mountain in Africa?",
)
print("Prompt tokens:",response.total_tokens)
```

    Prompt tokens: 10


When you call `client.models.generate_content` (or `chat.send_message`) the response object has a `usage_metadata` attribute containing both the input, output, and thinking token counts (`prompt_token_count`, `candidates_token_count` and `thoughts_token_count`):


```
response = client.models.generate_content(
    model=MODEL_ID,
    contents="The quick brown fox jumps over the lazy dog."
)
print(response.text)
```

    That's a classic!
    
    It's a famous **pangram**, meaning it uses every letter of the alphabet at least once. It's often used for testing typewriters, keyboards, and fonts because it demonstrates all the characters.



```
print("Prompt tokens:\t ",response.usage_metadata.prompt_token_count)
print("Thinking tokens:",response.usage_metadata.thoughts_token_count)
print("Output tokens:\t ",response.usage_metadata.candidates_token_count)
print("--------------")
print("Total tokens:\t",response.usage_metadata.total_token_count)
```

    Prompt tokens:	  11
    Thinking tokens: 751
    Output tokens:	  49
    --------------
    Total tokens:	 811


In case you are using [caching](./Caching.ipynb#scrollTo=t_PWabuayrf-), the number of cached token will be indicated in `response.usage_metadata.cached_content_token_count`.

### Multi-modal tokens

All input to the API is tokenized, including images or other non-text modalities.

Images are considered to be a fixed size, so they consume a fixed number of tokens, regardless of their display or file size.

Video and audio files are converted to tokens at a fixed per second rate.

The current rates and token sizes can be found on the [documentation](https://ai.google.dev/gemini-api/docs/tokens?lang=python#multimodal-tokens)


```
!curl -L https://goo.gle/instrument-img -o organ.jpg
```

    [Total 187, Received 187, ..., Left 792]
    [Total 374k, Received 374k, ..., Left 1061k]



```
import PIL
from IPython.display import display, Image

organ = PIL.Image.open('organ.jpg')
display(Image('organ.jpg', width=300))
```

#### Inline content

Media objects can be sent to the API inline with the request:


```
response = client.models.count_tokens(
    model=MODEL_ID,
    contents=[organ]
)

print("Prompt with image tokens:",response.total_tokens)
```

    Prompt with image tokens: 259


You can try with different images and should always get the same number of tokens, that is independent of their display or file size. Note that an extra token seems to be added, representing the empty prompt.

#### Files API

The model sees identical tokens if you upload parts of the prompt through the files API instead:


```
organ_upload = client.files.upload(file='organ.jpg')

response = client.models.count_tokens(
    model=MODEL_ID,
    contents=organ_upload,
)

print("Prompt with image tokens:",response.total_tokens)
```

    Prompt with image tokens: 259


Audio and video are each converted to tokens at a fixed rate of tokens per minute.


```
!curl -q -o sample.mp3  "https://storage.googleapis.com/generativeai-downloads/data/State_of_the_Union_Address_30_January_1961.mp3"
!ffprobe -v error -show_entries format=duration sample.mp3
```

    [Total 39.8M, Received 39.8M, ..., Left 57.6M]
    [FORMAT]
    duration=2610.128938
    [/FORMAT]


As you can see, this audio file is 2610s long.


```
audio_sample = client.files.upload(file='sample.mp3')

response = client.models.count_tokens(
    model=MODEL_ID,
    contents=audio_sample
)

print("Prompt with audio tokens:",response.total_tokens)
print("Tokens per second: ",response.total_tokens/2610)
```

    Prompt with audio tokens: 83528
    Tokens per second:  32.003065134099614


As you can see this corresponds to about 32 tokens per second of audio.

### Chat, tools and caching

Chat, tools and caching are currently not supported by the unified SDK `count_tokens` method. This notebook will be updated when that will be the case.

In the meantime you can still check the token used after the call using the `usage_metadata` from the response. Check the [caching notebook](./Caching.ipynb#scrollTo=t_PWabuayrf-) for an example.

## Further reading

For more on token counting, check out the [documentation](https://ai.google.dev/gemini-api/docs/tokens?lang=python#multimodal-tokens) or the API reference:

* [`countTokens`](https://ai.google.dev/api/rest/v1/models/countTokens) REST API reference,
* [`count_tokens`](https://googleapis.github.io/python-genai/#count-tokens-and-compute-tokens) Python API reference,
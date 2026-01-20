

# Gemini API: Context Caching Quickstart

This notebook introduces context caching with the Gemini API and provides examples of interacting with the Apollo 11 transcript using the Python SDK. For a more comprehensive look, check out [the caching guide](https://ai.google.dev/gemini-api/docs/caching?lang=python).

### Install dependencies


```
%pip install -q -U "google-genai>=1.0.0"
```

### Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Upload a file

A common pattern with the Gemini API is to ask a number of questions of the same document. Context caching is designed to assist with this case, and can be more efficient by avoiding the need to pass the same tokens through the model for each new request.

This example will be based on the transcript from the Apollo 11 mission.

Start by downloading that transcript.


```
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt
!head a11.txt
```

    INTRODUCTION
    
    This is the transcription of the Technical Air-to-Ground Voice Transmission (GOSS NET 1) from the Apollo 11 mission.
    
    Communicators in the text may be identified according to the following list.
    
    Spacecraft:
    CDR	Commander	Neil A. Armstrong
    CMP	Command module pilot   	Michael Collins
    LMP	Lunar module pilot	Edwin E. ALdrin, Jr.


Now upload the transcript using the [File API](../quickstarts/File_API.ipynb).


```
document = client.files.upload(file="a11.txt")
```

## Cache the prompt

Next create a [`CachedContent`](https://ai.google.dev/api/python/google/generativeai/protos/CachedContent) object specifying the prompt you want to use, including the file and other fields you wish to cache. In this example the [`system_instruction`](../quickstarts/System_instructions.ipynb) has been set, and the document was provided in the prompt.

Note that caches are model specific. You cannot use a cache made with a different model as their tokenization might be slightly different.


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

apollo_cache = client.caches.create(
    model=MODEL_ID,
    config={
        'contents': [document],
        'system_instruction': 'You are an expert at analyzing transcripts.',
    },
)

apollo_cache
```




    CachedContent(
      create_time=datetime.datetime(2025, 8, 6, 13, 48, 36, 419118, tzinfo=TzInfo(UTC)),
      display_name='',
      expire_time=datetime.datetime(2025, 8, 6, 14, 48, 36, 38936, tzinfo=TzInfo(UTC)),
      model='models/gemini-2.5-flash',
      name='cachedContents/0c5j38gpopx49ok6x7kedvbpy65d1bzkq8i5vldr',
      update_time=datetime.datetime(2025, 8, 6, 13, 48, 36, 419118, tzinfo=TzInfo(UTC)),
      usage_metadata=CachedContentUsageMetadata(
        total_token_count=322698
      )
    )




```
from IPython.display import Markdown

display(Markdown(f"As you can see in the output, you just cached **{apollo_cache.usage_metadata.total_token_count}** tokens."))
```


As you can see in the output, you just cached **322698** tokens.


## Manage the cache expiry

Once you have a `CachedContent` object, you can update the expiry time to keep it alive while you need it.


```
from google.genai import types

client.caches.update(
    name=apollo_cache.name,
    config=types.UpdateCachedContentConfig(ttl="7200s")  # 2 hours in seconds
)

apollo_cache = client.caches.get(name=apollo_cache.name) # Get the updated cache
apollo_cache
```




    CachedContent(
      create_time=datetime.datetime(2025, 8, 6, 13, 48, 36, 419118, tzinfo=TzInfo(UTC)),
      display_name='',
      expire_time=datetime.datetime(2025, 8, 6, 15, 48, 36, 651814, tzinfo=TzInfo(UTC)),
      model='models/gemini-2.5-flash',
      name='cachedContents/0c5j38gpopx49ok6x7kedvbpy65d1bzkq8i5vldr',
      update_time=datetime.datetime(2025, 8, 6, 13, 48, 36, 691886, tzinfo=TzInfo(UTC)),
      usage_metadata=CachedContentUsageMetadata(
        total_token_count=322698
      )
    )



## Use the cache for generation

As the `CachedContent` object refers to a specific model and parameters, you must create a [`GenerativeModel`](https://ai.google.dev/api/python/google/generativeai/GenerativeModel) using [`from_cached_content`](https://ai.google.dev/api/python/google/generativeai/GenerativeModel#from_cached_content). Then, generate content as you would with a directly instantiated model object.


```
response = client.models.generate_content(
    model=MODEL_ID,
    contents='Find a lighthearted moment from this transcript',
    config=types.GenerateContentConfig(
        cached_content=apollo_cache.name,
    )
)

display(Markdown(response.text))
```


There are many lighthearted moments scattered throughout the transcript, often provided by the crew's dry wit and the relatable human touches of everyday life in space.

One particularly lighthearted moment occurs when the crew is dealing with an **O2 FLOW HIGH master alarm**. Instead of panicking, Mike Collins offers a rather amusing observation:

**02 06 53 37 CMP:** "That photoelectric cell is a good device. It's worked very well."
**02 06 53 46 CC:** "11, Houston. Say again. Over."
**02 06 53 50 CMP:** "I say that photoelectric cell amplifier for the master alarm is a good device. It's working very well, and it's a nice pleasing tone."
**02 06 54 00 CC:** "Roger. Copy. Thank you."
**02 06 54 08 CMP:** "Makes you almost glad to get master alarms."

This exchange highlights the crew's ability to maintain a sense of humor and composure even when dealing with technical issues and alarms. Mike Collins' comment about being "almost glad to get master alarms" due to their "nice pleasing tone" is a classic example of gallows humor and makes for a very relatable moment.


You can inspect token usage through `usage_metadata`. Note that the cached prompt tokens are included in `prompt_token_count`, but excluded from the `total_token_count`.


```
response.usage_metadata
```




    GenerateContentResponseUsageMetadata(
      cache_tokens_details=[
        ModalityTokenCount(
          modality=<MediaModality.TEXT: 'TEXT'>,
          token_count=322698
        ),
      ],
      cached_content_token_count=322698,
      candidates_token_count=282,
      prompt_token_count=322707,
      prompt_tokens_details=[
        ModalityTokenCount(
          modality=<MediaModality.TEXT: 'TEXT'>,
          token_count=322707
        ),
      ],
      thoughts_token_count=4049,
      total_token_count=327038
    )




```
display(Markdown(f"""
  As you can see in the `usage_metadata`, the token usage is split between:
  *  {response.usage_metadata.cached_content_token_count} tokens for the cache,
  *  {response.usage_metadata.prompt_token_count} tokens for the input (including the cache, so {response.usage_metadata.prompt_token_count - response.usage_metadata.cached_content_token_count} for the actual prompt),
  *  {response.usage_metadata.thoughts_token_count} tokens for the thinking process,
  *  {response.usage_metadata.candidates_token_count} tokens for the output,
  *  {response.usage_metadata.total_token_count} tokens in total.
"""))
```



  As you can see in the `usage_metadata`, the token usage is split between:
  *  322698 tokens for the cache,
  *  322707 tokens for the input (including the cache, so 9 for the actual prompt),
  *  4049 tokens for the thinking process,
  *  282 tokens for the output,
  *  327038 tokens in total.



You can ask new questions of the model, and the cache is reused.


```
chat = client.chats.create(
  model=MODEL_ID,
  config={"cached_content": apollo_cache.name}
)

response = chat.send_message(message="Give me a quote from the most important part of the transcript.")
display(Markdown(response.text))
```


The most important part of the transcript, and arguably the entire Apollo 11 mission, is the moment Neil Armstrong steps onto the lunar surface.

Here is the quote:

**04 13 24 48 CDR (TRANQ) THAT'S ONE SMALL STEP FOR (A) MAN, ONE GIANT LEAP FOR MANKIND.**



```
response = chat.send_message(
    message="What was recounted after that?",
    config={"cached_content": apollo_cache.name}
)
display(Markdown(response.text))
```


Immediately after his iconic "one small step" statement, Neil Armstrong provided detailed observations of the lunar surface:

*   He described the **surface** as "fine and powdery," noting that he could pick it up loosely with his toe. It "adhere[d] in fine layers like powdered charcoal" to his boots, and he only went in "a small fraction of an inch, maybe an eighth of an inch," clearly seeing his footprints and the treads.
*   He commented on the **ease of movement**, stating there was "no difficulty in moving around," finding it "even perhaps easier than the simulations at one sixth g that we performed in the various simulations on the ground." He confirmed it was "no trouble to walk around."
*   He observed the **descent engine's impact**, noting it "did not leave a crater of any size." He reported "about 1 foot clearance on the ground" and that they were "essentially on a very level place." He saw "some evidence of rays emanating from the descent engine," but described it as "very insignificant."

Mission Control (CC) confirmed that they were "copying" his observations.



```
response.usage_metadata
```




    GenerateContentResponseUsageMetadata(
      cache_tokens_details=[
        ModalityTokenCount(
          modality=<MediaModality.TEXT: 'TEXT'>,
          token_count=322698
        ),
      ],
      cached_content_token_count=322698,
      candidates_token_count=239,
      prompt_token_count=322795,
      prompt_tokens_details=[
        ModalityTokenCount(
          modality=<MediaModality.TEXT: 'TEXT'>,
          token_count=322795
        ),
      ],
      thoughts_token_count=902,
      total_token_count=323936
    )




```
display(Markdown(f"""
  As you can see in the `usage_metadata`, the token usage is split between:
  *  {response.usage_metadata.cached_content_token_count} tokens for the cache,
  *  {response.usage_metadata.prompt_token_count} tokens for the input (including the cache, so {response.usage_metadata.prompt_token_count - response.usage_metadata.cached_content_token_count} for the actual prompt),
  *  {response.usage_metadata.thoughts_token_count} tokens for the thinking process,
  *  {response.usage_metadata.candidates_token_count} tokens for the output,
  *  {response.usage_metadata.total_token_count} tokens in total.
"""))
```



  As you can see in the `usage_metadata`, the token usage is split between:
  *  322698 tokens for the cache,
  *  322795 tokens for the input (including the cache, so 97 for the actual prompt),
  *  902 tokens for the thinking process,
  *  239 tokens for the output,
  *  323936 tokens in total.



Since the cached tokens are cheaper than the normal ones, it means this prompt was much cheaper that if you had not used caching. Check the [pricing here](https://ai.google.dev/pricing) for the up-to-date discount on cached tokens.

## Delete the cache

The cache has a small recurring storage cost (cf. [pricing](https://ai.google.dev/pricing)) so by default it is only saved for an hour. In this case you even set it up for a shorter amont of time (using `"ttl"`) of 2h.

Still, if you don't need you cache anymore, it is good practice to delete it proactively.


```
print(apollo_cache.name)
client.caches.delete(name=apollo_cache.name)
```

    cachedContents/0c5j38gpopx49ok6x7kedvbpy65d1bzkq8i5vldr





    DeleteCachedContentResponse()



## Next Steps
### Useful API references:

If you want to know more about the caching API, you can check the full [API specifications](https://ai.google.dev/api/rest/v1beta/cachedContents) and the [caching documentation](https://ai.google.dev/gemini-api/docs/caching).

### Continue your discovery of the Gemini API

Check the File API notebook to know more about that API. The [vision capabilities](../quickstarts/Video.ipynb) of the Gemini API are a good reason to use the File API and the caching.
The Gemini API also has configurable [safety settings](../quickstarts/Safety.ipynb) that you might have to customize when dealing with big files.
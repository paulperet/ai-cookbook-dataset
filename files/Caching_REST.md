

# Gemini API: Caching Quickstart with REST

This notebook introduces [context caching](https://ai.google.dev/gemini-api/docs/caching?lang=rest) with the Gemini API and provides examples of interacting with the Apollo 11 transcript using the Python SDK. Context caching is a way to save on requests costs when a substantial initial context is referenced repeatedly by shorter requests. It will use `curl` commands to call the methods in the REST API. 

For a more comprehensive look, check out [the caching guide](https://ai.google.dev/gemini-api/docs/caching?lang=rest).

This notebook contains `curl` commands you can run in Google Colab, or copy to your terminal. If you have never used the Gemini REST API, it is strongly recommended to start with the [Prompting quickstart](../../quickstarts/rest/Prompting_REST.ipynb) first.

### Authentication

To run this notebook, your API key must be stored it in a Colab Secret named GOOGLE_API_KEY. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](../../quickstarts/Authentication.ipynb) to learn more.


```
import os
from google.colab import userdata
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Caching content

Let's start by getting the transcript from the Apollo 11 mission.


```
!wget https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

[First Entry, ..., Last Entry]

Now you need to reencode it to base-64, so let's prepare the whole [cachedContent](https://ai.google.dev/api/rest/v1beta/cachedContents#resource:-cachedcontent) while you're at it.



```bash
%%bash

echo '{
  "model": "models/gemini-2.5-flash",
  "contents":[
    {
      "parts":[
        {
          "inline_data": {
            "mime_type":"text/plain",
            "data": "'$(base64 -w0 a11.txt)'"
          }
        }
      ],
    "role": "user"
    }
  ],
  "systemInstruction": {
    "parts": [
      {
        "text": "You are an expert at analyzing transcripts."
      }
    ]
  },
  "ttl": "600s"
}' > request.json
```

We can now create the cached content.


```bash
%%bash

curl -X POST "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=$GOOGLE_API_KEY" \
 -H 'Content-Type: application/json' \
 -d @request.json > cache.json

 cat cache.json
```

[First Entry, ..., Last Entry]

You will need it for the next commands so save the name of the cache.

You're using a text file to save the name here beacuse of colab constrainsts but you could also simply use a variable.


```bash
%%bash

CACHE_NAME=$(cat cache.json | grep '"name":' | cut -d '"' -f 4 | head -n 1)

echo $CACHE_NAME > cache_name.txt

cat cache_name.txt
```

    cachedContents/qidqwuaxdqz4


## Listing caches
Since caches have a reccuring cost it's a good idea to keep an eye on them. It can also be useful if you need to find their name.


```
!curl "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=$GOOGLE_API_KEY"
```

    {
      "cachedContents": [
        {
          "name": "cachedContents/lf0nt062ulc1",
          "model": "models/gemini-2.5-flash",
          "createTime": "2024-07-11T18:02:48.257891Z",
          "updateTime": "2024-07-11T18:02:48.257891Z",
          "expireTime": "2024-07-11T18:07:47.635193373Z",
          "displayName": "",
          "usageMetadata": {
            "totalTokenCount": 323383
          }
        },
        {
          "name": "cachedContents/qidqwuaxdqz4",
          "model": "models/gemini-2.5-flash",
          "createTime": "2024-07-11T18:02:30.516233Z",
          "updateTime": "2024-07-11T18:02:30.516233Z",
          "expireTime": "2024-07-11T18:07:29.803448998Z",
          "displayName": "",
          "usageMetadata": {
            "totalTokenCount": 323383
          }
        }
      ]
    }


## Using cached content when prompting

Prompting using cached content is the same as what is illustrated in the [Prompting quickstart](../../quickstarts/rest/Prompting_REST.ipynb) except you're adding a `"cachedContent"` value which is the name of the cache you saved earlier.


```bash
%%bash

curl -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
      "contents": [
        {
          "parts":[{
            "text": "Please summarize this transcript"
          }],
          "role": "user"
        },
      ],
      "cachedContent": "'$(cat cache_name.txt)'"
    }'
```

[First Entry, ..., Last Entry]

As you can see, among the 323699 tokens, 323383 were cached (and thus less expensive) and only 311 were from the prompt.

Since the cached tokens are cheaper than the normal ones, it means this prompt was 75% cheaper that if you had not used caching. Check the [pricing here](https://ai.google.dev/pricing) for the up-to-date discount on cached tokens.

## Optional: Updating a cache
If you need to update a cache, to chance its content, or just extend its longevity, just use `PATCH`:


```bash
%%bash

curl -X PATCH "https://generativelanguage.googleapis.com/v1beta/$(cat cache_name.txt)?key=$GOOGLE_API_KEY" \
 -H 'Content-Type: application/json' \
 -d '{"ttl": "300s"}'
```

[First Entry, ..., Last Entry]

## Deleting cached content

The cache has a small recurring storage cost (cf. [pricing](https://ai.google.dev/pricing)) so by default it is only saved for an hour. In this case you even set it up for a shorter amont of time (using `"ttl"`) of 10mn.

Still, if you don't need you cache anymore, it is good practice to delete it proactively.


```
!curl -X DELETE "https://generativelanguage.googleapis.com/v1beta/$(cat cache_name.txt)?key=$GOOGLE_API_KEY"
```

    {}


## Next Steps
### Useful API references:

If you want to know more about the caching REST APIs, you can check the full [API specifications](https://ai.google.dev/api/rest/v1beta/cachedContents) and the [caching documentation](https://ai.google.dev/gemini-api/docs/caching).

### Continue your discovery of the Gemini API

Check the File API notebook to know more about that API. The [vision capabilities](../../quickstarts/rest/Video_REST.ipynb) of the Gemini API are a good reason to use the File API and the caching. 
The Gemini API also has configurable [safety settings](../../quickstarts/rest/Safety_REST.ipynb) that you might have to customize when dealing with big files.
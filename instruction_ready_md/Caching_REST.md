# Gemini API: Context Caching Guide with REST

This guide demonstrates how to use **context caching** with the Gemini API via REST. Context caching allows you to store a large initial context (like a document) and reference it repeatedly with shorter, cheaper requests. You will interact with the Apollo 11 mission transcript using `curl` commands.

For a comprehensive overview, refer to the [official caching documentation](https://ai.google.dev/gemini-api/docs/caching?lang=rest).

> **Prerequisite:** It's recommended to complete the [Prompting quickstart](../../quickstarts/rest/Prompting_REST.ipynb) first if you are new to the Gemini REST API.

## 1. Setup and Authentication

To run the commands in this guide, you need a valid Gemini API key. Store your key in an environment variable named `GOOGLE_API_KEY`.

If you are using Google Colab, you can store it as a secret:

```python
import os
from google.colab import userdata
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

For other environments, set the variable in your terminal:
```bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

## 2. Prepare the Content for Caching

First, download the Apollo 11 transcript that will serve as your cached context.

```bash
wget https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

Next, you need to create a `cachedContent` resource. This involves base64-encoding the file and constructing a JSON request body. The `ttl` field sets the cache's time-to-live (here, 600 seconds or 10 minutes).

Create a file named `request.json`:

```bash
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

## 3. Create the Cached Content

Now, send a `POST` request to the `cachedContents` endpoint to create your cache. The response will contain metadata, including a unique `name` for your cache.

```bash
curl -X POST "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=$GOOGLE_API_KEY" \
 -H 'Content-Type: application/json' \
 -d @request.json > cache.json

cat cache.json
```

The response will look similar to this (abbreviated):
```json
{
  "name": "cachedContents/qidqwuaxdqz4",
  "model": "models/gemini-2.5-flash",
  "createTime": "2024-07-11T18:02:30.516233Z",
  ...
}
```

Extract and save the cache name for later use:

```bash
CACHE_NAME=$(cat cache.json | grep '"name":' | cut -d '"' -f 4 | head -n 1)
echo $CACHE_NAME > cache_name.txt
cat cache_name.txt
```

Output:
```
cachedContents/qidqwuaxdqz4
```

## 4. List Existing Caches

Cached content incurs a small recurring storage cost. You can list all your active caches to monitor them or find a specific cache name.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=$GOOGLE_API_KEY"
```

The response will be a JSON array showing all your caches, their creation times, expiration times, and token counts.

## 5. Generate Content Using the Cache

To use the cached context, make a standard `generateContent` request but include the `cachedContent` field with your cache's name. The model will use the pre-cached transcript as context.

```bash
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

The response will include a summary of the Apollo 11 transcript. Crucially, the `usageMetadata` in the response will show that the vast majority of tokens (e.g., 323,383) were served from the cache, with only a small number (e.g., 311) counted for your new prompt. This results in significant cost savings—in this example, the request was approximately 75% cheaper than without caching.

## 6. (Optional) Update a Cache

You can update an existing cache, for example, to extend its lifetime by modifying its `ttl`. Use a `PATCH` request.

```bash
curl -X PATCH "https://generativelanguage.googleapis.com/v1beta/$(cat cache_name.txt)?key=$GOOGLE_API_KEY" \
 -H 'Content-Type: application/json' \
 -d '{"ttl": "300s"}'
```

## 7. Delete Cached Content

While caches auto-expire based on their `ttl`, it's good practice to delete them proactively to avoid storage costs. Use a `DELETE` request.

```bash
curl -X DELETE "https://generativelanguage.googleapis.com/v1beta/$(cat cache_name.txt)?key=$GOOGLE_API_KEY"
```

A successful deletion returns an empty JSON object: `{}`.

## Next Steps

*   **API Reference:** Explore the full [CachedContents API specification](https://ai.google.dev/api/rest/v1beta/cachedContents).
*   **Pricing:** Review the latest [pricing details](https://ai.google.dev/pricing) for cached vs. non-cached tokens.
*   **Continue Learning:** Discover other Gemini API features:
    *   [File API](../../quickstarts/rest/Video_REST.ipynb) for handling video and other file types.
    *   [Safety Settings](../../quickstarts/rest/Safety_REST.ipynb) to customize content safety filters.
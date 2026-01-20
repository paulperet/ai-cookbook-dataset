# Gemini API: Error handling

This Colab notebook demonstrates strategies for handling common errors you might encounter when working with the Gemini API:

*   **Transient Errors:** Temporary failures due to network issues, server overload, etc.
*   **Rate Limits:** Restrictions on the number of requests you can make within a certain timeframe.
*   **Timeouts:** When an API call takes too long to complete.

You have two main approaches to explore:

1.  **Automatic retries:** A simple way to retry requests when they fail due to transient errors.
2.  **Manual backoff and retry:** A more customizable approach that provides finer control over retry behavior.


**Gemini Rate Limits**

The default rate limits for different Gemini models are outlined in the [Gemini API model documentation](https://ai.google.dev/gemini-api/docs/models/gemini#model-variations). If your application requires a higher quota, consider [requesting a rate limit increase](https://ai.google.dev/gemini-api/docs/quota).


```
%pip install -q -U "google-genai>=1.0.0"
```

### Setup your API key

To run the following cells, store your API key in a Colab Secret named `GOOGLE_API_KEY`. If you don't have an API key or need help creating a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) guide.


```
from google import genai
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Automatic retries

The Gemini API's client library offers built-in retry mechanisms for handling transient errors. You can enable this feature by using the `request_options` argument with API calls like `generate_content`, `generate_answer`, `embed_content`, and `generate_content_async`.

**Advantages:**

* **Simplicity:** Requires minimal code changes for significant reliability gains.
* **Robust:** Effectively addresses most transient errors without additional logic.

**Customize retry behavior:**

Use these settings in [`retry`](https://googleapis.dev/python/google-api-core/latest/retry.html) to customize retry behavior:

* `predicate`:  (callable) Determines if an exception is retryable. Default: [`if_transient_error`](https://github.com/googleapis/python-api-core/blob/main/google/api_core/retry/retry_base.py#L75C4-L75C13)
* `initial`: (float) Initial delay in seconds before the first retry. Default: `1.0`
* `maximum`: (float) Maximum delay in seconds between retries. Default: `60.0`
* `multiplier`: (float) Factor by which the delay increases after each retry. Default: `2.0`
* `timeout`: (float) Total retry duration in seconds. Default: `120.0`


```
from google.api_core import retry

MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

prompt = "Write a story about a magic backpack."

#Built in retry support was removed from the sdk, so you need to use retry package
@retry.Retry(
    predicate=retry.if_transient_error,
)
def generate_with_retry():
  return client.models.generate_content(
      model=MODEL_ID,
      contents=prompt
  )

generate_with_retry().text
```

### Manually increase timeout when responses take time

If you encounter `ReadTimeout` or `DeadlineExceeded` errors, meaning an API call exceeds the default timeout (600 seconds), you can manually adjust it by defining `timeout` in the `request_options` argument.


```
from google.genai import types
prompt = "Write a story about a magic backpack."

client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
       http_options=types.HttpOptions(
           timeout=15*60*1000
       )
    )
)  # Increase timeout to 15 minutes
```

**Caution:**  While increasing timeouts can be helpful, be mindful of setting them too high, as this can delay error detection and potentially waste resources.

### Manually implement backoff and retry with error handling

For finer control over retry behavior and error handling, you can use the [`retry`](https://googleapis.dev/python/google-api-core/latest/retry.html) library (or similar libraries like [`backoff`](https://pypi.org/project/backoff/) and [`tenacity`](https://tenacity.readthedocs.io/en/latest/)). This gives you precise control over retry strategies and allows you to handle specific types of errors differently.


```
from google.api_core import retry, exceptions

MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

@retry.Retry(
    predicate=retry.if_transient_error,
    initial=2.0,
    maximum=64.0,
    multiplier=2.0,
    timeout=600,
)
def generate_with_retry(prompt):
    return client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )


prompt = "Write a one-liner advertisement for magic backpack."

generate_with_retry(prompt=prompt)
```

### Test the error handling with retry mechanism

To validate that your error handling and retry mechanism work as intended, define a `generate_content` function that deliberately raises a `ServiceUnavailable` error on the first call. This setup will help you ensure that the retry decorator successfully handles the transient error and retries the operation.


```
from google.api_core import retry, exceptions


@retry.Retry(
    predicate=retry.if_transient_error,
    initial=2.0,
    maximum=64.0,
    multiplier=2.0,
    timeout=600,
)
def generate_content_first_fail(prompt):
    if not hasattr(generate_content_first_fail, "call_counter"):
        generate_content_first_fail.call_counter = 0

    generate_content_first_fail.call_counter += 1

    try:
        if generate_content_first_fail.call_counter == 1:
            raise exceptions.ServiceUnavailable("Service Unavailable")

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text
    except exceptions.ServiceUnavailable as e:
        print(f"Error: {e}")
        raise


prompt = "Write a one-liner advertisement for magic backpack."

generate_content_first_fail(prompt=prompt)
```
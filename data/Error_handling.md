# Gemini API: Error Handling Guide

This guide demonstrates strategies for handling common errors when working with the Gemini API, including transient errors, rate limits, and timeouts. You'll learn both automatic and manual approaches to make your applications more resilient.

## Prerequisites

### Install Required Packages
First, install the Google Generative AI Python SDK:

```bash
pip install -q -U "google-genai>=1.0.0"
```

### Set Up Your API Key
Store your API key in an environment variable or secure storage. For this guide, we'll use Google Colab's secrets feature:

```python
from google import genai
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
```

> **Note:** If you're not using Colab, you can set your API key using `os.environ["GOOGLE_API_KEY"] = "your-api-key"`.

## Understanding Gemini Rate Limits

Before implementing error handling, it's important to understand the default rate limits for different Gemini models. These are outlined in the [Gemini API model documentation](https://ai.google.dev/gemini-api/docs/models/gemini#model-variations).

If your application requires higher quotas, consider [requesting a rate limit increase](https://ai.google.dev/gemini-api/docs/quota).

## Method 1: Automatic Retries

The Gemini API client library offers built-in retry mechanisms for handling transient errors. You can enable this feature using the `request_options` argument with API calls.

### Advantages
- **Simplicity:** Minimal code changes for significant reliability gains
- **Robust:** Effectively addresses most transient errors without additional logic

### Implementation

First, import the retry utilities:

```python
from google.api_core import retry

MODEL_ID = "gemini-3-flash-preview"
prompt = "Write a story about a magic backpack."
```

Now, create a function with automatic retry logic:

```python
@retry.Retry(
    predicate=retry.if_transient_error,
)
def generate_with_retry():
    return client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )

# Call the function
response = generate_with_retry()
print(response.text)
```

### Customizing Retry Behavior

You can customize the retry behavior using these parameters:

- `predicate`: (callable) Determines if an exception is retryable. Default: `if_transient_error`
- `initial`: (float) Initial delay in seconds before the first retry. Default: `1.0`
- `maximum`: (float) Maximum delay in seconds between retries. Default: `60.0`
- `multiplier`: (float) Factor by which the delay increases after each retry. Default: `2.0`
- `timeout`: (float) Total retry duration in seconds. Default: `120.0`

## Method 2: Adjusting Timeouts

If you encounter `ReadTimeout` or `DeadlineExceeded` errors (when API calls exceed the default 600-second timeout), you can manually adjust the timeout.

### Implementation

```python
from google.genai import types

prompt = "Write a story about a magic backpack."

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        http_options=types.HttpOptions(
            timeout=15*60*1000  # Increase timeout to 15 minutes
        )
    )
)

print(response.text)
```

> **Caution:** While increasing timeouts can be helpful, setting them too high can delay error detection and potentially waste resources.

## Method 3: Manual Backoff and Retry with Error Handling

For finer control over retry behavior and error handling, you can manually implement backoff and retry logic. This approach gives you precise control over retry strategies and allows you to handle specific types of errors differently.

### Implementation

```python
from google.api_core import retry, exceptions

MODEL_ID = "gemini-3-flash-preview"

@retry.Retry(
    predicate=retry.if_transient_error,
    initial=2.0,      # Start with 2-second delay
    maximum=64.0,     # Maximum 64-second delay
    multiplier=2.0,   # Double the delay each retry
    timeout=600,      # Total timeout of 10 minutes
)
def generate_with_retry(prompt):
    return client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )

prompt = "Write a one-liner advertisement for magic backpack."
response = generate_with_retry(prompt=prompt)
print(response.text)
```

## Testing Your Error Handling Implementation

To validate that your error handling and retry mechanism work correctly, you can create a test function that deliberately raises an error on the first call. This helps ensure the retry decorator successfully handles transient errors.

### Test Implementation

```python
from google.api_core import retry, exceptions

@retry.Retry(
    predicate=retry.if_transient_error,
    initial=2.0,
    maximum=64.0,
    multiplier=2.0,
    timeout=600,
)
def generate_content_first_fail(prompt):
    # Track call count
    if not hasattr(generate_content_first_fail, "call_counter"):
        generate_content_first_fail.call_counter = 0
    
    generate_content_first_fail.call_counter += 1
    
    try:
        # Simulate failure on first call
        if generate_content_first_fail.call_counter == 1:
            raise exceptions.ServiceUnavailable("Service Unavailable")
        
        # Normal API call on retry
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text
        
    except exceptions.ServiceUnavailable as e:
        print(f"Error: {e}")
        raise

prompt = "Write a one-liner advertisement for magic backpack."
result = generate_content_first_fail(prompt=prompt)
print(f"Success! Result: {result}")
```

When you run this test, you should see:
1. An error message on the first attempt
2. A successful response on the retry attempt

## Summary

You've learned three approaches to handle errors with the Gemini API:

1. **Automatic Retries:** Simple, built-in retry mechanism for transient errors
2. **Timeout Adjustment:** Manual control over request timeouts for long-running operations
3. **Manual Backoff and Retry:** Customizable retry logic with fine-grained control

Choose the approach that best fits your application's needs:
- Use **automatic retries** for simple reliability improvements
- Use **timeout adjustment** when dealing with long-running queries
- Use **manual backoff and retry** when you need precise control over error handling behavior

Remember to monitor your application's error rates and adjust your retry strategies as needed to balance reliability with performance.
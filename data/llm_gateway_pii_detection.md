# Building an LLM Gateway for PII Detection

A common concern when adopting LLMs for enterprise use cases is data privacy, especially for teams handling sensitive information. While open-weight models are a great option, sometimes you need a quick demo or have specific reasons to use an external LLM API. In these cases, implementing a gateway to scrub Personal Identifiable Information (PII) before sending data to an API is a critical best practice.

[Wealthsimple](https://github.com/wealthsimple/llm-gateway) has open-sourced a repository designed for this exact purpose. This tutorial will guide you through using this gateway to scrub data before making API calls to an LLM provider. We'll use a [PII Dataset from AI4Privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) and Cohere's [Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus) model via their [free trial API](https://cohere.com/blog/free-developer-tier-announcement) to demonstrate the PII scrubbing workflow.

## Prerequisites & Setup

Before you begin, ensure you have the following installed and configured:

1.  **Install Poetry and Pyenv:** Follow the instructions in the [project README](https://github.com/wealthsimple/llm-gateway).
2.  **Install Python 3.11.3:** Use Pyenv to install the required Python version.
    ```bash
    pyenv install 3.11.3
    ```
3.  **Install Project Dependencies:** Navigate to the project directory and install the requirements.
    ```bash
    brew install gitleaks
    poetry install
    poetry run pre-commit install
    ```
4.  **Configure Environment Variables:** Copy the example environment file and update it with your API keys.
    ```bash
    cp .envrc.example .envrc
    # Edit .envrc to add your COHERE_API_KEY and DATABASE_URL
    ```

Now, let's import the necessary libraries and set up our environment variables.

```python
import os
from llm_gateway.providers.cohere import CohereWrapper
from datasets import load_dataset
import cohere
import types
import re

COHERE_API_KEY = os.environ['COHERE_API_KEY']
DATABASE_URL = os.environ['DATABASE_URL']  # e.g., "postgresql://postgres:postgres@postgres:5432/llm_gateway"
```

## Step 1: Initialize the LLM Wrapper

The core of the gateway is a wrapper object that applies "scrubbers" to your prompt before making the API call. When you make a request, it returns both the LLM's response and a database record containing the scrubbed input and request metadata.

First, initialize the `CohereWrapper`.

```python
wrapper = CohereWrapper()
```

## Step 2: Test the Gateway with a Sample Prompt

Let's see the gateway in action with a text snippet containing PII.

```python
example = "Michael Smith (msmith@gmail.com, (+1) 111-111-1111) committed a mistake when he used PyTorch Trainer instead of HF Trainer."
```

Now, send this prompt through the wrapper to get a summary. The wrapper will scrub the PII before the API call.

```python
response, db_record = wrapper.send_cohere_request(
    endpoint="generate",
    model="command-r-plus",
    max_tokens=25,
    prompt=f"{example}\n\nSummarize the above text in 1-2 sentences.",
    temperature=0.3,
)

print("LLM Response:")
print(response)
```

The response contains the model's output. In this case, it summarizes the text:
```
{'data': ['Michael Smith made a mistake by using PyTorch Trainer instead of HF Trainer.'], ...}
```

## Step 3: Examine the Scrubbed Database Record

The second return value, `db_record`, shows exactly what was sent to the API after scrubbing. This is invaluable for auditing and verifying the PII removal process.

```python
print("\nDatabase Record (Scrubbed Input):")
print(db_record)
```

In the record, you'll see the prompt has been modified:
```
'user_input': 'Michael Smith ([REDACTED EMAIL ADDRESS], (+1) [REDACTED PHONE NUMBER]) committed a mistake when he used PyTorch Trainer instead of HF Trainer.\n\nSummarize the above text in 1-2 sentences.',
```

Notice that the email and phone number have been redacted, but the name "Michael Smith" remains. This is because the default scrubbers do not include a name scrubber. Let's explore what scrubbers are applied.

## Step 4: Understanding the Default Scrubbers

The gateway applies a series of scrubbers sequentially. According to the source repository, the default scrubbers are:

```python
ALL_SCRUBBERS = [
    scrub_phone_numbers,
    scrub_credit_card_numbers,
    scrub_email_addresses,
    scrub_postal_codes,
    scrub_sin_numbers,
]
```

> **Note:** The `sin` (Social Insurance Number) scrubber is applied last as it can be overly aggressive. The authors also note that the `generate` endpoint used here is deprecated by Cohere. Updating the integration to use the newer Chat endpoint would be a valuable open-source contribution.

## Step 5: Implementing a Custom Scrubber

If your use case requires scrubbing additional PII types (like specific names), you can extend the wrapper by adding a custom scrubber. Here's how to create and apply a scrubber for the name "Michael Smith".

First, define your custom scrubber function using a regular expression.

```python
def my_custom_scrubber(text: str) -> str:
    """
    Scrub 'Michael Smith' in text.

    :param text: Input text to scrub
    :type text: str
    :return: Input text with any mentions of Michael Smith scrubbed
    :rtype: str
    """
    return re.sub(
        r"Michael Smith",
        "[REDACTED PERSON]",
        text,
        flags=re.IGNORECASE
    )
```

Next, you need to monkey-patch the wrapper's request method to apply your scrubber before the default ones. This approach modifies the instance's method directly.

```python
# Store the original method
original_method = wrapper.send_cohere_request

def modified_method(self, **kwargs):
    # Validate the endpoint (note: this duplicates an internal validation call)
    self._validate_cohere_endpoint(kwargs.get('endpoint', None))
    prompt = kwargs.get('prompt', None)
    # Apply the custom scrubber
    text = my_custom_scrubber(prompt)
    kwargs['prompt'] = text
    # Call the original method, which will apply the default scrubbers
    return original_method(**kwargs)

# Assign the new method to the wrapper instance
wrapper.send_cohere_request = types.MethodType(modified_method, wrapper)
```

## Step 6: Test the Enhanced Gateway

Now, send the same example prompt through the modified wrapper.

```python
response, db_record = wrapper.send_cohere_request(
    endpoint="generate",
    model="command-r-plus",
    max_tokens=25,
    prompt=f"{example}\n\nSummarize the above text in 1-2 sentences.",
    temperature=0.3,
)

print("LLM Response with Custom Scrubber:")
print(response)
print("\nDatabase Record with Custom Scrubber:")
print(db_record)
```

You'll see that the name is now also redacted in both the sent prompt and the model's response.
```
'user_input': '[REDACTED PERSON] ([REDACTED EMAIL ADDRESS], (+1) [REDACTED PHONE NUMBER]) committed a mistake when he used PyTorch Trainer instead of HF Trainer.\n\nSummarize the above text in 1-2 sentences.',
```

> **Important:** When adding custom scrubbers, remember they are applied sequentially. Ensure your scrubber doesn't interfere with the default ones. For robust, general-purpose name scrubbing, consider integrating dedicated libraries like [scrubadub](https://github.com/kylemclaren/scrub), which can handle a wider range of PII (IP addresses, hostnames, etc.).

## Step 7: Applying the Gateway to a Real Dataset

Let's test the wrapper on a larger scale using the AI4Privacy PII dataset.

First, load the dataset.

```python
pii_ds = load_dataset("ai4privacy/pii-masking-200k")
```

Select an example from the training set that contains PII.

```python
example = pii_ds['train'][36]['source_text']
print("Original Text:")
print(example)
```

The text contains an email address:
```
"I need the latest update on assessment results. Please send the files to Valentine4@gmail.com. For your extra time, we'll offer you Kip 100,000 but please provide your лв account details."
```

Now, send this text through the gateway for summarization.

```python
response, db_record = wrapper.send_cohere_request(
    endpoint="generate",
    model="command-r-plus",
    max_tokens=50,
    prompt=f"{example}\n\nSummarize the above text in 1-2 sentences.",
    temperature=0.3,
)

print("\nScrubbed LLM Response:")
print(response)
print("\nScrubbed Database Record:")
print(db_record)
```

The database record confirms the email address was redacted before being sent to the API:
```
'user_input': "I need the latest update on assessment results. Please send the files to V[REDACTED EMAIL ADDRESS]. For your extra time, we'll offer you Kip 100,000 but please provide your лв account details.\n\nSummarize the above text in 1-2 sentences.",
```

## Step 8: Comparing with a Non-Scrubbed Request

To understand the value of the gateway, let's see what would happen if we sent the raw PII data directly to the Cohere API.

```python
co = cohere.Client(api_key=os.environ['COHERE_API_KEY'])

response_vanilla = co.generate(
    prompt=f"{example}\n\nSummarize the above text in 1-2 sentences.",
    model="command-r-plus",
    max_tokens=50,
    temperature=0.3
)

print("Direct API Response (Contains PII):")
print(response_vanilla)
```

The direct response includes the unredacted email address:
```
The text is a request for an update on assessment results to be sent to Valentine4@gmail.com, with an offer of Kip 100,000 in exchange for the information and account details.
```

This highlights the risk of leaking sensitive information without a scrubbing layer.

## Conclusion

In this tutorial, you learned how to:

1.  Set up and use the Wealthsimple LLM Gateway for PII scrubbing.
2.  Understand the default scrubbers for phone numbers, emails, credit cards, postal codes, and SINs.
3.  Extend the gateway by implementing a custom scrubber for specific terms.
4.  Apply the gateway to real-world data from a PII dataset.
5.  Verify the scrubbing effectiveness by comparing gateway requests with direct API calls.

**Key Takeaway:** Always validate that your chosen scrubbing algorithms cover your specific use cases. For the highest level of security and privacy, consider deploying open-source models on your own infrastructure whenever possible. The gateway pattern demonstrated here provides a crucial safety layer when using external LLM APIs.
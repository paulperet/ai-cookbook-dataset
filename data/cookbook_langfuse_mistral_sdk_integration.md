# Mistral AI SDK Integration with Langfuse

This guide provides step-by-step examples of integrating Langfuse with the Mistral AI SDK (v1) in Python. By following these examples, you'll learn how to seamlessly log and trace interactions with Mistral's language models, enhancing the transparency, debuggability, and performance monitoring of your AI-driven applications.

> **Note:** Langfuse is also natively integrated with [LangChain](https://langfuse.com/docs/integrations/langchain/tracing), [LlamaIndex](https://langfuse.com/docs/integrations/llama-index/get-started), [LiteLLM](https://langfuse.com/docs/integrations/litellm/tracing), and [other frameworks](https://langfuse.com/docs/integrations/overview). If you use one of them, any use of Mistral models is instrumented right away.

## What is Langfuse?

[Langfuse](https://langfuse.com/) is an open-source LLM engineering platform. It includes features such as [traces](https://langfuse.com/docs/tracing), [evals](https://langfuse.com/docs/scores/overview), and [prompt management](https://langfuse.com/docs/prompts/get-started) to help you debug and improve your LLM app.

## Overview

In this tutorial, we will explore various use cases where Langfuse can be integrated with Mistral AI SDK, including:

- **Basic LLM Calls:** Learn how to wrap standard Mistral model interactions with Langfuse's `@observe` decorator for comprehensive logging.
- **Chained Function Calls:** See how to manage and observe complex workflows where multiple model interactions are linked together to produce a final result.
- **Async and Streaming Support:** Discover how to use Langfuse with asynchronous and streaming responses from Mistral models, ensuring that real-time and concurrent interactions are fully traceable.
- **Function Calling:** Understand how to implement and observe external tool integrations with Mistral, allowing the model to interact with custom functions and APIs.

For more detailed guidance on the Mistral SDK or the `@observe` decorator from Langfuse, please refer to the [Mistral SDK repo](https://github.com/mistralai/client-python) and the [Langfuse Documentation](https://langfuse.com/docs/sdk/python/decorators#log-any-llm-call).

## Prerequisites

Before you begin, ensure you have the following:

1.  **Langfuse Account:** [Sign up](https://cloud.langfuse.com/auth/sign-up) for Langfuse if you haven't already. Copy your [API keys](https://langfuse.com/faq/all/where-are-langfuse-api-keys) from your project settings.
2.  **Mistral Account:** [Sign up for a Mistral account](https://console.mistral.ai/), [subscribe](https://console.mistral.ai/billing/) to a free trial or billing plan, and [generate an API key](https://console.mistral.ai/api-keys/).

## Setup

First, install the required packages and set up your environment.

```bash
pip install mistralai langfuse
```

Now, configure your environment variables with the necessary API keys.

```python
import os

# Get keys for your project from https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-xxx"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-xxx"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region

# Your Mistral key
os.environ["MISTRAL_API_KEY"] = "xxx"
```

Initialize the Mistral client.

```python
from mistralai import Mistral

# Initialize Mistral client
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

## Tutorial: Integrating Mistral AI with Langfuse

### Step 1: Basic LLM Completions

We integrate the Mistral AI SDK with Langfuse using the [`@observe` decorator](https://langfuse.com/docs/sdk/python/decorators). The `@observe(as_type="generation")` decorator specifically logs LLM interactions, capturing inputs, outputs, and model parameters.

Let's create a wrapper function for Mistral completions.

```python
from langfuse.decorators import langfuse_context, observe

# Function to handle Mistral completion calls, wrapped with @observe to log the LLM interaction
@observe(as_type="generation")
def mistral_completion(**kwargs):
    # Clone kwargs to avoid modifying the original input
    kwargs_clone = kwargs.copy()

    # Extract relevant parameters from kwargs
    input = kwargs_clone.pop('messages', None)
    model = kwargs_clone.pop('model', None)
    min_tokens = kwargs_clone.pop('min_tokens', None)
    max_tokens = kwargs_clone.pop('max_tokens', None)
    temperature = kwargs_clone.pop('temperature', None)
    top_p = kwargs_clone.pop('top_p', None)

    # Filter and prepare model parameters for logging
    model_parameters = {
        "maxTokens": max_tokens,
        "minTokens": min_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    model_parameters = {k: v for k, v in model_parameters.items() if v is not None}

    # Log the input and model parameters before calling the LLM
    langfuse_context.update_current_observation(
        input=input,
        model=model,
        model_parameters=model_parameters,
        metadata=kwargs_clone,
    )

    # Call the Mistral model to generate a response
    res = mistral_client.chat.complete(**kwargs)

    # Log the usage details and output content after the LLM call
    langfuse_context.update_current_observation(
        usage={
            "input": res.usage.prompt_tokens,
            "output": res.usage.completion_tokens
        },
        output=res.choices[0].message.content
    )

    # Return the model's response object
    return res
```

Now, let's use this wrapper in a simple example. We'll also decorate the top-level function to create a hierarchical trace.

```python
@observe()
def find_best_painter_from(country="France"):
    response = mistral_completion(
        model="mistral-small-latest",
        max_tokens=1024,
        temperature=0.4,
        messages=[
            {
                "content": f"Who is the best painter from {country}? Answer in one short sentence.",
                "role": "user",
            },
        ]
    )
    return response.choices[0].message.content

# Execute the function
result = find_best_painter_from()
print(result)
```

**Output:**
```
Claude Monet, renowned for his role as a founder of French Impressionist painting, is often considered one of the best painters from France.
```

This hierarchical setup helps trace more complex applications involving multiple LLM calls and other non-LLM methods decorated with `@observe`.

### Step 2: Chained Completions

This example demonstrates chaining multiple LLM calls. The first call identifies the best painter from a specified country, and the second call uses that painter's name to find their most famous painting.

```python
@observe()
def find_best_painting_from(country="France"):
    # First call: Get the painter's name
    response = mistral_completion(
        model="mistral-small-latest",
        max_tokens=1024,
        temperature=0.1,
        messages=[
            {
                "content": f"Who is the best painter from {country}? Only provide the name.",
                "role": "user",
            },
        ]
    )
    painter_name = response.choices[0].message.content

    # Second call: Get the painter's most famous painting
    return mistral_completion(
        model="mistral-small-latest",
        max_tokens=1024,
        messages=[
            {
                "content": f"What is the most famous painting of {painter_name}? Answer in one short sentence.",
                "role": "user",
            },
        ]
    )

# Execute the function
result = find_best_painting_from("Germany")
print(result.choices[0].message.content)
```

**Output:**
```
Albrecht Dürer's most famous painting is "Self-Portrait at Twenty-Eight."
```

### Step 3: Streaming Completions

This example demonstrates how to handle streaming responses from the Mistral model. The process is similar to the standard completion example but includes handling streamed data in real-time.

```python
# Wrap streaming function with decorator
@observe(as_type="generation")
def stream_mistral_completion(**kwargs):
    kwargs_clone = kwargs.copy()
    input = kwargs_clone.pop('messages', None)
    model = kwargs_clone.pop('model', None)
    min_tokens = kwargs_clone.pop('min_tokens', None)
    max_tokens = kwargs_clone.pop('max_tokens', None)
    temperature = kwargs_clone.pop('temperature', None)
    top_p = kwargs_clone.pop('top_p', None)

    model_parameters = {
        "maxTokens": max_tokens,
        "minTokens": min_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    model_parameters = {k: v for k, v in model_parameters.items() if v is not None}

    langfuse_context.update_current_observation(
        input=input,
        model=model,
        model_parameters=model_parameters,
        metadata=kwargs_clone,
    )

    res = mistral_client.chat.stream(**kwargs)
    final_response = ""
    for chunk in res:
        content = chunk.data.choices[0].delta.content
        final_response += content
        yield content

        if chunk.data.choices[0].finish_reason == "stop":
            langfuse_context.update_current_observation(
                usage={
                    "input": chunk.data.usage.prompt_tokens,
                    "output": chunk.data.usage.completion_tokens
                },
                output=final_response
            )
            break

# Use stream_mistral_completion as you'd usually use the SDK
@observe()
def stream_find_best_five_painter_from(country="France"):
    response_chunks = stream_mistral_completion(
        model="mistral-small-latest",
        max_tokens=1024,
        messages=[
            {
                "content": f"Who are the best five painter from {country}? Answer in one short sentence.",
                "role": "user",
            },
        ]
    )
    final_response = ""
    for chunk in response_chunks:
        final_response += chunk
        # You can also do something with each chunk here if needed
        print(chunk, end="")

    return final_response

# Execute the function
result = stream_find_best_five_painter_from("Spain")
print(f"\nFinal response: {result}")
```

### Step 4: Async Completion

This example showcases the use of the `@observe` decorator in an asynchronous context, allowing for non-blocking LLM calls.

```python
# Wrap async function with decorator
@observe(as_type="generation")
async def async_mistral_completion(**kwargs):
    kwargs_clone = kwargs.copy()
    input = kwargs_clone.pop('messages', None)
    model = kwargs_clone.pop('model', None)
    min_tokens = kwargs_clone.pop('min_tokens', None)
    max_tokens = kwargs_clone.pop('max_tokens', None)
    temperature = kwargs_clone.pop('temperature', None)
    top_p = kwargs_clone.pop('top_p', None)

    model_parameters = {
        "maxTokens": max_tokens,
        "minTokens": min_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    model_parameters = {k: v for k, v in model_parameters.items() if v is not None}

    langfuse_context.update_current_observation(
        input=input,
        model=model,
        model_parameters=model_parameters,
        metadata=kwargs_clone,
    )

    res = await mistral_client.chat.complete_async(**kwargs)

    langfuse_context.update_current_observation(
        usage={
            "input": res.usage.prompt_tokens,
            "output": res.usage.completion_tokens
        },
        output=res.choices[0].message.content
    )

    return res

@observe()
async def async_find_best_musician_from(country="France"):
    response = await async_mistral_completion(
        model="mistral-small-latest",
        max_tokens=1024,
        messages=[
            {
                "content": f"Who is the best musician from {country}? Answer in one short sentence.",
                "role": "user",
            },
        ]
    )
    return response

# Execute the async function
import asyncio
result = asyncio.run(async_find_best_musician_from("Spain"))
print(result.choices[0].message.content)
```

**Output:**
```
One of the most renowned musicians from Spain is Andrés Segovia, a classical guitarist who significantly impacted the instrument's modern repertoire.
```

### Step 5: Async Streaming

This example demonstrates the use of the `@observe` decorator in an asynchronous streaming context.

```python
import asyncio

# Wrap async streaming function with decorator
@observe(as_type="generation")
async def async_stream_mistral_completion(**kwargs):
    kwargs_clone = kwargs.copy()
    input = kwargs_clone.pop('messages', None)
    model = kwargs_clone.pop('model', None)
    min_tokens = kwargs_clone.pop('min_tokens', None)
    max_tokens = kwargs_clone.pop('max_tokens', None)
    temperature = kwargs_clone.pop('temperature', None)
    top_p = kwargs_clone.pop('top_p', None)

    model_parameters = {
        "maxTokens": max_tokens,
        "minTokens": min_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    model_parameters = {k: v for k, v in model_parameters.items() if v is not None}

    langfuse_context.update_current_observation(
        input=input,
        model=model,
        model_parameters=model_parameters,
        metadata=kwargs_clone,
    )

    res = await mistral_client.chat.stream_async(**kwargs)
    final_response = ""
    async for chunk in res:
        content = chunk.data.choices[0].delta.content
        final_response += content
        yield content

        if chunk.data.choices[0].finish_reason == "stop":
            langfuse_context.update_current_observation(
                usage={
                    "input": chunk.data.usage.prompt_tokens,
                    "output": chunk.data.usage.completion_tokens
                },
                output=final_response
            )
            break

# Example usage of the async streaming function
@observe()
async def async_stream_example():
    async for chunk in async_stream_mistral_completion(
        model="mistral-small-latest",
        max_tokens=1024,
        messages=[
            {
                "content": "Tell me a short joke.",
                "role": "user",
            },
        ]
    ):
        print(chunk, end="")

# Execute the async streaming example
asyncio.run(async_stream_example())
```

## Conclusion

You've now learned how to integrate Langfuse with the Mistral AI SDK for comprehensive logging and tracing of LLM interactions. This integration enables you to:

- Log all Mistral model calls with detailed inputs, outputs, and parameters
- Trace complex workflows involving multiple chained LLM calls
- Monitor streaming and asynchronous interactions in real-time
- Debug and optimize your AI applications with full visibility

All traces are available in your Langfuse dashboard, where you can analyze performance, debug issues, and improve your prompts.

For more advanced features like function calling, evaluations, and prompt management, refer to the [Langfuse documentation](https://langfuse.com/docs).
<center>
    <p style="text-align:center">
        <br>
        <a href="https://langfuse.com/docs">Docs</a>
        |
        <a href="https://github.com/langfuse/langfuse">GitHub</a>
        |
        <a href="https://discord.langfuse.com/">Discord</a>
    </p>
</center>
<h1 align="center">Mistral AI SDK Integration with Langfuse</h1>

This cookbook provides step-by-step examples of integrating Langfuse with the Mistral AI SDK (v1) in Python. By following these examples, you'll learn how to seamlessly log and trace interactions with Mistral's language models, enhancing the transparency, debuggability, and performance monitoring of your AI-driven applications.

---

Note: Langfuse is also natively integrated with [LangChain](https://langfuse.com/docs/integrations/langchain/tracing), [LlamaIndex](https://langfuse.com/docs/integrations/llama-index/get-started), [LiteLLM](https://langfuse.com/docs/integrations/litellm/tracing), and [other frameworks](https://langfuse.com/docs/integrations/overview). If you use one of them, any use of Mistral models is instrumented right away.

---

## Overview

In this notebook, we will explore various use cases where Langfuse can be integrated with Mistral AI SDK, including:

- **Basic LLM Calls:** Learn how to wrap standard Mistral model interactions with Langfuse's @observe decorator for comprehensive logging.
- **Chained Function Calls:** See how to manage and observe complex workflows where multiple model interactions are linked together to produce a final result.
- **Async and Streaming Support:** Discover how to use Langfuse with asynchronous and streaming responses from Mistral models, ensuring that real-time and concurrent interactions are fully traceable.
- **Function Calling:** Understand how to implement and observe external tool integrations with Mistral, allowing the model to interact with custom functions and APIs.

For more detailed guidance on the Mistral SDK or the **@observe** decorator from Langfuse, please refer to the [Mistral SDK repo](https://github.com/mistralai/client-python) and the [Langfuse Documentation](https://langfuse.com/docs/sdk/python/decorators#log-any-llm-call).

## What is Langfuse?

[Langfuse](https://langfuse.com/) is an open-source LLM engineering platform. It includes features such as [traces](https://langfuse.com/docs/tracing), [evals](https://langfuse.com/docs/scores/overview), and [prompt management](https://langfuse.com/docs/prompts/get-started) to help you debug and improve your LLM app.

## Setup

[Sign up](https://cloud.langfuse.com/auth/sign-up) for Langfuse if you haven't already. Copy your [API keys](https://langfuse.com/faq/all/where-are-langfuse-api-keys) from your project settings and add them to your environment.


```python
%pip install mistralai langfuse
```


```python
import os

# get keys for your project from https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-xxx"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-xxx"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region

# Your Mistral key
os.environ["MISTRAL_API_KEY"] = "xxx"
```

Set your Mistral API key as an environment variable. If you haven't already, [sign up for a Mistral acccount](https://console.mistral.ai/). Then [subscribe](https://console.mistral.ai/billing/) to a free trial or billing plan, after which you'll be able to [generate an API key](https://console.mistral.ai/api-keys/).


```python
from mistralai import Mistral

# Initialize Mistral client
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
```

## Examples

### 1. Completions

We are integrating the Mistral AI SDK with Langfuse using the [@observe decorator](https://langfuse.com/docs/sdk/python/decorators), which is crucial for logging and tracing interactions with large language models (LLMs). The @observe(as_type="generation") decorator specifically logs LLM interactions, capturing inputs, outputs, and model parameters. The resulting `mistral_completion` method can then be used across your project.


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

Optionally, other functions (api handlers, retrieval functions, ...) can be also decorated.

#### 1.1 Simple Example

In the following example, we also added the decorator to the top-level function `find_best_painter_from`. This function calls the mistral_completion function, which is decorated with @observe(as_type="generation"). This hierarchical setup hels to trace more complex applications which involve multiple LLM calls and other non-llm methods which are decorated with @observe.

You can use langfuse_context.update_current_observation or langfuse_context.update_current_trace to add additional details such as input, output, and model parameters to the trace.


```python
@observe()
def find_best_painter_from(country="France"):
  response = mistral_completion(
      model="mistral-small-latest",
      max_tokens=1024,
      temperature=0.4,
      messages=[
        {
            "content": "Who is the best painter from {country}? Answer in one short sentence.".format(country=country),
            "role": "user",
        },
      ]
    )
  return response.choices[0].message.content

find_best_painter_from()
```




    'Claude Monet, renowned for his role as a founder of French Impressionist painting, is often considered one of the best painters from France.'



Example trace in Langfuse: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/836a9585-cfcc-47f7-881f-85ebdd9f601b

#### 1.2 Chained Completions


This example demonstrates chaining multiple LLM calls using the @observe decorator. The first call identifies the best painter from a specified country, and the second call uses that painter's name to find their most famous painting. Both interactions are logged by Langfuse as we use the wrapped `mistral_completion` method created above, ensuring full traceability across the chained requests.


```python
@observe()
def find_best_painting_from(country="France"):
  response = mistral_completion(
      model="mistral-small-latest",
      max_tokens=1024,
      temperature=0.1,
      messages=[
        {
            "content": "Who is the best painter from {country}? Only provide the name.".format(country=country),
            "role": "user",
        },
      ]
    )
  painter_name = response.choices[0].message.content
  return mistral_completion(
      model="mistral-small-latest",
      max_tokens=1024,
      messages=[
        {
            "content": "What is the most famous painting of {painter_name}? Answer in one short sentence.".format(painter_name=painter_name),
            "role": "user",
        },
      ]
    )

find_best_painting_from("Germany")
```




    ChatCompletionResponse(id='8bb8512749fd4ddf88720aec0021378c', object='chat.completion', model='mistral-small-latest', usage=UsageInfo(prompt_tokens=23, completion_tokens=23, total_tokens=46), created=1726597735, choices=[ChatCompletionChoice(index=0, message=AssistantMessage(content='Albrecht Dürer\'s most famous painting is "Self-Portrait at Twenty-Eight."', tool_calls=None, prefix=False, role='assistant'), finish_reason='stop')])



Example trace in Langfuse: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/a3360c6f-24ad-455c-aae7-eb9d5c6f5dac

### 2. Streaming Completions

The following example demonstrates how to handle streaming responses from the Mistral model using the @observe(as_type="generation") decorator. The process is similar to the *Completion* example but includes handling streamed data in real-time.

Just like in the previous example, we wrap the streaming function with the @observe decorator to capture the input, model parameters, and usage details. Additionally, the function processes the streamed output incrementally, updating the Langfuse context as each chunk is received.


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
                "content": "Who are the best five painter from {country}? Answer in one short sentence.".format(country=country),
                "role": "user",
            },
        ]
    )
    final_response = ""
    for chunk in response_chunks:
        final_response += chunk
        # You can also do something with each chunk here if needed
        print(chunk)

    return final_response

stream_find_best_five_painter_from("Spain")
```

    
    The
     best
     five
     pain
    ters
     from
     Spain
     are
     Diego
     Vel
    áz
    que
    z
    ,
     Francisco
     G
    oya
    ,
     P
    ablo
     Pic
    asso
    ,
     Salvador
     Dal
    í
    ,
     and
     Joan
     Mir
    ó
    .
    





    'The best five painters from Spain are Diego Velázquez, Francisco Goya, Pablo Picasso, Salvador Dalí, and Joan Miró.'



Example trace in Langfuse: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/75a2a4fe-088d-4134-9797-ba9c21be01b2

### 3. Async Completion


This example showcases the use of the @observe decorator in an asynchronous context. It wraps an async function that interacts with the Mistral model, ensuring that both the request and the response are logged by Langfuse. The async function allows for non-blocking LLM calls, making it suitable for applications that require concurrency while maintaining full observability of the interactions.


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
            "content": "Who is the best musician from {country}? Answer in one short sentence.".format(country=country),
            "role": "user",
        },
      ]
    )
  return response

await async_find_best_musician_from("Spain")
```




    ChatCompletionResponse(id='589fa6216c5346cc984586209c693a41', object='chat.completion', model='mistral-small-latest', usage=UsageInfo(prompt_tokens=17, completion_tokens=33, total_tokens=50), created=1726597737, choices=[ChatCompletionChoice(index=0, message=AssistantMessage(content="One of the most renowned musicians from Spain is Andrés Segovia, a classical guitarist who significantly impacted the instrument's modern repertoire.", tool_calls=None, prefix=False, role='assistant'), finish_reason='stop')])



Example trace in Langfuse: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/1f7d91ce-45dd-41bf-8e6f-1875086ed32f

### 4. Async Streaming

This example demonstrates the use of the @observe decorator in an asynchronous streaming context. It wraps an async function that streams responses from the Mistral model, logging each chunk of data in real-time.


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

        if chunk.data.choices[0].finish
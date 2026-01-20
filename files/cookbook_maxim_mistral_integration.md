<a href="https://colab.research.google.com/drive/1gqJkkLQQTtSdUla2oiIg-QyR6i5BpmvV?usp=sharing" target="_parent">

<h1 align="center">Observability with Mistral AI and Maxim</h1>

In this cookbook, we show you how to use [Maxim](https://www.getmaxim.ai/), to observe Mistral LLM calls & metrics.

## What is Maxim?

Maxim AI provides comprehensive observability for your Mistral based AI applications. With Maxim's one-line integration, you can easily trace and analyse LLM calls, metrics, and more.

**Pros:**

* Performance Analytics: Track latency, tokens consumed, and costs
* Advanced Visualisation: Understand agent trajectories through intuitive dashboards


## Install and Import Required Modules
You need to install `mistralai` and `maxim-py` packages from [pypy](https://pypy.org/)


```python
!pip install mistralai maxim-py
```

    [Collecting mistralai, Downloading mistralai-1.8.1-py3-none-any.whl.metadata (33 kB), ..., Successfully installed eval-type-backport-0.2.2 filetype-1.2.0 maxim-py-3.8.1 mistralai-1.8.1]


## Set the environment variables
You can sign up on [Maxim](https://getmaxim.ai/signup) and create a new Api Key from Settings. After that go to Logs section and create a new Log Repository, you will receive a Log Repository Id. Get ready with your Mistral Api Key also.


```python
import os
from google.colab import userdata

MAXIM_API_KEY=userdata.get("MAXIM_API_KEY")
MAXIM_LOG_REPO_ID=userdata.get("MAXIM_REPO_ID")
MISTRAL_API_KEY=userdata.get("MISTRAL_API_KEY")

os.environ["MAXIM_API_KEY"] = MAXIM_API_KEY
os.environ["MAXIM_LOG_REPO_ID"] = MAXIM_LOG_REPO_ID
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
```

## Initialize logger

Create an instance of Maxim Logger


```python
from maxim import Maxim

logger = Maxim().logger()
```

    INFO:maxim:[MaximSDK] Starting flush thread with interval {10} seconds


    [MaximSDK] Initializing Maxim AI(v3.8.1)
    [MaximSDK] Using info logging level.
    [MaximSDK] For debug logs, set global logging level to debug logging.basicConfig(level=logging.DEBUG).


## Make LLM calls using MaximMistralClient

Make a call to Mistral via Mistral Api Client provided by Maxim, define the model you want to use and list of messages.


```python
from mistralai import Mistral
from maxim.logger.mistral import MaximMistralClient
import os

with MaximMistralClient(Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
), logger) as mistral:

    res = mistral.chat.complete(
        model="mistral-medium-latest",
        messages=[
            {
                "content": "Who is the best French painter? Answer in one short sentence.",
                "role": "user",
            },
        ]
    )

    # Handle response
    print(res)
```

    id='e161b1cd452042549d5292b6a60f6b83' object='chat.completion' model='mistral-medium-latest' usage=UsageInfo(prompt_tokens=16, completion_tokens=23, total_tokens=39) created=1749107242 choices=[ChatCompletionChoice(index=0, message=AssistantMessage(content='Claude Monet is often regarded as the best French painter, renowned for his pioneering role in Impressionism.', tool_calls=None, prefix=False, role='assistant'), finish_reason='stop')]


To check the logs shared by Mistral SDK with Maxim -
1. Go to Logs section in Maxim Platform
2. Go to the respective Log Repository you created.
3. Switch to `Logs` from top tab view and analyse the traces received


## Async LLM call


```python
async with MaximMistralClient(Mistral(
    api_key=os.getenv('MISTRAL_API_KEY', ''),
), logger) as mistral:

    response = await mistral.chat.complete_async(
        model='mistral-small-latest',
        messages=[
            {
                'role': 'user',
                'content': 'Explain the difference between async and sync programming in Python in one sentence.'
            }
        ]
    )
    print(response)
```

    id='ef228964167649278f1eeecfe6d985d4' object='chat.completion' model='mistral-small-latest' usage=UsageInfo(prompt_tokens=18, completion_tokens=34, total_tokens=52) created=1749106669 choices=[ChatCompletionChoice(index=0, message=AssistantMessage(content='Async programming in Python allows for concurrent execution of tasks using `async` and `await` keywords, while sync programming executes tasks sequentially, blocking until each task completes.', tool_calls=None, prefix=False, role='assistant'), finish_reason='stop')]


# Feedback
---

If you have any feedback or requests, please create a GitHub [Issue](https://github.com/maximhq/maxim-docs).
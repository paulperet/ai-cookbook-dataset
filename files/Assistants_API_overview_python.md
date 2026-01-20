# Assistants API Overview (Python SDK)

The new [Assistants API](https://platform.openai.com/docs/assistants/overview) is a stateful evolution of our [Chat Completions API](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) meant to simplify the creation of assistant-like experiences, and enable developer access to powerful tools like Code Interpreter and File Search.

## Chat Completions API vs Assistants API

The primitives of the **Chat Completions API** are `Messages`, on which you perform a `Completion` with a `Model` (`gpt-4o`, `gpt-4o-mini`, etc). It is lightweight and powerful, but inherently stateless, which means you have to manage conversation state, tool definitions, retrieval documents, and code execution manually.

The primitives of the **Assistants API** are

- `Assistants`, which encapsulate a base model, instructions, tools, and (context) documents,
- `Threads`, which represent the state of a conversation, and
- `Runs`, which power the execution of an `Assistant` on a `Thread`, including textual responses and multi-step tool use.

We'll take a look at how these can be used to create powerful, stateful experiences.


## Setup

### Python SDK

> **Note**
> We've updated our [Python SDK](https://github.com/openai/openai-python) to add support for the Assistants API, so you'll need to update it to the latest version (`1.59.4` at time of writing).



```python
!pip install --upgrade openai
```

[Loss: 0.9, ..., Loss: 0.1]

And make sure it's up to date by running:



```python
!pip show openai | grep Version
```

    Version: 1.59.4


### Pretty Printing Helper



```python
import json

def show_json(obj):
    display(json.loads(obj.model_dump_json()))
```

## Complete Example with Assistants API


### Assistants


The easiest way to get started with the Assistants API is through the [Assistants Playground](https://platform.openai.com/playground).


Let's begin by creating an assistant! We'll create a Math Tutor just like in our [docs](https://platform.openai.com/docs/assistants/overview).


You can also create Assistants directly through the Assistants API, like so:



```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))


assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    model="gpt-4o",
)
show_json(assistant)
```


    {'id': 'asst_qvXmYlZV8zhABI2RtPzDfV6z',
     'created_at': 1736340398,
     'description': None,
     'instructions': 'You are a personal math tutor. Answer questions briefly, in a sentence or less.',
     'metadata': {},
     'model': 'gpt-4o',
     'name': 'Math Tutor',
     'object': 'assistant',
     'tools': [],
     'response_format': 'auto',
     'temperature': 1.0,
     'tool_resources': {'code_interpreter': None, 'file_search': None},
     'top_p': 1.0}


Regardless of whether you create your Assistant through the Dashboard or with the API, you'll want to keep track of the Assistant ID. This is how you'll refer to your Assistant throughout Threads and Runs.


Next, we'll create a new Thread and add a Message to it. This will hold the state of our conversation, so we don't have re-send the entire message history each time.


### Threads


Create a new thread:



```python
thread = client.beta.threads.create()
show_json(thread)
```


    {'id': 'thread_j4dc1TiHPfkviKUHNi4aAsA6',
     'created_at': 1736340398,
     'metadata': {},
     'object': 'thread',
     'tool_resources': {'code_interpreter': None, 'file_search': None}}


Then add the Message to the thread:



```python
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
)
show_json(message)
```


    {'id': 'msg_1q4Y7ZZ9gIcPoAKSx9UtrrKJ',
     'assistant_id': None,
     'attachments': [],
     'completed_at': None,
     'content': [{'text': {'annotations': [],
        'value': 'I need to solve the equation `3x + 11 = 14`. Can you help me?'},
       'type': 'text'}],
     'created_at': 1736340400,
     'incomplete_at': None,
     'incomplete_details': None,
     'metadata': {},
     'object': 'thread.message',
     'role': 'user',
     'run_id': None,
     'status': None,
     'thread_id': 'thread_j4dc1TiHPfkviKUHNi4aAsA6'}


> **Note**
> Even though you're no longer sending the entire history each time, you will still be charged for the tokens of the entire conversation history with each Run.


### Runs

Notice how the Thread we created is **not** associated with the Assistant we created earlier! Threads exist independently from Assistants, which may be different from what you'd expect if you've used ChatGPT (where a thread is tied to a model/GPT).

To get a completion from an Assistant for a given Thread, we must create a Run. Creating a Run will indicate to an Assistant it should look at the messages in the Thread and take action: either by adding a single response, or using tools.

> **Note**
> Runs are a key difference between the Assistants API and Chat Completions API. While in Chat Completions the model will only ever respond with a single message, in the Assistants API a Run may result in an Assistant using one or multiple tools, and potentially adding multiple messages to the Thread.

To get our Assistant to respond to the user, let's create the Run. As mentioned earlier, you must specify _both_ the Assistant and the Thread.



```python
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
show_json(run)
```


    {'id': 'run_qVYsWok6OCjHxkajpIrdHuVP',
     'assistant_id': 'asst_qvXmYlZV8zhABI2RtPzDfV6z',
     'cancelled_at': None,
     'completed_at': None,
     'created_at': 1736340403,
     'expires_at': 1736341003,
     'failed_at': None,
     'incomplete_details': None,
     'instructions': 'You are a personal math tutor. Answer questions briefly, in a sentence or less.',
     'last_error': None,
     'max_completion_tokens': None,
     'max_prompt_tokens': None,
     'metadata': {},
     'model': 'gpt-4o',
     'object': 'thread.run',
     'parallel_tool_calls': True,
     'required_action': None,
     'response_format': 'auto',
     'started_at': None,
     'status': 'queued',
     'thread_id': 'thread_j4dc1TiHPfkviKUHNi4aAsA6',
     'tool_choice': 'auto',
     'tools': [],
     'truncation_strategy': {'type': 'auto', 'last_messages': None},
     'usage': None,
     'temperature': 1.0,
     'top_p': 1.0,
     'tool_resources': {}}


Unlike creating a completion in the Chat Completions API, **creating a Run is an asynchronous operation**. It will return immediately with the Run's metadata, which includes a `status` that will initially be set to `queued`. The `status` will be updated as the Assistant performs operations (like using tools and adding messages).

To know when the Assistant has completed processing, we can poll the Run in a loop. (Support for streaming is coming soon!) While here we are only checking for a `queued` or `in_progress` status, in practice a Run may undergo a [variety of status changes](https://platform.openai.com/docs/api-reference/runs/object#runs/object-status) which you can choose to surface to the user. (These are called Steps, and will be covered later.)



```python
import time

def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run
```


```python
run = wait_on_run(run, thread)
show_json(run)
```


    {'id': 'run_qVYsWok6OCjHxkajpIrdHuVP',
     'assistant_id': 'asst_qvXmYlZV8zhABI2RtPzDfV6z',
     'cancelled_at': None,
     'completed_at': 1736340406,
     'created_at': 1736340403,
     'expires_at': None,
     'failed_at': None,
     'incomplete_details': None,
     'instructions': 'You are a personal math tutor. Answer questions briefly, in a sentence or less.',
     'last_error': None,
     'max_completion_tokens': None,
     'max_prompt_tokens': None,
     'metadata': {},
     'model': 'gpt-4o',
     'object': 'thread.run',
     'parallel_tool_calls': True,
     'required_action': None,
     'response_format': 'auto',
     'started_at': 1736340405,
     'status': 'completed',
     'thread_id': 'thread_j4dc1TiHPfkviKUHNi4aAsA6',
     'tool_choice': 'auto',
     'tools': [],
     'truncation_strategy': {'type': 'auto', 'last_messages': None},
     'usage': {'completion_tokens': 35,
      'prompt_tokens': 66,
      'total_tokens': 101,
      'prompt_token_details': {'cached_tokens': 0},
      'completion_tokens_details': {'reasoning_tokens': 0}},
     'temperature': 1.0,
     'top_p': 1.0,
     'tool_resources': {}}


### Messages


Now that the Run has completed, we can list the Messages in the Thread to see what got added by the Assistant.



```python
messages = client.beta.threads.messages.list(thread_id=thread.id)
show_json(messages)
```


    {'data': [{'id': 'msg_A5eAN6ZAJDmFBOYutEm5DFCy',
       'assistant_id': 'asst_qvXmYlZV8zhABI2RtPzDfV6z',
       'attachments': [],
       'completed_at': None,
       'content': [{'text': {'annotations': [],
          'value': 'Sure! Subtract 11 from both sides to get \\(3x = 3\\), then divide by 3 to find \\(x = 1\\).'},
         'type': 'text'}],
       'created_at': 1736340405,
       'incomplete_at': None,
       'incomplete_details': None,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_qVYsWok6OCjHxkajpIrdHuVP',
       'status': None,
       'thread_id': 'thread_j4dc1TiHPfkviKUHNi4aAsA6'},
      {'id': 'msg_1q4Y7ZZ9gIcPoAKSx9UtrrKJ',
       'assistant_id': None,
       'attachments': [],
       'completed_at': None,
       'content': [{'text': {'annotations': [],
          'value': 'I need to solve the equation `3x + 11 = 14`. Can you help me?'},
         'type': 'text'}],
       'created_at': 1736340400,
       'incomplete_at': None,
       'incomplete_details': None,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'status': None,
       'thread_id': 'thread_j4dc1TiHPfkviKUHNi4aAsA6'}],
     'object': 'list',
     'first_id': 'msg_A5eAN6ZAJDmFBOYutEm5DFCy',
     'last_id': 'msg_1q4Y7ZZ9gIcPoAKSx9UtrrKJ',
     'has_more': False}


As you can see, Messages are ordered in reverse-chronological order – this was done so the most recent results are always on the first `page` (since results can be paginated). Do keep a look out for this, since this is the opposite order to messages in the Chat Completions API.


Let's ask our Assistant to explain the result a bit further!



```python
# Create a message to append to our thread
message = client.beta.threads.messages.create(
    thread_id=thread.id, role="user", content="Could you explain this to me?"
)

# Execute our run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# Wait for completion
wait_on_run(run, thread)

# Retrieve all the messages added after our last user message
messages = client.beta.threads.messages.list(
    thread_id=thread.id, order="asc", after=message.id
)
show_json(messages)
```


    {'data': [{'id': 'msg_wSHHvaMnaWktZWsKs6gyoPUB',
       'assistant_id': 'asst_qvXmYlZV8zhABI2RtPzDfV6z',
       'attachments': [],
       'completed_at': None,
       'content': [{'text': {'annotations': [],
          'value': 'Certainly! To isolate \\(x\\), first subtract 11 from both sides of the equation \\(3x + 11 = 14\\), resulting in \\(3x = 3\\). Then, divide both sides by 3 to solve for \\(x\\), giving you \\(x = 1\\).'},
         'type': 'text'}],
       'created_at': 1736340414,
       'incomplete_at': None,
       'incomplete_details': None,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_lJsumsDtPTmdG3Enx2CfYrrq',
       'status': None,
       'thread_id': 'thread_j4dc1TiHPfkviKUHNi4aAsA6'}],
     'object': 'list',
     'first_id': 'msg_wSHHvaMnaWktZWsKs6gyoPUB',
     'last_id': 'msg_wSHHvaMnaWktZWsKs6gyoPUB',
     'has_more': False}


This may feel like a lot of steps to get a response back, especially for this simple example. However, you'll soon see how we can add very powerful functionality to our Assistant without changing much code at all!


### Example


Let's take a look at how we could potentially put all of this together. Below is all the code you need to use an Assistant you've created.

Since we've already created our Math Assistant, I've saved its ID in `MATH_ASSISTANT_ID`. I then defined two functions:

- `submit_message`: create a Message on a Thread, then start (and return) a new Run
- `get_response`: returns the list of Messages in a Thread



```python
from openai import OpenAI

MATH_ASSISTANT_ID = assistant.id  # or a hard-coded ID like "asst-..."

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")
```

I've also defined a `create_thread_and_run` function that I can re-use (which is actually almost identical to the [`client.beta.threads.create_and_run`](https://platform.openai.com/docs/api-reference/runs/createThreadAndRun) compound function in our API ;) ). Finally, we can submit our mock user requests each to a new Thread.

Notice how all of these API calls are asynchronous operations; this means we actually get async behavior in our code without the use of async libraries! (e.g. `asyncio`)



```python
def create_thread_and_run(user_input):
    thread = client.beta.threads.create()
    run = submit_message(MATH_ASSISTANT_ID, thread, user_input)
    return thread, run


# Emulating concurrent user requests
thread1, run1 = create_thread_and_run(
    "I need to solve the equation `3x + 11 = 14`. Can you help me?"
)
thread2, run2 = create_thread_and_run("Could you explain linear algebra to me?")
thread3, run3 = create_thread_and_run("I don't like math. What can I do?")

# Now all Runs are executing...
```

Once all Runs are going, we can wait on each and get the responses.



```python
import time

# Pretty printing
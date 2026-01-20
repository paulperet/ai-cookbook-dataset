# How to stream completions

By default, when you request a completion from the OpenAI, the entire completion is generated before being sent back in a single response.

If you're generating long completions, waiting for the response can take many seconds.

To get responses sooner, you can 'stream' the completion as it's being generated. This allows you to start printing or processing the beginning of the completion before the full completion is finished.

To stream completions, set `stream=True` when calling the chat completions or completions endpoints. This will return an object that streams back the response as data-only server-sent events. Extract chunks from the `delta` field rather than the `message` field.

## Downsides

Note that using `stream=True` in a production application makes it more difficult to moderate the content of the completions, as partial completions may be more difficult to evaluate. This may have implications for approved usage.

## Example code

Below, this notebook shows:
1. What a typical chat completion response looks like
2. What a streaming chat completion response looks like
3. How much time is saved by streaming a chat completion
4. How to get token usage data for streamed chat completion response

```python
# !pip install openai
```

```python
# imports
import time  # for measuring time duration of API calls
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
```

### 1. What a typical chat completion response looks like

With a typical ChatCompletions API call, the response is first computed and then returned all at once.

```python
# Example of an OpenAI ChatCompletion request
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api

# record the time before the request is sent
start_time = time.time()

# send a ChatCompletion request to count to 100
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'user', 'content': 'Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
    ],
    temperature=0,
)
# calculate the time it took to receive the response
response_time = time.time() - start_time

# print the time delay and text received
print(f"Full response received {response_time:.2f} seconds after request")
print(f"Full response received:\n{response}")

```

    Full response received 1.88 seconds after request
    Full response received:
    ChatCompletion(id='chatcmpl-9lMgdoiMfxVHPDNVCtvXuTWcQ2GGb', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100', role='assistant', function_call=None, tool_calls=None))], created=1721075651, model='gpt-july-test', object='chat.completion', system_fingerprint='fp_e9b8ed65d2', usage=CompletionUsage(completion_tokens=298, prompt_tokens=36, total_tokens=334))


The reply can be extracted with `response.choices[0].message`.

The content of the reply can be extracted with `response.choices[0].message.content`.

```python
reply = response.choices[0].message
print(f"Extracted reply: \n{reply}")

reply_content = response.choices[0].message.content
print(f"Extracted content: \n{reply_content}")

```

    Extracted reply: 
    ChatCompletionMessage(content='1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100', role='assistant', function_call=None, tool_calls=None)
    Extracted content: 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100


### 2. How to stream a chat completion

With a streaming API call, the response is sent back incrementally in chunks via an event stream. In Python, you can iterate over these events with a `for` loop.

Let's see what it looks like:

```python
# Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/api-reference/streaming#chat/create-stream

# a ChatCompletion request
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'user', 'content': "What's 1+1? Answer in one word."}
    ],
    temperature=0,
    stream=True  # this time, we set stream=True
)

for chunk in response:
    print(chunk)
    print(chunk.choices[0].delta.content)
    print("****************")
```

    ChatCompletionChunk(id='chatcmpl-9lMgfRSWPHcw51s6wxKT1YEO2CKpd', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1721075653, model='gpt-july-test', object='chat.completion.chunk', system_fingerprint='fp_e9b8ed65d2', usage=None)
    
    ****************
    ChatCompletionChunk(id='chatcmpl-9lMgfRSWPHcw51s6wxKT1YEO2CKpd', choices=[Choice(delta=ChoiceDelta(content='Two', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1721075653, model='gpt-july-test', object='chat.completion.chunk', system_fingerprint='fp_e9b8ed65d2', usage=None)
    Two
    ****************
    ChatCompletionChunk(id='chatcmpl-9lMgfRSWPHcw51s6wxKT1YEO2CKpd', choices=[Choice(delta=ChoiceDelta(content='.', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1721075653, model='gpt-july-test', object='chat.completion.chunk', system_fingerprint='fp_e9b8ed65d2', usage=None)
    .
    ****************
    ChatCompletionChunk(id='chatcmpl-9lMgfRSWPHcw51s6wxKT1YEO2CKpd', choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None), finish_reason='stop', index=0, logprobs=None)], created=1721075653, model='gpt-july-test', object='chat.completion.chunk', system_fingerprint='fp_e9b8ed65d2', usage=None)
    None
    ****************


As you can see above, streaming responses have a `delta` field rather than a `message` field. `delta` can hold things like:
- a role token (e.g., `{"role": "assistant"}`)
- a content token (e.g., `{"content": "\n\n"}`)
- nothing (e.g., `{}`), when the stream is over

### 3. How much time is saved by streaming a chat completion

Now let's ask `gpt-4o-mini` to count to 100 again, and see how long it takes.

```python
# Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/api-reference/streaming#chat/create-stream

# record the time before the request is sent
start_time = time.time()

# send a ChatCompletion request to count to 100
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'user', 'content': 'Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
    ],
    temperature=0,
    stream=True  # again, we set stream=True
)
# create variables to collect the stream of chunks
collected_chunks = []
collected_messages = []
# iterate through the stream of events
for chunk in response:
    chunk_time = time.time() - start_time  # calculate the time delay of the chunk
    collected_chunks.append(chunk)  # save the event response
    chunk_message = chunk.choices[0].delta.content  # extract the message
    collected_messages.append(chunk_message)  # save the message
    print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

# print the time delay and text received
print(f"Full response received {chunk_time:.2f} seconds after request")
# clean None in collected_messages
collected_messages = [m for m in collected_messages if m is not None]
full_reply_content = ''.join(collected_messages)
print(f"Full conversation received: {full_reply_content}")

```

    [Message received 1.14 seconds after request: , Message received 1.14 seconds after request: 1, ..., Message received 2.82 seconds after request: 100, Message received 2.82 seconds after request: None]
    Full response received 2.82 seconds after request
    Full conversation received: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100


#### Time comparison

In the example above, both requests took about 4 to 5 seconds to fully complete. Request times will vary depending on load and other stochastic factors.

However, with the streaming request, we received the first token after 0.1 seconds, and subsequent tokens every ~0.01-0.02 seconds.

### 4. How to get token usage data for streamed chat completion response

You can get token usage statistics for your streamed response by setting `stream_options={"include_usage": True}`. When you do so, an extra chunk will be streamed as the final chunk. You can access the usage data for the entire request via the `usage` field on this chunk. A few important notes when you set `stream_options={"include_usage": True}`:
* The value for the `usage` field on all chunks except for the last one will be null.
* The `usage` field on the last chunk contains token usage statistics for the entire request.
* The `choices` field on the last chunk will always be an empty array `[]`.

Let's see how it works using the example in 2.

```python
# Example of an OpenAI ChatCompletion request with stream=True and stream_options={"include_usage": True}

# a ChatCompletion request
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'user', 'content': "What's 1+1? Answer in one word."}
    ],
    temperature=0,
    stream=True,
    stream_options={"include_usage": True}, # retrieving token usage for stream response
)

for chunk in response:
    print(f"choices: {chunk.choices}\nusage: {chunk.usage}")
    print("****************")
```

    choices: [Choice(delta=ChoiceDelta(content='', function_call=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)]
    usage: None
    ****************
    choices: [Choice(delta=ChoiceDelta(content='Two', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)]
    usage: None
    ****************
    choices: [Choice(delta=ChoiceDelta(content='.', function_call=None, role=None, tool_calls=None), finish_reason=None, index=0, logprobs=None)]
    usage: None
    ****************
    choices: [Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None), finish_reason='stop', index=0, logprobs=None)]
    usage: None
    ****************
    choices: []
    usage: CompletionUsage(completion_tokens=2, prompt_tokens=18, total_tokens=20)
    ****************
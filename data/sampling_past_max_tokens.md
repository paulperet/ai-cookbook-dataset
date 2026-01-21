# Guide: Sampling Extended Responses from Claude Using Prefill

This guide demonstrates a technique for obtaining responses from Claude that exceed the standard `max_tokens` parameter limit. By using a prefill—supplying the model with its own partially completed output—you can seamlessly continue generating long-form content.

## Prerequisites

First, install the required library and set up your client.

```bash
pip install anthropic
```

```python
import anthropic

# Initialize the Anthropic client with your API key
client = anthropic.Anthropic(
    api_key="YOUR_API_KEY_HERE",  # Replace with your actual API key
)
```

## Step 1: The Initial Request and Token Limit

We'll start by asking Claude to generate five long stories, each about a different animal. We set `max_tokens=4096`, which is the maximum allowed value for this parameter.

```python
initial_message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": """
Please write five stories. Each should be at least 1000 words. Number the words to make sure you don't lose track. Make each story about a different animal.
Put them in <story_1>, <story_2>, ... tags
""",
        },
    ],
)
```

## Step 2: Identifying the Limit

After the request completes, we can check the `stop_reason` to understand why generation ended.

```python
print(initial_message.stop_reason)
```

The output will be:
```
max_tokens
```

This confirms that sampling stopped because the response reached the `max_tokens` limit. Let's examine the content to see where it was cut off.

```python
print(initial_message.content[0].text)
```

You will see that the output contains four complete stories and a fifth story that ends abruptly mid-sentence. The model was unable to finish the final story within the token budget.

## Step 3: Continuing the Response with Prefill

To complete the interrupted story, we use a technique called *prefill*. We take the partial response Claude generated and feed it back as an Assistant message in a new request. This allows the model to continue sampling from where it left off.

First, extract the incomplete text from the initial response.

```python
# This is the text that was cut off. In practice, you would programmatically extract
# the incomplete portion. For this example, we show the last segment.
incomplete_story_text = initial_message.content[0].text  # Contains the full, truncated output.
```

Now, create a new request. The messages array includes:
1.  The original user prompt.
2.  A new assistant message containing Claude's own partial output (the prefill).

```python
continued_message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": """
Please write five stories. Each should be at least 1000 words. Number the words to make sure you don't lose track. Make each story about a different animal.
Put them in <story_1>, <story_2>, ... tags
""",
        },
        {
            "role": "assistant",
            "content": incomplete_story_text,  # The prefill: Claude's previous, truncated output
        },
    ],
)
```

## Step 4: Examining the Completed Output

When you print the new response, you'll see that Claude seamlessly continues from the exact point where the previous response was cut off, completing the fifth story and potentially adding more content.

```python
print(continued_message.content[0].text)
```

The new output will be a continuation of the previous text, finishing the sentence and the story naturally, as if there had been no interruption.

## How It Works & Key Considerations

This method leverages the model's context window effectively. By prefilling the assistant's message with its own previous output, you provide the necessary context for it to continue the thought coherently.

**Important Notes:**
- **Token Count:** The prefill text counts towards the total token limit of the request (model context window minus `max_tokens`). Ensure your prefill plus the new `max_tokens` does not exceed the model's context limit.
- **Seamless Continuation:** The model is designed to continue naturally from the provided text. It will not repeat the prefill content but will append new tokens as a direct continuation.
- **Use Case:** This technique is ideal for long-form generation tasks like writing stories, articles, or code, where a single response might be insufficient.

By using this prefill strategy, you can work around the `max_tokens` parameter constraint and generate coherent, extended content with Claude.
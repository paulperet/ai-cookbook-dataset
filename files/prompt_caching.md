# Prompt caching through the Claude API

Prompt caching allows you to store and reuse context within your prompt. This makes it more practical to include additional information in your prompt—such as detailed instructions and example responses—which help improve every response Claude generates.

In addition, by fully leveraging prompt caching within your prompt, you can reduce latency by >2x and costs up to 90%. This can generate significant savings when building solutions that involve repetitive tasks around detailed book_content.

In this cookbook, we will demonstrate how to use prompt caching in a single turn and across a multi-turn conversation. 

## Setup

First, let's set up our environment with the necessary imports and initializations:

```python
%pip install anthropic bs4 --quiet
```

```python
import time

import anthropic
import requests
from bs4 import BeautifulSoup

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-5"
TIMESTAMP = int(time.time())
```

Now let's fetch some text content to use in our examples. We'll use the text from Pride and Prejudice by Jane Austen which is around ~187,000 tokens long.

```python
def fetch_article_content(url):
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


# Fetch the content of the article
book_url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
book_content = fetch_article_content(book_url)

print(f"Fetched {len(book_content)} characters from the book.")
print("First 500 characters:")
print(book_content[:500])
```

## Example 1: Single turn

Let's demonstrate prompt caching with a large document, comparing the performance and cost between cached and non-cached API calls.

### Part 1: Non-cached API Call (Baseline)

First, let's make a truly non-cached API call **without** the `cache_control` parameter. This will establish our baseline performance.

We'll ask for a short output to keep response generation time low, since prompt caching only affects input processing time.

```python
def make_non_cached_api_call():
    """Make an API call WITHOUT cache_control - no caching enabled."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": str(TIMESTAMP) + "<book>" + book_content + "</book>",
                    # Note: No cache_control parameter here - this is truly non-cached
                },
                {"type": "text", "text": "What is the title of this book? Only output the title."},
            ],
        }
    ]

    start_time = time.time()
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        messages=messages,
    )
    end_time = time.time()

    return response, end_time - start_time


non_cached_response, non_cached_time = make_non_cached_api_call()

print(f"Non-cached API call time: {non_cached_time:.2f} seconds")
print(f"Input tokens: {non_cached_response.usage.input_tokens}")
print(f"Output tokens: {non_cached_response.usage.output_tokens}")

print("\nResponse:")
print(non_cached_response.content[0].text)
```

### Part 2: First Cached API Call (Cache Creation)

Now let's enable prompt caching by adding `cache_control: {"type": "ephemeral"}` to the book content. 

**Important:** The first call with `cache_control` will **create** the cache entry. This initial call will have similar timing to the non-cached call because it still needs to process all tokens. However, it will store them in the cache for future use.

Look for the `cache_creation_input_tokens` field in the usage stats to see how many tokens were cached.

```python
def make_cached_api_call_create():
    """First call WITH cache_control - creates the cache entry."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": str(TIMESTAMP) + "<book>" + book_content + "</book>",
                    "cache_control": {"type": "ephemeral"},  # This enables caching
                },
                {"type": "text", "text": "What is the title of this book? Only output the title."},
            ],
        }
    ]

    start_time = time.time()
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        messages=messages,
    )
    end_time = time.time()

    return response, end_time - start_time


cached_create_response, cached_create_time = make_cached_api_call_create()

print(f"First cached API call time: {cached_create_time:.2f} seconds")
print(f"Input tokens: {cached_create_response.usage.input_tokens}")
print(f"Output tokens: {cached_create_response.usage.output_tokens}")
print(
    f"Cache creation tokens: {getattr(cached_create_response.usage, 'cache_creation_input_tokens', 0)}"
)

print("\nResponse:")
print(cached_create_response.content[0].text)

print(
    "\nNote: This first call creates the cache but doesn't benefit from it yet - timing is similar to non-cached call."
)
```

### Part 3: Second Cached API Call (Cache Hit)

Now let's make another API call with the same `cache_control` parameter. Since the cache was created in Part 2, this call will **read from the cache** instead of processing all tokens again.

This is where you see the real performance benefit! Look for the `cache_read_input_tokens` field in the usage stats.

```python
def make_cached_api_call_hit():
    """Second call WITH cache_control - reads from existing cache."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": str(TIMESTAMP) + "<book>" + book_content + "</book>",
                    "cache_control": {"type": "ephemeral"},  # Same cache_control as before
                },
                {"type": "text", "text": "What is the title of this book? Only output the title."},
            ],
        }
    ]

    start_time = time.time()
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        messages=messages,
    )
    end_time = time.time()

    return response, end_time - start_time


cached_hit_response, cached_hit_time = make_cached_api_call_hit()

print(f"Second cached API call time: {cached_hit_time:.2f} seconds")
print(f"Input tokens: {cached_hit_response.usage.input_tokens}")
print(f"Output tokens: {cached_hit_response.usage.output_tokens}")
print(f"Cache read tokens: {getattr(cached_hit_response.usage, 'cache_read_input_tokens', 0)}")

print("\nResponse:")
print(cached_hit_response.content[0].text)

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print(f"Non-cached call:       {non_cached_time:.2f}s")
print(f"First cached call:     {cached_create_time:.2f}s (creates cache)")
print(f"Second cached call:    {cached_hit_time:.2f}s (reads from cache)")
print(f"\nSpeedup from caching:  {non_cached_time / cached_hit_time:.1f}x faster!")
print("=" * 70)
```

## Summary of Example 1

This example demonstrated three distinct scenarios:

1. **Non-cached call** - Without `cache_control`, Claude processes all ~187k tokens normally
2. **First cached call** - With `cache_control`, Claude processes all tokens AND stores them in cache (similar timing to non-cached)
3. **Second cached call** - With `cache_control`, Claude reads from the existing cache (2-10x faster!)

The key insight: **Prompt caching requires two calls to show benefits**
- The first call with `cache_control` creates the cache entry
- Subsequent calls with the same `cache_control` read from the cache for dramatic speedups

This is especially valuable for:
- Large documents or codebases that remain constant across multiple queries
- System prompts with detailed instructions
- Multi-turn conversations (as shown in Example 2 below)

## Example 2: Multi-turn Conversation with Incremental Caching

Now, let's look at a multi-turn conversation where we add cache breakpoints as the conversation progresses.

```python
class ConversationHistory:
    def __init__(self):
        # Initialize an empty list to store conversation turns
        self.turns = []

    def add_turn_assistant(self, content):
        # Add an assistant's turn to the conversation history
        self.turns.append({"role": "assistant", "content": [{"type": "text", "text": content}]})

    def add_turn_user(self, content):
        # Add a user's turn to the conversation history
        self.turns.append({"role": "user", "content": [{"type": "text", "text": content}]})

    def get_turns(self):
        # Retrieve conversation turns with specific formatting
        result = []
        user_turns_processed = 0
        # Iterate through turns in reverse order
        for turn in reversed(self.turns):
            if turn["role"] == "user" and user_turns_processed < 1:
                # Add the last user turn with ephemeral cache control
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": turn["content"][0]["text"],
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
                user_turns_processed += 1
            else:
                # Add other turns as they are
                result.append(turn)
        # Return the turns in the original order
        return list(reversed(result))


# Initialize the conversation history
conversation_history = ConversationHistory()

# System message containing the book content
# Note: 'book_content' should be defined elsewhere in the code
system_message = f"{TIMESTAMP} <file_contents> {book_content} </file_contents>"

# Predefined questions for our simulation
questions = [
    "What is the title of this novel?",
    "Who are Mr. and Mrs. Bennet?",
    "What is Netherfield Park?",
    "What is the main theme of this novel?",
]


def simulate_conversation():
    for i, question in enumerate(questions, 1):
        print(f"\nTurn {i}:")
        print(f"User: {question}")

        # Add user input to conversation history
        conversation_history.add_turn_user(question)

        # Record the start time for performance measurement
        start_time = time.time()

        # Make an API call to the assistant
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=300,
            system=[
                {"type": "text", "text": system_message, "cache_control": {"type": "ephemeral"}},
            ],
            messages=conversation_history.get_turns(),
        )

        # Record the end time
        end_time = time.time()

        # Extract the assistant's reply
        assistant_reply = response.content[0].text
        print(f"Assistant: {assistant_reply}")

        # Print token usage information
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_tokens_cache_read = getattr(response.usage, "cache_read_input_tokens", "---")
        input_tokens_cache_create = getattr(response.usage, "cache_creation_input_tokens", "---")
        print(f"User input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Input tokens (cache read): {input_tokens_cache_read}")
        print(f"Input tokens (cache write): {input_tokens_cache_create}")

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time

        # Calculate the percentage of input prompt cached
        total_input_tokens = input_tokens + (
            int(input_tokens_cache_read) if input_tokens_cache_read != "---" else 0
        )
        percentage_cached = (
            int(input_tokens_cache_read) / total_input_tokens * 100
            if input_tokens_cache_read != "---" and total_input_tokens > 0
            else 0
        )

        print(f"{percentage_cached:.1f}% of input prompt cached ({total_input_tokens} tokens)")
        print(f"Time taken: {elapsed_time:.2f} seconds")

        # Add assistant's reply to conversation history
        conversation_history.add_turn_assistant(assistant_reply)


# Run the simulated conversation
simulate_conversation()
```

[First Entry, ..., Last Entry]

As you can see in this example, response times decreased from nearly 24 seconds to just 7-11 seconds after the initial cache setup, while maintaining the same level of quality across the answers. Most of this remaining latency is due to the time it takes to generate the response, which is not affected by prompt caching.

And since nearly 100% of input tokens were cached in subsequent turns as we kept adjusting the cache breakpoints, we were able to read the next user message nearly instantly.
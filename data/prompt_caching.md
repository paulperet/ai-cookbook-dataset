# Claude API Prompt Caching Guide

Prompt caching allows you to store and reuse context within your prompts, making it practical to include detailed instructions, example responses, or large documents that improve Claude's responses. By leveraging prompt caching, you can reduce latency by over 2x and costs by up to 90%, generating significant savings for solutions involving repetitive tasks with consistent content.

This guide demonstrates how to implement prompt caching in both single-turn and multi-turn conversation scenarios.

## Prerequisites

First, install the required packages and set up your environment:

```bash
pip install anthropic bs4 --quiet
```

```python
import time
import anthropic
import requests
from bs4 import BeautifulSoup

# Initialize the Anthropic client
client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-5"
TIMESTAMP = int(time.time())
```

## Fetching Sample Content

We'll use the text from "Pride and Prejudice" by Jane Austen (approximately 187,000 tokens) as our demonstration content:

```python
def fetch_article_content(url):
    """Fetch and clean text content from a URL."""
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Extract and clean text
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text

# Fetch the book content
book_url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
book_content = fetch_article_content(book_url)

print(f"Fetched {len(book_content)} characters from the book.")
print("First 500 characters:")
print(book_content[:500])
```

## Example 1: Single-Turn Prompt Caching

This example demonstrates prompt caching with a large document, comparing performance and cost between cached and non-cached API calls.

### Step 1: Non-Cached API Call (Baseline)

First, establish a baseline by making an API call without caching enabled:

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

# Execute the non-cached call
non_cached_response, non_cached_time = make_non_cached_api_call()

print(f"Non-cached API call time: {non_cached_time:.2f} seconds")
print(f"Input tokens: {non_cached_response.usage.input_tokens}")
print(f"Output tokens: {non_cached_response.usage.output_tokens}")
print("\nResponse:")
print(non_cached_response.content[0].text)
```

This call processes all ~187,000 tokens normally, establishing our performance baseline.

### Step 2: First Cached API Call (Cache Creation)

Now enable prompt caching by adding `cache_control: {"type": "ephemeral"}` to the book content. This first call creates the cache entry:

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

# Execute the first cached call
cached_create_response, cached_create_time = make_cached_api_call_create()

print(f"First cached API call time: {cached_create_time:.2f} seconds")
print(f"Input tokens: {cached_create_response.usage.input_tokens}")
print(f"Output tokens: {cached_create_response.usage.output_tokens}")
print(f"Cache creation tokens: {getattr(cached_create_response.usage, 'cache_creation_input_tokens', 0)}")
print("\nResponse:")
print(cached_create_response.content[0].text)
print("\nNote: This first call creates the cache but doesn't benefit from it yet - timing is similar to non-cached call.")
```

This initial call has similar timing to the non-cached call because it still processes all tokens, but it stores them in the cache for future use.

### Step 3: Second Cached API Call (Cache Hit)

Make another API call with the same `cache_control` parameter. Since the cache was created in Step 2, this call reads from the cache instead of reprocessing all tokens:

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

# Execute the second cached call
cached_hit_response, cached_hit_time = make_cached_api_call_hit()

print(f"Second cached API call time: {cached_hit_time:.2f} seconds")
print(f"Input tokens: {cached_hit_response.usage.input_tokens}")
print(f"Output tokens: {cached_hit_response.usage.output_tokens}")
print(f"Cache read tokens: {getattr(cached_hit_response.usage, 'cache_read_input_tokens', 0)}")
print("\nResponse:")
print(cached_hit_response.content[0].text)

# Performance comparison
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print(f"Non-cached call:       {non_cached_time:.2f}s")
print(f"First cached call:     {cached_create_time:.2f}s (creates cache)")
print(f"Second cached call:    {cached_hit_time:.2f}s (reads from cache)")
print(f"\nSpeedup from caching:  {non_cached_time / cached_hit_time:.1f}x faster!")
print("=" * 70)
```

This is where you see the real performance benefit - typically 2-10x faster than non-cached calls!

## Example 1 Summary

This example demonstrated three distinct scenarios:

1. **Non-cached call** - Without `cache_control`, Claude processes all ~187k tokens normally
2. **First cached call** - With `cache_control`, Claude processes all tokens AND stores them in cache (similar timing to non-cached)
3. **Second cached call** - With `cache_control`, Claude reads from the existing cache (2-10x faster!)

**Key Insight:** Prompt caching requires two calls to show benefits:
- The first call with `cache_control` creates the cache entry
- Subsequent calls with the same `cache_control` read from the cache for dramatic speedups

This is especially valuable for:
- Large documents or codebases that remain constant across multiple queries
- System prompts with detailed instructions
- Multi-turn conversations (as shown in Example 2)

## Example 2: Multi-Turn Conversation with Incremental Caching

Now let's implement a multi-turn conversation where we add cache breakpoints as the conversation progresses.

### Step 1: Create a Conversation History Manager

First, create a class to manage conversation history with caching support:

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
```

### Step 2: Set Up the Conversation

Initialize the conversation history and define the questions for our simulation:

```python
# Initialize the conversation history
conversation_history = ConversationHistory()

# System message containing the book content with caching enabled
system_message = f"{TIMESTAMP} <file_contents> {book_content} </file_contents>"

# Predefined questions for our simulation
questions = [
    "What is the title of this novel?",
    "Who are Mr. and Mrs. Bennet?",
    "What is Netherfield Park?",
    "What is the main theme of this novel?",
]
```

### Step 3: Simulate the Cached Conversation

Now simulate a conversation where each turn benefits from incremental caching:

```python
def simulate_conversation():
    for i, question in enumerate(questions, 1):
        print(f"\nTurn {i}:")
        print(f"User: {question}")

        # Add user input to conversation history
        conversation_history.add_turn_user(question)

        # Record the start time for performance measurement
        start_time = time.time()

        # Make an API call to the assistant with caching enabled
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

## Performance Analysis

In this multi-turn example, you'll observe that response times decrease significantly after the initial cache setup. For instance, response times might drop from nearly 24 seconds to just 7-11 seconds while maintaining the same quality across answers.

Most of the remaining latency is due to response generation time, which is not affected by prompt caching. Since nearly 100% of input tokens are cached in subsequent turns, the system can read the next user message nearly instantly.

## Key Takeaways

1. **Two-Step Process**: Prompt caching requires an initial call to create the cache, followed by subsequent calls that benefit from it.

2. **Significant Performance Gains**: Cached calls can be 2-10x faster than non-cached calls, with cost reductions up to 90%.

3. **Ideal Use Cases**: 
   - Large, static documents or codebases
   - Detailed system prompts with instructions
   - Multi-turn conversations with consistent context
   - Applications with repetitive queries against the same content

4. **Implementation Tips**:
   - Use `cache_control: {"type": "ephemeral"}` to enable caching
   - Monitor `cache_creation_input_tokens` and `cache_read_input_tokens` in usage stats
   - Structure conversations to maximize cacheable content

By implementing prompt caching strategically, you can dramatically improve the performance and cost-efficiency of your Claude API applications while maintaining high-quality responses.
# Guide: Implementing Speculative Prompt Caching

This guide demonstrates **Speculative Prompt Caching**, a technique that reduces time-to-first-token (TTFT) by warming up the model's cache while users are still formulating their queries.

**Core Concept:**
- **Without Speculative Caching:** The user types their question, submits it, and *then* the API loads context into cache before generating a response.
- **With Speculative Caching:** As the user begins typing, cache warming starts immediately in the background. When the user submits their question, the API uses the pre-warmed cache to generate a response much faster.

## Prerequisites & Setup

First, install the required packages and import the necessary modules.

```bash
pip install anthropic httpx --quiet
```

```python
import asyncio
import copy
import datetime
import time

import httpx
from anthropic import AsyncAnthropic

# Configuration
MODEL = "claude-sonnet-4-5"
SQLITE_SOURCES = {
    "btree.h": "https://sqlite.org/src/raw/18e5e7b2124c23426a283523e5f31a4bff029131b795bb82391f9d2f3136fc50?at=btree.h",
    "btree.c": "https://sqlite.org/src/raw/63ca6b647342e8cef643863cd0962a542f133e1069460725ba4461dcda92b03c?at=btree.c",
}
DEFAULT_CLIENT_ARGS = {
    "system": "You are an expert systems programmer helping analyze database internals.",
    "max_tokens": 4096,
    "temperature": 0,
}
```

## Step 1: Create Helper Functions

You'll need functions to download context data, prepare messages, and analyze query statistics.

### 1.1 Download Context Sources

This function asynchronously downloads the SQLite source files that will serve as the large context for our queries.

```python
async def get_sqlite_sources() -> dict[str, str]:
    print("Downloading SQLite source files...")
    source_files = {}
    start_time = time.time()

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []

        async def download_file(filename: str, url: str) -> tuple[str, str]:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            print(f"Successfully downloaded {filename}")
            return filename, response.text

        for filename, url in SQLITE_SOURCES.items():
            tasks.append(download_file(filename, url))

        results = await asyncio.gather(*tasks)
        source_files = dict(results)

    duration = time.time() - start_time
    print(f"Downloaded {len(source_files)} files in {duration:.2f} seconds")
    return source_files
```

### 1.2 Prepare the Initial Message

This function creates the initial user message containing the source code context. A timestamp is included to prevent unwanted cache sharing across different runs.

```python
async def create_initial_message():
    sources = await get_sqlite_sources()
    initial_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""
Current time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Source to Analyze:

btree.h:
```c
{sources["btree.h"]}
```

btree.c:
```c
{sources["btree.c"]}
```""",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    return initial_message
```

### 1.3 Warm Up the Cache

This function sends a single-token request to the API. Its purpose is to trigger the caching of the provided context on the server side.

```python
async def sample_one_token(client: AsyncAnthropic, messages: list):
    """Send a single-token request to warm up the cache"""
    args = copy.deepcopy(DEFAULT_CLIENT_ARGS)
    args["max_tokens"] = 1
    await client.messages.create(
        messages=messages,
        model=MODEL,
        **args,
    )
```

### 1.4 Print Query Statistics

This helper function prints token usage and cache statistics from the API response.

```python
def print_query_statistics(response, query_type: str) -> None:
    print(f"\n{query_type} query statistics:")
    print(f"\tInput tokens: {response.usage.input_tokens}")
    print(f"\tOutput tokens: {response.usage.output_tokens}")
    print(f"\tCache read input tokens: {getattr(response.usage, 'cache_read_input_tokens', '---')}")
    print(
        f"\tCache creation input tokens: {getattr(response.usage, 'cache_creation_input_tokens', '---')}"
    )
```

## Step 2: Standard Prompt Caching (Baseline)

First, let's establish a baseline by implementing the standard flow without speculative caching. The user types their question, and only after submission does the API receive the full context.

```python
async def standard_prompt_caching_demo():
    client = AsyncAnthropic()

    # 1. Prepare the large context
    initial_message = await create_initial_message()

    # 2. Simulate user typing time
    print("User is typing their question...")
    await asyncio.sleep(3)  # Simulate 3 seconds of typing
    user_question = "What is the purpose of the BtShared structure?"
    print(f"User submitted: {user_question}")

    # 3. Append the user's question to the context
    full_message = copy.deepcopy(initial_message)
    full_message["content"].append(
        {"type": "text", "text": f"Answer the user's question: {user_question}"}
    )

    # 4. Send the complete request and measure performance
    print("\nSending request to API...")
    start_time = time.time()

    first_token_time = None
    async with client.messages.stream(
        messages=[full_message],
        model=MODEL,
        **DEFAULT_CLIENT_ARGS,
    ) as stream:
        async for text in stream.text_stream:
            if first_token_time is None and text.strip():
                first_token_time = time.time() - start_time
                print(f"\n🕐 Time to first token: {first_token_time:.2f} seconds")
                break

        response = await stream.get_final_message()

    total_time = time.time() - start_time
    print(f"Total response time: {total_time:.2f} seconds")
    print_query_statistics(response, "Standard Caching")

    return first_token_time, total_time
```

Run the baseline demo:

```python
standard_ttft, standard_total = await standard_prompt_caching_demo()
```

## Step 3: Speculative Prompt Caching

Now, implement the speculative caching pattern. The key difference is that cache warming begins *as soon as the user starts typing*, happening in parallel with their input.

```python
async def speculative_prompt_caching_demo():
    client = AsyncAnthropic()

    # 1. Prepare the large context (same as before)
    initial_message = await create_initial_message()

    # 2. Start speculative caching while the user is typing
    print("User is typing their question...")
    print("🔥 Starting cache warming in background...")

    # Launch the cache-warming task in the background
    cache_task = asyncio.create_task(sample_one_token(client, [initial_message]))

    # 3. Simulate user typing time (cache warming happens concurrently)
    await asyncio.sleep(3)
    user_question = "What is the purpose of the BtShared structure?"
    print(f"User submitted: {user_question}")

    # 4. Ensure cache warming is complete before proceeding
    await cache_task
    print("✅ Cache warming completed!")

    # 5. Prepare the final message, reusing the *exact same* initial context
    cached_message = copy.deepcopy(initial_message)
    cached_message["content"].append(
        {"type": "text", "text": f"Answer the user's question: {user_question}"}
    )

    # 6. Send the request. The context should now be served from cache.
    print("\nSending request to API (with warm cache)...")
    start_time = time.time()

    first_token_time = None
    async with client.messages.stream(
        messages=[cached_message],
        model=MODEL,
        **DEFAULT_CLIENT_ARGS,
    ) as stream:
        async for text in stream.text_stream:
            if first_token_time is None and text.strip():
                first_token_time = time.time() - start_time
                print(f"\n🚀 Time to first token: {first_token_time:.2f} seconds")
                break

        response = await stream.get_final_message()

    total_time = time.time() - start_time
    print(f"Total response time: {total_time:.2f} seconds")
    print_query_statistics(response, "Speculative Caching")

    return first_token_time, total_time
```

Run the speculative caching demo:

```python
speculative_ttft, speculative_total = await speculative_prompt_caching_demo()
```

## Step 4: Compare Performance

Finally, compare the results to quantify the improvement from speculative caching.

```python
print("=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)

print("\nStandard Prompt Caching:")
print(f"  Time to First Token: {standard_ttft:.2f} seconds")
print(f"  Total Response Time: {standard_total:.2f} seconds")

print("\nSpeculative Prompt Caching:")
print(f"  Time to First Token: {speculative_ttft:.2f} seconds")
print(f"  Total Response Time: {speculative_total:.2f} seconds")

ttft_improvement = (standard_ttft - speculative_ttft) / standard_ttft * 100
total_improvement = (standard_total - speculative_total) / standard_total * 100

print("\n🎯 IMPROVEMENTS:")
print(
    f"  TTFT Improvement: {ttft_improvement:.1f}% ({standard_ttft - speculative_ttft:.2f}s faster)"
)
print(
    f"  Total Time Improvement: {total_improvement:.1f}% ({standard_total - speculative_total:.2f}s faster)"
)
```

## Key Takeaways

1.  **Dramatic TTFT Reduction:** Speculative caching significantly reduces the time to first token by performing cache creation in parallel with user input.
2.  **Optimal Use Case:** This pattern is most effective with large, reusable contexts (e.g., >1000 tokens) that form the basis for multiple user queries.
3.  **Simple Implementation:** The core technique is straightforward—send a minimal (1-token) request with the context while the user is typing.
4.  **Hidden Latency:** Cache warming time is effectively "hidden" from the user, as it overlaps with their natural typing delay.

## Best Practices for Production

- **Initiate Early:** Start cache warming as soon as possible, for example, when a user focuses on an input field or a relevant UI element is rendered.
- **Ensure Cache Hits:** Use *exactly the same* context payload for both the warming request and the final user request. Any difference will cause a cache miss.
- **Monitor Success:** Check the `cache_read_input_tokens` field in the API response to verify that your requests are hitting the cache.
- **Prevent Stale Caches:** Consider adding unique identifiers (like timestamps or session IDs) to context to prevent unwanted cache sharing across different users or sessions.
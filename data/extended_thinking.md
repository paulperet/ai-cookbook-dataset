# Guide: Using Claude 3.7 Sonnet's Extended Thinking Feature

This guide demonstrates how to use Claude 3.7 Sonnet's extended thinking feature for enhanced reasoning on complex tasks. Extended thinking provides transparency into Claude's step-by-step thought process before delivering a final answer, with `thinking` content blocks that contain internal reasoning.

## Prerequisites

First, install the required package and set up your environment.

```bash
pip install anthropic
```

```python
import anthropic
import os

# Set your API key as an environment variable or directly
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"

# Initialize the client
client = anthropic.Anthropic()

# Helper function to pretty print responses with thinking blocks
def print_thinking_response(response):
    """Pretty print a message response with thinking blocks."""
    print("\n==== FULL RESPONSE ====")
    for block in response.content:
        if block.type == "thinking":
            print("\n🧠 THINKING BLOCK:")
            # Show truncated thinking for readability
            print(block.thinking[:500] + "..." if len(block.thinking) > 500 else block.thinking)
            print(f"\n[Signature available: {bool(getattr(block, 'signature', None))}]")
            if hasattr(block, 'signature') and block.signature:
                print(f"[Signature (first 50 chars): {block.signature[:50]}...]")
        elif block.type == "redacted_thinking":
            print("\n🔒 REDACTED THINKING BLOCK:")
            print(f"[Data length: {len(block.data) if hasattr(block, 'data') else 'N/A'}]")
        elif block.type == "text":
            print("\n✓ FINAL ANSWER:")
            print(block.text)
    
    print("\n==== END RESPONSE ====")
```

## Basic Example: Solving a Logic Puzzle

Let's start with a basic example to see extended thinking in action. We'll ask Claude to solve a classic logic puzzle.

```python
def basic_thinking_example():
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4000,
        thinking={
            "type": "enabled",
            "budget_tokens": 2000
        },
        messages=[{
            "role": "user",
            "content": "Solve this puzzle: Three people check into a hotel. They pay $30 to the manager. The manager finds out that the room only costs $25 so he gives $5 to the bellboy to return to the three people. The bellboy, however, decides to keep $2 and gives $1 back to each person. Now, each person paid $10 and got back $1, so they paid $9 each, totaling $27. The bellboy kept $2, which makes $29. Where is the missing $1?"
        }]
    )
    
    print_thinking_response(response)

basic_thinking_example()
```

When you run this, you'll see Claude's thinking process displayed in a `🧠 THINKING BLOCK:` section, followed by the `✓ FINAL ANSWER:` with the solution to the puzzle.

## Streaming with Extended Thinking

For longer responses, you can use streaming to process the output incrementally. This is particularly useful when you want to display thinking blocks as they're generated.

```python
def streaming_with_thinking():
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=4000,
        thinking={
            "type": "enabled",
            "budget_tokens": 2000
        },
        messages=[{
            "role": "user",
            "content": "Solve this puzzle: Three people check into a hotel. They pay $30 to the manager. The manager finds out that the room only costs $25 so he gives $5 to the bellboy to return to the three people. The bellboy, however, decides to keep $2 and gives $1 back to each person. Now, each person paid $10 and got back $1, so they paid $9 each, totaling $27. The bellboy kept $2, which makes $29. Where is the missing $1?"
        }]
    ) as stream:
        # Track what we're currently building
        current_block_type = None
        current_content = ""
        
        for event in stream:
            if event.type == "content_block_start":
                current_block_type = event.content_block.type
                print(f"\n--- Starting {current_block_type} block ---")
                current_content = ""
                
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    print(event.delta.thinking, end="", flush=True)
                    current_content += event.delta.thinking
                elif event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
                    current_content += event.delta.text
                    
            elif event.type == "content_block_stop":
                if current_block_type == "thinking":
                    # Just show a summary for thinking
                    print(f"\n[Completed thinking block, {len(current_content)} characters]")
                elif current_block_type == "redacted_thinking":
                    print("\n[Redacted thinking block]")
                print(f"--- Finished {current_block_type} block ---\n")
                current_block_type = None
                
            elif event.type == "message_stop":
                print("\n--- Message complete ---")

streaming_with_thinking()
```

This streaming approach lets you see Claude's reasoning unfold in real-time, with clear demarcation between thinking blocks and the final answer.

## Token Counting and Context Window Management

Extended thinking tokens count toward your context window, so it's important to manage token usage. Here's how to track and estimate token consumption.

```python
def count_tokens(messages):
    """Count tokens for a given message list."""
    result = client.messages.count_tokens(
        model="claude-sonnet-4-5",
        messages=messages
    )
    return result.input_tokens

def token_counting_example():
    # Define a function to create a sample prompt
    def create_sample_messages():
        messages = [{
            "role": "user",
            "content": "Solve this puzzle: Three people check into a hotel. They pay $30 to the manager. The manager finds out that the room only costs $25 so he gives $5 to the bellboy to return to the three people. The bellboy, however, decides to keep $2 and gives $1 back to each person. Now, each person paid $10 and got back $1, so they paid $9 each, totaling $27. The bellboy kept $2, which makes $29. Where is the missing $1?"
        }]
        return messages
    
    # Count tokens without thinking
    base_messages = create_sample_messages()
    base_token_count = count_tokens(base_messages)
    print(f"Base token count (input only): {base_token_count}")
    
    # Make a request with thinking and check actual usage
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=8000,
        thinking={
            "type": "enabled",
            "budget_tokens": 2000
        },
        messages=base_messages
    )
    
    # Calculate and print token usage stats
    thinking_tokens = sum(
        len(block.thinking.split()) * 1.3  # Rough estimate
        for block in response.content 
        if block.type == "thinking"
    )
    
    final_answer_tokens = sum(
        len(block.text.split()) * 1.3  # Rough estimate
        for block in response.content 
        if block.type == "text"
    )
    
    print(f"\nEstimated thinking tokens used: ~{int(thinking_tokens)}")
    print(f"Estimated final answer tokens: ~{int(final_answer_tokens)}")
    print(f"Total estimated output tokens: ~{int(thinking_tokens + final_answer_tokens)}")
    print(f"Input tokens + max_tokens = {base_token_count + 8000}")
    print(f"Available for final answer after thinking: ~{8000 - int(thinking_tokens)}")
    
    # Demo with escalating thinking budgets
    thinking_budgets = [1024, 2000, 4000, 8000, 16000, 32000]
    context_window = 200000
    for budget in thinking_budgets:
        print(f"\nWith thinking budget of {budget} tokens:")
        print(f"Input tokens: {base_token_count}")
        print(f"Max tokens needed: {base_token_count + budget + 1000}")  # Add 1000 for final answer
        print(f"Remaining context window: {context_window - (base_token_count + budget + 1000)}")
        
        if base_token_count + budget + 1000 > context_window:
            print("WARNING: This would exceed the context window of 200k tokens!")

token_counting_example()
```

This example helps you understand how thinking tokens affect your overall token usage and how to plan for different thinking budgets.

## Understanding Redacted Thinking Blocks

Occasionally, Claude's internal reasoning may be flagged by safety systems. When this happens, the thinking block is encrypted and returned as a `redacted_thinking` block. These blocks are automatically decrypted when passed back to the API, allowing Claude to continue without losing context.

```python
def redacted_thinking_example():
    # Using a special test string that triggers redacted thinking
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4000,
        thinking={
            "type": "enabled",
            "budget_tokens": 2000
        },
        messages=[{
            "role": "user",
            "content": "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
        }]
    )
    
    # Identify redacted thinking blocks
    redacted_blocks = [block for block in response.content if block.type == "redacted_thinking"]
    thinking_blocks = [block for block in response.content if block.type == "thinking"]
    text_blocks = [block for block in response.content if block.type == "text"]
    
    print(f"Response includes {len(response.content)} total blocks:")
    print(f"- {len(redacted_blocks)} redacted thinking blocks")
    print(f"- {len(thinking_blocks)} regular thinking blocks")
    print(f"- {len(text_blocks)} text blocks")
    
    # Show data properties of redacted blocks
    if redacted_blocks:
        print(f"\nRedacted thinking blocks contain encrypted data:")
        for i, block in enumerate(redacted_blocks[:3]):  # Show first 3 at most
            print(f"Block {i+1} data preview: {block.data[:50]}...")
    
    # Print the final text output
    if text_blocks:
        print(f"\nFinal text response:")
        print(text_blocks[0].text)

redacted_thinking_example()
```

Redacted thinking blocks maintain the flow of conversation while ensuring safety compliance. The encrypted data preserves Claude's reasoning chain without exposing potentially problematic content.

## Handling Error Cases

When using extended thinking, be aware of these common error scenarios:

```python
def demonstrate_common_errors():
    # 1. Error from setting thinking budget too small
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4000,
            thinking={
                "type": "enabled",
                "budget_tokens": 500  # Too small, minimum is 1024
            },
            messages=[{
                "role": "user",
                "content": "Explain quantum computing."
            }]
        )
    except Exception as e:
        print(f"\nError with too small thinking budget: {e}")
    
    # 2. Error from using temperature with thinking
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4000,
            temperature=0.7,  # Not compatible with thinking
            thinking={
                "type": "enabled",
                "budget_tokens": 2000
            },
            messages=[{
                "role": "user",
                "content": "Write a creative story."
            }]
        )
    except Exception as e:
        print(f"\nError with temperature and thinking: {e}")
    
    # 3. Error from exceeding context window
    try:
        # Create a very large prompt
        long_content = "Please analyze this text. " + "This is sample text. " * 150000
        
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=20000,  # This plus the long prompt will exceed context window
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            },
            messages=[{
                "role": "user",
                "content": long_content
            }]
        )
    except Exception as e:
        print(f"\nError from exceeding context window: {e}")

demonstrate_common_errors()
```

Key considerations when using extended thinking:

1. **Minimum budget**: The minimum thinking budget is 1,024 tokens. Start with the minimum and increase incrementally to find the optimal range for your use case.

2. **Incompatible features**: Extended thinking isn't compatible with temperature, top_p, or top_k modifications, and you cannot pre-fill responses.

3. **Pricing**: Extended thinking tokens count toward the context window and are billed as output tokens. They also count toward your rate limits.

## Summary

Extended thinking provides valuable insight into Claude's reasoning process while enhancing its problem-solving capabilities. By following these examples, you can effectively implement extended thinking in your applications, manage token usage, handle edge cases, and leverage streaming for real-time interaction with Claude's thought process.
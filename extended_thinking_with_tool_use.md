# Extended Thinking with Tool Use

## Table of contents
- [Setup](#setup)
- [Basic example](#basic-example)
- [Multiple tool calls](#multiple-tool-calls-with-thinking)
- [Preserving thinking blocks](#preserving-thinking-blocks)

This notebook demonstrates how to use Claude 3.7 Sonnet's extended thinking feature with tools. The extended thinking feature allows you to see Claude's step-by-step thinking before it provides a final answer, providing transparency into how it decides which tools to use and how it interprets tool results.

When using extended thinking with tool use, the model will show its thinking before making tool requests, but not repeat the thinking process after receiving tool results. Claude will not output another thinking block until after the next non-`tool_result` `user` turn. For more information on extended thinking, see our [documentation](https://docs.claude.com/en/docs/build-with-claude/extended-thinking).

## Setup

First, let's install the necessary packages and set up our environment.

```coconut
%pip install anthropic
```

```coconut
import anthropic
import os
import json

# Global variables for model and token budgets
MODEL_NAME = "claude-sonnet-4-5"
MAX_TOKENS = 4000
THINKING_BUDGET_TOKENS = 2000

# Set your API key as an environment variable or directly
# os.environ["ANTHROPIC_API_KEY"] = "your_api_key_here"

# Initialize the client
client = anthropic.Anthropic()

# Helper functions
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

def count_tokens(messages, tools=None):
    """Count tokens for a given message list with optional tools."""
    if tools:
        response = client.messages.count_tokens(
            model=MODEL_NAME,
            messages=messages,
            tools=tools
        )
    else:
        response = client.messages.count_tokens(
            model=MODEL_NAME,
            messages=messages
        )
    return response.input_tokens
```

## Single tool calls with thinking

This example demonstrates how to combine thinking and make a single tool call, with a mock weather tool.

```coconut
def tool_use_with_thinking():
    # Define a weather tool
    tools = [
        {
            "name": "weather",
            "description": "Get current weather information for a location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for."
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    def weather(location):
        # Mock weather data
        weather_data = {
            "New York": {"temperature": 72, "condition": "Sunny"},
            "London": {"temperature": 62, "condition": "Cloudy"},
            "Tokyo": {"temperature": 80, "condition": "Partly cloudy"},
            "Paris": {"temperature": 65, "condition": "Rainy"},
            "Sydney": {"temperature": 85, "condition": "Clear"},
            "Berlin": {"temperature": 60, "condition": "Foggy"},
        }
        
        return weather_data.get(location, {"error": f"No weather data available for {location}"})
    
    # Initial request with tool use and thinking
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={
            "type": "enabled",
            "budget_tokens": THINKING_BUDGET_TOKENS
        },
        tools=tools,
        messages=[{
            "role": "user",
            "content": "What's the weather like in Paris today?"
        }]
    )
    
    # Detailed diagnostic output of initial response
    print("\n=== INITIAL RESPONSE ===")
    print(f"Response ID: {response.id}")
    print(f"Stop reason: {response.stop_reason}")
    print(f"Model: {response.model}")
    print(f"Content blocks: {len(response.content)} blocks")
    
    for i, block in enumerate(response.content):
        print(f"\nBlock {i+1}: Type = {block.type}")
        if block.type == "thinking":
            print(f"Thinking content: {block.thinking[:150]}...")
            print(f"Signature available: {bool(getattr(block, 'signature', None))}")
        elif block.type == "text":
            print(f"Text content: {block.text}")
        elif block.type == "tool_use":
            print(f"Tool: {block.name}")
            print(f"Tool input: {block.input}")
            print(f"Tool ID: {block.id}")
    print("=== END INITIAL RESPONSE ===\n")
    
    # Extract thinking blocks to include in the conversation history
    assistant_blocks = []
    for block in response.content:
        if block.type in ["thinking", "redacted_thinking", "tool_use"]:
            assistant_blocks.append(block)
            
    # Handle tool use if required
    full_conversation = [{
        "role": "user",
        "content": "What's the weather like in Paris today?"
    }]
    
    if response.stop_reason == "tool_use":
        # Add entire assistant response with thinking blocks and tool use
        full_conversation.append({
            "role": "assistant",
            "content": assistant_blocks
        })
        
        # Find the tool_use block
        tool_use_block = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use_block:
            # Execute the tool
            print(f"\n=== EXECUTING TOOL ===")
            print(f"Tool name: {tool_use_block.name}")
            print(f"Location to check: {tool_use_block.input['location']}")
            tool_result = weather(tool_use_block.input["location"])
            print(f"Result: {tool_result}")
            print("=== TOOL EXECUTION COMPLETE ===\n")
            
            # Add tool result to conversation
            full_conversation.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": json.dumps(tool_result)
                }]
            })
            
            # Continue the conversation with the same thinking configuration
            print("\n=== SENDING FOLLOW-UP REQUEST WITH TOOL RESULT ===")
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                thinking={
                    "type": "enabled",
                    "budget_tokens": THINKING_BUDGET_TOKENS
                },
                tools=tools,
                messages=full_conversation
            )
            print(f"Follow-up response received. Stop reason: {response.stop_reason}")
    
    print_thinking_response(response)

# Run the example
tool_use_with_thinking()
```

## Multiple tool calls with thinking

This example demonstrates how to handle multiple tool calls, such as a mock news and weather service, while observing the thinking process.

```coconut
def multiple_tool_calls_with_thinking():
    # Define tools
    tools = [
        {
            "name": "weather",
            "description": "Get current weather information for a location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for."
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "news",
            "description": "Get latest news headlines for a topic.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to get news about."
                    }
                },
                "required": ["topic"]
            }
        }
    ]
    
    def weather(location):
        # Mock weather data
        weather_data = {
            "New York": {"temperature": 72, "condition": "Sunny"},
            "London": {"temperature": 62, "condition": "Cloudy"},
            "Tokyo": {"temperature": 80, "condition": "Partly cloudy"},
            "Paris": {"temperature": 65, "condition": "Rainy"},
            "Sydney": {"temperature": 85, "condition": "Clear"},
            "Berlin": {"temperature": 60, "condition": "Foggy"},
        }
        
        return weather_data.get(location, {"error": f"No weather data available for {location}"})
    
    def news(topic):
        # Mock news data
        news_data = {
            "technology": [
                "New AI breakthrough announced by research lab",
                "Tech company releases latest smartphone model",
                "Quantum computing reaches milestone achievement"
            ],
            "sports": [
                "Local team wins championship game",
                "Star player signs record-breaking contract",
                "Olympic committee announces host city for 2036"
            ],
            "weather": [
                "Storm system developing in the Atlantic",
                "Record temperatures recorded across Europe",
                "Climate scientists release new research findings"
            ]
        }
        
        return {"headlines": news_data.get(topic.lower(), ["No news available for this topic"])}
    
    # Initial request
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={
                "type": "enabled",
                "budget_tokens": THINKING_BUDGET_TOKENS
        },
        tools=tools,
        messages=[{
            "role": "user",
            "content": "What's the weather in London, and can you also tell me the latest news about technology?"
        }]
    )
    
    # Print detailed information about initial response
    print("\n=== INITIAL RESPONSE ===")
    print(f"Response ID: {response.id}")
    print(f"Stop reason: {response.stop_reason}")
    print(f"Model: {response.model}")
    print(f"Content blocks: {len(response.content)} blocks")
    
    # Print each content block
    for i, block in enumerate(response.content):
        print(f"\nBlock {i+1}: Type = {block.type}")
        if block.type == "thinking":
            print(f"Thinking content: {block.thinking[:150]}...")
            print(f"Signature available: {bool(getattr(block, 'signature', None))}")
        elif block.type == "text":
            print(f"Text content: {block.text}")
        elif block.type == "tool_use":
            print(f"Tool: {block.name}")
            print(f"Tool input: {block.input}")
            print(f"Tool ID: {block.id}")
    print("=== END INITIAL RESPONSE ===\n")
    
    # Handle potentially multiple tool calls
    full_conversation = [{
        "role": "user",
        "content": "What's the weather in London, and can you also tell me the latest news about technology?"
    }]
    
    # Track iteration count for multi-turn tool use
    iteration = 0
    
    while response.stop_reason == "tool_use":
        iteration += 1
        print(f"\n=== TOOL USE ITERATION {iteration} ===")
        
        # Extract thinking blocks and tool use to include in conversation history
        assistant_blocks = []
        for block in response.content:
            if block.type in ["thinking", "redacted_thinking", "tool_use"]:
                assistant_blocks.append(block)
        
        # Add assistant response with thinking blocks and tool use
        full_conversation.append({
            "role": "assistant",
            "content": assistant_blocks
        })
        
        # Find the tool_use block
        tool_use_block = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use_block:
            print(f"\n=== EXECUTING TOOL ===")
            print(f"Tool name: {tool_use_block.name}")
            
            # Execute the appropriate tool
            if tool_use_block.name == "weather":
                print(f"Location to check: {tool_use_block.input['location']}")
                tool_result = weather(tool_use_block.input["location"])
            elif tool_use_block.name == "news":
                print(f"Topic to check: {tool_use_block.input['topic']}")
                tool_result = news(tool_use_block.input["topic"])
            else:
                tool_result = {"error": "Unknown tool"}
                
            print(f"Result: {tool_result}")
            print("=== TOOL EXECUTION COMPLETE ===\n")
            
            # Add tool result to conversation
            full_conversation.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": json.dumps(tool_result)
                }]
            })
            
            # Continue the conversation
            print("\n=== SENDING FOLLOW-UP REQUEST WITH TOOL RESULT ===")
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                thinking={
                        "type": "enabled",
                        "budget_tokens": THINKING_BUDGET_TOKENS
                },
                tools=tools,
                messages=full_conversation
            )
            
            # Print follow-up response details
            print(f"\n=== FOLLOW-UP RESPONSE (ITERATION {iteration}) ===")
            print(f"Response ID: {response.id}")
            print(f"Stop reason: {response.stop_reason}")
            print(f"Content blocks: {len(response.content)} blocks")
            
            for i, block in enumerate(response.content):
                print(f"\nBlock {i+1}: Type = {block.type}")
                if block.type == "thinking":
                    print(f"Thinking content preview: {block.thinking[:100]}...")
                elif block.type == "text":
                    print(f"Text content preview: {block.text[:100]}...")
                elif block.type == "tool_use":
                    print(f"Tool: {block.name}")
                    print(f"Tool input preview: {str(block.input)[:100]}")
            print(f"=== END FOLLOW-UP RESPONSE (ITERATION {iteration}) ===\n")
            
            if response.stop_reason != "tool_use":
                print("\n=== FINAL RESPONSE ===")
                print_thinking_response(response)
                print("=== END FINAL RESPONSE ===")
        else:
            print("No tool_use block found in response.")
            break

# Run the example
multiple_tool_calls_with_thinking()
```

## Preserving thinking blocks

When working with extended thinking and tools, make sure to:

1. **Preserve thinking block signatures**: Each thinking block contains a cryptographic signature that validates the conversation context. These signatures must be included when passing thinking blocks back to Claude.

2. **Avoid modifying prior context**: The API will reject requests if any previous content (including thinking blocks) is modified when submitting a new request with tool results.

3. **Handle both thinking and redacted_thinking blocks**: Both types of blocks must be preserved in the conversation history, even if the content of redacted blocks is not human readable.

For more details on extended thinking without tools, see the main "Extended Thinking" notebook.

```coconut
def thinking_block_preservation_example():
    # Define a simple weather tool
    tools = [
        {
            "name": "weather",
            "description": "Get current weather information for a location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for."
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    def weather(location):
        # Mock weather data
        weather_data = {
            "New York": {"temperature": 72, "condition": "Sunny"},
            "London": {"temperature": 62, "condition": "Cloudy"},
            "Tokyo": {"temperature": 80, "condition": "Partly cloudy"},
            "Paris": {"temperature": 65, "condition": "Rainy"},
            "Sydney": {"temperature": 85, "condition": "Clear"},
            "Berlin": {"temperature": 60, "condition": "Foggy"},
        }
        
        return weather_data.get(location, {"error": f"No weather data available for {location}"})
    
    # Initial request with tool use and thinking
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={
            "type": "enabled",
            "budget_tokens": THINKING_BUDGET_TOKENS
        },
        tools=tools,
        messages=[{
            "role": "user",
            "content": "What's the weather like in Berlin right now?"
        }]
    )
    
    # Extract blocks from response
    thinking_blocks = [b for b in response.content if b.type == "thinking"]
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    
    print("\n=== INITIAL RESPONSE ===")
    print(f"Response contains:")
    print(f"- {len(thinking_blocks)} thinking blocks")
    print(f"- {len(tool_use_blocks)} tool use blocks")
    
    # Check if tool use was triggered
    if tool_use_blocks:
        tool_block = tool_use_blocks[0]
        print(f"\nTool called: {tool_block.name}")
        print(f"Location to check: {tool_block.input['location']}")
        
        # Execute the
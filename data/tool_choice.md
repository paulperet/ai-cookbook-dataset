# Guide: Mastering the `tool_choice` Parameter with Claude

This guide will walk you through the three modes of the `tool_choice` parameter when working with Claude's tool‑use capabilities: `auto`, `tool`, and `any`. You'll learn how to control when and which tools Claude calls, enabling you to build more predictable and robust AI workflows.

## Prerequisites

Ensure you have the Anthropic SDK installed and your API key configured.

```bash
pip install anthropic
```

## Setup

First, import the necessary library and initialize the client.

```python
from anthropic import Anthropic

client = Anthropic()
MODEL_NAME = "claude-sonnet-4-5"
```

## Part 1: `auto` – Let Claude Decide

Setting `tool_choice` to `"auto"` allows Claude to decide whether to use any of the provided tools. This is the default behavior.

### Step 1: Define a Mock Web Search Tool

We'll create a simple tool definition. For demonstration, the tool won't perform an actual search.

```python
def web_search(topic):
    print(f"pretending to search the web for {topic}")

web_search_tool = {
    "name": "web_search",
    "description": "A tool to retrieve up to date information on a given topic by searching the web",
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "The topic to search the web for"},
        },
        "required": ["topic"],
    },
}
```

### Step 2: Create a Chat Function with `tool_choice="auto"`

Now, build a function that sends a user query to Claude, includes the tool, and lets Claude decide whether to use it.

```python
from datetime import date

def chat_with_web_search(user_query):
    messages = [{"role": "user", "content": user_query}]

    system_prompt = f"""
    Answer as many questions as you can using your existing knowledge.
    Only search the web for queries that you can not confidently answer.
    Today's date is {date.today().strftime("%B %d %Y")}
    If you think a user's question involves something in the future that hasn't happened yet, use the search tool.
    """

    response = client.messages.create(
        system=system_prompt,
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1000,
        tool_choice={"type": "auto"},
        tools=[web_search_tool],
    )
    last_content_block = response.content[-1]
    if last_content_block.type == "text":
        print("Claude did NOT call a tool")
        print(f"Assistant: {last_content_block.text}")
    elif last_content_block.type == "tool_use":
        print("Claude wants to use a tool")
        print(last_content_block)
```

### Step 3: Test the `auto` Behavior

Let's test with questions Claude can answer from its knowledge and questions that require up‑to‑date information.

```python
# Claude should answer from knowledge
chat_with_web_search("What color is the sky?")
```

```
Claude did NOT call a tool
Assistant: The sky appears blue during the day. This is because the Earth's atmosphere scatters more blue light from the sun than other colors, making the sky look blue.
```

```python
# Claude should decide to search for recent events
chat_with_web_search("Who won the 2024 Miami Grand Prix?")
```

```
Claude wants to use a tool
ToolUseBlock(id='toolu_staging_018nwaaRebX33pHqoZZXDaSw', input={'topic': '2024 Miami Grand Prix winner'}, name='web_search', type='tool_use')
```

**Key Insight:** With `auto`, the quality of your system prompt is critical. Clear instructions help Claude decide when a tool is truly necessary.

## Part 2: `tool` – Force a Specific Tool

You can force Claude to always use a particular tool by setting `tool_choice` to `{"type": "tool", "name": "tool_name"}`. This is useful for deterministic workflows like data extraction.

### Step 1: Define Multiple Tools

We'll define two tools: one for sentiment analysis (structured output) and a simple calculator.

```python
tools = [
    {
        "name": "print_sentiment_scores",
        "description": "Prints the sentiment scores of a given tweet or piece of text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "positive_score": {
                    "type": "number",
                    "description": "The positive sentiment score, ranging from 0.0 to 1.0.",
                },
                "negative_score": {
                    "type": "number",
                    "description": "The negative sentiment score, ranging from 0.0 to 1.0.",
                },
                "neutral_score": {
                    "type": "number",
                    "description": "The neutral sentiment score, ranging from 0.0 to 1.0.",
                },
            },
            "required": ["positive_score", "negative_score", "neutral_score"],
        },
    },
    {
        "name": "calculator",
        "description": "Adds two number",
        "input_schema": {
            "type": "object",
            "properties": {
                "num1": {"type": "number", "description": "first number to add"},
                "num2": {"type": "number", "description": "second number to add"},
            },
            "required": ["num1", "num2"],
        },
    },
]
```

### Step 2: Demonstrate the Problem with `auto`

First, see what happens when Claude can choose freely with a poorly‑designed prompt.

```python
def analyze_tweet_sentiment(query):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": query}],
    )
    print(response)

analyze_tweet_sentiment("Holy cow, I just made the most incredible meal!")
```

```
ToolsBetaMessage(id='msg_staging_01ApgXx7W7qsDugdaRWh6p21', content=[TextBlock(text="That's great to hear! I don't actually have the capability to assess sentiment from text...", type='text')], ...)
```

Claude responds in plain text instead of using the tool.

### Step 3: Force the Sentiment Tool

Now, modify the function to force the `print_sentiment_scores` tool.

```python
def analyze_tweet_sentiment(query):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "tool", "name": "print_sentiment_scores"},
        messages=[{"role": "user", "content": query}],
    )
    print(response)

analyze_tweet_sentiment("Holy cow, I just made the most incredible meal!")
```

```
ToolsBetaMessage(id='msg_staging_018GtYk8Xvee3w8Eeh6pbgoq', content=[ToolUseBlock(id='toolu_staging_01FMRQ9pZniZqFUGQwTcFU4N', input={'positive_score': 0.9, 'negative_score': 0.0, 'neutral_score': 0.1}, name='print_sentiment_scores', type='tool_use')], ...)
```

Now Claude always calls the sentiment tool, producing structured JSON output. Even a math‑like tweet triggers the forced tool.

```python
analyze_tweet_sentiment("I love my cats! I had four and just adopted 2 more! Guess how many I have now?")
```

```
ToolsBetaMessage(id='msg_staging_01RACamfrHdpvLxWaNwDfZEF', content=[ToolUseBlock(id='toolu_staging_01Wb6ZKSwKvqVSKLDAte9cKU', input={'positive_score': 0.8, 'negative_score': 0.0, 'neutral_score': 0.2}, name='print_sentiment_scores', type='tool_use')], ...)
```

**Note:** Even when forcing a tool, good prompt engineering improves results. Wrap the user input in a clear instruction.

```python
def analyze_tweet_sentiment(query):
    prompt = f"""
    Analyze the sentiment in the following tweet:
    <tweet>{query}</tweet>
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "tool", "name": "print_sentiment_scores"},
        messages=[{"role": "user", "content": prompt}],
    )
    print(response)
```

## Part 3: `any` – Require Any Tool

The `any` setting tells Claude it must call one of the provided tools, but it can choose which one. This is ideal for interfaces where all responses must be tool calls, like a chatbot that only acts through defined actions.

### Step 1: Build an SMS Chatbot with Two Tools

We'll create a chatbot that can only communicate by sending texts or fetching customer info.

```python
def send_text_to_user(text):
    # In a real implementation, this would send an SMS
    print(f"TEXT MESSAGE SENT: {text}")

def get_customer_info(username):
    # Mock customer data lookup
    return {
        "username": username,
        "email": f"{username}@email.com",
        "purchases": [
            {"id": 1, "product": "computer mouse"},
            {"id": 2, "product": "screen protector"},
            {"id": 3, "product": "usb charging cable"},
        ],
    }

tools = [
    {
        "name": "send_text_to_user",
        "description": "Sends a text message to a user",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The piece of text to be sent to the user via text message",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "get_customer_info",
        "description": "gets information on a customer based on the customer's username. Response includes email, username, and previous purchases. Only call this tool once a user has provided you with their username",
        "input_schema": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "The username of the user in question.",
                },
            },
            "required": ["username"],
        },
    },
]

system_prompt = """
All your communication with a user is done via text message.
Only call tools when you have enough information to accurately call them.
Do not call the get_customer_info tool until a user has provided you with their username. This is important.
If you do not know a user's username, simply ask a user for their username.
"""
```

### Step 2: Implement the Chatbot with `tool_choice="any"`

The function will process a user message and force Claude to pick a tool.

```python
def sms_chatbot(user_message):
    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        system=system_prompt,
        model=MODEL_NAME,
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "any"},
        messages=messages,
    )
    if response.stop_reason == "tool_use":
        last_content_block = response.content[-1]
        if last_content_block.type == "tool_use":
            tool_name = last_content_block.name
            tool_inputs = last_content_block.input
            print(f"=======Claude Wants To Call The {tool_name} Tool=======")
            if tool_name == "send_text_to_user":
                send_text_to_user(tool_inputs["text"])
            elif tool_name == "get_customer_info":
                # In a full implementation, you would call the function and handle the result
                print(f"Would fetch info for: {tool_inputs['username']}")
```

### Step 3: Test the `any` Behavior

```python
# Claude lacks the username, so it should ask for it via send_text_to_user
sms_chatbot("What did I buy last?")
```

```
=======Claude Wants To Call The send_text_to_user Tool=======
TEXT MESSAGE SENT: I'd be happy to help you with information about your previous purchases! To look up your account details, I'll need your username. Could you please provide your username?
```

```python
# Now provide a username
sms_chatbot("My username is alice123")
```

```
=======Claude Wants To Call The get_customer_info Tool=======
Would fetch info for: alice123
```

With `tool_choice="any"`, Claude is forced to pick a tool every time, ensuring the chatbot only acts through its defined interfaces.

## Summary

- **`auto`**: Let Claude decide based on the prompt and its knowledge. Best for flexible assistants.
- **`tool`**: Force a specific tool. Essential for deterministic output like data extraction.
- **`any`**: Require any tool call. Perfect for agent‑like systems where all actions must be through tools.

Choose the `tool_choice` strategy that matches your application's need for flexibility versus control.
# OpenAI Prompt Caching Cookbook

## Introduction

OpenAI's prompt caching feature automatically reduces latency and costs for prompts exceeding 1024 tokens, with discounts reaching up to 80% for prompts over 10,000 tokens. This is achieved by caching repetitive information across LLM API requests at the organization level, enabling shared caches among team members while maintaining zero data retention.

The system activates automatically for qualifying prompts. When you make an API request, it checks if the beginning portion (prefix) of your prompt matches a cached version. A cache hit uses the cached content, reducing processing time and cost. If no match exists, the full prompt is processed and its prefix is cached for future use.

Key use cases include:
- **Agents using tools and structured outputs**: Cache extended tool lists and schemas
- **Coding and writing assistants**: Cache large codebase sections or workspace summaries
- **Chatbots**: Cache static portions of multi-turn conversations to maintain context efficiently

## Prerequisites

Before starting, ensure you have:
- An OpenAI API key with access to GPT-4 models
- Python 3.7+ installed
- The OpenAI Python library

### Setup

```bash
pip install openai
```

```python
import os
import json
import time
from openai import OpenAI

# Initialize the client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(organization='org-l89177bnhkme4a44292n5r3j', api_key=api_key)
```

## Example 1: Caching Tools and Multi-Turn Conversations

In this example, we'll create a customer support assistant with multiple tools and demonstrate how caching works across a multi-turn conversation. The key principle is that tool definitions and their order must remain identical for caching to work effectively.

### Step 1: Define Your Tools

First, define the function tools your assistant will use. Notice that each tool has a detailed description and parameter schema:

```python
# Define tools for customer support assistant
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an order that has not yet been shipped. Use this when a customer requests order cancellation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID."
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason for cancelling the order."
                    }
                },
                "required": ["order_id", "reason"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "return_item",
            "description": "Process a return for an order. This should be called when a customer wants to return an item and the order has already been delivered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID."
                    },
                    "item_id": {
                        "type": "string",
                        "description": "The specific item ID the customer wants to return."
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason for returning the item."
                    }
                },
                "required": ["order_id", "item_id", "reason"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_shipping_address",
            "description": "Update the shipping address for an order that hasn't been shipped yet. Use this if the customer wants to change their delivery address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID."
                    },
                    "new_address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string", "description": "The new street address."},
                            "city": {"type": "string", "description": "The new city."},
                            "state": {"type": "string", "description": "The new state."},
                            "zip": {"type": "string", "description": "The new zip code."},
                            "country": {"type": "string", "description": "The new country."}
                        },
                        "required": ["street", "city", "state", "zip", "country"],
                        "additionalProperties": False
                    }
                },
                "required": ["order_id", "new_address"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_payment_method",
            "description": "Update the payment method for an order that hasn't been completed yet. Use this if the customer wants to change their payment details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID."
                    },
                    "payment_method": {
                        "type": "object",
                        "properties": {
                            "card_number": {"type": "string", "description": "The new credit card number."},
                            "expiry_date": {"type": "string", "description": "The new credit card expiry date in MM/YY format."},
                            "cvv": {"type": "string", "description": "The new credit card CVV code."}
                        },
                        "required": ["card_number", "expiry_date", "cvv"],
                        "additionalProperties": False
                    }
                },
                "required": ["order_id", "payment_method"],
                "additionalProperties": False
            }
        }
    }
]
```

### Step 2: Create System Message and Initial Conversation

Next, create a comprehensive system message with guardrails and start the conversation:

```python
# Enhanced system message with guardrails
messages = [
    {
        "role": "system", 
        "content": (
            "You are a professional, empathetic, and efficient customer support assistant. Your mission is to provide fast, clear, "
            "and comprehensive assistance to customers while maintaining a warm and approachable tone. "
            "Always express empathy, especially when the user seems frustrated or concerned, and ensure that your language is polite and professional. "
            "Use simple and clear communication to avoid any misunderstanding, and confirm actions with the user before proceeding. "
            "In more complex or time-sensitive cases, assure the user that you're taking swift action and provide regular updates. "
            "Adapt to the user’s tone: remain calm, friendly, and understanding, even in stressful or difficult situations."
            "\n\n"
            "Additionally, there are several important guardrails that you must adhere to while assisting users:"
            "\n\n"
            "1. **Confidentiality and Data Privacy**: Do not share any sensitive information about the company or other users. When handling personal details such as order IDs, addresses, or payment methods, ensure that the information is treated with the highest confidentiality. If a user requests access to their data, only provide the necessary information relevant to their request, ensuring no other user's information is accidentally revealed."
            "\n\n"
            "2. **Secure Payment Handling**: When updating payment details or processing refunds, always ensure that payment data such as credit card numbers, CVVs, and expiration dates are transmitted and stored securely. Never display or log full credit card numbers. Confirm with the user before processing any payment changes or refunds."
            "\n\n"
            "3. **Respect Boundaries**: If a user expresses frustration or dissatisfaction, remain calm and empathetic but avoid overstepping professional boundaries. Do not make personal judgments, and refrain from using language that might escalate the situation. Stick to factual information and clear solutions to resolve the user's concerns."
            "\n\n"
            "4. **Legal Compliance**: Ensure that all actions you take comply with legal and regulatory standards. For example, if the user requests a refund, cancellation, or return, follow the company’s refund policies strictly. If the order cannot be canceled due to being shipped or another restriction, explain the policy clearly but sympathetically."
            "\n\n"
            "5. **Consistency**: Always provide consistent information that aligns with company policies. If unsure about a company policy, communicate clearly with the user, letting them know that you are verifying the information, and avoid providing false promises. If escalating an issue to another team, inform the user and provide a realistic timeline for when they can expect a resolution."
            "\n\n"
            "6. **User Empowerment**: Whenever possible, empower the user to make informed decisions. Provide them with relevant options and explain each clearly, ensuring that they understand the consequences of each choice (e.g., canceling an order may result in loss of loyalty points, etc.). Ensure that your assistance supports their autonomy."
            "\n\n"
            "7. **No Speculative Information**: Do not speculate about outcomes or provide information that you are not certain of. Always stick to verified facts when discussing order statuses, policies, or potential resolutions. If something is unclear, tell the user you will investigate further before making any commitments."
            "\n\n"
            "8. **Respectful and Inclusive Language**: Ensure that your language remains inclusive and respectful, regardless of the user's tone. Avoid making assumptions based on limited information and be mindful of diverse user needs and backgrounds."
        )
    },
    {
        "role": "user", 
        "content": (
            "Hi, I placed an order three days ago and haven't received any updates on when it's going to be delivered. "
            "Could you help me check the delivery date? My order number is #9876543210. I'm a little worried because I need this item urgently."
        )
    }
]

# Define follow-up query
user_query2 = {
    "role": "user", 
    "content": (
        "Since my order hasn't actually shipped yet, I would like to cancel it. "
        "The order number is #9876543210, and I need to cancel because I've decided to purchase it locally to get it faster. "
        "Can you help me with that? Thank you!"
    )
}
```

### Step 3: Create the Completion Function

Now create a helper function to run completions and return usage data:

```python
# Function to run completion with the provided message history and tools
def completion_run(messages, tools):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=tools,
        messages=messages,
        tool_choice="required"
    )
    usage_data = json.dumps(completion.to_dict(), indent=4)
    return usage_data
```

### Step 4: Run the Multi-Turn Conversation

Execute two consecutive runs to demonstrate caching. The first run processes the full prompt, while the second run benefits from cached tokens:

```python
# Main function to handle the two runs
def main(messages, tools, user_query2):
    # Run 1: Initial query
    print("Run 1:")
    run1 = completion_run(messages, tools)
    print(run1)

    # Delay for 7 seconds
    time.sleep(7)

    # Append user_query2 to the message history
    messages.append(user_query2)

    # Run 2: With appended query
    print("\nRun 2:")
    run2 = completion_run(messages, tools)
    print(run2)

# Run the main function
main(messages, tools, user_query2)
```

### Step 5: Analyze the Results

Examine the output to understand caching behavior:

**Run 1 Output (First Request):**
```json
{
    "usage": {
        "completion_tokens": 17,
        "prompt_tokens": 1079,
        "total_tokens": 1096,
        "prompt_tokens_details": {
            "cached_tokens": 0
        }
    }
}
```

**Run 2 Output (Second Request):**
```json
{
    "usage": {
        "completion_tokens": 64,
        "prompt_tokens": 1136,
        "total_tokens": 1200,
        "prompt_tokens_details": {
            "cached_tokens": 1024
        }
    }
}
```

**Key Observations:**
1. **Run 1** shows `"cached_tokens": 0` - no cache hit since this is the first request
2. **Run 2** shows `"cached_tokens": 1024` - significant cache hit, indicating the prompt prefix was successfully cached
3. The system message, tools, and initial user message (1079 tokens) formed the cached prefix
4. Only the new user message in Run 2 needed fresh processing

## Example 2: Caching with Images

Images in prompts also qualify for caching. Whether linked via URL or encoded in base64, images count toward your token total and rate limits. The `detail` parameter must remain consistent, as it affects how images are tokenized.

### Step 1: Define Image URLs

First, define the image URLs you'll use:

```python
sauce_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/12-04-20-saucen-by-RalfR-15.jpg/800px-12-04-20-saucen-by-RalfR-15.jpg"
veggie_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Veggies.jpg/800px-Veggies.jpg"
eggs_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Egg_shelf.jpg/450px-Egg_shelf.jpg"
milk_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Lactaid_brand.jpg/800px-Lactaid_brand.jpg"
```

### Step 2: Create Image Completion Function

Create a function that sends multiple images with a text query:

```python
def multiimage_completion(url1, url2, user_query):
    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": url1, "detail": "high"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url2, "detail": "high"}
                    },
                    {"type": "text", "text": user_query}
                ]
            }
        ],
        max_tokens=300,
    )
    print(json.dumps(completion.to_dict(), indent=4))
```

### Step 3: Run Multiple Image Completions

Execute three runs with different image combinations to demonstrate caching behavior:

```python
# First run with sauce and veggies
print("Run 1: Sauce + Veggies")
multiimage_completion(sauce_url, veggie_url, "What can I make with these ingredients?")

# Wait 7 seconds
time.sleep(7)

# Second run with same images (should cache)
print("\nRun 2: Sauce + Veggies (same as Run 1)")
multiimage_completion(sauce_url, veggie_url, "What can I make with these ingredients?")

# Wait 7 seconds
time.sleep(7)

# Third run with different first image (should not cache)
print("\nRun 3: Eggs + Veggies (different first image)")
multiimage_completion(eggs_url, veggie_url, "What can I make with these ingredients?")
```

### Step 4: Understand Image Caching Results

The output will show:
- **Run 1**: `"cached_tokens": 0` (first request, no cache)
- **Run 2**: `"cached_tokens": [some number]` (cache hit since images are identical)
- **Run 3**: `"cached_tokens": 0` (no cache hit because first image changed, breaking the prefix match)

**Important Note**: Even though Run 3 has the same second image and text query, caching requires the entire prompt prefix to match. Changing the first image breaks the prefix, preventing a cache hit.

## Best Practices for Prompt Caching

1. **Structure Your Prompts**: Place static content (instructions, examples, tool definitions) at the beginning of your prompt. Put variable content (user-specific information) at the end.

2. **Maintain Consistency**: For tools, ensure definitions and their order remain identical across requests. For images, keep the `detail` parameter consistent.

3. **Monitor Cache Performance**: Check the `cached_tokens` field in `usage.prompt_tokens_details` to verify caching is working.

4. **Understand Token Counting**: All requests display `cached_tokens`, but it will be zero for requests under 1024 tokens. Caching discounts apply to the actual number of tokens processed.

5. **Organization Scope**: Caches are shared within your organization but isolated from other organizations.

6. **Image Considerations**: Images count toward token totals and rate limits. GPT-4o-mini adds extra tokens for image processing costs.

By following these guidelines and examples, you can effectively leverage prompt caching to reduce latency and costs in your AI applications.
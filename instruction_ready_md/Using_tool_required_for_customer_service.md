# Building a Deterministic Customer Service Agent with Required Tool Calls

## Overview

This guide demonstrates how to build a customer service agent using OpenAI's ChatCompletion API with mandatory tool usage. By setting `tool_choice='required'`, you ensure the agent always uses predefined tools, creating predictable workflows ideal for customer service scenarios. We'll implement a system that consults specific instructions based on customer problems and conclude with automated testing using a simulated customer.

## Prerequisites

Ensure you have the OpenAI Python package installed and your API key configured.

```bash
pip install openai
```

## Setup

Import the required libraries and initialize the OpenAI client.

```python
import json
from openai import OpenAI
import os

client = OpenAI()
GPT_MODEL = 'gpt-4-turbo'
```

## Step 1: Define Tools and Instructions

First, define the tools your customer service agent can use and the instructions it will consult for different problem types.

### 1.1 Define the Tools

Create two tools: one for speaking to the user and another for fetching problem-specific instructions.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "speak_to_user",
            "description": "Use this to speak to the user to give them information and to ask for anything required for their case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Text of message to send to user. Can cover multiple topics."
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_instructions",
            "description": "Used to get instructions to deal with the user's problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "enum": ["fraud", "refund", "information"],
                        "description": """The type of problem the customer has. Can be one of:
                        - fraud: Required to report and resolve fraud.
                        - refund: Required to submit a refund request.
                        - information: Used for any other informational queries."""
                    }
                },
                "required": ["problem"]
            }
        }
    }
]
```

### 1.2 Define Problem-Specific Instructions

Create a set of instructions for each problem type. The agent will retrieve these based on the customer's issue.

```python
INSTRUCTIONS = [
    {
        "type": "fraud",
        "instructions": """• Ask the customer to describe the fraudulent activity, including the date and items involved in the suspected fraud.
• Offer the customer a refund.
• Report the fraud to the security team for further investigation.
• Thank the customer for contacting support and invite them to reach out with any future queries."""
    },
    {
        "type": "refund",
        "instructions": """• Confirm the customer's purchase details and verify the transaction in the system.
• Check the company's refund policy to ensure the request meets the criteria.
• Ask the customer to provide a reason for the refund.
• Submit the refund request to the accounting department.
• Inform the customer of the expected time frame for the refund processing.
• Thank the customer for contacting support and invite them to reach out with any future queries."""
    },
    {
        "type": "information",
        "instructions": """• Greet the customer and ask how you can assist them today.
• Listen carefully to the customer's query and clarify if necessary.
• Provide accurate and clear information based on the customer's questions.
• Offer to assist with any additional questions or provide further details if needed.
• Ensure the customer is satisfied with the information provided.
• Thank the customer for contacting support and invite them to reach out with any future queries."""
    }
]
```

## Step 2: Create the Assistant System Prompt

Define the system prompt that sets the behavior and constraints for your customer service agent.

```python
assistant_system_prompt = """You are a customer service assistant. Your role is to answer user questions politely and competently.
You should follow these instructions to solve the case:
- Understand their problem and get the relevant instructions.
- Follow the instructions to solve the customer's problem. Get their confirmation before performing a permanent operation like a refund or similar.
- Help them with any other problems or close the case.

Only call a tool once in a single message.
If you need to fetch a piece of information from a system or document that you don't have access to, give a clear, confident answer with some dummy values."""
```

## Step 3: Implement the Message Handling Function

Create the `submit_user_message` function that manages the conversation loop, ensuring tools are always called.

```python
def submit_user_message(user_query, conversation_messages=[]):
    """Message handling function which loops through tool calls until it reaches one that requires a response.
    Once it receives respond=True it returns the conversation_messages to the user."""

    respond = False
    user_message = {"role": "user", "content": user_query}
    conversation_messages.append(user_message)

    print(f"User: {user_query}")

    while respond is False:
        messages = [{"role": "system", "content": assistant_system_prompt}]
        [messages.append(x) for x in conversation_messages]

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0,
            tools=tools,
            tool_choice='required'
        )

        conversation_messages.append(response.choices[0].message)
        respond, conversation_messages = execute_function(response.choices[0].message, conversation_messages)

    return conversation_messages
```

## Step 4: Implement the Tool Execution Function

Create the `execute_function` function that processes tool calls and updates the conversation.

```python
def execute_function(function_calls, messages):
    """Wrapper function to execute the tool calls"""

    for function_call in function_calls.tool_calls:
        function_id = function_call.id
        function_name = function_call.function.name
        print(f"Calling function {function_name}")
        function_arguments = json.loads(function_call.function.arguments)

        if function_name == 'get_instructions':
            respond = False
            instruction_name = function_arguments['problem']
            instructions = [inst for inst in INSTRUCTIONS if inst['type'] == instruction_name][0]

            messages.append({
                "tool_call_id": function_id,
                "role": "tool",
                "name": function_name,
                "content": instructions['instructions'],
            })

        elif function_name != 'get_instructions':
            respond = True
            messages.append({
                "tool_call_id": function_id,
                "role": "tool",
                "name": function_name,
                "content": function_arguments['message'],
            })
            print(f"Assistant: {function_arguments['message']}")

    return (respond, messages)
```

## Step 5: Test the Customer Service Agent

Now, let's test the agent with a simulated customer conversation about a stolen item.

### 5.1 Initial Customer Query

```python
messages = submit_user_message("Hi, I have had an item stolen that was supposed to be delivered to me yesterday.")
```

**Output:**
```
User: Hi, I have had an item stolen that was supposed to be delivered to me yesterday.
Calling function get_instructions
Calling function speak_to_user
Assistant: I'm sorry to hear about the stolen item. Could you please provide me with more details about the fraudulent activity, including the date and the items involved? This information will help us to investigate the issue further and proceed with the necessary actions, including offering you a refund.
```

### 5.2 Provide Additional Details

```python
messages = submit_user_message("For sure, it was a shirt, it was supposed to be delivered yesterday but it never arrived.", messages)
```

**Output:**
```
User: For sure, it was a shirt, it was supposed to be delivered yesterday but it never arrived.
Calling function speak_to_user
Assistant: Thank you for providing the details. I will now proceed to report this incident to our security team for further investigation and arrange a refund for the stolen shirt. Please confirm if you would like me to go ahead with the refund.
Calling function speak_to_user
Assistant: Thank you for contacting us about this issue. Please don't hesitate to reach out if you have any more questions or need further assistance in the future.
```

### 5.3 Confirm Refund Request

```python
messages = submit_user_message("Yes I would like to proceed with the refund.", messages)
```

**Output:**
```
User: Yes I would like to proceed with the refund.
Calling function get_instructions
Calling function speak_to_user
Assistant: Thank you for confirming. I have processed the refund for the stolen shirt. The amount should be reflected in your account within 5-7 business days. If you have any more questions or need further assistance, please feel free to contact us.
```

### 5.4 Conclude the Conversation

```python
messages = submit_user_message("Thanks very much.", messages)
```

**Output:**
```
User: Thanks very much.
Calling function speak_to_user
Assistant: You're welcome! If you need any more help in the future, don't hesitate to reach out. Have a great day!
```

## Step 6: Automated Evaluation with Simulated Customers

To thoroughly test your agent, create an automated evaluation where a GPT model simulates customers with different queries.

### 6.1 Define the Customer System Prompt

```python
customer_system_prompt = """You are a user calling in to customer service.
You will talk to the agent until you have a resolution to your query.
Your query is {query}.
You will be presented with a conversation - provide answers for any assistant questions you receive. 
Here is the conversation - you are the "user" and you are speaking with the "assistant":
{chat_history}

If you don't know the details, respond with dummy values.
Once your query is resolved, respond with "DONE" """

questions = [
    'I want to get a refund for the suit I ordered last Friday.',
    'Can you tell me what your policy is for returning damaged goods?',
    'Please tell me what your complaint policy is'
]
```

### 6.2 Implement the Conversation Execution Function

```python
def execute_conversation(objective):
    conversation_messages = []
    done = False
    user_query = objective

    while done is False:
        conversation_messages = submit_user_message(user_query, conversation_messages)

        messages_string = ''
        for x in conversation_messages:
            if isinstance(x, dict):
                if x['role'] == 'user':
                    messages_string += 'User: ' + x['content'] + '\n'
                elif x['role'] == 'tool':
                    if x['name'] == 'speak_to_user':
                        messages_string += 'Assistant: ' + x['content'] + '\n'
            else:
                continue

        messages = [
            {
                "role": "system",
                "content": customer_system_prompt.format(query=objective, chat_history=messages_string)
            },
            {
                "role": "user",
                "content": "Continue the chat to solve your query. Remember, you are in the user in this exchange. Do not provide User: or Assistant: in your response"
            }
        ]

        user_response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.5
        )

        conversation_messages.append({
            "role": "user",
            "content": user_response.choices[0].message.content
        })

        if 'DONE' in user_response.choices[0].message.content:
            done = True
            print("Achieved objective, closing conversation\n\n")
        else:
            user_query = user_response.choices[0].message.content
```

### 6.3 Run the Automated Evaluation

```python
for question in questions:
    execute_conversation(question)
```

The evaluation will run through each query, with the simulated customer interacting with your agent until resolution. Each conversation concludes when the customer responds with "DONE".

## Conclusion

You've successfully built a deterministic customer service agent using required tool calls. This approach ensures predictable behavior by mandating tool usage at each step, making it ideal for structured workflows like customer service. The automated evaluation demonstrates how you can test your agent with simulated customers, providing a robust testing framework for your applications.

Key takeaways:
- Use `tool_choice='required'` to guarantee tool usage in every API call
- Structure your tools to match specific workflow steps
- Implement conversation loops that handle tool execution and response management
- Create automated testing with simulated users to validate agent performance

This pattern can be extended to various domains where deterministic, tool-driven interactions are valuable.
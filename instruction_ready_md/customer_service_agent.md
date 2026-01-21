# Guide: Creating a Customer Service Agent with Client-Side Tools

In this guide, you will build a customer service chatbot using Claude 3 and client-side tools. The agent will be able to retrieve customer information, fetch order details, and cancel orders by calling simulated backend functions. You will define the tools, handle the tool-calling loop, and test the complete workflow.

## Prerequisites

Ensure you have the Anthropic Python library installed and your API key configured.

```bash
pip install anthropic
```

## Step 1: Import Libraries and Initialize the Client

Begin by importing the necessary library and setting up the Claude client.

```python
import anthropic
import json

# Initialize the Anthropic client
client = anthropic.Client()
MODEL_NAME = "claude-3-opus-20240229"  # Or your preferred Claude 3 model
```

## Step 2: Define the Client-Side Tools

Define the tools your agent can use. Each tool includes a name, description, and an input schema that Claude will follow.

```python
tools = [
    {
        "name": "get_customer_info",
        "description": "Retrieves customer information based on their customer ID. Returns the customer's name, email, and phone number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The unique identifier for the customer.",
                }
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_order_details",
        "description": "Retrieves the details of a specific order based on the order ID. Returns the order ID, product name, quantity, price, and order status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique identifier for the order.",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancels an order based on the provided order ID. Returns a confirmation message if the cancellation is successful.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique identifier for the order to be cancelled.",
                }
            },
            "required": ["order_id"],
        },
    },
]
```

## Step 3: Simulate Backend Tool Functions

Since this is a demonstration, you will simulate the backend functions that would normally query a database or API.

```python
def get_customer_info(customer_id):
    # Simulated customer data
    customers = {
        "C1": {"name": "John Doe", "email": "john@example.com", "phone": "123-456-7890"},
        "C2": {"name": "Jane Smith", "email": "jane@example.com", "phone": "987-654-3210"},
    }
    return customers.get(customer_id, "Customer not found")

def get_order_details(order_id):
    # Simulated order data
    orders = {
        "O1": {
            "id": "O1",
            "product": "Widget A",
            "quantity": 2,
            "price": 19.99,
            "status": "Shipped",
        },
        "O2": {
            "id": "O2",
            "product": "Gadget B",
            "quantity": 1,
            "price": 49.99,
            "status": "Processing",
        },
    }
    return orders.get(order_id, "Order not found")

def cancel_order(order_id):
    # Simulated order cancellation
    if order_id in ["O1", "O2"]:
        return True
    else:
        return False
```

## Step 4: Create a Tool Call Processor

Create a helper function that routes the tool name and input to the correct simulated function.

```python
def process_tool_call(tool_name, tool_input):
    if tool_name == "get_customer_info":
        return get_customer_info(tool_input["customer_id"])
    elif tool_name == "get_order_details":
        return get_order_details(tool_input["order_id"])
    elif tool_name == "cancel_order":
        return cancel_order(tool_input["order_id"])
```

## Step 5: Build the Chatbot Interaction Loop

Now, construct the main interaction function. This function sends the user message to Claude, processes any tool calls, and continues the conversation until Claude provides a final answer.

```python
def chatbot_interaction(user_message):
    print(f"\n{'=' * 50}\nUser Message: {user_message}\n{'=' * 50}")

    # Start the conversation with the user's message
    messages = [{"role": "user", "content": user_message}]

    # Get the initial response from Claude
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        tools=tools,
        messages=messages
    )

    print(f"\nInitial Stop Reason: {response.stop_reason}")

    # If Claude wants to use a tool, enter the tool-calling loop
    while response.stop_reason == "tool_use":
        # Extract the tool use block from the response
        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"\nTool Used: {tool_name}")
        print("Tool Input:")
        print(json.dumps(tool_input, indent=2))

        # Execute the tool and get the result
        tool_result = process_tool_call(tool_name, tool_input)

        print("\nTool Result:")
        print(json.dumps(tool_result, indent=2))

        # Append the assistant's tool-use and the tool result to the message history
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(tool_result),
                }
            ],
        })

        # Send the updated conversation back to Claude
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        print(f"\nResponse Stop Reason: {response.stop_reason}")

    # Extract the final text response from Claude
    final_response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )

    print(f"\nFinal Response: {final_response}")
    return final_response
```

## Step 6: Test the Customer Service Agent

Test your agent with a variety of customer service queries to see it in action.

```python
# Test 1: Retrieve customer information
print("Test 1: Customer Info Lookup")
chatbot_interaction("Can you tell me the email address for customer C1?")

# Test 2: Check order status
print("\n\nTest 2: Order Details")
chatbot_interaction("What is the status of order O2?")

# Test 3: Cancel an order
print("\n\nTest 3: Order Cancellation")
chatbot_interaction("Please cancel order O1 for me.")
```

## Conclusion

You have successfully built a customer service agent using Claude 3 and client-side tools. The agent can:
1.  Look up customer details by ID.
2.  Retrieve order information.
3.  Process order cancellations.

In a production environment, you would replace the simulated functions with real API calls or database queries. You can extend this system by adding more tools, such as updating customer profiles, checking inventory, or initiating returns.

This pattern of defining tools, processing calls, and managing a conversational loop is a powerful foundation for building capable, tool-using AI agents.
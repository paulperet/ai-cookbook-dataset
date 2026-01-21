# Building a Barista Bot Agent with Gemini API and Automatic Function Calling

This guide walks you through building an AI-powered barista agent using the Gemini API's automatic function calling capability. You'll create a coffee ordering system where an LLM agent can interact with users, understand natural language requests, and execute specific functions to manage orders.

## Prerequisites

First, install the required package and set up authentication:

```bash
pip install -qU "google-genai>=1.0.0"
```

```python
from google import genai
from google.colab import userdata

# Initialize the client with your API key
client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))
```

## Step 1: Define the Order Management System

Create the core functions that will manage the customer's order. These functions track both the in-progress order and the confirmed order sent to the kitchen.

```python
from typing import Optional
from random import randint

# Global variables to track orders
order = []  # The in-progress order
placed_order = []  # The confirmed, completed order

def add_to_order(drink: str, modifiers: Optional[list[str]] = None) -> None:
    """Adds the specified drink to the customer's order, including any modifiers."""
    if modifiers is None:
        modifiers = []
    order.append((drink, modifiers))

def get_order() -> list[tuple[str, list[str]]]:
    """Returns the customer's order."""
    return order

def remove_item(n: int) -> str:
    """Removes the nth (one-based) item from the order.
    
    Returns:
        The item that was removed.
    """
    item, _ = order.pop(n - 1)
    return item

def clear_order() -> None:
    """Removes all items from the customer's order."""
    order.clear()

def confirm_order() -> str:
    """Asks the customer if the order is correct.
    
    Returns:
        The user's free-text response.
    """
    print("Your order:")
    if not order:
        print("  (no items)")
    
    for drink, modifiers in order:
        print(f"  {drink}")
        if modifiers:
            print(f'   - {", ".join(modifiers)}')
    
    return input("Is this correct? ")

def place_order() -> int:
    """Submit the order to the kitchen.
    
    Returns:
        The estimated number of minutes until the order is ready.
    """
    placed_order[:] = order.copy()
    clear_order()
    
    # Simulate order fulfillment time
    return randint(1, 10)
```

## Step 2: Test the Order Management Functions

Before integrating with the LLM, verify that your order management functions work correctly:

```python
# Test the ordering system
clear_order()
add_to_order("Latte", ["Extra shot"])
add_to_order("Tea")
remove_item(2)
add_to_order("Tea", ["Earl Grey", "hot"])
confirm_order()
```

When you run this, you'll see the order displayed and be prompted for confirmation.

## Step 3: Define the Barista Bot Prompt

Create a comprehensive system prompt that defines the barista's behavior, menu, and constraints:

```python
COFFEE_BOT_PROMPT = """You are a coffee order taking system and you are restricted to talk only about drinks on the MENU. Do not talk about anything but ordering MENU drinks for the customer, ever.
Your goal is to do place_order after understanding the menu items and any modifiers the customer wants.
Add items to the customer's order with add_to_order, remove specific items with remove_item, and reset the order with clear_order.
To see the contents of the order so far, call get_order (by default this is shown to you, not the user)
Always confirm_order with the user (double-check) before calling place_order. Calling confirm_order will display the order items to the user and returns their response to seeing the list. Their response may contain modifications.
Always verify and respond with drink and modifier names from the MENU before adding them to the order.
If you are unsure a drink or modifier matches those on the MENU, ask a question to clarify or redirect.
You only have the modifiers listed on the menu below: Milk options, espresso shots, caffeine, sweeteners, special requests.
Once the customer has finished ordering items, confirm_order and then place_order.

Hours: Tues, Wed, Thurs, 10am to 2pm
Prices: All drinks are free.

MENU:
Coffee Drinks:
Espresso
Americano
Cold Brew

Coffee Drinks with Milk:
Latte
Cappuccino
Cortado
Macchiato
Mocha
Flat White

Tea Drinks:
English Breakfast Tea
Green Tea
Earl Grey

Tea Drinks with Milk:
Chai Latte
Matcha Latte
London Fog

Other Drinks:
Steamer
Hot Chocolate

Modifiers:
Milk options: Whole, 2%, Oat, Almond, 2% Lactose Free; Default option: whole
Espresso shots: Single, Double, Triple, Quadruple; default: Double
Caffeine: Decaf, Regular; default: Regular
Hot-Iced: Hot, Iced; Default: Hot
Sweeteners (option to add one or more): vanilla sweetener, hazelnut sweetener, caramel sauce, chocolate sauce, sugar free vanilla sweetener
Special requests: any reasonable modification that does not involve items not on the menu, for example: 'extra hot', 'one pump', 'half caff', 'extra foam', etc.

"dirty" means add a shot of espresso to a drink that doesn't usually have it, like "Dirty Chai Latte".
"Regular milk" is the same as 'whole milk'.
"Sweetened" means add some regular sugar, not a sweetener.

Soy milk has run out of stock today, so soy is not available.
"""
```

## Step 4: Configure the Gemini Model with Tools

Now, configure the Gemini model with your ordering system functions as tools:

```python
from google.genai import types
from google.api_core import retry

# Bundle all ordering functions into a system
ordering_system = [
    add_to_order,
    get_order,
    remove_item,
    clear_order,
    confirm_order,
    place_order,
]

# Choose your model
model_name = "gemini-3-flash-preview"

# Create the chat session with tools and system instruction
chat = client.chats.create(
    model=model_name,
    config=types.GenerateContentConfig(
        tools=ordering_system,
        system_instruction=COFFEE_BOT_PROMPT,
    ),
)

# Reset order tracking
placed_order = []
order = []
```

## Step 5: Implement the Chat Loop

Create an interactive loop that connects user input to the model and displays responses:

```python
from IPython.display import display, Markdown

print("Welcome to Barista bot!\n\n")

# Main interaction loop
while not placed_order:
    # Get user input
    user_input = input("> ")
    
    # Send to the model
    response = chat.send_message(user_input)
    
    # Display the model's response
    display(Markdown(response.text))

# Order has been placed
print("\n\n[barista bot session over]")
print()
print("Your order:")
print(f"  {placed_order}\n")
print("- Thanks for using Barista Bot!")
```

## Step 6: Test Your Barista Bot

Now you can interact with your barista bot! Here's an example conversation:

```
Welcome to Barista bot!

> i would like to have a cappuccino with almond milk

Ok, I've added a Cappuccino with Almond Milk to your order. Anything else?

> do you have soy milk?

I'm sorry, I do not have soy milk. We only have Whole, 2%, Oat, Almond, and 2% Lactose Free milk options.

> do you have long black

I am sorry, I do not have Long Black. Would you like to order an Americano instead?

> no, that's all

Your order:
  Cappuccino
   - Almond Milk
Is this correct? yes

Okay, just to confirm, you would like to order: 1 Cappuccino with Almond Milk. Is that correct?

> yes

Okay, I've placed your order. It will be ready in approximately 8 minutes.

[barista bot session over]

Your order:
  [('Cappuccino', ['Almond Milk'])]

- Thanks for using Barista Bot!
```

## Experimentation Ideas

Try these scenarios to explore your barista bot's capabilities:

1. **Menu inquiries**: Ask "what coffee drinks are available?" or "what tea options do you have?"
2. **Unspecified terms**: Try "a strong latte" or "an EB tea" to see how the bot handles ambiguous requests
3. **Order modifications**: Change your mind mid-order with "uhh cancel the latte sorry"
4. **Off-menu requests**: Ask for "a babycino" to see how the bot handles unavailable items

## Key Takeaways

This tutorial demonstrated how to:

1. **Define a functional API** that represents a real-world system (coffee ordering)
2. **Use automatic function calling** to connect LLM reasoning with executable code
3. **Create a system prompt** that constrains the AI to specific domain knowledge
4. **Build an interactive agent loop** that maintains conversation context

The approach shown here provides a practical pattern for integrating traditional software systems with AI agents, allowing for natural language interaction while maintaining precise control over system behavior.

## Next Steps

To deepen your understanding of this implementation:

- Explore different system instruction strategies to modify the bot's interaction style
- Experiment with more complex function signatures and return types
- Add error handling and validation to your ordering functions
- Implement persistent order storage instead of using global variables

This barista bot example demonstrates a powerful pattern for building AI agents that can understand natural language while executing precise, controlled actions in your applications.
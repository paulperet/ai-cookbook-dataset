# Guide: Extracting a Structured Shopping List from a Recipe with Gemini

This guide demonstrates how to use the Gemini API's Python SDK to extract information from unstructured text and format it into a structured output. We'll take a recipe, extract the ingredients, and then organize them into a categorized shopping list.

## Prerequisites & Setup

First, ensure you have the required library installed and your API key configured.

### 1. Install the SDK

```bash
pip install -U -q "google-genai>=1.0.0"
```

### 2. Configure the API Client

Store your API key in an environment variable named `GOOGLE_API_KEY`. Then, initialize the Gemini client.

```python
from google import genai
import os

# Retrieve your API key from an environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 3. Select a Model

Choose a Gemini model for this task. For this example, we'll use `gemini-2.5-flash-lite`.

```python
MODEL_ID = "gemini-2.5-flash-lite"
```

## Step 1: Extract Ingredients from a Recipe

Our first task is to instruct the model to identify and list all ingredients from a provided recipe.

### Define the System Prompt

We create a system instruction that tells the model its specific role.

```python
from google.genai import types

groceries_system_prompt = """
Your task is to extract to a list all the groceries with its quantities based on the provided recipe.
Make sure that groceries are in the order of appearance.
"""

grocery_extraction_config = types.GenerateContentConfig(
    system_instruction=groceries_system_prompt
)
```

### Provide the Recipe and Generate the List

Now, we'll pass our recipe text to the model using the configuration we just defined.

```python
recipe = """
Step 1:
Grind 3 garlic cloves, knob of fresh ginger, roughly chopped, 3 spring onions to a paste in a food processor.
Add 2 tbsp of clear honey, juice from one orange, 1 tbsp of light soy sauce and 2 tbsp of vegetable oil, then blend again.
Pour the mixture over the cubed chicken from 4 small breast fillets and leave to marinate for at least 1hr.
Toss in the 20 button mushrooms for the last half an hour so they take on some of the flavour, too.

Step 2:
Thread the chicken, 20 cherry tomatoes, mushrooms and 2 large red peppers onto 20 wooden skewers,
then cook on a griddle pan for 7-8 mins each side or until the chicken is thoroughly cooked and golden brown.
Turn the kebabs frequently and baste with the marinade from time to time until evenly cooked.
Arrange on a platter, and eat with your fingers.
"""

grocery_list = client.models.generate_content(
    model=MODEL_ID,
    contents=recipe,
    config=grocery_extraction_config
)

print(grocery_list.text)
```

**Expected Output:**
```
- 3 garlic cloves
- knob of fresh ginger
- 3 spring onions
- 2 tbsp of clear honey
- 1 orange
- 1 tbsp of light soy sauce
- 2 tbsp of vegetable oil
- 4 small chicken breast fillets
- 20 button mushrooms
- 20 cherry tomatoes
- 2 large red peppers
```

Great! The model has successfully extracted a raw list of ingredients from the recipe text.

## Step 2: Format the List into a Categorized Shopping List

The raw list is useful, but for a practical shopping trip, we want it organized by category (e.g., Vegetables, Meat) and formatted with checkboxes.

### Define a New System Prompt

We instruct the model to act as a shopping list organizer.

```python
shopping_list_system_prompt = """
You are given a list of groceries. Complete the following:
- Organize groceries into categories for easier shopping.
- List each item one under another with a checkbox [].
"""

shopping_list_config = types.GenerateContentConfig(
    system_instruction=shopping_list_system_prompt
)
```

### Use Few-Shot Prompting for Desired Format

To ensure the model understands our exact formatting preference, we provide an example. This technique is called few-shot prompting.

```python
# Construct a prompt with an example format, followed by our actual grocery list.
shopping_list_prompt = f"""
LIST: 3 tomatoes, 1 turkey, 4 tomatoes
OUTPUT:
## VEGETABLES
- [ ] 7 tomatoes
## MEAT
- [ ] 1 turkey

LIST: {grocery_list.text}
OUTPUT:
"""

# Generate the final, formatted shopping list
shopping_list = client.models.generate_content(
    model=MODEL_ID,
    contents=shopping_list_prompt,
    config=shopping_list_config
)

print(shopping_list.text)
```

**Final Output:**
```
## VEGETABLES
- [ ] 3 garlic cloves
- [ ] knob of fresh ginger
- [ ] 3 spring onions
- [ ] 20 button mushrooms
- [ ] 20 cherry tomatoes
- [ ] 2 large red peppers

## FRUITS
- [ ] 1 orange

## MEAT
- [ ] 4 small chicken breast fillets

## SAUCES & OILS
- [ ] 2 tbsp of clear honey
- [ ] 1 tbsp of light soy sauce
- [ ] 2 tbsp of vegetable oil
```

## Summary

You have successfully built a two-step information extraction pipeline:
1.  **Extraction:** Used a system prompt to instruct the Gemini model to pull all ingredients and quantities from a block of recipe text.
2.  **Transformation & Formatting:** Used few-shot prompting to guide the model in categorizing the raw list and outputting it in a clean, markdown-friendly shopping list format.

This pattern—breaking a complex task into discrete steps and using clear instructions with examples—is a powerful way to get reliable, structured outputs from LLMs. You can adapt this workflow for other information extraction tasks, such as pulling dates from documents, summarizing key points from articles, or structuring data from meeting notes.
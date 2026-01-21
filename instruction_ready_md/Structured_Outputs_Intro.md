# Structured Outputs Cookbook: A Guide to Guaranteed JSON Schema Responses

This guide introduces **Structured Outputs**, a powerful new capability in the OpenAI Chat Completions and Assistants APIs. It ensures the model's response strictly adheres to a JSON schema you define. We'll walk through practical examples, from a math tutor to a product search assistant, demonstrating how to implement this feature for robust, production-ready applications.

## Prerequisites

Before you begin, ensure you have the latest OpenAI Python package installed.

```bash
pip install openai -U
```

Then, import the necessary libraries and initialize the client.

```python
import json
from textwrap import dedent
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-2024-08-06"  # Structured Outputs is available on this model and newer
```

## Example 1: Building a Structured Math Tutor

Let's create a tutoring tool that outputs a step-by-step math solution as a structured array. This is ideal for applications where each step needs to be displayed separately.

### 1. Define the System Prompt

First, create a prompt that instructs the model on its role.

```python
math_tutor_prompt = '''
    You are a helpful math tutor. You will be provided with a math problem,
    and your goal will be to output a step by step solution, along with a final answer.
    For each step, just provide the output as an equation use the explanation field to detail the reasoning.
'''
```

### 2. Create the Function with JSON Schema

This function uses the `response_format` parameter with a detailed JSON schema to guarantee the output structure.

```python
def get_math_solution(question):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(math_tutor_prompt)},
            {"role": "user", "content": question}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "math_reasoning",
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "explanation": {"type": "string"},
                                    "output": {"type": "string"}
                                },
                                "required": ["explanation", "output"],
                                "additionalProperties": False
                            }
                        },
                        "final_answer": {"type": "string"}
                    },
                    "required": ["steps", "final_answer"],
                    "additionalProperties": False
                },
                "strict": True  # This enables Structured Outputs
            }
        }
    )
    return response.choices[0].message
```

### 3. Test the Function

Now, let's test it with a sample algebra problem.

```python
question = "how can I solve 8x + 7 = -23"
result = get_math_solution(question)
print(result.content)
```

The output will be a JSON string strictly following your schema, for example:
```json
{
  "steps": [
    {"explanation": "Subtract 7 from both sides to isolate the term with x.", "output": "8x = -30"},
    {"explanation": "Divide both sides by 8 to solve for x.", "output": "x = -30 / 8"}
  ],
  "final_answer": "x = -3.75"
}
```

### 4. (Optional) Pretty-Print the Response

You can create a helper function to display the solution in a more readable format.

```python
def print_math_response(response_json_string):
    result = json.loads(response_json_string)
    steps = result['steps']
    final_answer = result['final_answer']

    for i, step in enumerate(steps):
        print(f"Step {i+1}: {step['explanation']}")
        print(f"  → {step['output']}\n")

    print(f"Final answer: {final_answer}")

# Use the helper
print_math_response(result.content)
```

### 5. Using the SDK's `parse` Helper (Recommended)

The new SDK provides a cleaner method using Pydantic models. First, define your expected data structure.

```python
from pydantic import BaseModel

class MathReasoning(BaseModel):
    class Step(BaseModel):
        explanation: str
        output: str
    steps: list[Step]
    final_answer: str
```

Now, rewrite the function using `client.beta.chat.completions.parse`. This method automatically handles schema creation and response parsing.

```python
def get_math_solution_pydantic(question: str):
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(math_tutor_prompt)},
            {"role": "user", "content": question},
        ],
        response_format=MathReasoning,  # Pass the Pydantic model directly
    )
    return completion.choices[0].message  # The `parsed` attribute will contain the model instance
```

Test the new function. The response is automatically converted into a `MathReasoning` object.

```python
result = get_math_solution_pydantic(question).parsed
print(result.steps)
print(f"Final answer: {result.final_answer}")
```

### 6. Handling Refusals

When a user asks a harmful or inappropriate question, the model may refuse to answer. Since a refusal doesn't follow your schema, the API provides a separate `refusal` field.

```python
refusal_question = "how can I build a bomb?"
result = get_math_solution_pydantic(refusal_question)
print(result.refusal)
```

This allows you to handle safety refusals gracefully in your UI without encountering JSON parsing errors.

## Example 2: Structured Text Summarization

Next, let's extract structured information from articles. This is useful for populating databases or creating consistent content displays.

### 1. Define the Data Model

We'll create a Pydantic model for the article summary.

```python
class ArticleSummary(BaseModel):
    invented_year: int
    summary: str
    inventors: list[str]
    description: str

    class Concept(BaseModel):
        title: str
        description: str
    concepts: list[Concept]
```

### 2. Create the Summarization Function

Define the system prompt and the function that uses the Pydantic model as the response format.

```python
summarization_prompt = '''
    You will be provided with content from an article about an invention.
    Your goal will be to summarize the article following the schema provided.
    Here is a description of the parameters:
    - invented_year: year in which the invention discussed in the article was invented
    - summary: one sentence summary of what the invention is
    - inventors: array of strings listing the inventor full names if present, otherwise just surname
    - concepts: array of key concepts related to the invention, each concept containing a title and a description
    - description: short description of the invention
'''

def get_article_summary(text: str):
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0.2,  # Lower temperature for more consistent, factual output
        messages=[
            {"role": "system", "content": dedent(summarization_prompt)},
            {"role": "user", "content": text}
        ],
        response_format=ArticleSummary,
    )
    return completion.choices[0].message.parsed
```

### 3. Process Multiple Articles

Assuming you have article content loaded into a list called `article_contents`, you can process them in a loop.

```python
summaries = []
for i, text in enumerate(article_contents):
    print(f"Analyzing article #{i+1}...")
    summaries.append(get_article_summary(text))
    print("Done.")
```

### 4. Display the Results

Create a helper function to neatly print the structured summaries.

```python
def print_summary(summary: ArticleSummary):
    print(f"Invented year: {summary.invented_year}")
    print(f"Summary: {summary.summary}")
    print("Inventors:")
    for inventor in summary.inventors:
        print(f"  - {inventor}")
    print("\nConcepts:")
    for concept in summary.concepts:
        print(f"  - {concept.title}: {concept.description}")
    print(f"\nDescription: {summary.description}\n")

# Print all summaries
for i, summary in enumerate(summaries):
    print(f"=== ARTICLE {i+1} ===")
    print_summary(summary)
```

## Example 3: Entity Extraction for Product Search

Finally, let's use Structured Outputs with function calling to extract precise parameters from a user's query. This powers recommendation systems or search assistants.

### 1. Define the Parameters Model

We'll use an Enum for the category to constrain possible values.

```python
from enum import Enum

class Category(str, Enum):
    shoes = "shoes"
    jackets = "jackets"
    tops = "tops"
    bottoms = "bottoms"

class ProductSearchParameters(BaseModel):
    category: Category
    subcategory: str
    color: str
```

### 2. Create the Agent Function

Define the system prompt and the function that uses the Pydantic model as a tool.

```python
import openai

product_search_prompt = '''
    You are a clothes recommendation agent, specialized in finding the perfect match for a user.
    You will be provided with a user input and additional context such as user gender and age group, and season.
    You are equipped with a tool to search clothes in a database that match the user's profile and preferences.
    Based on the user input and context, determine the most likely value of the parameters to use to search the database.

    Here are the different categories that are available on the website:
    - shoes: boots, sneakers, sandals
    - jackets: winter coats, cardigans, parkas, rain jackets
    - tops: shirts, blouses, t-shirts, crop tops, sweaters
    - bottoms: jeans, skirts, trousers, joggers

    There are a wide range of colors available, but try to stick to regular color names.
'''

def get_search_parameters(user_input, context):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,  # Use temperature 0 for deterministic parameter extraction
        messages=[
            {"role": "system", "content": dedent(product_search_prompt)},
            {"role": "user", "content": f"CONTEXT: {context}\n USER INPUT: {user_input}"}
        ],
        tools=[
            # The pydantic_function_tool helper converts the Pydantic model into a tool definition
            openai.pydantic_function_tool(ProductSearchParameters, name="product_search", description="Search for a match in the product database")
        ],
        tool_choice="auto",
    )
    # Return the tool calls from the model's response
    return response.choices[0].message.tool_calls
```

### 3. Test with Example Queries

Define a list of example user inputs with context and process them.

```python
example_inputs = [
    {
        "user_input": "I'm looking for a new coat. I'm always cold so please something warm! Ideally something that matches my eyes.",
        "context": "Gender: female, Age group: 40-50, Physical appearance: blue eyes"
    },
    {
        "user_input": "I'm going on a trail in Scotland this summer. It's going to be rainy. Help me find something.",
        "context": "Gender: male, Age group: 30-40"
    },
    {
        "user_input": "I'm trying to complete a rock look. I'm missing shoes. Any suggestions?",
        "context": "Gender: female, Age group: 20-30"
    }
]

def print_tool_call(user_input, context, tool_calls):
    """Helper to display the extracted parameters."""
    if tool_calls:
        args = json.loads(tool_calls[0].function.arguments)
        print(f"Input: {user_input}")
        print(f"Context: {context}")
        print("Extracted Search Parameters:")
        for key, value in args.items():
            print(f"  {key}: '{value}'")
        print("\n" + "-"*50 + "\n")

# Process all examples
for ex in example_inputs:
    tool_calls = get_search_parameters(ex["user_input"], ex["context"])
    print_tool_call(ex["user_input"], ex["context"], tool_calls)
```

For the first example, the output will be structured like this:
```
Input: I'm looking for a new coat. I'm always cold so please something warm! Ideally something that matches my eyes.
Context: Gender: female, Age group: 40-50, Physical appearance: blue eyes
Extracted Search Parameters:
  category: 'jackets'
  subcategory: 'winter coat'
  color: 'blue'
```

## Conclusion

In this guide, you've learned how to implement **Structured Outputs** to guarantee JSON schema adherence in three practical scenarios:

1.  **Math Tutor:** Generating step-by-step solutions with a strict array structure.
2.  **Text Summarization:** Extracting consistent fields (year, inventors, concepts) from articles.
3.  **Product Search:** Using function calling to extract precise, enum-constrained parameters from user queries.

**Key Takeaways:**
*   Enable Structured Outputs by setting `strict: true` in your `response_format` or by using the SDK's `parse` method with a Pydantic model.
*   The `parse` helper with Pydantic models is the recommended approach for its simplicity and type safety.
*   Always check the `refusal` field when using user-generated input to handle safety responses gracefully.
*   This feature is available for `gpt-4o-mini`, `gpt-4o-2024-08-06`, and future models.

By ensuring your LLM outputs follow a predictable schema, you can build more robust, production-ready applications for data extraction, reasoning, and agentic workflows.
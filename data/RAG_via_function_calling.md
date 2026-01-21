# Building a Multi-Database RAG Router with Function Calling

This guide demonstrates how to build a Retrieval-Augmented Generation (RAG) agent that uses function calling to intelligently route user queries to the appropriate database. Instead of implementing the full RAG pipeline, we focus on the routing logic, which provides several advantages:

*   **Scalability:** Easily add new databases without rewriting core logic.
*   **Flexibility:** Seamlessly switch between RAG queries and standard chat using the "auto" tool choice mode.
*   **Evaluation:** The routing component is simple to test, evaluate, and fine-tune.

We'll simulate a scenario for a fictional Lemonade company with three knowledge bases: Human Resources (HR), Product Information, and Finance.

## Prerequisites & Setup

First, install the required library and set up the Mistral client.

```bash
pip install mistralai
```

```python
from mistralai import Mistral
from getpass import getpass
import json

# Securely input your API key
api_key = getpass("Enter your Mistral API Key: ")
client = Mistral(api_key=api_key)
```

## Step 1: Generate Mock Questions for Testing

To evaluate our router, we need a diverse set of test questions. We'll use the LLM to generate questions categorized for our three databases, plus a set of "Other" unrelated questions.

```python
def generate_questions():
    """
    Generates a labeled dataset of questions for HR, Product, Finance, and Other topics.

    Returns:
        dict: A JSON object containing a list of [question, label] pairs.
    """
    chat_response = client.chat.complete(
        model="mistral-large-latest",
        response_format={"type": "json_object"},
        temperature=1,  # High temperature for variety
        messages=[
            {
                "role": "user",
                "content": """
### Context
A Lemonade company is building an internal RAG system with three data sources:
- **HR:** Information on holidays, perks, salary, office policies.
- **Product:** Details on lemonade tastes, pricing, packaging colors.
- **Finance:** Data on revenue, production costs, liabilities.

### Task
Generate 30 questions a new employee might ask. Categorize each as "HR", "Product", "Finance", or "Other". "Other" questions should be completely unrelated to the context.

### Output Format
Return a JSON object with a key "pairs". Its value should be a list of lists, where each inner list has two elements: [question, label].

Example:
{"pairs": [["How much money did the company make in 2024?", "Finance"], ["How many holiday days do I have?", "HR"]]}
"""
            }
        ]
    )
    return json.loads(chat_response.choices[0].message.content)

# Generate and store the test questions
question_labels = generate_questions()
```

## Step 2: Build the Router Agent

The core of our system is an agent that analyzes a user's question and decides which database to search. It uses function calling to make this decision explicit.

```python
def get_response(question):
    """
    Routes a user's question to the appropriate database using function calling.

    Args:
        question (str): The user's input question.

    Returns:
        str: The label of the database the agent decided to search ("HR", "Product", "Finance", "Other").
    """
    # Define the function/tool the agent can call
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_in_database",
                "description": "Search for an answer in the specified knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The exact question asked by the user.",
                        },
                        "source": {
                            "type": "string",
                            "description": "The knowledge source to search. Must be 'HR', 'Product', 'Finance', or 'Other'."
                        }
                    },
                    "required": ["source", "question"],
                },
            },
        }
    ]

    # System prompt defines the agent's role and constraints
    system_prompt = """
    You are an AI assistant for a Lemonade company.
    Your job is to help employees find information by routing their questions to the correct knowledge base.
    You have access to a single tool function called "search_in_database".
    You must call this tool for every user query.
    The "source" parameter must be exactly one of: "HR", "Product", "Finance", or "Other".
    Once you receive content from the tool, you will use it to formulate the final answer.
    Be helpful, factual, and professional. Respond in the same language as the user.
    """

    # Initialize the conversation
    chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # Get the model's response, forcing it to use the provided tool
    chat_response = client.chat.complete(
        model="mistral-large-latest",
        temperature=0.3,  # Low temperature for consistent routing
        messages=chat_history,
        tools=tools,
        tool_choice='any'  # The model must choose to use the tool
    )

    # Extract the agent's decision from the function call arguments
    chat_history.append(chat_response.choices[0].message)
    agent_decision = json.loads(chat_response.choices[0].message.tool_calls[0].function.arguments)
    db_to_search_in = agent_decision['source']

    # Print the routing decision for clarity
    print(f"Question: '{question}'")
    print(f"Agent's Decision: Search in {db_to_search_in} database\n")
    return db_to_search_in

# Test the router with a single example
get_response('What are the different tastes of lemonade?')
```

## Step 3: Evaluate the Router Agent

Now, let's run our generated test questions through the router to see how accurately it categorizes them.

```python
print("Starting Router Evaluation...\n")

for question_label in question_labels['pairs']:
    question = question_label[0]
    expected_db = question_label[1]

    # Get the agent's prediction
    predicted_db = get_response(question)

    # Check if the prediction matches the expected label
    if predicted_db == expected_db:
        print("✅ CORRECT")
    else:
        print(f"❌ INCORRECT. Expected: {expected_db}")
    print("---")
```

This evaluation loop will print each question, the agent's routing decision, and whether it was correct based on our generated labels. This provides a clear measure of the router's performance and highlights any categories where it might struggle.

## Summary

You have successfully built a scalable RAG routing system using function calling. The agent analyzes incoming questions and makes explicit decisions about which specialized knowledge base to query. This architecture separates the routing logic from the retrieval and generation steps, making the system easier to maintain, extend, and evaluate. You can now integrate this router with actual RAG backends for each database (`HR`, `Product`, `Finance`) and a general-purpose LLM for `Other` queries.
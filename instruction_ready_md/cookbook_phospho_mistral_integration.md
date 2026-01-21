# Cookbook: Integrating Mistral AI with phospho Analytics

This guide provides step-by-step instructions for integrating the phospho analytics platform with Mistral AI. You will learn how to log conversations and run an A/B test to compare model performance.

## Overview

This tutorial demonstrates how to log your Mistral AI chatbot conversations to the phospho platform. After logging, you'll use phospho's analytics to compare two Mistral models: **mistral-tiny** (formerly mistral-7b) and **mistral-large-latest**.

### What is phospho?

**phospho** is an open-source analytics platform for LLM applications. It provides automatic clustering to help discover use cases and topics, along with no-code analytics to gain insights into how your application is being used.

## Prerequisites

Ensure you have Python 3.9 or higher installed. You'll also need API keys for both Mistral AI and phospho.

### 1. Install Required Packages

First, install the necessary Python libraries:

```bash
pip install phospho==0.3.40
pip install mistralai==1.1.0
pip install python-dotenv==1.0.1
pip install tqdm==4.66.5
```

### 2. Set Up Environment Variables

Create a `.env` file in your project directory and add your API keys:

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
PHOSPHO_API_KEY=your_phospho_api_key_here
PHOSPHO_PROJECT_ID=your_phospho_project_id_here
```

Then, load these variables in your Python script:

```python
import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PHOSPHO_API_KEY = os.getenv("PHOSPHO_API_KEY")
PHOSPHO_PROJECT_ID = os.getenv("PHOSPHO_PROJECT_ID")
```

## Part 1: Logging Your Messages

### Step 1: Initialize phospho

Start by initializing the phospho client. By default, it looks for environment variables, but you can also pass parameters directly:

```python
import phospho

phospho.init(api_key=PHOSPHO_API_KEY, project_id=PHOSPHO_PROJECT_ID)
```

### Step 2: Log a Simple Completion

Now, let's log a single query-response pair. First, create a function that calls the Mistral API:

```python
import mistralai

def one_mistral_call(input_text: str) -> str:
    """Make a single call to Mistral AI and return the response."""
    client = mistralai.Mistral(api_key=MISTRAL_API_KEY)
    completion = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ],
    )
    return completion.choices[0].message.content
```

Next, call the function and log the interaction:

```python
input_text = "Thank you for your last message!"
output_text = one_mistral_call(input_text)

print(f"Assistant: {output_text}")

# Log the interaction to phospho
log_result = phospho.log(
    input=input_text,
    output=output_text,
    user_id="user-123",  # Replace with actual user ID
    version_id="one_mistral_call",
)

print(f"Logged task ID: {log_result['task_id']}")
```

When you run this, you'll see the assistant's response and confirmation that the message was logged. The message is now accessible on the phospho platform, where sentiment evaluation and language detection are performed automatically.

### Step 3: Log Streaming Completions

phospho also supports streaming responses. Here's how to log a streaming chat conversation:

```python
import phospho
import mistralai
from phospho import MutableGenerator
import uuid

def simple_mistral_chat():
    """Run an interactive chat session with streaming logging."""
    phospho.init(api_key=PHOSPHO_API_KEY, project_id=PHOSPHO_PROJECT_ID)
    client = mistralai.Mistral(api_key=MISTRAL_API_KEY)
    messages = []
    
    # Create a unique session ID for this conversation
    session_id = str(uuid.uuid4())
    
    print("Ask anything (Type /exit to quit)")

    while True:
        prompt = input("\n> ")
        if prompt == "/exit":
            break
            
        messages.append({"role": "user", "content": prompt})
        query = {
            "messages": messages,
            "model": "mistral-small-latest",
        }
        
        # Get streaming response from Mistral
        response = client.chat.stream(**query)
        
        # Wrap the response for phospho logging
        mutable_response = MutableGenerator(response, stop=lambda x: x == "")
        
        # Log the streaming response
        phospho.log(
            input=query,
            output=mutable_response,
            stream=True,
            session_id=session_id,
            user_id="user-123",
            version_id="simple_mistral_chat",
            output_to_str_function=lambda x: x["data"]["choices"][0]["delta"].get("content", ""),
        )
        
        # Display the response as it streams
        print("\nAssistant: ", end="")
        for chunk in mutable_response:
            text = chunk.data.choices[0].delta.content
            if text is not None:
                print(text, end="", flush=True)
                
        # Add assistant's response to message history
        messages.append({"role": "assistant", "content": mutable_response.collected_string})
```

Run the chat function:

```python
simple_mistral_chat()
```

The full conversation will be logged to phospho and available for analysis in the platform.

**Note:** For more logging options (async, async streaming, decorators), refer to the [phospho documentation](https://docs.phospho.ai/integrations/python/logging).

## Part 2: Evaluating Model Performance with A/B Testing

Now that you know how to log conversations, let's use phospho analytics to compare two Mistral models. We'll create a mathematical AI tutor and evaluate which model provides better pedagogical responses.

### Step 1: Define the Test Dataset

We'll use 50 mathematical problems from the MetaMath dataset. Here's a subset of the questions:

```python
maths_questions = [
    "Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?",
    "What is the total cost of purchasing equipment for all sixteen players on the football team, considering that each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80?",
    "Diego baked 12 cakes for his sister's birthday. Donald also baked 4 cakes, but ate x while waiting for the party to start. There are 15 cakes left. What is the value of unknown variable x?",
    "Convert $10101_3$ to a base 10 integer.",
    # ... (45 more questions)
]
```

### Step 2: Create Model Evaluation Functions

Define functions to query both models and log the responses:

```python
def query_mistral_model(question: str, model_name: str) -> str:
    """Query a specific Mistral model with a math question."""
    client = mistralai.Mistral(api_key=MISTRAL_API_KEY)
    
    completion = client.chat.complete(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful math tutor. Explain your reasoning step by step."},
            {"role": "user", "content": question},
        ],
        temperature=0.3,  # Lower temperature for more consistent outputs
    )
    
    return completion.choices[0].message.content

def evaluate_models_on_dataset(questions: list, model_a: str, model_b: str):
    """Run A/B test by querying both models for each question."""
    phospho.init(api_key=PHOSPHO_API_KEY, project_id=PHOSPHO_PROJECT_ID)
    
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}/{len(questions)}: {question[:50]}...")
        
        # Query Model A
        response_a = query_mistral_model(question, model_a)
        phospho.log(
            input=question,
            output=response_a,
            user_id=f"test-user-{i}",
            version_id=model_a,
            metadata={"question_index": i, "model": model_a}
        )
        
        # Query Model B
        response_b = query_mistral_model(question, model_b)
        phospho.log(
            input=question,
            output=response_b,
            user_id=f"test-user-{i}",
            version_id=model_b,
            metadata={"question_index": i, "model": model_b}
        )
        
        print(f"  {model_a}: Response logged")
        print(f"  {model_b}: Response logged")
```

### Step 3: Run the A/B Test

Execute the evaluation with your chosen models:

```python
# Define the models to compare
MODEL_A = "mistral-tiny"  # Formerly mistral-7b
MODEL_B = "mistral-large-latest"

print(f"Starting A/B test: {MODEL_A} vs {MODEL_B}")
evaluate_models_on_dataset(maths_questions[:10], MODEL_A, MODEL_B)  # Test with first 10 questions
print("\nA/B test completed! Check your phospho dashboard for analytics.")
```

### Step 4: Analyze Results in phospho Dashboard

After running the test:

1. Go to your phospho project dashboard
2. Navigate to the "Analytics" section
3. Use the version filter to compare `mistral-tiny` vs `mistral-large-latest`
4. Analyze metrics like:
   - Response quality scores
   - Response lengths
   - User engagement patterns
   - Error rates

You can also set up custom evaluations in phospho to automatically score responses based on criteria like:
- Mathematical correctness
- Clarity of explanation
- Pedagogical effectiveness

## Conclusion

You've successfully learned how to:

1. **Log Mistral AI conversations** to phospho for both simple and streaming completions
2. **Set up an A/B test** to compare different Mistral models
3. **Analyze model performance** using phospho's analytics dashboard

This workflow enables you to make data-driven decisions about which models work best for your specific use case. You can extend this approach to test different prompts, parameters, or even compare Mistral with other LLM providers.

For next steps, consider:
- Setting up automated evaluations in phospho
- Creating custom metrics for your specific domain
- Implementing continuous A/B testing in production

Remember to check the [phospho documentation](https://docs.phospho.ai) for advanced features and best practices.
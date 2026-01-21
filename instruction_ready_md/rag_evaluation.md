# Evaluating RAG with Mistral Models: LLM as a Judge Using Structured Outputs

This guide demonstrates how to use Mistral AI's models to evaluate a Retrieval-Augmented Generation (RAG) system. You will implement an "LLM as a Judge" pattern using structured outputs to assess three key metrics: context relevance, answer relevance, and groundedness.

## Prerequisites

Before you begin, ensure you have the following:

1.  A Mistral AI API key. You can obtain one from the [Mistral AI console](https://console.mistral.ai/).
2.  The required Python packages installed.

### Step 1: Install Dependencies and Set Up

First, install the necessary libraries and configure your environment.

```bash
pip install mistralai==1.5.1 httpx==0.28.1 pydantic==2.10.6 python-dateutil==2.9.0.post0 jsonpath-python==1.0.6 typing-inspect==0.9.0
```

Now, import the required modules and securely set your API key.

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import List
from getpass import getpass
from mistralai import Mistral

# Securely input your API key
api_key = getpass("Enter Mistral AI API Key: ")

# Initialize the Mistral client
client = Mistral(api_key=api_key)
model = "mistral-large-latest"
```

### Step 2: Define the Evaluation Schema

To get structured, consistent evaluations, you will define Pydantic models. These models enforce a specific output format for the LLM judge, making the results easy to parse and analyze programmatically.

First, create an `Enum` to represent the scoring scale and a constant description for clarity.

```python
class Score(str, Enum):
    no_relevance = "0"
    low_relevance = "1"
    medium_relevance = "2"
    high_relevance = "3"

# Description used across all scoring criteria
SCORE_DESCRIPTION = (
    "Score as a string between '0' and '3'. "
    "0: No relevance/Not grounded/Irrelevant - The context/answer is completely unrelated or not based on the context. "
    "1: Low relevance/Low groundedness/Somewhat relevant - The context/answer has minimal relevance or grounding. "
    "2: Medium relevance/Medium groundedness/Mostly relevant - The context/answer is somewhat relevant or grounded. "
    "3: High relevance/High groundedness/Fully relevant - The context/answer is highly relevant or grounded."
)
```

Next, define separate models for each evaluation criterion. Each model includes an `explanation` field for the judge's reasoning and a `score` field.

```python
class ContextRelevance(BaseModel):
    explanation: str = Field(..., description=("Step-by-step reasoning explaining how the retrieved context aligns with the user's query. "
                    "Consider the relevance of the information to the query's intent and the appropriateness of the context "
                    "in providing a coherent and useful response."))
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class AnswerRelevance(BaseModel):
    explanation: str = Field(..., description=("Step-by-step reasoning explaining how well the generated answer addresses the user's original query. "
                    "Consider the helpfulness and on-point nature of the answer, aligning with the user's intent and providing valuable insights."))
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class Groundedness(BaseModel):
    explanation: str = Field(..., description=("Step-by-step reasoning explaining how faithful the generated answer is to the retrieved context. "
                    "Consider the factual accuracy and reliability of the answer, ensuring it is grounded in the retrieved information."))
    score: Score = Field(..., description=SCORE_DESCRIPTION)
```

Finally, create a main model that encapsulates all three criteria into a single structured response.

```python
class RAGEvaluation(BaseModel):
    context_relevance: ContextRelevance = Field(..., description="Evaluation of the context relevance to the query, considering how well the retrieved context aligns with the user's intent." )
    answer_relevance: AnswerRelevance = Field(..., description="Evaluation of the answer relevance to the query, assessing how well the generated answer addresses the user's original query." )
    groundedness: Groundedness = Field(..., description="Evaluation of the groundedness of the generated answer, ensuring it is faithful to the retrieved context." )
```

### Step 3: Implement the Evaluation Function

With the schema defined, you can now create the core function that calls the Mistral model to perform the evaluation. This function takes the user's query, the retrieved context, and the generated answer as input.

```python
def evaluate_rag(query: str, retrieved_context: str, generated_answer: str):
    chat_response = client.chat.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a judge for evaluating a Retrieval-Augmented Generation (RAG) system. "
                    "Evaluate the context relevance, answer relevance, and groundedness based on the following criteria: "
                    "Provide a reasoning and a score as a string between '0' and '3' for each criterion. "
                    "Context Relevance: How relevant is the retrieved context to the query? "
                    "Answer Relevance: How relevant is the generated answer to the query? "
                    "Groundedness: How faithful is the generated answer to the retrieved context?"
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\nRetrieved Context: {retrieved_context}\nGenerated Answer: {generated_answer}"
            },
        ],
        response_format=RAGEvaluation,  # Instructs the model to output the structured format
        temperature=0  # Use deterministic output for consistent evaluations
    )
    # Return the parsed Pydantic object from the model's response
    return chat_response.choices[0].message.parsed
```

### Step 4: Run an Example Evaluation

Let's test the evaluation function with a sample query, context, and answer.

```python
# Define the example inputs
query = "What are the benefits of renewable energy?"
retrieved_context = "Renewable energy includes solar, wind, hydro, and geothermal energy, which are naturally replenished."
generated_answer = "Renewable energy sources like solar and wind are environmentally friendly and reduce carbon emissions."

# Call the evaluation function
evaluation = evaluate_rag(query, retrieved_context, generated_answer)

# Print the results in a readable format
print("🏆 RAG Evaluation:")
print("\nCriteria: Context Relevance")
print(f"Reasoning: {evaluation.context_relevance.explanation}")
print(f"Score: {evaluation.context_relevance.score.value}/3")

print("\nCriteria: Answer Relevance")
print(f"Reasoning: {evaluation.answer_relevance.explanation}")
print(f"Score: {evaluation.answer_relevance.score.value}/3")

print("\nCriteria: Groundedness")
print(f"Reasoning: {evaluation.groundedness.explanation}")
print(f"Score: {evaluation.groundedness.score.value}/3")
```

**Expected Output:**

The LLM judge will analyze the inputs and return a structured evaluation. For the example above, you might see output similar to this:

```
🏆 RAG Evaluation:

Criteria: Context Relevance
Reasoning: The retrieved context is relevant to the query as it defines renewable energy and lists various types such as solar, wind, hydro, and geothermal energy. It provides a basic understanding of what renewable energy encompasses, which is useful for addressing the benefits of renewable energy.
Score: 3/3

Criteria: Answer Relevance
Reasoning: The generated answer addresses the user's query by highlighting the environmental benefits of renewable energy, specifically mentioning solar and wind energy. It discusses the reduction of carbon emissions, which is a key benefit of renewable energy. However, it does not mention other types of renewable energy like hydro and geothermal, which were included in the context.
Score: 2/3

Criteria: Groundedness
Reasoning: The generated answer is mostly grounded in the retrieved context. It mentions solar and wind energy, which are part of the context. However, it does not mention hydro and geothermal energy, which were also included in the context. Additionally, the answer introduces the benefit of reducing carbon emissions, which is not explicitly stated in the context but is a well-known benefit of renewable energy.
Score: 2/3
```

## Summary

You have successfully built an automated evaluator for RAG systems using Mistral AI. By leveraging structured outputs, you can programmatically obtain detailed, multi-criteria assessments of your system's performance. You can now integrate this `evaluate_rag` function into your development pipeline to systematically test and improve your RAG applications.
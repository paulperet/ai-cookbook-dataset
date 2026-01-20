# Evaluating RAG: Using Mistral Models for LLM as a Judge (With Structured Outputs)

This cookbook shows an example of using the Mistral AI models for LLM As A Judge using structured outputs.

## Imports & API Key Setting
You can get your api key from: https://console.mistral.ai/


```python
!pip install mistralai==1.5.1 httpx==0.28.1 pydantic==2.10.6 python-dateutil==2.9.0.post0 jsonpath-python==1.0.6 typing-inspect==0.9.0
from pydantic import BaseModel, Field
from enum import Enum
from typing import List
from getpass import getpass
from mistralai import Mistral

# Define the API key and model
api_key = getpass("Enter Mistral AI API Key")
```

    Requirement already satisfied: mistralai==1.5.1 in /opt/anaconda3/lib/python3.12/site-packages (1.5.1)
    Requirement already satisfied: httpx==0.28.1 in /opt/anaconda3/lib/python3.12/site-packages (0.28.1)
    Requirement already satisfied: pydantic==2.10.6 in /opt/anaconda3/lib/python3.12/site-packages (2.10.6)
    Requirement already satisfied: python-dateutil==2.9.0.post0 in /opt/anaconda3/lib/python3.12/site-packages (2.9.0.post0)
    Requirement already satisfied: jsonpath-python==1.0.6 in /opt/anaconda3/lib/python3.12/site-packages (1.0.6)
    Requirement already satisfied: typing-inspect==0.9.0 in /opt/anaconda3/lib/python3.12/site-packages (0.9.0)
    [First Entry, ..., Last Entry]
    Enter Mistral AI API Key ········

## Main Code For LLM As A Judge For RAG (With Structured Outputs)


```python
from pydantic import BaseModel, Field
from enum import Enum
from getpass import getpass
from mistralai import Mistral

# Initialize the Mistral client with the API key
client = Mistral(api_key=api_key)
model = "mistral-large-latest"

# Define Enum for scores
class Score(str, Enum):
    no_relevance = "0"
    low_relevance = "1"
    medium_relevance = "2"
    high_relevance = "3"

# Define a constant for the score description
SCORE_DESCRIPTION = (
    "Score as a string between '0' and '3'. "
    "0: No relevance/Not grounded/Irrelevant - The context/answer is completely unrelated or not based on the context. "
    "1: Low relevance/Low groundedness/Somewhat relevant - The context/answer has minimal relevance or grounding. "
    "2: Medium relevance/Medium groundedness/Mostly relevant - The context/answer is somewhat relevant or grounded. "
    "3: High relevance/High groundedness/Fully relevant - The context/answer is highly relevant or grounded."
)

# Define separate classes for each criterion with detailed descriptions
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

class RAGEvaluation(BaseModel):
    context_relevance: ContextRelevance = Field(..., description="Evaluation of the context relevance to the query, considering how well the retrieved context aligns with the user's intent." )
    answer_relevance: AnswerRelevance = Field(..., description="Evaluation of the answer relevance to the query, assessing how well the generated answer addresses the user's original query." )
    groundedness: Groundedness = Field(..., description="Evaluation of the groundedness of the generated answer, ensuring it is faithful to the retrieved context." )

# Function to evaluate RAG metrics
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
        response_format=RAGEvaluation,
        temperature=0
    )
    return chat_response.choices[0].message.parsed

# Example usage
query = "What are the benefits of renewable energy?"
retrieved_context = "Renewable energy includes solar, wind, hydro, and geothermal energy, which are naturally replenished."
generated_answer = "Renewable energy sources like solar and wind are environmentally friendly and reduce carbon emissions."
evaluation = evaluate_rag(query, retrieved_context, generated_answer)

# Print the evaluation
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



```python

```
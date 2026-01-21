# HuatuoGPT-o1 Medical RAG and Reasoning Guide

_Authored by: [Alan Ponnachan](https://huggingface.co/AlanPonnachan)_

## Introduction

This guide demonstrates how to build a medical question-answering system using HuatuoGPT-o1, a specialized medical Large Language Model (LLM), within a Retrieval-Augmented Generation (RAG) framework. The system retrieves relevant information from a medical knowledge base and uses the model's advanced reasoning capabilities to generate detailed, structured responses.

## Prerequisites

Before you begin, ensure you have a Python environment with GPU support for optimal performance. The following libraries are required.

### Installation

Install the necessary packages using pip:

```bash
pip install transformers datasets sentence-transformers scikit-learn --upgrade -q
```

### Imports

Once installed, import the required modules:

```python
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```

## Step 1: Load the Medical Dataset

We will use the **"ChatDoctor-HealthCareMagic-100k"** dataset from Hugging Face, which contains 100,000 real-world patient-doctor dialogues. This forms our knowledge base.

```python
# Load the dataset
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
```

## Step 2: Initialize the Models

We need two models: the HuatuoGPT-o1 LLM for generation and a Sentence Transformer for creating text embeddings.

```python
# Initialize the HuatuoGPT-o1 model and tokenizer
model_name = "FreedomIntelligence/HuatuoGPT-o1-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
```

## Step 3: Prepare the Knowledge Base

To enable semantic search, we will convert the dataset's question-answer pairs into vector embeddings.

```python
# Convert the dataset to a pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset["train"])

# Combine the 'input' (question) and 'output' (answer) columns to create a context
df["combined"] = df["input"] + " " + df["output"]

# Generate embeddings for all combined contexts
print("Generating embeddings for the knowledge base...")
embeddings = embed_model.encode(
    df["combined"].tolist(), show_progress_bar=True, batch_size=128
)
print("Embeddings generated!")
```

## Step 4: Implement the Retrieval Function

This function takes a user query, embeds it, and retrieves the most similar contexts from the knowledge base using cosine similarity.

```python
def retrieve_relevant_contexts(query: str, k: int = 3) -> list:
    """
    Retrieves the k most relevant contexts to a given query.

    Args:
        query (str): The user's medical query.
        k (int): The number of relevant contexts to retrieve.

    Returns:
        list: A list of dictionaries, each containing a relevant context,
              its original question, answer, and similarity score.
    """
    # Generate the embedding for the query
    query_embedding = embed_model.encode([query])[0]

    # Calculate cosine similarity between the query and all stored embeddings
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Get the indices of the top k most similar contexts
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    # Compile the results
    contexts = []
    for idx in top_k_indices:
        contexts.append(
            {
                "question": df.iloc[idx]["input"],
                "answer": df.iloc[idx]["output"],
                "similarity": similarities[idx],
            }
        )

    return contexts
```

## Step 5: Implement the Response Generation Function

This function constructs a detailed prompt using the retrieved contexts and instructs HuatuoGPT-o1 to generate a reasoned response.

```python
def generate_structured_response(query: str, contexts: list) -> str:
    """
    Generates a detailed response using the retrieved contexts.

    Args:
        query (str): The user's medical query.
        contexts (list): A list of relevant contexts.

    Returns:
        str: The generated response.
    """
    # Format the retrieved contexts into the prompt
    context_prompt = "\n".join(
        [
            f"Reference {i+1}:"
            f"\nQuestion: {ctx['question']}"
            f"\nAnswer: {ctx['answer']}"
            for i, ctx in enumerate(contexts)
        ]
    )

    # Construct the final instruction prompt
    prompt = f"""Based on the following references and your medical knowledge, provide a detailed response:

References:
{context_prompt}

Question: {query}

By considering:
1. The key medical concepts in the question.
2. How the reference cases relate to this question.
3. What medical principles should be applied.
4. Any potential complications or considerations.

Give the final response:
"""

    # Format the prompt for the chat model and generate a response
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        num_beams=1,
        do_sample=True,
    )

    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the final response part
    final_response = response.split("Give the final response:\n")[-1]

    return final_response
```

## Step 6: Create an End-to-End Query Processor

Now, combine the retrieval and generation steps into a single, easy-to-use function.

```python
def process_query(query: str, k: int = 3) -> tuple:
    """
    Processes a medical query end-to-end.

    Args:
        query (str): The user's medical query.
        k (int): The number of relevant contexts to retrieve.

    Returns:
        tuple: The generated response and the retrieved contexts.
    """
    contexts = retrieve_relevant_contexts(query, k)
    response = generate_structured_response(query, contexts)
    return response, contexts
```

## Step 7: Run an Example

Let's test the complete system with a sample medical query.

```python
# Define a sample query
query = "I've been experiencing persistent headaches and dizziness for the past week. What could be the cause?"

# Process the query
response, contexts = process_query(query)

# Display the retrieved contexts
print("\nQuery:", query)
print("\nRelevant Contexts:")
for i, ctx in enumerate(contexts, 1):
    print(f"\nReference {i} (Similarity: {ctx['similarity']:.3f}):")
    print(f"Q: {ctx['question']}")
    print(f"A: {ctx['answer']}")

# Display the generated response
print("\nGenerated Response:")
print(response)
```

**Expected Output (Truncated for Brevity):**

```
Query: I've been experiencing persistent headaches and dizziness for the past week. What could be the cause?

Relevant Contexts:

Reference 1 (Similarity: 0.687):
Q: Dizziness, sometimes severe, nausea, sometimes severe...
A: Hello! Thank you for asking on Chat Doctor! I carefully read your question...

...

Generated Response:
assistant
## Thinking
Alright, let's think about this. So, we're dealing with someone who's been having these bouts of dizziness and headaches...
## Final Response
The symptoms of dizziness, headaches, and occasional nausea you are experiencing could be related to several underlying conditions...
```

## Conclusion and Next Steps

You have successfully built a medical RAG system using HuatuoGPT-o1. This system retrieves relevant information and leverages the model's reasoning to provide structured answers.

To enhance this system further, consider:

1.  **Parameter Tuning:** Experiment with the number of retrieved contexts (`k`) and generation parameters like `temperature`.
2.  **Domain Fine-tuning:** Fine-tune HuatuoGPT-o1 on a specialized medical corpus for improved accuracy.
3.  **Evaluation:** Assess performance using medical QA benchmarks.
4.  **Deployment:** Add a user interface (e.g., a web app using Gradio or Streamlit) for easier interaction.
5.  **Robustness:** Implement error handling and logging for production use.

Feel free to adapt this foundation to create powerful, domain-specific AI assistants.
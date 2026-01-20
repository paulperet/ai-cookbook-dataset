# HuatuoGPT-o1 Medical RAG and Reasoning

_Authored by: [Alan Ponnachan](https://huggingface.co/AlanPonnachan)_

This notebook demonstrates an end-to-end example of using HuatuoGPT-o1 for medical question answering with Retrieval-Augmented Generation (RAG) and reasoning. We'll leverage the HuatuoGPT-o1 model, a medical Large Language Model (LLM) designed for advanced medical reasoning, to provide detailed and well-structured answers to medical queries.

## Introduction

HuatuoGPT-o1 is a medical LLM that excels at identifying mistakes, exploring alternative strategies, and refining its answers. It utilizes verifiable medical problems and a specialized medical verifier to enhance its reasoning capabilities. This notebook showcases how to use HuatuoGPT-o1 in a RAG setting, where we retrieve relevant information from a medical knowledge base and then use the model to generate a reasoned response.

##  Notebook Setup


**Important:** Before running the code, ensure you are using a GPU runtime for faster performance. Go to **"Runtime" -> "Change runtime type"** and select **"GPU"** under "Hardware accelerator."

Let's start by installing the necessary libraries.


```python
!pip install transformers datasets sentence-transformers scikit-learn --upgrade -q
```

    [First Entry, ..., Last Entry]

##  Load the Dataset

We'll use the **"ChatDoctor-HealthCareMagic-100k"** dataset from the Hugging Face Datasets library. This dataset contains 100,000 real-world patient-doctor interactions, providing a rich knowledge base for our RAG system.


```python
from datasets import load_dataset

dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
```

    [First Entry, ..., Last Entry]

## Step 3: Initialize the Models

We need to initialize two models:

1. **HuatuoGPT-o1**: The medical LLM for generating responses.
2. **Sentence Transformer**: An embedding model for creating vector representations of text, which we'll use for retrieval.


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Initialize HuatuoGPT-o1
model_name = "FreedomIntelligence/HuatuoGPT-o1-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize Sentence Transformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
```

    [First Entry, ..., Last Entry]

## Prepare the Knowledge Base

We'll create a knowledge base by generating embeddings for the combined question-answer pairs from the dataset.


```python
import pandas as pd
import numpy as np

# Convert dataset to DataFrame
df = pd.DataFrame(dataset["train"])

# Combine question and answer for context
df["combined"] = df["input"] + " " + df["output"]

# Generate embeddings
print("Generating embeddings for the knowledge base...")
embeddings = embed_model.encode(
    df["combined"].tolist(), show_progress_bar=True, batch_size=128
)
print("Embeddings generated!")
```

    Generating embeddings for the knowledge base...
    [First Entry, ..., Last Entry]
    Embeddings generated!

## Implement Retrieval

This function retrieves the `k` most relevant contexts to a given query using cosine similarity.


```python
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_contexts(query: str, k: int = 3) -> list:
    """
    Retrieves the k most relevant contexts to a given query.

    Args:
        query (str): The user's medical query.
        k (int): The number of relevant contexts to retrieve.

    Returns:
        list: A list of dictionaries, each containing a relevant context.
    """
    # Generate query embedding
    query_embedding = embed_model.encode([query])[0]

    # Calculate similarities
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Get top k similar contexts
    top_k_indices = np.argsort(similarities)[-k:][::-1]

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

## Implement Response Generation

This function generates a detailed response using the retrieved contexts.


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
    # Prepare prompt with retrieved contexts
    context_prompt = "\n".join(
        [
            f"Reference {i+1}:"
            f"\nQuestion: {ctx['question']}"
            f"\nAnswer: {ctx['answer']}"
            for i, ctx in enumerate(contexts)
        ]
    )

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

    # Generate response
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

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the final response portion
    final_response = response.split("Give the final response:\n")[-1]

    return final_response
```

## Putting It All Together

Let's define a function to process a query end-to-end and then use it with an example.


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

# Example query
query = "I've been experiencing persistent headaches and dizziness for the past week. What could be the cause?"

# Process query
response, contexts = process_query(query)

# Print results
print("\nQuery:", query)
print("\nRelevant Contexts:")
for i, ctx in enumerate(contexts, 1):
    print(f"\nReference {i} (Similarity: {ctx['similarity']:.3f}):")
    print(f"Q: {ctx['question']}")
    print(f"A: {ctx['answer']}")

print("\nGenerated Response:")
print(response)
```

    
    Query: I've been experiencing persistent headaches and dizziness for the past week. What could be the cause?
    
    Relevant Contexts:
    
    Reference 1 (Similarity: 0.687):
    Q: Dizziness, sometimes severe, nausea, sometimes severe. Very close to throwing up at times, but not actually doing it. Headache. No pain anywhere, and it comes and goes a couple times in a day. I v had this about a week. I am well hydrated. I v been diagnosed with vertigo years ago, but it went away years ago, and this is nothing like that was. I feel okay between episodes, but tired. I have been laying down and sleeping when it happens, and seem ok when I get back up. It s been hit and miss, meaning not everyday. I haven t changed my diet or products
    A: Hello! Thank you for asking on Chat Doctor! I carefully read your question and would explain that your symptoms could be related to an inner ear disorder or an inflammatory disorder, causing the headache. Coming to this point, I would recommend consulting with an ENT specialist for a careful physical exam and labyrinthine tests to exclude possible inner ear disorder. Further, tests to be done are
    
    Reference 2 (Similarity: 0.673):
    Q: I have been having dizzy spells , bad headache I collapsed on the train the other day and went to hospital but hey couldnt find anything in my blood or brain scan the headache has been coming and going for about one month but te dizziness only started three days ago
    A: Hello! Welcome and thank you for asking on Chat Doctor ! Your symptoms could be related to low blood pressure or orthostatic hypotension. An inner ear disorder can not be excluded too, considering the dizzy spells. For this reason, I would recommend first consulting with an ENT specialist for a physical check up and labyrinthine tests. Other tests to consider would be a Head Up Tilt test for orthostatic hypotension, especially if your blood pressure values Chat Doctor.  Hope you will find this answer helpful! Best wishes,
    
    Reference 3 (Similarity: 0.672):
    Q: over the past two weeks or so I have had an experience of what I believe is vertigo. The first time I was mowing my lawn on a riding tractor and made a turn in the yard and felt like I was swaying back and forth. It lasted just a few minutes and thankfully I had a good grip on the stearing wheel. The second time was today, I was sitting at my desk at work and all of a sudden it seemed as though my desk was wobbiling back and forth. It wasn t the desk it was me. The first time it happened I do not recall having a headache but today I have had just a slight headache most of the day. Any suggestions?
    A: Hi, There can be many causes of vertigo. One of the most common causes is diseases associated with ear like labyrinthine (infection of the ear), vestibular neuritis (inflammation of the nerves) or BPPV (benign positional vertigo). It can also be related to diseases of brain (infection or swelling) or heart disorders (arrhythmia-rhythm disturbances) or cervical spondylosis (neck posture related issues). Besides this, there are simpler causes like anemia (low hemoglobin), hypoglycemia (low sugar), prolonged fasting, excessive heat, stress, anxiety or lack of proper sleep. Hence, I feel, first, focus on lifestyle modifications. Have a good balanced diet with lots of fruits and vegetables and less of tea and coffee. Maintain proper posture while working and sleeping, take good sleep for 7-8 hours, do some meditation or go out for a walk. If still the symptoms persist then do go for some investigations like-complete blood count, sugar levels, electrolytes, ECG, X-ray cervical spine and MRI brain. This will help us to make a proper diagnosis. Take care. Hope I have answered your question. Let me know if I can assist you further.
    
    Generated Response:
    
    assistant
    ## Thinking
    
    Alright, let's think about this. So, we're dealing with someone who's been having these bouts of dizziness and headaches for about a week now. That sounds pretty uncomfortable. Dizziness and headaches can come from a bunch of different things, right? Like, maybe it's something to do with the inner ear, or maybe it's a bit more systemic, like a problem with blood pressure or even something neurological.
    
    Okay, let's break it down. Inner ear problems, like vertigo, are pretty common culprits here. They can definitely cause dizziness and sometimes headaches, although they usually don't last forever. But since this person says their symptoms are hitting and missing, it might not be exactly the same as their old vertigo.
    
    Now, let's consider the possibility of something like orthostatic hypotension. That's where your blood pressure drops when you stand up, and it can make you dizzy. But the thing is, if this were orthostatic hypotension, we'd expect the dizziness to happen every time they stand up, which isn't quite the case here. Plus, the headaches are a bit of a wildcard.
    
    Hmm, what else could it be? Maybe anemia or hypoglycemia. Those can cause dizziness and headaches too. But again, without any major changes in diet or lifestyle, it's hard to say if that's really it.
    
    Let's see, what else should we think about? Oh, right, the person mentions they've had their blood checked and a brain scan, but nothing showed up. That rules out a lot of serious stuff like infections or brain issues, which is good news. But it also means we have to keep looking at other possibilities.
    
    Given all this, it seems like the best course of action is to consult an ENT specialist. They can do some tests specific to inner ear disorders, which might shed some light on what's going on. And if those tests don't reveal anything, maybe we should look into things like lifestyle changes, especially around diet and hydration.
    
    So, in summary, it looks like we need to keep an eye on things. The dizziness and headaches could be due to an inner ear issue or something systemic. Consulting a specialist and making some lifestyle adjustments might help figure out what's causing these symptoms.
    
    ## Final Response
    
    The symptoms of dizziness, headaches, and occasional nausea you are experiencing could be related to several underlying conditions. Based on the information provided, it appears that an inner ear disorder, such as benign paroxysmal positional vertigo (BPPV) or vestibular neuritis, is a plausible explanation. These conditions can cause episodes of dizziness and sometimes headaches, although they typically resolve on their own or improve with treatment.
    
    Another consideration is orthostatic hypotension, which involves a drop in blood pressure upon standing, potentially causing dizziness. However, given that your symptoms do not consistently occur with changes in position, this is less likely.
    
    Systemic factors, such as anemia or hypoglycemia, could also contribute to dizziness and headaches. Since these conditions can be influenced by dietary and lifestyle factors, maintaining a balanced diet, staying hydrated, and ensuring adequate rest may help alleviate symptoms.
    
    To better understand the nature of your symptoms, it would be advisable to consult with an ENT specialist for a thorough examination and possibly labyrinthine tests to assess any inner ear issues. Additionally, considering a Head-Up Tilt test for orthostatic hypotension and evaluating other systemic factors through appropriate blood tests and scans could provide further insights. 
    
    In summary, while the exact cause remains unclear, exploring options like an ENT consultation and adjusting lifestyle factors may aid in managing your symptoms.


## Conclusion

This notebook demonstrates a practical application of HuatuoGPT-o1 for medical question answering using RAG and reasoning. By combining retrieval from a relevant knowledge base with the advanced reasoning capabilities of HuatuoGPT-o1, we can build a system that provides detailed and well-structured answers to complex medical queries.

You can further enhance this system by:

*   Experimenting with different values of `k` (number of retrieved contexts).
*   Fine-tuning HuatuoGPT-o1 on a specific medical domain.
*   Evaluating the system's performance using medical benchmarks.
*   Adding a user interface for easier interaction.
*   Improving upon existing code by handling edge cases.

Feel free to adapt and expand upon this example to create even more powerful and helpful medical AI applications!
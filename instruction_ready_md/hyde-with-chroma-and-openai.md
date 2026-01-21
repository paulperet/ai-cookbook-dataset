# Robust Question Answering with Chroma and OpenAI: A Step-by-Step Guide

This guide walks you through building a robust question-answering system using [Chroma](https://trychroma.com) and OpenAI's APIs. You'll learn how to use embeddings for document retrieval and progressively improve the system's accuracy by adding context, filtering results, and implementing advanced techniques like Hypothetical Document Embeddings (HyDE).

## Prerequisites

Before you begin, ensure you have the following:

1.  An OpenAI API key. You can obtain one from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2.  Python installed on your system.

## Step 1: Environment Setup

First, install the necessary Python libraries and configure your OpenAI client.

```bash
pip install -qU openai chromadb pandas
```

```python
import os
from openai import OpenAI
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Configure your OpenAI API key
# Option 1: Set it as an environment variable before running the script.
# Option 2: Uncomment and set it directly here (not recommended for production).
# os.environ["OPENAI_API_KEY"] = 'sk-your-api-key-here'

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)
OPENAI_MODEL = "gpt-4o"  # Model to use for chat completions
print("OpenAI client is ready.")
```

## Step 2: Load and Explore the Dataset

We'll use the [SciFact dataset](https://github.com/allenai/scifact), which contains scientific claims and a corresponding corpus of paper abstracts. This provides a ground truth for evaluating our system's performance.

```python
# Adjust the path to your data directory
data_path = '../../data'

# Load the claims dataset
claim_df = pd.read_json(f'{data_path}/scifact_claims.jsonl', lines=True)
print("Claims dataset preview:")
print(claim_df.head())

# Load the corpus (documents)
corpus_df = pd.read_json(f'{data_path}/scifact_corpus.jsonl', lines=True)
print("\nCorpus dataset preview:")
print(corpus_df.head())
```

## Step 3: Establish a Baseline (Zero-Shot Evaluation)

First, let's see how well the LLM performs on the claims without any additional context. This establishes our baseline accuracy.

We'll sample 50 claims and ask the model to assess each as 'True', 'False', or 'NEE' (Not Enough Evidence).

```python
def build_prompt(claim):
    """Creates a prompt for zero-shot claim assessment."""
    return [
        {
            "role": "system",
            "content": "I will ask you to assess a scientific claim. Output only the text 'True' if the claim is true, 'False' if the claim is false, or 'NEE' if there's not enough evidence."
        },
        {
            "role": "user",
            "content": f"""
Example:

Claim:
0-dimensional biomaterials show inductive properties.

Assessment:
False

Claim:
1/2000 in UK have abnormal PrP positivity.

Assessment:
True

Claim:
Aspirin inhibits the production of PGE2.

Assessment:
False

End of examples. Assess the following claim:

Claim:
{claim}

Assessment:
"""
        }
    ]

def assess_claims(claims):
    """Queries the OpenAI API to assess a list of claims."""
    responses = []
    for claim in claims:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=build_prompt(claim),
            max_tokens=3,
        )
        # Clean the response
        assessment = response.choices[0].message.content.strip('., ')
        responses.append(assessment)
    return responses

# Sample 50 claims
samples = claim_df.sample(50, random_state=42)  # Set seed for reproducibility
claims = samples['claim'].tolist()

# Get ground truth labels from the dataset
def get_groundtruth(evidence):
    groundtruth = []
    for e in evidence:
        if len(e) == 0:
            groundtruth.append('NEE')
        else:
            # All evidence for a claim is consistent (SUPPORT or CONTRADICT)
            if list(e.values())[0][0]['label'] == 'SUPPORT':
                groundtruth.append('True')
            else:
                groundtruth.append('False')
    return groundtruth

evidence = samples['evidence'].tolist()
groundtruth = get_groundtruth(evidence)

# Assess claims with the model
gpt_inferred = assess_claims(claims)

# Evaluate performance
def confusion_matrix(inferred, groundtruth):
    """Prints a confusion matrix comparing model predictions to ground truth."""
    assert len(inferred) == len(groundtruth)
    confusion = {
        'True': {'True': 0, 'False': 0, 'NEE': 0},
        'False': {'True': 0, 'False': 0, 'NEE': 0},
        'NEE': {'True': 0, 'False': 0, 'NEE': 0},
    }
    for i, g in zip(inferred, groundtruth):
        confusion[i][g] += 1

    print('\tGroundtruth')
    print('\tTrue\tFalse\tNEE')
    for i in confusion:
        print(i, end='\t')
        for g in confusion[i]:
            print(confusion[i][g], end='\t')
        print()
    return confusion

print("Baseline (Zero-Shot) Performance:")
confusion_matrix(gpt_inferred, groundtruth)
```

**Result Analysis:** The baseline model often shows a bias, frequently labeling claims as "True" even when they are false. It also tends to be uncertain, marking many claims as "NEE." This highlights the need for providing relevant context.

## Step 4: Add Context with a Vector Database (Chroma)

We'll now load the document corpus into Chroma, a vector database, to retrieve relevant context for each claim.

### 4.1 Initialize Chroma and Create a Collection

```python
# Initialize the embedding function and Chroma client
embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()  # Ephemeral client for this session
scifact_corpus_collection = chroma_client.create_collection(
    name='scifact_corpus',
    embedding_function=embedding_function
)
```

### 4.2 Populate the Collection with Documents

We'll add documents in batches to manage memory usage. Each document is a concatenation of a paper's title and abstract.

```python
batch_size = 100
print("Loading corpus into Chroma...")
for i in range(0, len(corpus_df), batch_size):
    batch_df = corpus_df[i:i+batch_size]
    scifact_corpus_collection.add(
        ids=batch_df['doc_id'].apply(lambda x: str(x)).tolist(),
        documents=(batch_df['title'] + '. ' + batch_df['abstract'].apply(lambda x: ' '.join(x))).to_list(),
        metadatas=[{"structured": structured} for structured in batch_df['structured'].to_list()]
    )
print("Corpus loaded successfully.")
```

### 4.3 Retrieve Context and Re-evaluate Claims

Now, for each claim, we retrieve the top 3 most relevant documents from the corpus and provide them as context to the LLM.

```python
# Retrieve relevant documents for each claim
claim_query_result = scifact_corpus_collection.query(
    query_texts=claims,
    include=['documents', 'distances'],
    n_results=3
)

def build_prompt_with_context(claim, context):
    """Creates a prompt that includes retrieved documents as context."""
    return [
        {
            'role': 'system',
            'content': "I will ask you to assess whether a particular scientific claim, based on evidence provided. Output only the text 'True' if the claim is true, 'False' if the claim is false, or 'NEE' if there's not enough evidence."
        },
        {
            'role': 'user',
            'content': f"""
The evidence is the following:

{' '.join(context)}

Assess the following claim on the basis of the evidence. Output only the text 'True' if the claim is true, 'False' if the claim is false, or 'NEE' if there's not enough evidence. Do not output any other text.

Claim:
{claim}

Assessment:
"""
        }
    ]

def assess_claims_with_context(claims, contexts):
    """Assesses claims using provided document contexts."""
    responses = []
    for claim, context in zip(claims, contexts):
        if len(context) == 0:
            responses.append('NEE')
            continue
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=build_prompt_with_context(claim=claim, context=context),
            max_tokens=3,
        )
        assessment = response.choices[0].message.content.strip('., ')
        responses.append(assessment)
    return responses

# Assess claims with the retrieved context
gpt_with_context_evaluation = assess_claims_with_context(claims, claim_query_result['documents'])
print("\nPerformance with Retrieved Context:")
confusion_matrix(gpt_with_context_evaluation, groundtruth)
```

**Result Analysis:** Adding context improves the model's ability to identify false claims. However, irrelevant documents can sometimes confuse the model, leading to incorrect assessments.

## Step 5: Filter Context by Relevance

Chroma returns a distance score for each retrieved document. We can filter out documents with a distance above a certain threshold to provide cleaner context.

```python
def filter_query_result(query_result, distance_threshold=0.25):
    """Filters query results, removing documents with a distance above the threshold."""
    filtered_result = {
        'ids': [],
        'documents': [],
        'distances': []
    }
    for ids, docs, distances in zip(query_result['ids'], query_result['documents'], query_result['distances']):
        filtered_ids = []
        filtered_docs = []
        filtered_dists = []
        for id_, doc, dist in zip(ids, docs, distances):
            if dist <= distance_threshold:
                filtered_ids.append(id_)
                filtered_docs.append(doc)
                filtered_dists.append(dist)
        filtered_result['ids'].append(filtered_ids)
        filtered_result['documents'].append(filtered_docs)
        filtered_result['distances'].append(filtered_dists)
    return filtered_result

# Apply the filter
filtered_claim_query_result = filter_query_result(claim_query_result)

# Re-assess claims with filtered context
gpt_with_filtered_context_evaluation = assess_claims_with_context(claims, filtered_claim_query_result['documents'])
print("\nPerformance with Filtered Context (Threshold=0.25):")
confusion_matrix(gpt_with_filtered_context_evaluation, groundtruth)
```

**Result Analysis:** Filtering reduces noise, making the model more cautious. It correctly labels more "NEE" cases but may become overly conservative, missing some true/false classifications. The threshold is a tunable hyperparameter.

## Step 6: Improve Retrieval with Hypothetical Document Embeddings (HyDE)

The core issue is that a short claim is structurally different from a long abstract, making retrieval suboptimal. The HyDE technique improves retrieval by having the LLM *hallucinate* a plausible abstract based on the claim. We then use this generated abstract as the query to find truly relevant documents.

### 6.1 Generate Hypothetical Abstracts

```python
def build_hallucination_prompt(claim):
    """Prompts the model to generate a hypothetical scientific abstract related to the claim."""
    return [
        {
            'role': 'system',
            'content': """I will ask you to write an abstract for a scientific paper which supports or refutes a given claim. It should be written in scientific language, include a title. Output only one abstract, then stop."""
        },
        {
            'role': 'user',
            'content': f"""
Claim:
{claim}

Abstract:
"""
        }
    ]

def generate_hypothetical_abstracts(claims):
    """Generates a hypothetical abstract for each claim."""
    abstracts = []
    for claim in claims:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=build_hallucination_prompt(claim),
            max_tokens=400,
        )
        abstract = response.choices[0].message.content
        abstracts.append(abstract)
    return abstracts

print("Generating hypothetical abstracts...")
hypothetical_abstracts = generate_hypothetical_abstracts(claims)
```

### 6.2 Query the Corpus with Hypothetical Abstracts

Now, use these generated abstracts—not the original claims—to search the vector database.

```python
hyde_query_result = scifact_corpus_collection.query(
    query_texts=hypothetical_abstracts,
    include=['documents', 'distances'],
    n_results=3
)

# (Optional) Apply the same distance filter as before
filtered_hyde_result = filter_query_result(hyde_query_result, distance_threshold=0.25)
```

### 6.3 Assess Claims with HyDE-Retrieved Context

Finally, evaluate the claims using the documents retrieved via the hypothetical abstracts.

```python
gpt_with_hyde_evaluation = assess_claims_with_context(claims, filtered_hyde_result['documents'])
print("\nPerformance with HyDE-Retrieved Context:")
confusion_matrix(gpt_with_hyde_evaluation, groundtruth)
```

**Result Analysis:** HyDE often improves retrieval quality because the generated abstract is structurally similar to the documents in the corpus. This typically leads to more relevant context being retrieved, which can improve the final assessment accuracy compared to using the raw claim as a query.

## Summary and Key Takeaways

You have built and iteratively improved a robust question-answering pipeline:

1.  **Baseline:** Evaluated the LLM's zero-shot capability, identifying inherent biases.
2.  **Basic RAG:** Added context via a vector database (Chroma), improving accuracy but introducing noise.
3.  **Filtered RAG:** Applied a distance threshold to filter irrelevant documents, increasing precision but potentially reducing recall.
4.  **Advanced RAG (HyDE):** Used the LLM to generate better search queries, aligning the query structure with the document corpus for improved retrieval.

**Next Steps for Exploration:**
*   Experiment with different `distance_threshold` values to find the optimal balance for your dataset.
*   Try different embedding models or chunking strategies when loading documents into Chroma.
*   Implement a re-ranking step to further refine the retrieved documents before passing them to the LLM.
*   Explore other advanced retrieval techniques like query expansion or multi-query retrieval.

This guide demonstrates the core principles of building a production-ready retrieval-augmented generation (RAG) system.
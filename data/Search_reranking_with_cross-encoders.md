# Building a Cross-Encoder for Search Reranking

This guide demonstrates how to improve search result accuracy by implementing a cross-encoder to rerank documents retrieved by a semantic search system. Cross-encoders provide higher relevance precision than bi-encoders (embedding-based search) but are computationally expensive. The hybrid approach—using a fast bi-encoder to retrieve candidates, then a precise cross-encoder to rerank them—delivers both speed and accuracy.

## Prerequisites

Install the required Python libraries:

```bash
pip install openai arxiv tenacity pandas tiktoken
```

Set up your environment by importing the necessary modules and configuring the OpenAI client:

```python
import arxiv
from math import exp
import openai
import os
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken

# Initialize the OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

OPENAI_MODEL = "gpt-4"
```

## Step 1: Retrieve Initial Search Results

We’ll use the arXiv API to fetch academic papers related to a query. This simulates the initial retrieval step performed by a bi-encoder or any fast search system.

Define your search query and fetch results:

```python
query = "how do bi-encoders work for sentence embeddings"
search = arxiv.Search(
    query=query,
    max_results=20,
    sort_by=arxiv.SortCriterion.Relevance
)
```

Extract the relevant metadata from each result:

```python
result_list = []

for result in search.results():
    result_dict = {
        "title": result.title,
        "summary": result.summary,
        "article_url": [x.href for x in result.links][0],  # First link is usually the abstract page
        "pdf_url": [x.href for x in result.links][1]       # Second link is typically the PDF
    }
    result_list.append(result_dict)
```

Let’s inspect the first result and print all titles to see what was retrieved:

```python
print("First result:")
print(result_list[0])

print("\nAll retrieved titles:")
for i, result in enumerate(result_list):
    print(f"{i + 1}: {result['title']}")
```

## Step 2: Prepare the Cross-Encoder

A cross-encoder evaluates the relevance of a document to a query by processing them together. We’ll build one using the OpenAI Completions API with few-shot examples to guide the model.

First, set up the tokenizer and define the tokens for our binary classification ("Yes" or "No"):

```python
tokens = [" Yes", " No"]
tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL)
token_ids = [tokenizer.encode(token) for token in tokens]
print(f"Token IDs: {token_ids}")
```

Next, construct the prompt with few-shot examples. These examples teach the model to recognize relevance in your domain:

```python
prompt = '''
You are an Assistant responsible for helping detect whether the retrieved document is relevant to the query. For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved document is relevant to the query.

Query: How to plant a tree?
Document: """Cars were invented in 1886, when German inventor Carl Benz patented his Benz Patent-Motorwagen.[3][4][5] Cars became widely available during the 20th century. One of the first cars affordable by the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced horse-drawn carriages.[6] In Europe and other parts of the world, demand for automobiles did not increase until after World War II.[7] The car is considered an essential part of the developed economy."""
Relevant: No

Query: Has the coronavirus vaccine been approved?
Document: """The Pfizer-BioNTech COVID-19 vaccine was approved for emergency use in the United States on December 11, 2020."""
Relevant: Yes

Query: What is the capital of France?
Document: """Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine. Beyond such landmarks as the Eiffel Tower and the 12th-century, Gothic Notre-Dame cathedral, the city is known for its cafe culture and designer boutiques along the Rue du Faubourg Saint-Honoré."""
Relevant: Yes

Query: What are some papers to learn about PPO reinforcement learning?
Document: """Proximal Policy Optimization and its Dynamic Version for Sequence Generation: In sequence generation task, many works use policy gradient for model optimization to tackle the intractable backpropagation issue when maximizing the non-differentiable evaluation metrics or fooling the discriminator in adversarial learning. In this paper, we replace policy gradient with proximal policy optimization (PPO), which is a proved more efficient reinforcement learning algorithm, and propose a dynamic approach for PPO (PPO-dynamic). We demonstrate the efficacy of PPO and PPO-dynamic on conditional sequence generation tasks including synthetic experiment and chit-chat chatbot. The results show that PPO and PPO-dynamic can beat policy gradient by stability and performance."""
Relevant: Yes

Query: Explain sentence embeddings
Document: """Inside the bubble: exploring the environments of reionisation-era Lyman-α emitting galaxies with JADES and FRESCO: We present a study of the environments of 16 Lyman-α emitting galaxies (LAEs) in the reionisation era (5.8<z<8) identified by JWST/NIRSpec as part of the JWST Advanced Deep Extragalactic Survey (JADES). Unless situated in sufficiently (re)ionised regions, Lyman-α emission from these galaxies would be strongly absorbed by neutral gas in the intergalactic medium (IGM). We conservatively estimate sizes of the ionised regions required to reconcile the relatively low Lyman-α velocity offsets (ΔvLyα<300kms−1) with moderately high Lyman-α escape fractions (fesc,Lyα>5%) observed in our sample of LAEs, indicating the presence of ionised ``bubbles'' with physical sizes of the order of 0.1pMpc≲Rion≲1pMpc in a patchy reionisation scenario where the bubbles are embedded in a fully neutral IGM. Around half of the LAEs in our sample are found to coincide with large-scale galaxy overdensities seen in FRESCO at z∼5.8-5.9 and z∼7.3, suggesting Lyman-α transmission is strongly enhanced in such overdense regions, and underlining the importance of LAEs as tracers of the first large-scale ionised bubbles. Considering only spectroscopically confirmed galaxies, we find our sample of UV-faint LAEs (MUV≳−20mag) and their direct neighbours are generally not able to produce the required ionised regions based on the Lyman-α transmission properties, suggesting lower-luminosity sources likely play an important role in carving out these bubbles. These observations demonstrate the combined power of JWST multi-object and slitless spectroscopy in acquiring a unique view of the early stages of Cosmic Reionisation via the most distant LAEs."""
Relevant: No

Query: {query}
Document: """{document}"""
Relevant:
'''
```

Now, define the function that calls the OpenAI API. We’ll use a retry decorator for robustness and request log probabilities to quantify the model’s confidence:

```python
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def document_relevance(query, document):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt.format(query=query, document=document),
        temperature=0,
        logprobs=True,
        logit_bias={3363: 1, 1400: 1},  # Bias for " Yes" and " No" tokens
        max_tokens=1
    )

    choice = response.choices[0]
    return (
        query,
        document,
        choice.text.strip(),
        choice.logprobs.token_logprobs[0]  # Log probability of the predicted token
    )
```

## Step 3: Test the Cross-Encoder on a Single Document

Before processing all results, test the function on one document to ensure it works correctly:

```python
# Combine title and summary for the first document
content = result_list[0]["title"] + ": " + result_list[0]["summary"]

response = client.completions.create(
    model="text-davinci-003",
    prompt=prompt.format(query=query, document=content),
    temperature=0,
    logprobs=1,
    logit_bias={3363: 1, 1400: 1},
    max_tokens=1,
)

result = response.choices[0]
print(f"Prediction: {result.text.strip()}")
print(f"Log probability: {result.logprobs.token_logprobs[0]}")
```

## Step 4: Rerank All Search Results

Apply the cross-encoder to each retrieved document. This step may take a few seconds per document, so consider batching or parallelization for production use.

```python
output_list = []

for doc in result_list:
    content = doc["title"] + ": " + doc["summary"]
    try:
        output_list.append(document_relevance(query, document=content))
    except Exception as e:
        print(f"Error processing document: {e}")
```

## Step 5: Process and Sort the Results

Convert the results into a DataFrame for easy manipulation. We’ll transform log probabilities into probabilities and compute a unified “yes probability” score for ranking.

```python
output_df = pd.DataFrame(
    output_list,
    columns=["query", "document", "prediction", "logprobs"]
).reset_index()

# Convert log probability to probability
output_df["probability"] = output_df["logprobs"].apply(exp)

# Create a unified score: probability for "Yes", (1 - probability) for "No"
output_df["yes_probability"] = output_df.apply(
    lambda row: row["probability"] if row["prediction"] == "Yes" else 1 - row["probability"],
    axis=1
)

print("Processed results:")
print(output_df.head())
```

Finally, rerank the documents by their “yes probability” in descending order:

```python
reranked_df = output_df.sort_values(by=["yes_probability"], ascending=False).reset_index(drop=True)

print("Top 10 reranked documents:")
print(reranked_df[["document", "prediction", "yes_probability"]].head(10))
```

Inspect the new top document to verify the reranking:

```python
print("New top document after reranking:")
print(reranked_df["document"][0])
```

## Conclusion

You’ve successfully built a cross-encoder to rerank search results. This approach is particularly effective when:

1.  **Domain-specific relevance** is critical and not fully captured by generic embeddings.
2.  **Initial retrieval** (e.g., via semantic search) narrows the candidate set to a manageable size (e.g., 20-100 documents).

### Practical Considerations and Next Steps

- **Fine-tuning**: For highly specialized domains, consider fine-tuning a smaller model (like `ada` or `babbage`) on a larger set of labeled examples. This can reduce latency and cost while maintaining accuracy.
- **Latency**: Cross-encoders are slower than bi-encoders. Always use them only on a pre-filtered subset of results.
- **Open-source alternatives**: Explore open-source cross-encoder models on platforms like Hugging Face (e.g., `jeffwan/mmarco-mMiniLMv2-L12-H384-v1`) for cost-effective, self-hosted solutions.

This hybrid retrieval-reranking pattern is widely used in production systems—from financial report analysis to personalized content recommendations—where precision is paramount.
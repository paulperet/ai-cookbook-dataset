# Using Log Probabilities for Classification and Q&A Evaluation

This guide demonstrates how to use the `logprobs` parameter in the OpenAI Chat Completions API. Enabling `logprobs` returns the log probabilities of each output token, providing insight into the model's confidence and the alternative tokens it considered. This is useful for:

1.  **Classification:** Gauging model confidence in category predictions.
2.  **Retrieval (Q&A) Evaluation:** Implementing self-evaluation to reduce hallucinations.
3.  **Autocomplete:** Powering dynamic suggestion systems.
4.  **Token Highlighting & Perplexity Calculation:** Enabling advanced text analysis.

## Prerequisites

First, ensure you have the necessary libraries installed and your OpenAI API key configured.

```bash
pip install openai numpy
```

```python
from openai import OpenAI
import numpy as np
import os

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
```

We'll also define a helper function to simplify API calls.

```python
def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
):
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion
```

## 1. Assessing Confidence for Classification Tasks

Classification models are more useful when we understand their confidence. `logprobs` provide token-level probabilities, allowing us to set confidence thresholds or flag uncertain predictions for review.

### Step 1: Define the Classification Prompt

We'll create a prompt that asks the model to classify news headlines into one of four categories.

```python
CLASSIFICATION_PROMPT = """You will be given a headline of a news article.
Classify the article into one of the following categories: Technology, Politics, Sports, and Art.
Return only the name of the category, and nothing else.
MAKE SURE your output is one of the four categories stated.
Article headline: {headline}"""
```

### Step 2: Test Headlines Without Log Probabilities

First, let's see the model's classifications without confidence scores.

```python
headlines = [
    "Tech Giant Unveils Latest Smartphone Model with Advanced Photo-Editing Features.",
    "Local Mayor Launches Initiative to Enhance Urban Public Transport.",
    "Tennis Champion Showcases Hidden Talents in Symphony Orchestra Debut",
]

for headline in headlines:
    print(f"\nHeadline: {headline}")
    API_RESPONSE = get_completion(
        [{"role": "user", "content": CLASSIFICATION_PROMPT.format(headline=headline)}],
        model="gpt-4o",
    )
    print(f"Category: {API_RESPONSE.choices[0].message.content}")
```

The output shows the predicted categories, but we don't know how confident the model was.

### Step 3: Enable Log Probabilities for Confidence Scoring

Now, let's rerun the same prompt with `logprobs=True` and `top_logprobs=2` to see the top two most likely tokens and their probabilities.

```python
for headline in headlines:
    print(f"\nHeadline: {headline}")
    API_RESPONSE = get_completion(
        [{"role": "user", "content": CLASSIFICATION_PROMPT.format(headline=headline)}],
        model="gpt-4o-mini",
        logprobs=True,
        top_logprobs=2,
    )
    top_two_logprobs = API_RESPONSE.choices[0].logprobs.content[0].top_logprobs
    for i, logprob in enumerate(top_two_logprobs, start=1):
        linear_prob = np.round(np.exp(logprob.logprob) * 100, 2)
        print(f"  Output token {i}: '{logprob.token}', "
              f"logprob: {logprob.logprob}, "
              f"linear probability: {linear_prob}%")
```

**Analysis:**
*   **Headline 1 (Technology):** The model is 100% confident.
*   **Headline 2 (Politics):** The model is 100% confident.
*   **Headline 3 (Art):** The model is 97.23% confident, with "Sports" as a distant second choice (1.39%). This reflects the headline's mixed themes.

This visibility allows you to build systems that automatically accept high-confidence classifications and route low-confidence ones for human review.

## 2. Retrieval Confidence Scoring to Reduce Hallucinations

In Retrieval-Augmented Generation (RAG) systems, you can use `logprobs` to have the model self-evaluate whether the provided context contains sufficient information to answer a question, reducing hallucinations.

### Step 1: Define the Context and Questions

We'll use a hardcoded article about Ada Lovelace and create two sets of questions: ones easily answered by the text and ones only partially covered.

```python
# Article retrieved
ada_lovelace_article = """Augusta Ada King, Countess of Lovelace (née Byron; 10 December 1815 – 27 November 1852) was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.
Ada Byron was the only legitimate child of poet Lord Byron and reformer Lady Byron. All Lovelace's half-siblings, Lord Byron's other children, were born out of wedlock to other women. Byron separated from his wife a month after Ada was born and left England forever. He died in Greece when Ada was eight. Her mother was anxious about her upbringing and promoted Ada's interest in mathematics and logic in an effort to prevent her from developing her father's perceived insanity. Despite this, Ada remained interested in him, naming her two sons Byron and Gordon. Upon her death, she was buried next to him at her request. Although often ill in her childhood, Ada pursued her studies assiduously. She married William King in 1835. King was made Earl of Lovelace in 1838, Ada thereby becoming Countess of Lovelace.
Her educational and social exploits brought her into contact with scientists such as Andrew Crosse, Charles Babbage, Sir David Brewster, Charles Wheatstone, Michael Faraday, and the author Charles Dickens, contacts which she used to further her education. Ada described her approach as "poetical science" and herself as an "Analyst (& Metaphysician)".
When she was eighteen, her mathematical talents led her to a long working relationship and friendship with fellow British mathematician Charles Babbage, who is known as "the father of computers". She was in particular interested in Babbage's work on the Analytical Engine. Lovelace first met him in June 1833, through their mutual friend, and her private tutor, Mary Somerville.
Between 1842 and 1843, Ada translated an article by the military engineer Luigi Menabrea (later Prime Minister of Italy) about the Analytical Engine, supplementing it with an elaborate set of seven notes, simply called "Notes".
Lovelace's notes are important in the early history of computers, especially since the seventh one contained what many consider to be the first computer program—that is, an algorithm designed to be carried out by a machine. Other historians reject this perspective and point out that Babbage's personal notes from the years 1836/1837 contain the first programs for the engine. She also developed a vision of the capability of computers to go beyond mere calculating or number-crunching, while many others, including Babbage himself, focused only on those capabilities. Her mindset of "poetical science" led her to ask questions about the Analytical Engine (as shown in her notes) examining how individuals and society relate to technology as a collaborative tool.
"""

# Questions that can be easily answered given the article
easy_questions = [
    "What nationality was Ada Lovelace?",
    "What was an important finding from Lovelace's seventh note?",
]

# Questions that are not fully covered in the article
medium_questions = [
    "Did Lovelace collaborate with Charles Dickens",
    "What concepts did Lovelace build with Charles Babbage",
]
```

### Step 2: Create a Self-Evaluation Prompt

The prompt instructs the model to output only `True` or `False` based on whether the article contains sufficient information.

```python
PROMPT = """You retrieved this article: {article}. The question is: {question}.
Before even answering the question, consider whether you have sufficient information in the article to answer the question fully.
Your output should JUST be the boolean true or false, of if you have sufficient information in the article to answer the question.
Respond with just one word, the boolean true or false. You must output the word 'True', or the word 'False', nothing else.
"""
```

### Step 3: Evaluate Confidence for Each Question

We'll call the API with `logprobs=True` to get the confidence score for the `True` or `False` output.

```python
print("Questions clearly answered in article")
for question in easy_questions:
    API_RESPONSE = get_completion(
        [
            {
                "role": "user",
                "content": PROMPT.format(
                    article=ada_lovelace_article, question=question
                ),
            }
        ],
        model="gpt-4o-mini",
        logprobs=True,
    )
    logprob = API_RESPONSE.choices[0].logprobs.content[0]
    linear_prob = np.round(np.exp(logprob.logprob) * 100, 2)
    print(f"  Question: {question}")
    print(f"    Answer: {logprob.token}, Confidence: {linear_prob}%")

print("\nQuestions only partially covered in the article")
for question in medium_questions:
    API_RESPONSE = get_completion(
        [
            {
                "role": "user",
                "content": PROMPT.format(
                    article=ada_lovelace_article, question=question
                ),
            }
        ],
        model="gpt-4o",
        logprobs=True,
        top_logprobs=3,
    )
    logprob = API_RESPONSE.choices[0].logprobs.content[0]
    linear_prob = np.round(np.exp(logprob.logprob) * 100, 2)
    print(f"  Question: {question}")
    print(f"    Answer: {logprob.token}, Confidence: {linear_prob}%")
```

**Analysis:**
*   **Easy Questions:** The model is ~100% confident the context is sufficient (`True`).
*   **Medium Questions:** Confidence drops slightly (to ~99% for `False` and `True`), indicating the model recognizes the ambiguity or incomplete information in the context.

You can integrate this check into your RAG pipeline to withhold answers or request clarification when confidence falls below a defined threshold, significantly reducing hallucinations.

## 3. Building an Autocomplete System

`logprobs` can power autocomplete systems by showing the model's confidence in predicting the next token. We can choose to suggest a completion only when the model is highly confident.

### Step 1: Define the Test Sentence Fragments

We'll break a sample sentence into progressively longer prefixes.

```python
sentence_list = [
    "My",
    "My least",
    "My least favorite",
    "My least favorite TV",
    "My least favorite TV show",
    "My least favorite TV show is",
    "My least favorite TV show is Breaking Bad",
]
```

### Step 2: Simulate Autocomplete with Confidence Checking

For each fragment, we ask the model to complete the sentence and examine the log probability of its first predicted token.

```python
PROMPT = """Complete this sentence. You are acting as auto-complete. {sentence}"""

high_prob_completions = {}
low_prob_completions = {}

for sentence in sentence_list:
    API_RESPONSE = get_completion(
        [{"role": "user", "content": PROMPT.format(sentence=sentence)}],
        model="gpt-4o-mini",
        logprobs=True,
        top_logprobs=5,
    )

    # Get the top predicted token for the completion
    top_token_info = API_RESPONSE.choices[0].logprobs.content[0].top_logprobs[0]
    linear_prob = np.exp(top_token_info.logprob)

    # Store based on a confidence threshold (e.g., 50%)
    if linear_prob > 0.5:
        high_prob_completions[sentence] = {
            "next_token": top_token_info.token,
            "confidence": np.round(linear_prob * 100, 2)
        }
    else:
        low_prob_completions[sentence] = {
            "next_token": top_token_info.token,
            "confidence": np.round(linear_prob * 100, 2)
        }

print("High Confidence Completions (Confidence > 50%):")
for sent, data in high_prob_completions.items():
    print(f"  '{sent}...' -> '{data['next_token']}' ({data['confidence']}% confident)")

print("\nLow Confidence Completions (Confidence <= 50%):")
for sent, data in low_prob_completions.items():
    print(f"  '{sent}...' -> '{data['next_token']}' ({data['confidence']}% confident)")
```

In a real autocomplete system, you would only display suggestions from the `high_prob_completions` dictionary, ensuring users receive helpful and accurate prompts.

## Summary

The `logprobs` parameter unlocks a deeper understanding of model outputs by providing token-level confidence scores. This guide demonstrated three practical applications:

1.  **Classification Confidence:** Set acceptance thresholds or flag uncertain predictions.
2.  **Retrieval Self-Evaluation:** Reduce hallucinations in Q&A systems by verifying context sufficiency.
3.  **Intelligent Autocomplete:** Suggest next tokens only when the model is highly confident.

By integrating `logprobs` into your workflows, you can build more reliable, transparent, and user-friendly AI applications.
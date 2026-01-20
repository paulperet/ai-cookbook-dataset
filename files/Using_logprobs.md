# Using logprobs for classification and Q&A evaluation

This notebook demonstrates the use of the `logprobs` parameter in the Chat Completions API. When `logprobs` is enabled, the API returns the log probabilities of each output token, along with a limited number of the most likely tokens at each token position and their log probabilities. The relevant request parameters are:
* `logprobs`: Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
* `top_logprobs`: An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. `logprobs` must be set to true if this parameter is used.

Log probabilities of output tokens indicate the likelihood of each token occurring in the sequence given the context. To simplify, a logprob is `log(p)`, where `p` = probability of a token occurring at a specific position based on the previous tokens in the context. Some key points about `logprobs`:
* Higher log probabilities suggest a higher likelihood of the token in that context. This allows users to gauge the model's confidence in its output or explore alternative responses the model considered.
* Logprob can be any negative number or `0.0`. `0.0` corresponds to 100% probability.
* Logprobs allow us to compute the joint probability of a sequence as the sum of the logprobs of the individual tokens. This is useful for scoring and ranking model outputs. Another common approach is to take the average per-token logprob of a sentence to choose the best generation.
* We can examine the `logprobs` assigned to different candidate tokens to understand what options the model considered plausible or implausible.

While there are a wide array of use cases for `logprobs`, this notebook will focus on its use for:

1. Classification tasks

* Large Language Models excel at many classification tasks, but accurately measuring the model's confidence in its outputs can be challenging. `logprobs` provide a probability associated with each class prediction, enabling users to set their own classification or confidence thresholds.

2. Retrieval (Q&A) evaluation

* `logprobs` can assist with self-evaluation in retrieval applications. In the Q&A example, the model outputs a contrived `has_sufficient_context_for_answer` boolean, which can serve as a confidence score of whether the answer is contained in the retrieved content. Evaluations of this type can reduce retrieval-based hallucinations and enhance accuracy.

3. Autocomplete
*  `logprobs` could help us decide how to suggest words as a user is typing.

4. Token highlighting and outputting bytes
*  Users can easily create a token highlighter using the built in tokenization that comes with enabling `logprobs`. Additionally, the bytes parameter includes the ASCII encoding of each output character, which is particularly useful for reproducing emojis and special characters.

5. Calculating perplexity
* `logprobs` can be used to help us assess the model's overall confidence in a result and help us compare the confidence of results from different prompts.

## 0. Imports and utils


```python
from openai import OpenAI
from math import exp
import numpy as np
from IPython.display import display, HTML
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
```


```python
def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4",
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
    top_logprobs=None,
) -> str:
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

## 1. Using `logprobs` to assess confidence for classification tasks

Let's say we want to create a system to classify news articles into a set of pre-defined categories. Without `logprobs`, we can use Chat Completions to do this, but it is much more difficult to assess the certainty with which the model made its classifications.

Now, with `logprobs` enabled, we can see exactly how confident the model is in its predictions, which is crucial for creating an accurate and trustworthy classifier. For example, if the log probability for the chosen category is high, this suggests the model is quite confident in its classification. If it's low, this suggests the model is less confident. This can be particularly useful in cases where the model's classification is not what you expected, or when the model's output needs to be reviewed or validated by a human.

We'll begin with a prompt that presents the model with four categories: **Technology, Politics, Sports, and Arts**. The model is then tasked with classifying articles into these categories based solely on their headlines.


```python
CLASSIFICATION_PROMPT = """You will be given a headline of a news article.
Classify the article into one of the following categories: Technology, Politics, Sports, and Art.
Return only the name of the category, and nothing else.
MAKE SURE your output is one of the four categories stated.
Article headline: {headline}"""

```

Let's look at three sample headlines, and first begin with a standard Chat Completions output, without `logprobs`


```python
headlines = [
    "Tech Giant Unveils Latest Smartphone Model with Advanced Photo-Editing Features.",
    "Local Mayor Launches Initiative to Enhance Urban Public Transport.",
    "Tennis Champion Showcases Hidden Talents in Symphony Orchestra Debut",
]

```


```python
for headline in headlines:
    print(f"\nHeadline: {headline}")
    API_RESPONSE = get_completion(
        [{"role": "user", "content": CLASSIFICATION_PROMPT.format(headline=headline)}],
        model="gpt-4o",
    )
    print(f"Category: {API_RESPONSE.choices[0].message.content}\n")
```

    
    Headline: Tech Giant Unveils Latest Smartphone Model with Advanced Photo-Editing Features.
    Category: Technology
    
    
    Headline: Local Mayor Launches Initiative to Enhance Urban Public Transport.
    Category: Politics
    
    
    Headline: Tennis Champion Showcases Hidden Talents in Symphony Orchestra Debut
    Category: Art
    


Here we can see the selected category for each headline. However, we have no visibility into the confidence of the model in its predictions. Let's rerun the same prompt but with `logprobs` enabled, and `top_logprobs` set to 2 (this will show us the 2 most likely output tokens for each token). Additionally we can also output the linear probability of each output token, in order to convert the log probability to the more easily interprable scale of 0-100%. 


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
    html_content = ""
    for i, logprob in enumerate(top_two_logprobs, start=1):
        html_content += (
            f"Output token {i}: {logprob.token}, "
            f"logprobs: {logprob.logprob}, "
            f"linear probability: {np.round(np.exp(logprob.logprob)*100,2)}%"
        )
    display(HTML(html_content))
    print("\n")
```

    
    Headline: Tech Giant Unveils Latest Smartphone Model with Advanced Photo-Editing Features.


Output token 1: Technology, logprobs: 0.0, linear probability: 100.0%<br>Output token 2:  Technology, logprobs: -18.75, linear probability: 0.0%


    
    
    
    Headline: Local Mayor Launches Initiative to Enhance Urban Public Transport.


Output token 1: Politics, logprobs: -3.1281633e-07, linear probability: 100.0%<br>Output token 2: Polit, logprobs: -16.0, linear probability: 0.0%


    
    
    
    Headline: Tennis Champion Showcases Hidden Talents in Symphony Orchestra Debut


Output token 1: Art, logprobs: -0.028133942, linear probability: 97.23%<br>Output token 2: Sports, logprobs: -4.278134, linear probability: 1.39%


    
    


As expected from the first two headlines, gpt-4o-mini is 100% confident in its classifications, as the content is clearly technology and politics focused, respectively. However, the third headline combines both sports and art-related themes, resulting in slightly lower confidence at 97%, while still demonstrating strong certainty in its classification.

`logprobs` are quite useful for classification tasks. They allow us to set confidence thresholds or output multiple potential tokens if the log probability of the selected output is not sufficiently high. For instance, when creating a recommendation engine to tag articles, we can automatically classify headlines that exceed a certain threshold and send less certain ones for manual review.

## 2. Retrieval confidence scoring to reduce hallucinations

To reduce hallucinations, and the performance of our RAG-based Q&A system, we can use `logprobs` to evaluate how confident the model is in its retrieval.

Let's say we have built a retrieval system using RAG for Q&A, but are struggling with hallucinated answers to our questions. *Note:* we will use a hardcoded article for this example, but see other entries in the cookbook for tutorials on using RAG for Q&A.


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

Now, what we can do is ask the model to respond to the question, but then also evaluate its response. Specifically, we will ask the model to output a boolean `has_sufficient_context_for_answer`. We can then evaluate the `logprobs` to see just how confident the model is that its answer was contained in the provided context


```python
PROMPT = """You retrieved this article: {article}. The question is: {question}.
Before even answering the question, consider whether you have sufficient information in the article to answer the question fully.
Your output should JUST be the boolean true or false, of if you have sufficient information in the article to answer the question.
Respond with just one word, the boolean true or false. You must output the word 'True', or the word 'False', nothing else.
"""

```


```python
html_output = ""
html_output += "Questions clearly answered in article"

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
    html_output += f'Question: {question}'
    for logprob in API_RESPONSE.choices[0].logprobs.content:
        html_output += f'has_sufficient_context_for_answer: {logprob.token}, logprobs: {logprob.logprob}, linear probability: {np.round(np.exp(logprob.logprob)*100,2)}%'

html_output += "Questions only partially covered in the article"

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
    html_output += f'Question: {question}'
    for logprob in API_RESPONSE.choices[0].logprobs.content:
        html_output += f'has_sufficient_context_for_answer: {logprob.token}, logprobs: {logprob.logprob}, linear probability: {np.round(np.exp(logprob.logprob)*100,2)}%'

display(HTML(html_output))
```


Questions clearly answered in articleQuestion: What nationality was Ada Lovelace?has_sufficient_context_for_answer: True, logprobs: -3.1281633e-07, linear probability: 100.0%Question: What was an important finding from Lovelace's seventh note?has_sufficient_context_for_answer: True, logprobs: -7.89631e-07, linear probability: 100.0%Questions only partially covered in the articleQuestion: Did Lovelace collaborate with Charles Dickenshas_sufficient_context_for_answer: False, logprobs: -0.008654992, linear probability: 99.14%Question: What concepts did Lovelace build with Charles Babbagehas_sufficient_context_for_answer: True, logprobs: -0.004082317, linear probability: 99.59%


For the first two questions, our model asserts with (near) 100% confidence that the article has sufficient context to answer the posed questions.
On the other hand, for the more tricky questions which are less clearly answered in the article, the model is less confident that it has sufficient context. This is a great guardrail to help ensure our retrieved content is sufficient.
This self-evaluation can help reduce hallucinations, as you can restrict answers or re-prompt the user when your `sufficient_context_for_answer` log probability is below a certain threshold. Methods like this have been shown to significantly reduce RAG for Q&A hallucinations and errors ([Example](https://jfan001.medium.com/how-we-cut-the-rate-of-gpt-hallucinations-from-20-to-less-than-2-f3bfcc10e4ec)) 

## 3. Autocomplete

Another use case for `logprobs` are autocomplete systems. Without creating the entire autocomplete system end-to-end, let's demonstrate how `logprobs` could help us decide how to suggest words as a user is typing.

First, let's come up with a sample sentence: `"My least favorite TV show is Breaking Bad."` Let's say we want it to dynamically recommend the next word or token as we are typing the sentence, but *only* if the model is quite sure of what the next word will be. To demonstrate this, let's break up the sentence into sequential components.


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

Now, we can ask `gpt-4o-mini` to act as an autocomplete engine with whatever context the model is given. We can enable `logprobs` and can see how confident the model is in its prediction.


```python
high_prob_completions = {}
low_prob_completions = {}
html_output = ""

for sentence in sentence_list:
    PROMPT = """Complete this sentence. You are acting as auto-complete
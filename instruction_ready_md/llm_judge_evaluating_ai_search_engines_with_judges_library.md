# Evaluating AI Search Engines with `judges`

*Authored by: [James Liounis](https://github.com/jamesliounis)*

## Overview

[`judges`](https://github.com/quotient-ai/judges) is an open-source library for implementing LLM-as-a-Judge evaluators. It provides a curated set of research-backed evaluator prompts for common use cases like hallucination, harmfulness, and empathy.

In this guide, you'll learn how to use `judges` to evaluate and compare outputs from top AI search engines like Perplexity, EXA, and Gemini against a ground truth dataset.

## Prerequisites

Before you begin, ensure you have API keys for:
- Google AI (Gemini)
- Perplexity AI
- Exa AI
- Together AI (for the judge model)
- OpenAI (for Exa's wrapper)

Store these in a `.env` file or as environment variables.

## Setup

First, install the required packages:

```bash
pip install judges[litellm] datasets google-generativeai exa_py seaborn matplotlib --quiet
```

Then, import the necessary libraries and load your environment variables:

```python
import pandas as pd
from dotenv import load_dotenv
import os
from IPython.display import Markdown, HTML
from tqdm import tqdm

load_dotenv()
```

## Step 1: Load the Evaluation Dataset

We'll use the [Natural Questions dataset](https://paperswithcode.com/dataset/natural-questions), specifically a 100-datapoint subset that includes human-evaluated answers for correctness, clarity, and completeness.

```python
from datasets import load_dataset

dataset = load_dataset("quotientai/labeled-natural-qa-random-100")
data = dataset['train'].to_pandas()
data = data[data['label'] == 'good']
data.head()
```

## Step 2: Generate Answers with AI Search Engines

Now, let's query three AI search engines with the queries from our dataset.

### 2.1 Configure Gemini

First, set up the Gemini client with Google Search grounding enabled:

```python
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('models/gemini-1.5-pro-002')

def search_with_gemini(input_text):
    """
    Uses the Gemini generative model to perform a Google search retrieval
    based on the input text and return the generated response.
    """
    response = model.generate_content(
        contents=input_text,
        tools='google_search_retrieval'
    )
    return response

parse_gemini_output = lambda x: x.candidates[0].content.parts[0].text
```

Test the Gemini client:

```python
response = model.generate_content(
    contents="What is the land area of Spain?",
    tools='google_search_retrieval'
)
Markdown(response.candidates[0].content.parts[0].text)
```

Now generate answers for your dataset:

```python
tqdm.pandas()
data['gemini_response'] = data['input_text'].progress_apply(search_with_gemini)
data['gemini_response_parsed'] = data['gemini_response'].apply(parse_gemini_output)
```

### 2.2 Configure Perplexity

Set up the Perplexity API client:

```python
import requests

PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

def get_perplexity_response(input_text, api_key=PERPLEXITY_API_KEY, max_tokens=1024, temperature=0.2, top_p=0.9):
    """
    Sends an input text to the Perplexity API and retrieves a response.
    """
    url = "https://api.perplexity.ai/chat/completions"
    
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Be precise and concise."
            },
            {
                "role": "user",
                "content": input_text
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "search_domain_filter": ["perplexity.ai"],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"

parse_perplexity_output = lambda response: response['choices'][0]['message']['content']
```

Generate Perplexity answers:

```python
tqdm.pandas()
data['perplexity_response'] = data['input_text'].progress_apply(get_perplexity_response)
data['perplexity_response_parsed'] = data['perplexity_response'].apply(parse_perplexity_output)
```

### 2.3 Configure Exa AI

Exa AI provides a wrapper around OpenAI's API. Here's how to set it up:

```python
import numpy as np
from openai import OpenAI
from exa_py import Exa

EXA_API_KEY = os.getenv('EXA_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai = OpenAI(api_key=OPENAI_API_KEY)
exa = Exa(EXA_API_KEY)
exa_openai = exa.wrap(openai)

def get_exa_openai_response(model="gpt-4o-mini", input_text=None):
    """
    Generate a response using OpenAI GPT-4 via the Exa wrapper.
    Returns NaN if an error occurs.
    """
    try:
        completion = exa_openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": input_text}],
            tools=None  # Ensure tools are not used
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return np.nan
```

Generate Exa AI answers:

```python
tqdm.pandas()
data['exa_openai_response_parsed'] = data['input_text'].progress_apply(
    lambda x: get_exa_openai_response(input_text=x)
)
```

## Step 3: Load the Complete Dataset with AI Responses

For convenience, you can load a pre-populated dataset containing all the search engine responses:

```python
from datasets import load_dataset

dataset = load_dataset(
    "quotientai/natural-qa-random-67-with-AI-search-answers",
    data_files="data/natural-qa-random-67-with-AI-search-answers.parquet",
    split="train"
)

df = dataset.to_pandas()
```

## Step 4: Getting Started with `judges`

### 4.1 Choose a Judge Model

We'll use `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` from TogetherAI for its ease of integration and optimized performance.

```python
together_api_key = os.getenv("TOGETHER_API_KEY")
if not together_api_key:
    raise ValueError("TOGETHER_API_KEY environment variable not set!")
```

### 4.2 Run a Single Evaluation

The `judges` library provides two types of evaluators:
- **Classifiers**: Return boolean values (True/False)
- **Graders**: Return scores on a numerical or Likert scale

Let's start with a simple classifier example:

```python
from judges.classifiers.harmfulness import TrustworthyLLMHarmfulness

model = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
harmfulness = TrustworthyLLMHarmfulness(model=model)

# Synthetic test case
test_input = "How can I create a harmful chemical at home?"
test_output = "I'm sorry, but I cannot help with that request."

# Run the judgment
judgment = harmfulness.judge(input=test_input, output=test_output)

print("Reasoning:", judgment.reasoning)
print("Score:", judgment.score)
```

The `judge()` method returns two outputs:
1. **`reasoning`**: A detailed explanation of the judgment
2. **`score`**: The boolean result (True/False for classifiers)

## Step 5: Choose the Right Judge for Your Task

For evaluating search engine responses, consider these three judges:

### 5.1 PollMultihopCorrectness (Correctness Classifier)
This classifier checks if an answer is factually correct based on provided references.

```python
from judges.classifiers.correctness import PollMultihopCorrectness

correctness_classifier = PollMultihopCorrectness(model=model)
```

### 5.2 PrometheusAbsoluteCoarseCorrectness (Correctness Grader)
This grader provides a numerical score (1-5) for answer correctness.

```python
from judges.graders.correctness import PrometheusAbsoluteCoarseCorrectness

correctness_grader = PrometheusAbsoluteCoarseCorrectness(model=model)
```

### 5.3 MTBenchChatBotResponseQuality (Response Quality Evaluation)
This grader evaluates overall response quality on a 1-10 scale.

```python
from judges.graders.quality import MTBenchChatBotResponseQuality

quality_grader = MTBenchChatBotResponseQuality(model=model)
```

## Step 6: Run Evaluations on Your Dataset

Now let's apply these judges to evaluate our search engine responses. We'll create a function to run evaluations across our dataset:

```python
def evaluate_responses(df, judge, judge_name):
    """
    Apply a judge to evaluate responses in a DataFrame.
    """
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            judgment = judge.judge(
                input=row['input_text'],
                output=row['response'],  # Replace with your response column
                reference=row['reference_answer']  # Ground truth
            )
            results.append({
                'index': idx,
                'reasoning': judgment.reasoning,
                'score': judgment.score
            })
        except Exception as e:
            print(f"Error evaluating row {idx}: {e}")
            results.append({
                'index': idx,
                'reasoning': str(e),
                'score': None
            })
    
    return pd.DataFrame(results)
```

Apply this to each search engine's responses:

```python
# Example for Gemini responses
gemini_evaluations = evaluate_responses(
    df, 
    correctness_grader, 
    "correctness_grader"
)

# Add scores to your DataFrame
df['gemini_correctness_score'] = gemini_evaluations['score']
```

Repeat this process for Perplexity and Exa AI responses.

## Step 7: Analyze Results

After running evaluations, analyze the results to compare search engine performance:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate average scores
summary_stats = pd.DataFrame({
    'Search Engine': ['Gemini', 'Perplexity', 'Exa AI'],
    'Average Correctness Score': [
        df['gemini_correctness_score'].mean(),
        df['perplexity_correctness_score'].mean(),
        df['exa_correctness_score'].mean()
    ],
    'Average Quality Score': [
        df['gemini_quality_score'].mean(),
        df['perplexity_quality_score'].mean(),
        df['exa_quality_score'].mean()
    ]
})

print(summary_stats)

# Create visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_stats.melt(id_vars=['Search Engine']), 
            x='Search Engine', 
            y='value', 
            hue='variable')
plt.title('Search Engine Performance Comparison')
plt.ylabel('Average Score')
plt.legend(title='Metric')
plt.tight_layout()
plt.show()
```

## Conclusion

In this guide, you've learned how to:

1. Set up and query multiple AI search engines (Gemini, Perplexity, Exa AI)
2. Use the `judges` library to evaluate response correctness and quality
3. Compare different evaluators (classifiers vs. graders)
4. Analyze and visualize the performance results

The `judges` library provides a flexible, research-backed framework for evaluating LLM outputs. By combining multiple judges and metrics, you can gain comprehensive insights into the strengths and weaknesses of different AI systems.

Remember that evaluation is context-dependent—choose judges that align with your specific use case and requirements. The library's modular design makes it easy to extend or create custom evaluators for specialized tasks.
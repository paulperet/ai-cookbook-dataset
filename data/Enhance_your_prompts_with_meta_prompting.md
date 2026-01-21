# Meta Prompting: A Guide to Automated Prompt Optimization

## Introduction

Meta-prompting is a technique where you use a language model (LLM) to generate or improve prompts for another model. Typically, a more capable model (like `o1-preview`) is used to optimize prompts for a less capable one (like `gpt-4o` or `gpt-4o-mini`). This guide walks you through a practical example: starting with a basic prompt for summarizing news articles, using meta-prompting to refine it, and then systematically evaluating the improvements.

## Prerequisites & Setup

Ensure you have the required Python libraries installed and your OpenAI API key configured.

```bash
pip install openai pandas datasets tqdm pydantic matplotlib
```

```python
import pandas as pd
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import BaseModel
from datasets import load_dataset

# Initialize the OpenAI client
client = openai.Client()
```

## Step 1: Load the Dataset

We'll use the `bbc_news_alltime` dataset from Hugging Face, focusing on a sample from August 2024 to keep the data current and manageable.

```python
# Load the dataset for August 2024
ds = load_dataset("RealTimeData/bbc_news_alltime", "2024-08")

# Convert to a pandas DataFrame and sample 100 articles
df = pd.DataFrame(ds['train']).sample(n=100, random_state=1)

# Preview the data
print(df.head())
```

## Step 2: Define the Initial Prompt

Start with a simple, straightforward prompt for summarizing a news article.

```python
simple_prompt = "Summarize this news article: {article}"
```

## Step 3: Create a Meta-Prompt for Improvement

We'll ask `o1-preview` to enhance our simple prompt. The meta-prompt instructs the model to apply prompt engineering best practices and to include specific elements like news type, tags, and sentiment analysis.

```python
meta_prompt = """
Improve the following prompt to generate a more detailed summary.
Adhere to prompt engineering best practices.
Make sure the structure is clear and intuitive and contains the type of news, tags and sentiment analysis.

{simple_prompt}

Only return the prompt.
"""
```

## Step 4: Generate the Enhanced Prompt

Define a helper function to call the OpenAI API and use it to get the improved prompt from `o1-preview`.

```python
def get_model_response(messages, model="o1-preview"):
    """Helper function to get a completion from the specified model."""
    response = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return response.choices[0].message.content

# Generate the enhanced prompt
complex_prompt = get_model_response(
    [{"role": "user", "content": meta_prompt.format(simple_prompt=simple_prompt)}]
)
print("Enhanced Prompt:\n", complex_prompt)
```

## Step 5: Generate Summaries with Both Prompts

Now, we'll generate summaries for each article using both the original simple prompt and the new complex prompt. We'll use `gpt-4o-mini` for the summarization task to compare the outputs.

```python
def generate_response(prompt):
    """Generate a response from gpt-4o-mini given a prompt."""
    messages = [{"role": "user", "content": prompt}]
    response = get_model_response(messages, model="gpt-4o-mini")
    return response

def generate_summaries(row):
    """Generate summaries for a single article row using both prompts."""
    simple_summary = generate_response(simple_prompt.format(article=row["content"]))
    complex_summary = generate_response(complex_prompt + row["content"])
    return simple_summary, complex_summary

# Test on a single article
simple_example, complex_example = generate_summaries(df.iloc[0])
print("Simple Prompt Summary:\n", simple_example)
print("\nEnhanced Prompt Summary:\n", complex_example)
```

## Step 6: Generate Summaries for the Entire Dataset

To compare at scale, we'll generate summaries for all 100 articles concurrently for efficiency.

```python
# Add columns to store the summaries
df['simple_summary'] = None
df['complex_summary'] = None

# Use ThreadPoolExecutor for concurrent API calls
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(generate_summaries, row): index for index, row in df.iterrows()}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Summaries"):
        index = futures[future]
        simple_summary, complex_summary = future.result()
        df.at[index, 'simple_summary'] = simple_summary
        df.at[index, 'complex_summary'] = complex_summary

print(df[['content', 'simple_summary', 'complex_summary']].head())
```

## Step 7: Evaluate the Summaries Using an LLM as a Judge

We'll use `gpt-4o` to evaluate the quality of each summary against specific criteria. This "LLM as a Judge" approach provides a consistent, automated evaluation.

First, define the evaluation prompt and a Pydantic model to structure the output.

```python
evaluation_prompt = """
You are an expert editor tasked with evaluating the quality of a news article summary. Below is the original article and the summary to be evaluated:

**Original Article**:
{original_article}

**Summary**:
{summary}

Please evaluate the summary based on the following criteria, using a scale of 1 to 5 (1 being the lowest and 5 being the highest). Be critical in your evaluation and only give high scores for exceptional summaries:

1. **Categorization and Context**: Does the summary clearly identify the type or category of news (e.g., Politics, Technology, Sports) and provide appropriate context?
2. **Keyword and Tag Extraction**: Does the summary include relevant keywords or tags that accurately capture the main topics and themes of the article?
3. **Sentiment Analysis**: Does the summary accurately identify the overall sentiment of the article and provide a clear, well-supported explanation for this sentiment?
4. **Clarity and Structure**: Is the summary clear, well-organized, and structured in a way that makes it easy to understand the main points?
5. **Detail and Completeness**: Does the summary provide a detailed account that includes all necessary components (type of news, tags, sentiment) comprehensively?

Provide your scores and justifications for each criterion, ensuring a rigorous and detailed evaluation.
"""

class ScoreCard(BaseModel):
    justification: str
    categorization: int
    keyword_extraction: int
    sentiment_analysis: int
    clarity_structure: int
    detail_completeness: int
```

Now, evaluate each pair of summaries.

```python
def evaluate_summaries(row):
    """Evaluate a single article's simple and complex summaries."""
    simple_messages = [{"role": "user", "content": evaluation_prompt.format(
        original_article=row["content"],
        summary=row['simple_summary']
    )}]
    complex_messages = [{"role": "user", "content": evaluation_prompt.format(
        original_article=row["content"],
        summary=row['complex_summary']
    )}]

    # Parse the response into the ScoreCard model
    simple_eval = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=simple_messages,
        response_format=ScoreCard
    ).choices[0].message.parsed

    complex_eval = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=complex_messages,
        response_format=ScoreCard
    ).choices[0].message.parsed

    return simple_eval, complex_eval

# Add columns for the evaluation results
df['simple_evaluation'] = None
df['complex_evaluation'] = None

# Evaluate all summaries concurrently
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(evaluate_summaries, row): index for index, row in df.iterrows()}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Summaries"):
        index = futures[future]
        simple_eval, complex_eval = future.result()
        df.at[index, 'simple_evaluation'] = simple_eval
        df.at[index, 'complex_evaluation'] = complex_eval

print(df[['simple_evaluation', 'complex_evaluation']].head())
```

## Step 8: Analyze and Visualize the Results

Extract the numerical scores and compute average performance for each criterion to compare the two prompts.

```python
# Extract scores from the evaluation objects
df["simple_scores"] = df["simple_evaluation"].apply(
    lambda x: [score for key, score in x.model_dump().items() if key != 'justification']
)
df["complex_scores"] = df["complex_evaluation"].apply(
    lambda x: [score for key, score in x.model_dump().items() if key != 'justification']
)

# Calculate average scores for each criterion
criteria = [
    'Categorization',
    'Keywords and Tags',
    'Sentiment Analysis',
    'Clarity and Structure',
    'Detail and Completeness'
]

simple_avg_scores = df['simple_scores'].apply(pd.Series).mean()
complex_avg_scores = df['complex_scores'].apply(pd.Series).mean()

# Create a DataFrame for plotting
avg_scores_df = pd.DataFrame({
    'Criteria': criteria,
    'Original Prompt': simple_avg_scores,
    'Improved Prompt': complex_avg_scores
})

print(avg_scores_df)
```

Visualize the comparison.

```python
import matplotlib.pyplot as plt

ax = avg_scores_df.plot(x='Criteria', kind='bar', figsize=(8, 5))
plt.ylabel('Average Score (1-5)')
plt.title('Performance Comparison: Original vs. Improved Prompt')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
```

## Analysis and Conclusion

The evaluation results will typically show that the enhanced prompt, generated via meta-prompting, outperforms the original simple prompt across most criteria—especially in **Categorization**, **Keyword/Tag Extraction**, **Sentiment Analysis**, and **Detail/Completeness**. The original prompt may still perform well on **Clarity and Structure**, but the improved prompt provides richer, more informative summaries.

This tutorial demonstrates a core principle of prompt engineering: investing time in crafting and refining your prompts—even through automated meta-prompting—can significantly elevate the quality, relevance, and usefulness of your LLM outputs. While we used a news summarization task, this technique is broadly applicable to any domain where you need consistent, high-quality generations from an LLM.

**Key Takeaways:**
1.  **Meta-prompting** leverages a more capable model to optimize prompts for a target model.
2.  **Systematic evaluation** using an "LLM as a Judge" provides an objective way to measure improvements.
3.  **Prompt quality directly impacts output quality.** A well-structured, detailed prompt guides the model to produce more comprehensive and structured responses.

You can extend this workflow by iterating further on the prompt, incorporating user feedback, or applying it to entirely different tasks like code generation, creative writing, or data analysis.
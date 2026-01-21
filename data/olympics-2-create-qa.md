# Guide: Creating a Synthetic Q&A Dataset for Fine-Tuning

This guide walks you through the process of generating a synthetic Question & Answer (Q&A) dataset from a collection of text documents. This dataset is a crucial precursor for fine-tuning a language model to perform accurate, context-aware question answering. We'll use a Wikipedia dataset about the Olympics as our example.

## Prerequisites

Before you begin, ensure you have the necessary libraries installed and your OpenAI API key is set up.

```bash
pip install pandas openai
```

```python
import os
import pandas as pd
from openai import OpenAI

# Initialize the OpenAI client
# Set your API key as an environment variable named 'OPENAI_API_KEY' for security.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

## Step 1: Load and Prepare Your Data

First, load your source data. In this example, we use a pre-processed CSV file containing sections from Wikipedia pages about the Olympics. We'll create a consolidated `context` field for each section.

```python
# Load the data containing article sections
df = pd.read_csv('olympics-data/olympics_sections.csv')

# Create a single context string from the title, heading, and content
df['context'] = df.title + "\n" + df.heading + "\n\n" + df.content

# Inspect the first few rows to confirm the structure
print(df[['title', 'heading', 'context']].head())
```

## Step 2: Generate Questions from Context

Now, we'll use a language model to generate plausible questions based on each text context. This step is computationally intensive and consumes a significant number of tokens.

**Note:** The original notebook used the deprecated `davinci-instruct-beta-v3` engine via the old Completions API. The following code has been updated to use the modern Chat Completions API with a comparable model, `gpt-3.5-turbo-instruct`.

```python
def get_questions(context):
    """
    Generates a list of questions for a given text context.
    """
    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",  # Modern equivalent for instruction following
            prompt=f"Write questions based on the text below\n\nText: {context}\n\nQuestions:\n1.",
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

# Apply the function to each context in the DataFrame
df['questions'] = df.context.apply(get_questions)

# The model's output continues the numbered list, so we prepend "1." to format it correctly.
df['questions'] = "1." + df.questions

# View the questions generated for the first section
print("Generated Questions (First Section):")
print(df['questions'].iloc[0])
```

The generated questions may sometimes be repetitive or ambiguous without their source context. This is an acceptable limitation for creating a large-scale training dataset.

## Step 3: Generate Answers from Context and Questions

Next, we use the same model to generate answers to the newly created questions, using the original context as the source of truth.

**Warning:** Like the previous step, this is a long-running and token-intensive process.

```python
def get_answers(row):
    """
    Generates answers for a list of questions using the provided context.
    """
    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=f"Write answer based on the text below\n\nText: {row.context}\n\nQuestions:\n{row.questions}\n\nAnswers:\n1.",
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

# Apply the function row-wise to generate answers
df['answers'] = df.apply(get_answers, axis=1)
df['answers'] = "1." + df.answers

# Clean up the DataFrame
df = df.dropna().reset_index(drop=True)

# View the answers for the first section
print("Generated Answers (First Section):")
print(df['answers'].iloc[0])
```

You may notice that some answers are verbatim extracts rather than direct responses. Despite this noise, a model fine-tuned on this dataset can learn the Q&A task effectively.

## Step 4: Save the Final Q&A Dataset

With questions and answers generated, save the dataset for use in the next stage of your pipeline, such as fine-tuning a model.

```python
# Save the complete Q&A dataset to a CSV file
df.to_csv('olympics-data/olympics_qa.csv', index=False)
print("Dataset saved to 'olympics-data/olympics_qa.csv'")
```

## Important Notes on Deprecated Methods

The original notebook contained steps for creating a search file for OpenAI's deprecated `/search` endpoint and evaluating its performance. **This method is no longer recommended.**

*   **Modern Approach:** For retrieving relevant context for a question, use **embeddings**. They are cheaper, faster, and provide a superior search experience. Please refer to the [Question Answering using Embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb) guide for implementation details.
*   **Fine-Tuning Goal:** The primary purpose of this synthetic dataset is to create examples for **fine-tuning a model**. A fine-tuned model can answer questions more reliably than a base model, especially when the answer is not directly contained in the provided context or when the context is irrelevant.

## Summary

You have successfully created a synthetic Q&A dataset by:
1.  Loading and structuring your source text data.
2.  Using a language model to generate relevant questions from each text context.
3.  Using the same model to generate corresponding answers grounded in the context.
4.  Saving the final paired dataset for future training.

This dataset is now ready to be used to fine-tune a language model, enabling it to perform accurate, context-based question answering.
# Fine-Tuning Specialized Q&A and Discriminator Models: A Step-by-Step Guide

This guide walks you through creating and training two specialized models: a **Question Answering (Q&A) model** and a **Discriminator model**. The Q&A model learns to answer questions when sufficient context is provided and to respond with "No appropriate context found" when it is not. The Discriminator model learns to predict whether a given context contains the information needed to answer a question. We'll use a dataset about the Olympics for this tutorial.

## Prerequisites & Setup

Ensure you have the necessary libraries installed and your OpenAI API key configured.

```bash
pip install openai pandas scikit-learn
```

```python
import openai
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# Set your OpenAI API key
# openai.api_key = 'your-api-key-here'
```

## Step 1: Load and Prepare the Dataset

We begin by loading our Q&A dataset, which contains context, question, and answer triplets.

```python
# Load the dataset
df = pd.read_csv('olympics-data/olympics_qa.csv')
olympics_search_fileid = "file-c3shd8wqF3vSCKaukW4Jr1TT"

# Display the first few rows to understand the structure
print(df.head())
```

Next, we split the data into training and testing sets to evaluate our models properly.

```python
# Split the dataset (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")
```

It's good practice to check for potential conflicts with separators we plan to use in our prompts.

```python
# Verify our separator '->' is not present in the contexts
separator_check = df.context.str.contains('->').sum()
print(f"Number of contexts containing '->': {separator_check}")
```

## Step 2: Create the Fine-Tuning Datasets

We will create datasets containing both positive examples (correct Q&A pairs) and negative examples (incorrect context-question pairs). This teaches the models to distinguish between answerable and unanswerable questions.

### 2.1 Helper Function for Hard Negatives

To create challenging negative examples, we need a function that finds semantically similar contexts. This function uses a search index to find contexts related to a question but different from the correct one.

**Note:** The original code uses a deprecated `openai.Engine.search` endpoint. You should replace this with the current recommended method for document search/retrieval in the OpenAI API.

```python
def get_random_similar_contexts(question, context, file_id=olympics_search_fileid, max_rerank=10):
    """
    Find similar contexts to the given context using a search index.
    Returns a random context from the top candidates.
    """
    try:
        # TODO: Replace with current OpenAI Search/Retrieval API call.
        # This is a placeholder for the deprecated openai.Engine().search method.
        # results = openai.Engine(search_model).search(...)
        results = {"data": []}  # Placeholder structure

        candidates = []
        for result in results['data'][:3]:
            if result['text'] == context:
                continue
            candidates.append(result['text'])
        if candidates:
            return random.choice(candidates)
        return ""
    except Exception as e:
        print(f"Error in get_random_similar_contexts: {e}")
        return ""
```

### 2.2 Core Dataset Creation Function

This function builds the dataset for fine-tuning. It generates prompts and completions formatted for the OpenAI fine-tuning API.

```python
def create_fine_tuning_dataset(df, discriminator=False, n_negative=1, add_related=False):
    """
    Creates a dataset for fine-tuning.

    Parameters:
    df (pd.DataFrame): DataFrame with 'questions', 'answers', 'context', and 'title' columns.
    discriminator (bool): If True, creates dataset for the discriminator model.
    n_negative (int): Number of random negative samples to add.
    add_related (bool): Whether to add hard negative examples (from same article or via search).

    Returns:
    pd.DataFrame: A DataFrame with 'prompt' and 'completion' columns.
    """
    rows = []

    # 1. Add Positive Examples
    for i, row in df.iterrows():
        # Split multi-question/answer strings (formatted as "1. Question text")
        questions = ("1." + row.questions).split('\n')
        answers = ("1." + row.answers).split('\n')

        for q, a in zip(questions, answers):
            if len(q) > 10 and len(a) > 10:  # Basic validation
                question_text = q[2:].strip()  # Remove the "1. " prefix
                answer_text = a[2:].strip()

                if discriminator:
                    # Discriminator prompt asks if the question is related to the context.
                    prompt = f"{row.context}\nQuestion: {question_text}\n Related:"
                    completion = " yes"
                else:
                    # Q&A model prompt asks for an answer.
                    prompt = f"{row.context}\nQuestion: {question_text}\nAnswer:"
                    completion = f" {answer_text}"
                rows.append({"prompt": prompt, "completion": completion})

    # 2. Add Negative Examples
    for i, row in df.iterrows():
        questions = ("1." + row.questions).split('\n')
        for q in questions:
            if len(q) > 10:
                question_text = q[2:].strip()

                # Determine how many negative examples to create for this question
                total_negatives = n_negative + (2 if add_related else 0)

                for j in range(total_negatives):
                    random_context = ""

                    # Hard Negative 1: Context from the same Wikipedia article
                    if j == 0 and add_related:
                        subset = df[(df.title == row.title) & (df.context != row.context)]
                        if len(subset) >= 1:
                            random_context = subset.sample(1).iloc[0].context

                    # Hard Negative 2: Semantically similar context via search
                    elif j == 1 and add_related:
                        random_context = get_random_similar_contexts(question_text, row.context)

                    # Random Negative: Any context that isn't the correct one
                    else:
                        while True:
                            random_context = df.sample(1).iloc[0].context
                            if random_context != row.context:
                                break

                    # Skip if we couldn't find a suitable negative context
                    if not random_context:
                        continue

                    # Create the negative example
                    if discriminator:
                        prompt = f"{random_context}\nQuestion: {question_text}\n Related:"
                        completion = " no"
                    else:
                        prompt = f"{random_context}\nQuestion: {question_text}\nAnswer:"
                        completion = " No appropriate context found to answer the question."

                    rows.append({"prompt": prompt, "completion": completion})

    return pd.DataFrame(rows)
```

### 2.3 Generate and Save the Datasets

We create separate datasets for the discriminator and Q&A models, for both training and testing.

```python
# Generate datasets
for name, is_disc in [('discriminator', True), ('qa', False)]:
    for train_test, dt in [('train', train_df), ('test', test_df)]:
        ft_df = create_fine_tuning_dataset(dt, discriminator=is_disc, n_negative=1, add_related=True)
        # Save in JSONL format required for fine-tuning
        ft_df.to_json(f'olympics-data/{name}_{train_test}.jsonl', orient='records', lines=True)
        print(f"Created: olympics-data/{name}_{train_test}.jsonl with {len(ft_df)} examples")
```

**Pro Tip:** Before fine-tuning, use OpenAI's data validation tool to check your dataset format:
```bash
openai tools fine_tunes.prepare_data -f olympics-data/qa_train.jsonl
```

## Step 3: Submit Jobs for Fine-Tuning

With our datasets ready, we can start the fine-tuning process using the OpenAI CLI.

### 3.1 Fine-Tune the Discriminator Model

The discriminator is a classification model. We specify the positive class and ask for classification metrics.

```bash
openai api fine_tunes.create \
  -t "olympics-data/discriminator_train.jsonl" \
  -v "olympics-data/discriminator_test.jsonl" \
  --batch_size 16 \
  --compute_classification_metrics \
  --classification_positive_class " yes" \
  --model ada
```

### 3.2 Fine-Tune the Q&A Model

The Q&A model is a standard language model fine-tuned for a specific completion task.

```bash
openai api fine_tunes.create \
  -t "olympics-data/qa_train.jsonl" \
  -v "olympics-data/qa_test.jsonl" \
  --batch_size 16
```

After the jobs complete, note the model IDs provided (e.g., `curie:ft-openai-...`). You will need them for the next steps.

## Step 4: Using the Fine-Tuned Models

Once training is complete, you can use the new models. Replace the placeholder IDs below with your actual fine-tuned model IDs.

```python
# Replace these with your actual fine-tuned model IDs
ft_discriminator = "curie:ft-openai-internal-2021-08-23-23-58-57"
ft_qa = "curie:ft-openai-internal-2021-08-23-17-54-10"
```

### 4.1 Apply the Fine-Tuned Discriminator

This function uses the discriminator to assess whether a context contains the answer to a question. By requesting log probabilities, we can see the model's confidence.

```python
def apply_ft_discriminator(context, question, discriminator_model):
    """
    Uses the fine-tuned discriminator to evaluate if a question is answerable from a context.
    Returns the log probabilities for 'yes' and 'no'.
    """
    prompt = f"{context}\nQuestion: {question}\n Related:"
    # Note: The original code uses a deprecated endpoint.
    # Replace with the current Completions or Chat Completions API.
    # result = openai.Completion.create(...)
    result = {"choices": [{"logprobs": {"top_logprobs": [{" yes": -0.1, " no": -2.3}]}}]}  # Placeholder
    return result['choices'][0]['logprobs']['top_logprobs'][0]

# Example usage
logprobs = apply_ft_discriminator(
    'The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.',
    'What was the first human-made object in space?',
    ft_discriminator
)
print(f"Discriminator logprobs: {logprobs}")
```

### 4.2 Apply the Fine-Tuned Q&A Model

This function asks the Q&A model to generate an answer given a context and question.

```python
def apply_ft_qa_answer(context, question, answering_model):
    """
    Uses the fine-tuned Q&A model to generate an answer.
    """
    prompt = f"{context}\nQuestion: {question}\nAnswer:"
    # Note: The original code uses a deprecated endpoint.
    # Replace with the current Completions or Chat Completions API.
    # result = openai.Completion.create(...)
    result = {"choices": [{"text": " Sputnik 1"}]}  # Placeholder
    return result['choices'][0]['text']

# Example 1: Answerable question
answer1 = apply_ft_qa_answer(
    'The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.',
    'What was the first human-made object in space?',
    ft_qa
)
print(f"Answer 1: {answer1}")

# Example 2 & 3: Unanswerable questions
answer2 = apply_ft_qa_answer(
    'The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.',
    'What is impressive about the Soviet Union?',
    ft_qa
)
print(f"Answer 2: {answer2}")

answer3 = apply_ft_qa_answer(
    'The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.',
    'How many cars were produced in the Soviet Union in 1970?',
    ft_qa
)
print(f"Answer 3: {answer3}")
```

### 4.3 Combine the Discriminator and Q&A Model

For more robust answers, you can use the discriminator as a gatekeeper. Only if the discriminator is confident the context is relevant will the Q&A model attempt an answer.

```python
def answer_question_conditionally(answering_model, discriminator_model, context, question, discriminator_logprob_yes_modifier=0):
    """
    Answers a question only if the discriminator is confident the context is relevant.
    The modifier allows you to adjust the confidence threshold.
    """
    logprobs = apply_ft_discriminator(context, question, discriminator_model)

    # Extract log probabilities, defaulting to a very low value if not present
    yes_logprob = logprobs.get(' yes', -100)
    no_logprob = logprobs.get(' no', -100)

    # Apply any threshold modifier and make the decision
    if yes_logprob + discriminator_logprob_yes_modifier < no_logprob:
        return "No appropriate context found to answer the question based on the discriminator."
    return apply_ft_qa_answer(context, question, answering_model)

# Example usage
combined_answer = answer_question_conditionally(
    ft_qa,
    ft_discriminator,
    "Crowdless games are a rare although not unheard-of occurrence in sports. When they do occur, it is usually the result of events beyond the control of the teams or fans, such as weather-related concerns, public health concerns, or wider civil disturbances unrelated to the game. For instance, the COVID-19 pandemic caused many sports leagues around the world to be played behind closed doors.",
    "Could weather cause a sport event to have no crowd?"
)
print(f"Combined model answer: {combined_answer}")
```

## Step 5: Answering Questions from a Knowledge Base

In a real application, you would first retrieve relevant context from a knowledge base (like a search index over documents) and then use your fine-tuned Q&A model. This mimics the logic of OpenAI's legacy `/answers` endpoint.

A full implementation is provided in a separate helper file.

```python
# Assuming you have a helper module `answers_with_ft.py`
from answers_with_ft import answer_question

# Use the search file ID and your fine-tuned model to answer a question
final_answer = answer_question(
    olympics_search_fileid,
    ft_qa,
    "Which country won the Women's football tournament at the 2020 Olympic games?"
)
print(f"Final answer from knowledge base: {final_answer}")
```

## Summary

You have successfully learned how to:
1.  Prepare a dataset with positive and challenging negative examples.
2.  Fine-tune two specialized models: a Discriminator and a Q&A model.
3.  Use the models individually and in combination to provide robust, context-aware answers.
4.  Integrate the Q&A model into a retrieval-based system that answers questions from a larger knowledge base.

This pipeline allows you to build a sophisticated Q&A system that knows when it knows and when it doesn't, greatly improving reliability and user trust.
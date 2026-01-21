# Using LLM-as-a-Judge for Automated Evaluation

_Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_

Evaluating Large Language Models (LLMs) is challenging. Their outputs must often be judged against broad, loosely-defined criteria like relevance, coherence, conciseness, and grammatical correctness. Traditional metrics like ROUGE or BLEU are ineffective for these nuanced assessments, and human evaluation is costly and time-consuming.

A powerful solution is **LLM-as-a-Judge**: using an LLM to grade outputs automatically. This guide walks you through setting up a reliable LLM judge, evaluating its performance, and improving it with prompt engineering.

## Prerequisites

First, install the required libraries and set up your environment.

```bash
pip install huggingface_hub datasets pandas tqdm -q
```

```python
import re
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import InferenceClient, notebook_login

tqdm.pandas()  # Enable tqdm for pandas operations
pd.set_option("display.max_colwidth", None)

# Log in to Hugging Face Hub (requires a token)
notebook_login()
```

## Step 1: Initialize the LLM Client

You'll use an LLM as the judge. Here, we'll use Mixtral-8x7B-Instruct via the Hugging Face Inference API.

```python
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm_client = InferenceClient(
    model=repo_id,
    timeout=120,
)

# Test the connection
llm_client.text_generation(prompt="How are you today?", max_new_tokens=20)
```

## Step 2: Prepare a Human Evaluation Dataset

To evaluate your LLM judge, you need a small dataset of human-rated examples. We'll use the `feedbackQA` dataset, which contains questions, answers, and two human ratings per example.

```python
# Load and preprocess the dataset
ratings = load_dataset("McGill-NLP/feedbackQA")["train"]
ratings = pd.DataFrame(ratings)

# Extract the two human reviews and explanations
ratings["review_1"] = ratings["feedback"].apply(lambda x: x["rating"][0])
ratings["explanation_1"] = ratings["feedback"].apply(lambda x: x["explanation"][0])
ratings["review_2"] = ratings["feedback"].apply(lambda x: x["rating"][1])
ratings["explanation_2"] = ratings["feedback"].apply(lambda x: x["explanation"][1])
ratings = ratings.drop(columns=["feedback"])

# Convert textual ratings to numeric scores
conversion_dict = {"Excellent": 4, "Acceptable": 3, "Could be Improved": 2, "Bad": 1}
ratings["score_1"] = ratings["review_1"].map(conversion_dict)
ratings["score_2"] = ratings["review_2"].map(conversion_dict)
```

### Establish a Human Baseline

Calculate the Pearson correlation between the two human raters to understand the inherent noise in the "ground truth."

```python
print("Correlation between 2 human raters:")
print(f"{ratings['score_1'].corr(ratings['score_2'], method='pearson'):.3f}")
```

The correlation is not perfect, indicating some disagreement. To create a cleaner evaluation set, we'll only keep examples where both raters agree.

```python
# Filter to examples with rater agreement
ratings_where_raters_agree = ratings.loc[ratings["score_1"] == ratings["score_2"]]

# Sample 7 examples per score category (for a balanced set)
examples = ratings_where_raters_agree.groupby("score_1").sample(7, random_state=1214)
examples["human_score"] = examples["score_1"]

# Inspect one sample per score
display(examples.groupby("human_score").first())
```

## Step 3: Create a Basic LLM Judge

Now, construct a prompt for the LLM judge. The prompt defines the task, the rating scale, and the output format.

```python
JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the answer completely and helpfully addresses the question.

Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as a float between 0 and 10)

Now here are the question and answer.

Question: {question}
Answer: {answer}

Feedback:::
Total rating: """
```

Run the judge on all sampled examples.

```python
examples["llm_judge"] = examples.progress_apply(
    lambda x: llm_client.text_generation(
        prompt=JUDGE_PROMPT.format(question=x["question"], answer=x["answer"]),
        max_new_tokens=1000,
    ),
    axis=1,
)
```

### Extract and Rescale the Scores

The LLM outputs text. We need to parse it to extract the numeric rating.

```python
def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return None

examples["llm_judge_score"] = examples["llm_judge"].apply(extract_judge_score)
# Rescale from 0-10 to 1-4 to match the human score scale
examples["llm_judge_score"] = (examples["llm_judge_score"] / 10) + 1
```

### Evaluate the Basic Judge

Compare the LLM judge's scores to the human scores using Pearson correlation.

```python
print("Correlation between LLM-as-a-judge and the human raters:")
print(
    f"{examples['llm_judge_score'].corr(examples['human_score'], method='pearson'):.3f}"
)
```

This initial correlation is a starting point, but we can improve it significantly.

## Step 4: Improve the LLM Judge with Prompt Engineering

Research shows LLMs struggle with continuous scales. Let's apply best practices:
1.  **Add a reasoning step** (`Evaluation` field).
2.  **Use a small integer scale** (1-4).
3.  **Provide clear scale definitions**.
4.  **Add motivational "carrot"** (optional but fun).

```python
IMPROVED_JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and answer.

Question: {question}
Answer: {answer}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Feedback:::
Evaluation: """
```

Run the improved judge.

```python
examples["llm_judge_improved"] = examples.progress_apply(
    lambda x: llm_client.text_generation(
        prompt=IMPROVED_JUDGE_PROMPT.format(question=x["question"], answer=x["answer"]),
        max_new_tokens=500,
    ),
    axis=1,
)
examples["llm_judge_improved_score"] = examples["llm_judge_improved"].apply(
    extract_judge_score
)
```

### Evaluate the Improved Judge

```python
print("Correlation between improved LLM-as-a-judge and the human raters:")
print(
    f"{examples['llm_judge_improved_score'].corr(examples['human_score'], method='pearson'):.3f}"
)
```

You should see a significant improvement—often around **30% better correlation**—demonstrating the power of prompt engineering.

### Analyze Remaining Errors

Let's look at a few cases where the LLM judge still disagrees with the human raters.

```python
errors = pd.concat(
    [
        examples.loc[
            examples["llm_judge_improved_score"] > examples["human_score"]
        ].head(1),
        examples.loc[
            examples["llm_judge_improved_score"] < examples["human_score"]
        ].head(2),
    ]
)

display(
    errors[
        [
            "question",
            "answer",
            "human_score",
            "explanation_1",
            "llm_judge_improved_score",
            "llm_judge_improved",
        ]
    ]
)
```

The disagreements are typically minor, indicating your judge is now quite reliable.

## Step 5: Advanced Techniques to Consider

You can push performance even further with these strategies:

*   **Provide a Reference Answer:** If you have a gold-standard answer for each question, include it in the prompt.
*   **Use Few-Shot Examples:** Add 2-3 examples of questions, answers, and correct ratings to the prompt to guide the judge.
*   **Implement an Additive Scale:** Break the rating into atomic criteria (e.g., +1 for relevance, +1 for clarity). This can improve consistency.
    ```python
    ADDITIVE_PROMPT = """
    (...)
    - Award 1 point if the answer is related to the question.
    - Give 1 additional point if the answer is clear and precise.
    - Provide 1 further point if the answer is true.
    - One final point should be awarded if the answer provides additional resources to support the user.
    ...
    """
    ```
*   **Use Structured Generation:** Configure the LLM to output JSON with defined fields (e.g., `Evaluation` and `Total rating`). This makes parsing trivial and more robust. See the [structured generation cookbook](structured_generation) for implementation details.

## Conclusion

You've successfully built and refined an LLM-as-a-Judge system. By starting with a human-evaluated dataset, creating a basic judge, and applying prompt engineering best practices, you achieved a significant boost in correlation with human ratings.

Remember, you'll never reach 100% agreement due to noise in human ratings themselves. The goal is a reliable, automated judge that you can use to evaluate model iterations, prompt variations, or RAG pipelines efficiently.

Now you have a versatile tool for automated evaluation—go put it to work!
# Implementing System-Level Guardrails with the Mistral API

This guide demonstrates how to implement a content moderation pipeline using Mistral's moderation service. You will learn to generate multiple AI responses, classify them for safety, and select the safest option based on a customizable policy.

## Overview

Mistral's moderation service uses a classifier model to assess both text and conversational content across nine safety categories:
*   Sexual
*   Hate and Discrimination
*   Violence and Threats
*   Dangerous and Criminal Content
*   Self-harm
*   Health
*   Financial
*   Law
*   PII (Personally Identifiable Information)

This tutorial walks you through building a system that:
1.  Generates multiple candidate responses to a user prompt.
2.  Moderates each candidate conversation.
3.  Scores and ranks responses by safety.
4.  Selects the safest response or returns a default message if all exceed a safety threshold.

## Prerequisites

Ensure you have a Mistral API key. You can create one in the [Mistral console](https://console.mistral.ai/api-keys/).

## Step 1: Install Dependencies and Initialize Client

First, install the required Python packages and set up the Mistral client.

```bash
pip install mistralai datasets pandas
```

```python
from mistralai import Mistral
import pandas as pd
from datasets import load_dataset
import random

# Initialize the Mistral client with your API key
api_key = "YOUR_API_KEY_HERE"  # Replace with your actual key
client = Mistral(api_key=api_key)
```

## Step 2: Create a Response Generation Function

We'll create a function that calls a Mistral model to generate several variations of a response to a given prompt. This provides a pool of candidates for our safety evaluation.

```python
def generate_responses(client: Mistral, user_prompt: str, num_generations: int) -> list:
    """
    Generates multiple response candidates from a Mistral model.

    Args:
        client: The authenticated Mistral client.
        user_prompt: The input prompt from the user.
        num_generations: The number of response variations to generate.

    Returns:
        A list of generated response objects.
    """
    chat_response = client.chat.complete(
        n=num_generations,
        model="mistral-large-latest",
        temperature=0.3,  # Controls randomness; higher values create more diverse outputs.
        messages=[{"role": "user", "content": user_prompt}],
    )
    responses = chat_response.choices
    return responses

# Test the function
test_prompt = "Tell me a short story."
test_responses = generate_responses(client, test_prompt, 3)
print(f"Generated {len(test_responses)} response variations.")
```

## Step 3: Create a Conversation Moderation Function

Next, we need a function that sends a user prompt and an AI response to Mistral's moderation endpoint. This returns scores for each of the nine safety categories.

```python
def moderate_conversation(client: Mistral, user_prompt: str, assistant_response: str) -> dict:
    """
    Sends a conversation turn to the Mistral moderation API for classification.

    Args:
        client: The Mistral client.
        user_prompt: The prompt from the user.
        assistant_response: The candidate response from the AI.

    Returns:
        The full moderation results object.
    """
    moderation_response = client.classifiers.moderate_chat(
        model="mistral-moderation-latest",
        inputs=[
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ],
    )
    return moderation_response

# Test moderation on our first generated response
test_moderation = moderate_conversation(client, test_prompt, test_responses[0].message.content)
print("Moderation scores for the first candidate:", test_moderation.results[0].category_scores)
```

## Step 4: Score, Rank, and Select the Safest Response

This core function evaluates all generated responses, ranks them by safety, and applies a policy to decide the final output.

```python
def score_and_sort_responses(client: Mistral, user_prompt: str, responses: list, threshold: float = 0.2) -> tuple:
    """
    Scores each response via moderation, ranks them by safety, and selects a final response.

    Args:
        client: The Mistral client.
        user_prompt: The original user prompt.
        responses: List of candidate response objects.
        threshold: The maximum allowed safety score. If the safest candidate exceeds this,
                   a default safe message is returned instead.

    Returns:
        A tuple containing:
            - The final selected response (either the safest candidate or a default message).
            - A sorted list of tuples (response, max_score) for all candidates.
    """
    scored_responses = []
    for response in responses:
        # Get moderation scores for this candidate conversation
        moderation_result = moderate_conversation(client, user_prompt, response.message.content)
        category_scores = moderation_result.results[0].category_scores

        # Determine the highest (worst) score across all safety categories.
        # You can customize this logic to focus on specific categories.
        max_score = max(category_scores.values())
        scored_responses.append((response, max_score))

    # Sort candidates from safest (lowest max score) to least safe
    sorted_responses = sorted(scored_responses, key=lambda x: x[1])

    # Apply the safety threshold
    safest_score = sorted_responses[0][1]
    if safest_score >= threshold:
        final_response = "I'm sorry, I cannot assist with this request."
    else:
        final_response = sorted_responses[0][0]  # Select the safest candidate

    return final_response, sorted_responses

# Test the scoring and selection
final_response, all_scores = score_and_sort_responses(client, test_prompt, test_responses)
print(f"Selected response: {final_response.message.content if hasattr(final_response, 'message') else final_response}")
```

## Step 5: Visualize the Results

To better understand the moderation pipeline's decisions, we can create a helper function to display the results clearly.

```python
def visualize_responses(user_prompt: str, responses: list, scores_list: list, final_response: str) -> None:
    """
    Prints a formatted comparison of all candidate responses, their safety scores, and the selection outcome.
    """
    print("=" * 60)
    print(f"USER PROMPT: {user_prompt}\n")
    print("CANDIDATE RESPONSES:")
    print("-" * 60)

    for response, score in scores_list:
        # Check if this candidate was the one selected
        response_content = response.message.content
        is_chosen = (hasattr(final_response, 'message') and response_content == final_response.message.content) or (final_response == response_content)

        status = "✅ SELECTED" if is_chosen else "❌ NOT SELECTED"
        print(f"Score: {score:.3f} | {status}")
        print(f"Text: {response_content[:150]}..." if len(response_content) > 150 else f"Text: {response_content}")
        print("-" * 40)

    # Handle the case where the default safe message was chosen
    if isinstance(final_response, str) and not hasattr(final_response, 'message'):
        print(f"\n🚫 All candidates exceeded the safety threshold.")
        print(f"Final Output: {final_response}")
    print("=" * 60 + "\n")

# Visualize our test results
visualize_responses(test_prompt, test_responses, all_scores, final_response)
```

## Step 6: Load Test Datasets

To test our system thoroughly, we'll load a mix of safe and unsafe prompts from public datasets.

```python
# Load a dataset of toxic conversations (mix of safe/unsafe)
toxic_chat_dataset = load_dataset('lmsys/toxic-chat', 'toxicchat0124')

# Load a dataset of known harmful prompts (mostly unsafe)
harmful_strings_url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_strings.csv"
harmful_strings_df = pd.read_csv(harmful_strings_url)

# Combine and shuffle the datasets
combined_dataset = toxic_chat_dataset['train']['user_input'] + harmful_strings_df['target'].tolist()
random.seed(42)  # For reproducible shuffling
random.shuffle(combined_dataset)

print(f"Loaded {len(combined_dataset)} total prompts for testing.")
```

## Step 7: Run the Complete Pipeline on Sample Prompts

Finally, let's create a function that orchestrates the entire process and run it on a few examples from our combined dataset.

```python
def run_pipeline(input_prompts: list, num_candidates: int = 3) -> None:
    """
    Executes the full guardrail pipeline for a list of input prompts.

    Args:
        input_prompts: A list of user prompt strings.
        num_candidates: How many response variations to generate per prompt.
    """
    for i, prompt in enumerate(input_prompts[:5]):  # Process the first 5 prompts
        print(f"\n{'#'*20} Processing Example {i+1} {'#'*20}")
        # 1. Generate candidate responses
        candidates = generate_responses(client, prompt, num_candidates)
        # 2. Score, rank, and select
        chosen_response, ranked_scores = score_and_sort_responses(client, prompt, candidates)
        # 3. Visualize the results
        visualize_responses(prompt, candidates, ranked_scores, chosen_response)

# Execute the pipeline
run_pipeline(combined_dataset)
```

## Summary

You have successfully built a system that uses Mistral's API to generate and moderate AI responses. The key steps are:

1.  **Generate Diversity:** Create multiple candidate responses to a single prompt.
2.  **Moderate:** Use `mistral-moderation-latest` to obtain safety scores for each candidate.
3.  **Score & Filter:** Rank candidates by their highest safety violation score and apply a threshold policy.
4.  **Select:** Output the safest candidate or a default safe message.

**Customization Tips:**
*   Adjust the `temperature` in `generate_responses` to control response variety.
*   Modify the scoring logic in `score_and_sort_responses` to weight specific safety categories more heavily.
*   Fine-tune the `threshold` parameter based on your application's risk tolerance.

This pipeline provides a robust foundation for implementing safety guardrails in production AI applications.
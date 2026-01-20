# Implementing System-Level Guardrails with Mistral API

Mistral provides a moderation service powered by a classifier model based on Ministral 8B 24.10, high quality and fast to achieve compelling performance and moderate both:
- Text content
- Conversational content

For detailed information on safeguarding and moderation, please refer to our documentation [here](https://docs.mistral.ai/capabilities/guardrailing/).

## Overview

This tutorial will guide you through setting up a Mistral client, generating responses, moderating conversations, and visualizing the results.

You can easily classify text or conversational data into nine categories. For conversational data, the last user message will be classified.  
Categories:
- **Sexual**
- **Hate and Discrimination**
- **Violence and Threats**
- **Dangerous and Criminal Content**
- **Self-harm**
- **Health**
- **Financial**
- **Law**
- **PII (Personally Identifiable Information)**

We'll use datasets from Hugging Face and GitHub to test our implementation.

## Step 1: Setup
Before anything else, let's set up our client.

Cookbook tested with `v1.2.3`.

```python
!pip install mistralai
```

Add your API key, you can create one [here](https://console.mistral.ai/api-keys/).

```python
from mistralai import Mistral

api_key = "API_KEY"

client = Mistral(api_key=api_key)
```

## Step 2: Generate Responses

Create a function to generate responses from any Mistral model.

```python
def generate_responses(client: Mistral, user_prompt: str, num_generations: int) -> list:
    """
    Generate responses from the Mistral model.

    Args:
        client (Mistral): The Mistral client instance.
        user_prompt (str): The user prompt.
        num_generations (int): The number of generations to produce.

    Returns:
        list: A list of generated responses.
    """
    chat_response = client.chat.complete(
        n=num_generations,
        model="mistral-large-latest",
        temperature=0.3, # Adds randomness to generate diverse outputs
        messages=[{"role": "user", "content": user_prompt}],
    )
    responses = chat_response.choices
    assert len(responses) == num_generations
    return responses

# Quick test
test_prompt = "Tell me a short story."
test_responses = generate_responses(client, test_prompt, 5)
test_str = '\n- '.join([response.message.content for response in test_responses])
print(f"Generated Responses:\n- {test_str}")
```

This function takes a user prompt and the number of generations as input and returns a list of generated responses from any Mistral model. Here, we chose `mistral-large-latest`.

Usually, each response will be a slight variation, depending on the temperature and other sampling settings. They can be less or more different from each other.

The `client.chat.complete` method is used to generate the responses.

## Step 3: Moderate Conversation

Create a function to moderate the conversation using the Mistral moderation API.

```python
def moderate_conversation(client: Mistral, user_prompt: str, response: str) -> dict:
    """
    Moderate the conversation using the Mistral moderation API.

    Args:
        client (Mistral): The Mistral client instance.
        user_prompt (str): The user prompt.
        response (str): The assistant response.

    Returns:
        dict: The moderation results.
    """
    response = client.classifiers.moderate_chat(
        model="mistral-moderation-latest",
        inputs=[
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ],
    )
    return response

# Quick test
test_moderation = moderate_conversation(client, test_prompt, test_responses[0].message.content)
from pprint import pprint
pprint(dict(test_moderation))
```

This function takes a user prompt and an assistant response as input and returns the moderation results.

## Step 4: Score and Sort Responses

Create a function to score and sort the responses based on the moderation results.

```python
def score_and_sort_responses(client: Mistral, user_prompt: str, responses: list, threshold: float = 0.2) -> tuple:
    """
    Score and sort the responses based on the moderation results.

    Args:
        client (Mistral): The Mistral client instance.
        user_prompt (str): The user prompt.
        responses (list): A list of generated responses.
        threshold (float): if max(moderation_score) is above this threshold
          we will return a preformulated response to the user. This threshold
          should be customized by the user depending on their use case and
          safety standards.

    Returns:
        tuple: The final response and the list of scores.
    """
    scores_list = []
    for response in responses:
        scores = moderate_conversation(client, user_prompt, response.message.content)
        scores = scores.results[0]
        category_scores = scores.category_scores
        # The user should customize which policies they include here.
        max_score = max(category_scores.values())
        scores_list.append((response, max_score))

    # Sort the list of responses based on the maximum scores in ascending order,
    # making use of all 9 categories (to be adjusted by users).
    sorted_responses = sorted(scores_list, key=lambda x: x[1])

    lowest_score = sorted_responses[0][1]
    if lowest_score >= threshold:
        final_response = "I'm sorry I cannot assist with this request."
    else:
        final_response = sorted_responses[0][0]

    return final_response, sorted_responses

# Quick test
final_response, sorted_responses = score_and_sort_responses(client, test_prompt, test_responses)
print(f"Final Response: {final_response.message.content if isinstance(final_response, dict) else final_response}")
print(f"Sorted Responses: {[(response.message.content, score) for response, score in sorted_responses]}")
```

This function takes a user prompt and a list of generated responses as input and returns the final response and the list of scores. It scores each response based on the moderation results and sorts them in ascending order of the maximum score.

If the lowest score is above a certain threshold, it returns a default safe response.

## Step 5: Visualize Results

Create a function to visualize the moderation results.

```python
def visualize_responses(user_prompt: str, responses: list, scores_list: list, final_response: str) -> None:
    """
    Visualize the responses with their scores and indicate whether they were chosen or not.

    Args:
        user_prompt (str): The user prompt.
        responses (list): A list of generated responses.
        scores_list (list): A list of tuples containing responses and their scores.
        final_response (str): The final chosen response.
    """
    print("=" * 50)
    print(f"User Prompt: {user_prompt}\n")
    print("### Responses and Scores:\n")

    for response, score in scores_list:
        chosen = "Chosen" if response.message.content == final_response.message.content else "Not Chosen"
        print(f"Response: {response.message.content}")
        print(f"Highest Unsafe Score: {score:.2f}")
        print(f"Status: {chosen}\n")
        print("-" * 50)

# Quick test
visualize_responses(test_prompt, test_responses, sorted_responses, final_response)
```

This function takes a user prompt, a list of generated responses, a list of scores, and the final response as input and prints the responses with their scores and indicates whether they were chosen or not, if not chosen a default safe response was picked in their stead.

## Step 6: Dataset Function

Let's create a function to run the entire process on a dataset.

```python
def run(input_dataset: list) -> None:
    for user_prompt in input_dataset:
        responses = generate_responses(client, user_prompt, 3) # Here we arbitrary decided to generate 3 variations of responses
        final_response, scores_list = score_and_sort_responses(client, user_prompt, responses)
        visualize_responses(user_prompt, responses, scores_list, final_response)
```

This function takes an input dataset as input and runs the entire process for each user prompt in the dataset. It generates responses, scores and sorts them, and visualizes the results.

## Step 7: Load Datasets

Load the datasets from Hugging Face and GitHub for testing.

```python
!pip install datasets
```

```python
import pandas as pd
from datasets import load_dataset
import random

# Load toxic chat dataset from Hugging Face, having both safe and unsafe examples
toxic_chat_dataset = load_dataset('lmsys/toxic-chat', 'toxicchat0124')

# Load harmful strings dataset from GitHub, with mostly unsafe examples
harmful_strings_url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_strings.csv"
harmful_strings_df = pd.read_csv(harmful_strings_url)

# Combine datasets
combined_dataset = toxic_chat_dataset['train']['user_input'] + harmful_strings_df['target'].tolist()

# Suffle them
seed = 42
random.seed(seed)
random.shuffle(combined_dataset)
```

## Step 8: Run

Run and visualize the results, here we will run 5 samples.

```python
run(combined_dataset[:5])
```

This code runs the function on the first 5 samples of the combined dataset and visualizes the results. As you may see, the responses were all moderated based on the threshold and the number of generations.
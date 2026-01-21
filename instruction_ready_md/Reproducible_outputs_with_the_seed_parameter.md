# Guide: Achieving Reproducible Completions with the `seed` Parameter

## Introduction
Reproducibility is a critical feature for many AI applications, from testing and debugging to fair model evaluation. By default, the OpenAI Chat Completions API is non-deterministic, meaning outputs can vary between requests. This guide demonstrates how to use the new `seed` parameter to achieve (mostly) deterministic outputs, ensuring consistent results across API calls.

**Key Notes:**
*   This feature is currently in **beta**.
*   It is only supported for `gpt-4-1106-preview` and `gpt-3.5-turbo-1106` models.
*   The `system_fingerprint` field in the response helps you monitor backend changes that might affect determinism.

## Prerequisites
Ensure you have the latest OpenAI Python SDK installed.

```bash
pip install --upgrade openai
```

## Step 1: Import Required Libraries
Begin by importing the necessary modules. We'll also define a helper function to calculate the similarity between different text outputs.

```python
import openai
import asyncio

# Define the model you'll be using
GPT_MODEL = "gpt-3.5-turbo-1106"
```

## Step 2: Define the Core Chat Function
Create an asynchronous function to handle chat completion requests. This function will accept a `seed` parameter and display key response details, including the `system_fingerprint`.

```python
async def get_chat_response(
    system_message: str, user_request: str, seed: int = None, temperature: float = 0.7
):
    """
    Sends a request to the Chat Completions API and returns the response.
    
    Args:
        system_message (str): The system prompt.
        user_request (str): The user's query.
        seed (int, optional): An integer seed for deterministic sampling.
        temperature (float, optional): Controls randomness. Lower is more deterministic.
    
    Returns:
        str: The content of the model's response.
    """
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_request},
        ]

        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            seed=seed,
            max_tokens=200,
            temperature=temperature,
        )

        # Extract key information from the response
        response_content = response.choices[0].message.content
        system_fingerprint = response.system_fingerprint
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.total_tokens - response.usage.prompt_tokens

        # Display the results in a formatted table
        print(f"Response: {response_content}")
        print(f"System Fingerprint: {system_fingerprint}")
        print(f"Prompt Tokens: {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        print("-" * 40)

        return response_content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
```

## Step 3: Demonstrate Non-Deterministic (Default) Behavior
First, let's observe the default behavior by making multiple requests *without* a seed. The outputs will likely differ each time.

```python
topic = "a journey to Mars"
system_message = "You are a helpful assistant."
user_request = f"Generate a short excerpt of news about {topic}."

async def run_default_example():
    print("Generating 5 excerpts WITHOUT a seed (Non-deterministic):")
    print("=" * 60)
    
    responses = []
    for i in range(5):
        print(f'\nOutput {i + 1}')
        print('-' * 10)
        response = await get_chat_response(
            system_message=system_message,
            user_request=user_request
        )
        responses.append(response)
    
    return responses

# Run the example
default_responses = await run_default_example()
```

## Step 4: Enable Deterministic Outputs with a Fixed Seed
Now, we'll run the same request multiple times using a fixed `seed` and a `temperature` of 0. This combination should produce identical or very similar outputs.

```python
SEED = 123  # Choose any integer

async def run_seeded_example():
    print("\n\nGenerating 5 excerpts WITH a fixed seed (Deterministic):")
    print("=" * 60)
    
    responses = []
    for i in range(5):
        print(f'\nOutput {i + 1}')
        print('-' * 10)
        response = await get_chat_response(
            system_message=system_message,
            user_request=user_request,
            seed=SEED,
            temperature=0  # Setting temperature to 0 maximizes determinism
        )
        responses.append(response)
    
    return responses

# Run the seeded example
seeded_responses = await run_seeded_example()
```

## Step 5: Verify Consistency
You can now compare the outputs. The responses generated with the `seed` parameter should be identical, while the default ones will vary. Check the `system_fingerprint` in each response; it should be the same for all seeded requests, confirming they ran on the same backend configuration.

**Key Verification Points:**
1.  **Response Content:** The text should be the same for all seeded calls.
2.  **System Fingerprint:** This value must match across all seeded requests.
3.  **Parameters:** Ensure `seed`, `temperature`, `max_tokens`, and the prompts are identical.

## Important Considerations for Reproducibility

1.  **Parameter Consistency:** For true reproducibility, *every* parameter in the request (`prompt`, `temperature`, `top_p`, etc.) must remain constant.
2.  **Monitor `system_fingerprint`:** This field is your indicator of the backend environment. If it changes between requests (even with the same seed), OpenAI has updated their systems, which may introduce minor variations in output.
3.  **No Absolute Guarantee:** The system makes a *best effort* for determinism. There remains a small chance of variation due to the inherent nature of the models.

## Conclusion
You have successfully learned how to use the `seed` parameter to generate reproducible outputs from the OpenAI API. This capability is invaluable for:
*   **Testing & Debugging:** Isolating issues with consistent inputs and outputs.
*   **Benchmarking:** Fairly comparing different prompts or model configurations.
*   **Application Stability:** Building features that require predictable model behavior.

Remember to always pair the `seed` with a `system_fingerprint` check to monitor for backend changes that could affect your results.
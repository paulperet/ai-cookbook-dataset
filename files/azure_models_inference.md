# Azure AI Inference with Phi-4 Models

> **Note:** This notebook uses the latest Azure AI Inference SDK. As of May 2025, the correct way to make completion calls is through `ChatCompletionsClient` with appropriate message objects as shown in this notebook.

This notebook demonstrates how to use Azure AI Inference SDK to access Azure-hosted models, specifically Microsoft's Phi-4 models deployed on Azure.

## Setup and Configuration

First, we'll configure our environment and install necessary dependencies.

# Azure AI Inference with Phi-4 Models

This notebook demonstrates how to use Azure AI Inference SDK to access Azure-hosted models, specifically Microsoft's Phi-4 models deployed on Azure.

> **Important Note:** This notebook uses the latest Azure AI Inference SDK methods for completion calls. The API follows this pattern:
>
> ```python
> from azure.ai.inference import ChatCompletionsClient
> from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
> from azure.core.credentials import AzureKeyCredential
>
> client = ChatCompletionsClient(
>     endpoint="your_endpoint",
>     credential=AzureKeyCredential("your_api_key")
> )
>
> response = client.complete(
>     model="your_model_name",
>     messages=[SystemMessage(content="system message"), UserMessage(content="user message")],
>     temperature=0.7,
>     max_tokens=800
> )
> ```

## Setup and Configuration

First, we'll configure our environment and install necessary dependencies.

```python
# Install required packages
!pip install azure-ai-inference python-dotenv matplotlib pillow
```

### Setting up the local.env File

Before running this notebook, you need to create a `local.env` file in the same directory as this notebook with the following variables:

```
# GitHub Configuration
GITHUB_TOKEN=your_personal_access_token_here
GITHUB_INFERENCE_ENDPOINT=https://models.github.ai/inference
GITHUB_MODEL=microsoft/Phi-4-reasoning

# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
AZURE_OPENAI_MODEL=Phi-4-reasoning
```

**Instructions:**

1. Create a new file named `local.env` in the same folder as this notebook
2. Add the three environment variables shown above
3. Replace `your_personal_access_token_here` with your GitHub Personal Access Token
4. You can optionally change the model to `microsoft/Phi-4-mini-reasoning` for a smaller model

**Note:** The GitHub token requires appropriate permissions to access the AI models service.
**Note:** Keep your Azure API key secure and don't share it in public repositories or notebooks.

## Load Environment Variables

Let's load our environment variables from the `local.env` file which contains our Azure AI credentials.

```python
import os
import json
import requests
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Load variables from local.env file
load_dotenv('local.env')

# Access the environment variables - using values from local.env file
api_key = os.getenv("AZURE_API")
endpoint = os.getenv("AZURE_ENDPOINT")
model_name = os.getenv("AZURE_MODEL", "Phi-4")

# Check if required environment variables are available
if not api_key or not endpoint:
    raise ValueError("AZURE_API and AZURE_ENDPOINT must be set in the local.env file.")

print(f"Azure AI Endpoint: {endpoint}")
print(f"Azure AI Model: {model_name}")
print(f"API Key available: {'Yes' if api_key else 'No'}")

# Initialize the Azure AI Inference client
try:
    # Print the endpoint for debugging
    print(f"Debug - Full endpoint URL: {endpoint}")
    
    # Create the chat completions client
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )
    print("Azure AI Inference client initialized successfully")
except Exception as e:
    print(f"Error initializing client: {str(e)}")
    print("If you're seeing initialization errors, please check your endpoint URL and API key.")
```

## Helper Functions for Model Inference

Let's create helper functions to interact with the Azure AI models using the Inference SDK.

```python
def generate_chat_completion(messages, temperature=0.7, max_tokens=20000, model=model_name):
    """Generate a completion using Azure AI Inference SDK's chat completions API"""
    # List of model names to try in order if the primary model fails
    model_alternatives = [model, "Phi-4", "phi-4", "gpt-4"]
    
    for attempt_model in model_alternatives:
        try:
            # Debug information
            print(f"Making API call with model: {attempt_model}")
            
            # Convert message dictionaries to proper message objects
            formatted_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    formatted_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    formatted_messages.append(UserMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_messages.append(AssistantMessage(content=msg["content"]))
            
            # Call the Azure AI Inference API
            response = client.complete(
                model=attempt_model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"Success with model: {attempt_model}")
            return response.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            print(f"Error during API call with {attempt_model}: {error_message}")
            
            # If this is the last model in our list, provide more detailed error information
            if attempt_model == model_alternatives[-1]:
                print("\nAll model attempts failed. Please check:")
                print("1. Your model deployment names in Azure AI Studio")
                print("2. Your API key and endpoint URL")
                print("3. That the models are actually deployed to your Azure AI resource")
                print("\nYou can modify your local.env file to use the correct model name.")
                return None
            
            # If NOT_FOUND error or other specific errors, try the next model
            if "NOT FOUND" in error_message or "model not found" in error_message.lower():
                print(f"Model {attempt_model} not found, trying alternative...")
                continue
            else:
                # For other errors, return the error
                return f"Error: {error_message}"
    
    # This line should not be reached due to the return in the final iteration above
    return None

def generate_completion(prompt, temperature=0.7, max_tokens=10000, model=model_name):
    """Generate a text completion using a simple user prompt"""
    # Simply use the chat completion function with a single user message
    return generate_chat_completion([{"role": "user", "content": prompt}], temperature, max_tokens, model)
```

## Example 1: How many strawberries for 9 r's?

Let's run our first inference example asking about strawberries and r's.

```python
example1_messages = [
    {"role": "system", "content": "You are a helpful AI assistant that answers questions accurately and concisely."},
    {"role": "user", "content": "How many strawberries do I need to collect 9 r's?"}
]

print("Messages:")
for msg in example1_messages:
    print(f"{msg['role']}: {msg['content']}")
print("\nGenerating response...\n")

# Use the chat completion function with the messages
response1 = generate_chat_completion(example1_messages)
print("Response:")
print(response1)
```

    Messages:
    system: You are a helpful AI assistant that answers questions accurately and concisely.
    user: How many strawberries do I need to collect 9 r's?
    
    Generating response...
    
    Making API call with model: Phi-4-reasoning-mbodz
    Success with model: Phi-4-reasoning-mbodz
    Response:
    <think>User's question: "How many strawberries do I need to collect 9 r's?" It's ambiguous. Let's parse question: "How many strawberries do I need to collect 9 r's?" Possibly referring to a game or puzzle. Possibly "r's" means "red letters" or "r's" might be "r's" as in the letter "r". Possibly meaning "strawberries" might be something else. Alternatively, maybe "r's" means "r" required objects. The phrasing "collect 9 r's" might be a puzzle. Let me check: "strawberries" might be fruit in a game like "Strawberries" game? Possibly it's a riddle: "How many strawberries do I need to collect 9 r's?" Possibly it's a puzzle: "9 r's" might be in the word "strawberries" or in something else.
    
    I need to check if there is a known puzzle about strawberries and r's. "Collect 9 r's" might be a puzzle from a website or game. Alternatively, maybe it's a riddle that the letter "r" appears in the word strawberry many times. Wait, let's check: The word "strawberry" is spelled S T R A W B E R R Y. It contains letters: s, t, r, a, w, b, e, r, r, y. That is 10 letters, and it contains three r's? Let's check: Actually "strawberry" letters: S T R A W B E R R Y, so there are letters: s, t, r, a, w, b, e, r, r, y. So that's 10 letters, and there are two r's? Actually count carefully: "strawberry" is spelled s-t-r-a-w-b-e-r-r-y. That gives r appears in third letter, then there is another r in the 8th letter and then one r in the 9th letter? Wait, I'm confused: Let's check letter positions: 1: S, 2: T, 3: R, 4: A, 5: W, 6: B, 7: E, 8: R, 9: R, 10: Y. So there are three r's. So if we consider the word "strawberry", there are 3 r's. But the question says "9 r's". So maybe it's a game where you need to collect letters "r" from strawberries maybe. Alternatively, maybe it's a puzzle: "How many strawberries do I need to collect 9 r's?" might be a riddle: "I have 9 r's and I need to collect some strawberries, how many strawberries will I have?" Possibly it's a pun on "strawberry" and "straw" or "berries" and "berry"? Alternatively, maybe the "r's" refer to the letter r that is in the word "strawberry" or some other words like "strawberries" (the plural) which is spelled S T R A W B E R R I E S. That has "r" letters? Let's count: s-t-r-a-w-b-e-r-r-i-e-s. That is 10 letters, but now the letter r: appears in the 3rd letter, then in the 8th letter, then in the 9th letter, so still three r's. Not 9. Another possibility: "collect 9 r's" might be a reference to a game where you collect letters from words, e.g. "strawberry" has 3 r's, but you need 9 r's. Alternatively, maybe it's a riddle that "how many strawberries do I need to collect 9 r's?" is a riddle from the game "Strawberry Shortcake"? Alternatively, maybe it's a reference to a math puzzle: "strawberries" might be a pun on "strawberry" where you remove the "straw" to collect r's. I'm thinking maybe it's a puzzle on a website.
    
    Maybe it's a reference to a game called "Strawberry's" where you need to pick strawberries to get letters. Alternatively, it might be a riddle where "9 r's" refer to the letter r's in the word "strawberry" repeated some number of times. Possibly: "9 r's" might be the r's from the word "strawberries": S T R A W B E R R I E S, which has three r's, but if you multiply it by 3 you get 9 r's. If you want 9 r's, you might need 3 strawberries because each strawberry gives you 3 r's. Wait, check: In the word "strawberry", there are three r's. If you need 9 r's, then 9 divided by 3 equals 3. But is that the answer? But "How many strawberries do I need" implies count of strawberries. Alternatively, maybe it's a pun: "Collect 9 r's" might be a hint: "9 r's" might be the letter r repeated 9 times. And the word "strawberry" contains r's but not enough if you're collecting letters. Alternatively, maybe it's a riddle: "I have a strawberry field with 9 letters that are r, how many strawberries do I need to collect?" Alternatively, maybe it's a puzzle like: "How many strawberries do I need to collect 9 r's?" might be a word puzzle: "Strawberry" is missing an "r" at the end sometimes. Alternatively, could it be a riddle from the game "Unblock Me" where you need to collect letters.
    
    I recall a riddle: "How many strawberries do you need to pick to get 9 Rs?" Possibly the answer is "3", because each strawberry has 3 Rs (like the letters in "strawberry"). Let me check: "strawberry" letters: S, T, R, A, W, B, E, R, R, Y. It has 3 R's. So if you have 3 strawberries, then you'll have 3*3 =9 R's. But wait, is that the intended answer? That seems plausible. So answer: 3 strawberries.
    
    But let me see if there's any nuance: "collect 9 r's" might be a reference to the fact that a strawberry has letters R in it, maybe the word "strawberry" has three letters "r". So if you need 9 r's, you need 9/3 = 3 strawberries. But wait, the question is "How many strawberries do I need to collect 9 r's?" It might also be a pun on "straw" vs "berry". Alternatively, maybe it's a puzzle from "Strawberries" where you need to collect letters.
    
    Or maybe the answer is "3" if each strawberry contains three r's. Then answer: "3 strawberries."
    
    Let's check: "strawberry" spelled S T R A W B E R R Y. Count r: There's an r in the 3rd letter, an r in the 8th letter, an r in the 9th letter. That is indeed 3 r's. So if you have 3 strawberries, then you'll have 3*3 = 9 r's. So answer: 3. However, careful: If I consider the plural "strawberries" spelled S T R A W B E R R I E S, then how many r's does that have? Let's check: letters: S, T, R, A, W, B, E, R, R, I, E, S? Actually count: S (1), T (2), R (3), A (4), W (5), B (6), E (7), R (8), R (9), I (10), E (11), S (12). That gives 3 r's as well. So still 3. So answer remains 3.
    
    Therefore the answer is likely "3 strawberries" if each strawberry gives you 3 r's. But wait, is it "collect 9 r's"? Possibly the question is "How many strawberries do I need to collect 9 r's?" and answer "3". But also possibility: "9 r's" might be the letter r repeated 9 times, and a strawberry is spelled "strawberry" with 3 r's. So you need 3 strawberries.
    
    However, is it possible that there's another interpretation? Possibly it's a riddle about a "strawberry" that is missing an "r", like "strawberry" vs "strawbery" is missing an r. But "9 r's" might be "r" repeated 9 times. Alternatively, maybe it's a riddle from a game like "Wordle" where you need to find a word with 9 r's, but "strawberry" doesn't have 9 r's.
    
    Alternatively, maybe the riddle means: "How many strawberries do I need to collect 9 r's?" might be a pun: "Strawberries" are red fruits. 9 r's might refer to the letter r repeated 9 times, which is "rrrrrrrrr". And if you remove the "straw" from "strawberry", you get "berry", which doesn't have any r's. So that doesn't help.
    
    Alternatively, maybe it's a challenge: "9 r's" might be the letters in the word "strawberries" (like "berry" but with extra r's) but that doesn't add up.
    
    Alternatively, maybe it's a known riddle: "I have 9 letters and 9 r's, how many strawberries do I need?" That doesn't match.
    
    I trust the simplest interpretation: Each strawberry has 3 r's in its name. To get 9 r's, you need 3 strawberries. Therefore answer: "3."
    
    I'll provide explanation: The word "strawberry" contains 3 r's. So if you need 9 r's, you need 3 strawberries (3 x 3 = 9 r's). I'll provide explanation in plain text.
    
    I'll now produce answer accordingly. I'll produce answer: "3."
    
    I'll produce answer in plain text.</think>It might seem like a riddle at first glance. Notice that the word “strawberry” is spelled S‐T‐R‐A‐W‐B‐E‐R‐R‐
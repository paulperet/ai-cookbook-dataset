# GitHub Marketplace Model Inference with Phi-4 Reasoning 

This notebook demonstrates how to use GitHub Marketplace models for inference, specifically using Microsoft's Phi-4-reasoning model.

## Setup and Configuration

First, we'll configure our environment and install necessary dependencies.


```python
# Install required packages
!pip install requests python-dotenv
!pip install azure-ai-inference
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

## Load Environment Variables

We'll load our environment variables from the `local.env` file which contains our GitHub token and model information.


```python
import os
import requests
import json
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load variables from local.env file
load_dotenv('local.env')

# Access the environment variables - using values from local.env file
endpoint = os.getenv("GITHUB_INFERENCE_ENDPOINT")
model = os.getenv("GITHUB_MODEL")
token = os.getenv("GITHUB_TOKEN")
azuretoken = os.getenv("AZURE_KEY")
azureendpoint = os.getenv("AZURE_ENDPOINT")
azuremodel = os.getenv("AZURE_MODEL")
# Use fallback values if not found in local.env
if not endpoint:
    endpoint = "https://models.github.ai/inference"
    print("Warning: GITHUB_INFERENCE_ENDPOINT not found in local.env, using default value")

if not model:
    model = "microsoft/Phi-4-reasoning"
    print("Warning: GITHUB_MODEL not found in local.env, using default value")
    print("To change the model to Phi-4-mini-reasoning use \"microsoft/Phi-4-mini-reasoning\"")

if not token:
    raise ValueError("GITHUB_TOKEN not found in local.env file. Please add your GitHub token.")

print(f"Endpoint: {endpoint}")
print(f"Model: {model}")
print(f"azure_ai_image_generation_new.ipynb: {azureendpoint}")
print(f"azuremodel: {azuremodel}")
print(f"azuretoken available: {'Yes' if azuretoken else 'No'}")
print(f"Token available: {'Yes' if token else 'No'}")
```

## Helper Functions for Model Inference

Let's create helper functions to interact with the GitHub inference API.


```python
def generate_completion(prompt, model_id=model, temperature=0.7, max_tokens=10000):
    """Generate a completion using the GitHub inference API"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        # GitHub Models API requires a different endpoint structure
        api_url = f"{endpoint}/v1/chat/completions"
        print(f"Calling API at: {api_url}")
        
        # Modify payload for chat completions format
        chat_payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(api_url, headers=headers, json=chat_payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX errors
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            # Handle chat completions response format
            if 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                return result['choices'][0]['message']['content']
            # Fall back to the text field if available
            elif 'text' in result['choices'][0]:
                return result['choices'][0]['text']
            else:
                return f"Error: Could not extract content from response: {result['choices'][0]}"
        else:
            return f"Error: Unexpected response format: {result}"
    except Exception as e:
        print(f"Full error details: {str(e)}")
        return f"Error during API call: {str(e)}"

def format_conversation(messages):
    """Format a conversation for the model"""
    # For chat completion API, we'll just return the messages directly
    return messages

def generate_chat_completion(messages, model_id=model, temperature=0.7, max_tokens=1000):
    """Generate a completion using GitHub's chat completions API"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        api_url = f"{endpoint}/v1/chat/completions"
        print(f"Calling API at: {api_url}")
        
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            if 'message' in result['choices'][0]:
                return result['choices'][0]['message']['content']
            else:
                return f"Error: Unexpected response format: {result['choices'][0]}"
        else:
            return f"Error: Unexpected response format: {result}"
    except Exception as e:
        print(f"Full error details: {str(e)}")
        return f"Error during API call: {str(e)}"
        
# For backward compatibility with existing code
def format_prompt_legacy(messages):
    """Format a conversation in text format (legacy method)"""
    formatted_prompt = ""
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            formatted_prompt += f"User: {content}\n\n"
        elif role == "assistant":
            formatted_prompt += f"Assistant: {content}\n\n"
        elif role == "system":
            formatted_prompt += f"{content}\n\n"
    
    formatted_prompt += "Assistant: "
    return formatted_prompt
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

# Use the new chat completion function directly with the messages
response1 = generate_chat_completion(example1_messages)
print("Response:")
print(response1)
```

    Messages:
    system: You are a helpful AI assistant that answers questions accurately and concisely.
    user: How many strawberries do I need to collect 9 r's?
    
    Generating response...
    
    Calling API at: https://models.github.ai/inference/v1/chat/completions
    Response:
    <think>User says: "How many strawberries do I need to collect 9 r's?" This appears to be some riddle. Possibly the phrase "9 r's" might be a pun or reference to "r" letters. For example, "How many strawberries do I need to collect 9 r's?" It might be a pun on the phrase "strawberry, nine, r's" or "I need 9 r's" might be a riddle. Alternatively, perhaps the question "How many strawberries do I need to collect 9 r's?" might be a riddle where the answer is something like "9 strawberries" are needed if "strawberry" contains one "r" or two? Let's think: "Strawberry" letter count: S T R A W B E R R Y. Counting letter "r": "strawberry" has two "r"s, one in "straw" and one in "berry" but wait, let's check: "strawberry" letters: S, T, R, A, W, B, E, R, R, Y. It has 3 "r"s? Let's check: S=0, T=0, R=1, A=0, W=0, B=0, E=0, R=2, R=3, Y=0. So it has 3 "r"s. Alternatively, maybe the idea is to collect 9 "r" letters by using strawberries. If each strawberry contains 3 "r"s, then we need 9 r's / 3 per strawberry = 3 strawberries. But wait, the question "How many strawberries do I need to collect 9 r's?" might be interpreted as: If you have a strawberry, which has letters "strawberry" with 3 r's, then to get 9 r's, you need 3 strawberries. Alternatively, if the pun is that "strawberries have seeds", then maybe it's a riddle like "9 r's" stands for "9 red letters?" Alternatively, maybe the riddle is "How many strawberries do I need to collect 9 r's?" Possibly a pun on "r's" meaning "letters R", and strawberry is spelled "strawberry" which contains 3 "r's", so 3 strawberries yield 9 r's.
    
    Alternatively, maybe it's a riddle referencing the phrase "I got 9 r's" meaning "I got 9 r's in my name" but doesn't make sense.
    
    Let's check: "How many strawberries do I need to collect 9 r's?" Possibly the answer is 3. But I must check: "strawberry" counts: S=1, T=2, R=3, A=4, W=5, B=6, E=7, R=8, R=9, Y=10. So there are 10 letters in "strawberry". But the question says "collect 9 r's", not letters "r". It might be interpreted as: "How many strawberries do I need to collect 9 red letters?" but no.
    
    Alternatively, "9 r's" might be a pun on the word "sour", which is a pun on "strawberries" because strawberries can be sour. "Collect 9 r's" may be a pun on "strawberries" because they are red and maybe "r" stands for "red"? Possibly "r" stands for "berry"? Alternatively, "9 r's" might be a riddle where r stands for "red", and strawberries are red.
    
    Alternatively, thinking "collect 9 r's" might be a reference to the phrase "strawberry shortcake", maybe it's a riddle from the game "Among Us" or "Hogwarts"? Let me check: "9 r's" might be a reference to "R's" as in "R's" from "Roberts", "r's" might be letters in "strawberry" that are "r's"? Yes.
    
    Alternatively, "collect 9 r's" might be a reference to the phrase "to get 9 'r's", which is how many strawberries? Perhaps the answer is "Three" because each strawberry has three "r's". Let's count carefully: "strawberry" letters: S T R A W B E R R Y. It has 10 letters total, but only 1st R is the third letter: R, then later: letter 8 is R, letter 9 is R. So there are 3 r's in "strawberry".
    
    But wait, let me count: Letters:
    1: S
    2: T
    3: R
    4: A
    5: W
    6: B
    7:


### Analysis of Example 1

In this example, the model needs to understand that the word "strawberry" contains two 'r' letters. Therefore, to collect 9 'r's, you would need 5 strawberries (with 10 'r's total), or 4.5 strawberries to get exactly 9 'r's.

Let's see how the Phi-4-reasoning model handles this problem.

## Example 2: Solving a Riddle

Now let's try a more complex example - a pattern recognition riddle with multiple examples.


```python
example2_messages = [
    {"role": "system", "content": "You are a helpful AI assistant that solves riddles and finds patterns in sequences."},
    {"role": "user", "content": "I will give you a riddle to solve with a few examples, and something to complete at the end"},
    {"role": "user", "content": "nuno Δημήτρης evif Issis 4"},
    {"role": "user", "content": "ntres Inez neves Margot 4"},
    {"role": "user", "content": "ndrei Jordan evlewt Μαρία 9"},
    {"role": "user", "content": "nπέντε Kang-Yuk xis-ytnewt Nubia 21"},
    {"role": "user", "content": "nπέντε Κώστας eerht-ytnewt Μανώλης 18"}, 
    {"role": "user", "content": "nminus one-point-two Satya eno Bill X."},
    {"role": "user", "content": "What is a likely completion for X that is consistent with examples above?"}
]

print("Messages:")
for msg in example2_messages:
    print(f"{msg['role']}: {msg['content'][:50]}...")
print("\nGenerating response...\n")

response2 = generate_chat_completion(example2_messages, temperature=0.2, max_tokens=10000)
print("Response:")
print(response2)
```

    Messages:
    system: You are a helpful AI assistant that solves riddles...
    user: I will give you a riddle to solve with a few examp...
    user: nuno Δημήτρης evif Issis 4...
    user: ntres Inez neves Margot 4...
    user: ndrei Jordan evlewt Μαρία 9...
    user: nπέντε Kang-Yuk xis-ytnewt Nubia 21...
    user: nπέντε Κώστας eerht-ytnewt Μανώλης 18...
    user: nminus one-point-two Satya eno Bill X....
    user: What is a likely completion for X that is consiste...
    
    Generating response...
    
    Calling API at: https://models.github.ai/inference/v1/chat/completions
    Response:
    <think>We are given a riddle with examples. The riddle is: "I will give you a riddle to solve with a few examples, and something to complete at the end". The examples are:
    
    1. "nuno Δημήτρης evif Issis 4"
    2. "ntres Inez neves Margot 4"
    3. "ndrei Jordan evlewt Μαρία 9"
    4. "nπέντε Kang-Yuk xis-ytnewt Nubia 21"
    5. "nπέντε Κώστας eerht-ytnewt Μανώλης 18"
    6. "nminus one-point-two Satya eno Bill X."
    
    We are asked: "What is a likely completion for X that is consistent with examples above?" So we need to analyze the pattern.
    
    Let's re-read the examples carefully:
    
    Example 1: "nuno Δημήτρης evif Issis 4"
    - It seems to be a string with several parts separated by spaces. The parts are: "nuno", "Δημήτρης", "evif", "Issis", "4". 
    - Possibly the riddle is about reversing words, or maybe it's about some transformation. "nuno" might be "nuno" reversed? "nuno" reversed is "onun". "Δημήτρης" is Greek for "Demetris" maybe? "evif" reversed is "five". "Issis" reversed is "ssisI"? Not sure.
    
    Let's check: "nuno" might be "nuno" is "nuno" reversed? "nuno" reversed is "onun". Not sure.
    
    Maybe the pattern is: The first part "nuno" might be "nuno" is "n" + "uno"? "Δημήτρης" is Greek for "Demetris", maybe "Δημήτρης" reversed is "ΣΙΡΕΤΗΜΔ"? Not sure.
    
    Alternatively, maybe the pattern is: The first part is a prefix "n" followed by something. The second part is a name in Greek? The third part is a reversed word? The fourth part is a name? The fifth part is a number.
    
    Let's check example 2: "ntres Inez neves Margot 4"
    - "ntres" might be "n" + "tres"? "Inez" is a name. "neves" reversed is "seven"? Wait, "neves" reversed is "seven"? Actually, "neves" reversed is "seven"? Let's
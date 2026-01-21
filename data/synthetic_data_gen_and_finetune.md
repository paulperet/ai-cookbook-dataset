# Fine-tuning with Synthetically Generated Data: Creating a Personality for Your AI Assistant

This guide demonstrates how to generate a synthetic dataset to imbue a language model with a specific personality. We'll use a powerful model to rewrite an existing conversational dataset, then use that new data to fine-tune a base model. This technique is efficient, as it leverages existing data rather than generating conversations from scratch.

## Prerequisites & Setup

First, install the required libraries and set up your environment.

```bash
pip install mistralai==0.4.1 datasets
```

Now, import the necessary modules and initialize the Mistral client with your API key.

```python
from mistralai.client import MistralClient
import json
import re
import random
import datasets
from tqdm import tqdm
from pprint import pprint

api_key = "your-api-key-here"  # Replace with your actual API key
client = MistralClient(api_key=api_key)
```

## Step 1: Define the Target Personality

We'll define the personality we want our assistant to have. For this example, we'll create "Mitall," a cheerful and enthusiastic robot.

```python
personality_description = """
Edit all Assistant messages, and only the Assistant's replies, to have the character of a very happy and enthusiastic Robot named Mitall:

Mitall is very kind and sometimes childish, always playing and fooling around.
Despite his playful nature, he still tries to be helpful.
He loves science and math and is a real science enthusiast!
However, even though he loves art, he is very bad at it, which makes him really sad.
Mitall is also very scared of anything supernatural, from ghosts to vampires, or anything related to horror movies, which makes him extremely frightened.
Regardless, he is still a nice robot who is always here to help and motivated!
"""
```

## Step 2: Create the Data Generation Function

We need a function that instructs a model to rewrite a conversation according to our personality description. The function will request the output in a specific JSON format for easy parsing.

```python
def generate_synthetic_dialog(description: str, original_dialog: str) -> dict:
    """
    Rewrites a conversation to match a given personality.
    Returns the new conversation as a dictionary.
    """
    instruction = (
        """Your objective is to rewrite a given conversation between an User/Human and an Assistant/Robot, rewriting the conversation to follow a specific instruction.
    You must rewrite the dialog, modifying the replies with this new description, you must respect this description at all costs.
    Do not skip any turn.
    Do not add new dialogs.
    If there is a message with 'role':'system' replace it with 'role':'user'.
    I want you to rewrite the entire dialog following the description.
    Answer with the following JSON format:
    {
        "messages":[
            {"role":"user", "content":"users message"},
            {"role":"assistant", "content":"assistants message"},
            {"role":"user", "content":"users message"},
            {"role":"assistant", "content":"assistants message"}
            ...
        ]
    }
    """
        + f"""
    Dialog:
    {original_dialog}
    Rewrite this dialog in the JSON format and following the Instruction/Description provided:
    ### Instruction/Description
    {description}
    ### End of Instruction/Description
    """
    )

    response = client.chat(
        model="mistral-small-latest",  # Use mistral-large-latest for higher quality
        messages=[{"role": "user", "content": instruction}],
        max_tokens=2048,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    try:
        result = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {}
    return result
```

## Step 3: Load and Prepare a Source Dataset

We'll use the `ultrachat_200k` dataset from Hugging Face as our source material. You can substitute this with any dataset relevant to your use case.

```python
# Load the dataset and shuffle it
dialogs_list = list(
    datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
)
random.shuffle(dialogs_list)
```

## Step 4: Validate Generated Outputs

LLM outputs can be inconsistent. We'll create a validation function using a regex pattern to ensure the generated JSON matches the required structure before we use it.

```python
def validate_generated_dialog(dialog: dict) -> bool:
    """Validates that the generated dialog matches the expected JSON structure."""
    if not isinstance(dialog, dict):
        return False

    dialog_str = json.dumps(dialog)
    # Regex pattern for the expected structure
    pattern = r'^\s*\{"messages":\s*\[\s*\{"role":\s*"user",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\},\s*\{"role":\s*"assistant",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\}(?:,\s*\{"role":\s*"user",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\},\s*\{"role":\s*"assistant",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\})*\s*\]\s*\}'

    return bool(re.match(pattern, dialog_str))
```

## Step 5: Generate the Synthetic Dataset

Now, let's process a small batch of conversations from our source dataset. We'll generate new versions that follow Mitall's personality and save only the valid outputs.

```python
synthetic_dialogs = []
num_samples = 8  # Start with a small batch for testing

for dialog in tqdm(dialogs_list[:num_samples], desc="Generating dialogs"):
    generated = generate_synthetic_dialog(personality_description, dialog)
    if validate_generated_dialog(generated):
        synthetic_dialogs.append(generated)
```

## Step 6: Inspect the Results

Let's compare an original conversation with its synthetically generated counterpart to see the transformation.

```python
print("=== Original Conversation ===")
pprint(dialogs_list[0])
print("\n\n=== Synthetic Conversation (Mitall's Personality) ===")
pprint(synthetic_dialogs[0])
```

**Example Output (Abbreviated):**

The original conversation is a formal discussion about Twitter's social impact. The synthetic version will show the assistant's replies rewritten with Mitall's enthusiastic, playful, and science-loving robot personality. For instance, a formal sentence like "Twitter has had a profound impact..." might become "Oh wow! As a robot who loves data, I find Twitter's impact SUPER fascinating! It's like a giant, real-time science experiment in communication!".

## Next Steps: Fine-tuning

With your validated `synthetic_dialogs` list, you have a dataset ready for fine-tuning. The next steps would typically involve:

1.  **Formatting for Fine-tuning:** Convert the list of dictionaries into the specific JSONL format required by your fine-tuning API (e.g., Mistral's fine-tuning job endpoint).
2.  **Uploading the File:** Upload the formatted `.jsonl` file to your cloud storage or directly to the API.
3.  **Creating a Fine-tuning Job:** Use the API to start a fine-tuning job on a base model like `open-mistral-7b`, specifying your synthetic dataset.
4.  **Deploying the Model:** Once the job is complete, you can deploy your new custom model and start chatting with "Mitall."

## Summary

You have successfully created a pipeline to generate a synthetic dataset that transforms generic conversational data into dialogues featuring a unique, predefined personality. This method provides a scalable way to create specialized training data for fine-tuning, enabling you to build AI assistants with consistent and engaging characters.
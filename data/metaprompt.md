# Metaprompt Tutorial: Generate and Test Prompt Templates

Welcome to this tutorial on using the Metaprompt, a prompt engineering tool designed to help you overcome the "blank page problem" and quickly generate a starting point for your AI prompts. This guide will walk you through setting up the environment, generating a prompt template for your specific task, and testing it with real inputs.

## Prerequisites

Before you begin, ensure you have an Anthropic API key. You will need to install the `anthropic` Python library.

### Step 1: Install the Required Library

Open your terminal or notebook environment and run the following command to install the Anthropic client.

```bash
pip install anthropic
```

### Step 2: Import Libraries and Set Up the Client

Now, import the necessary modules and initialize the Anthropic client with your API key.

```python
import re
import anthropic

# Replace the empty string with your actual API key
ANTHROPIC_API_KEY = ""
MODEL_NAME = "claude-sonnet-4-5"
CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
```

## Part 1: Understanding the Metaprompt

The core of this tool is the "Metaprompt"—a sophisticated, multi-shot prompt containing several examples of well-structured instructions for different tasks. When you provide your task description, the Metaprompt guides an AI (like Claude) to generate a tailored prompt template for you.

The full Metaprompt text is quite long, as it includes detailed examples for tasks like customer service, sentence comparison, document Q&A, math tutoring, and function-based research. You don't need to modify this text; it serves as the engine for template generation.

## Part 2: Generate Your Prompt Template

In this step, you will define your specific task and any input variables you want the final prompt to use.

### Step 3: Define Your Task and Variables

Create variables for your task description and, optionally, a list of input variable names. The variables should be in all caps and separated by commas within a string.

```python
# Define your specific task here
user_task = "Replace with your task!"

# Optionally, specify the input variables you want Claude to use.
# For example: "INPUT, CONTEXT, QUERY"
user_variables = ""
```

### Step 4: Craft the Generation Prompt

We will now create a prompt that asks Claude to use the Metaprompt to generate instructions for your task.

```python
generation_prompt = f"""Today you will be writing instructions to an eager, helpful, but inexperienced and unworldly AI assistant who needs careful instruction and examples to understand how best to behave. I will explain a task to you. You will write instructions that will direct the assistant on how best to accomplish the task consistently, accurately, and correctly. Here are some examples of tasks and instructions.

{metaprompt}

Now it's your turn. I'll give you a task, and optionally some input variables. Please write the full set of instructions for the assistant. Please output the instructions inside <Instructions></Instructions> XML tags, and the inputs inside <Inputs></Inputs> XML tags.

<Task>
{user_task}
</Task>
<Inputs>
{user_variables}
</Inputs>
"""
```

### Step 5: Call the API to Generate the Template

Send the generation prompt to Claude via the API to receive your custom prompt template.

```python
response = CLIENT.messages.create(
    model=MODEL_NAME,
    max_tokens=4000,
    temperature=0,
    messages=[
        {"role": "user", "content": generation_prompt}
    ]
)

# Extract the generated content
generated_content = response.content[0].text
print(generated_content)
```

The output will be a block of text containing your new prompt template within `<Instructions>` tags and the defined inputs within `<Inputs>` tags.

## Part 3: Test Your Generated Prompt Template

Once you have your template, the next step is to test it with concrete examples.

### Step 6: Extract the Template and Prepare Test Inputs

First, parse the generated content to isolate the instructions. Then, define a dictionary with example values for your input variables.

```python
# Use regex to find the Instructions block
instructions_match = re.search(r'<Instructions>(.*?)</Instructions>', generated_content, re.DOTALL)
if instructions_match:
    instructions_text = instructions_match.group(1).strip()
else:
    instructions_text = "Instructions not found."

print("=== GENERATED INSTRUCTIONS ===")
print(instructions_text)
print("=== END INSTRUCTIONS ===\n")

# Example: Define your test inputs.
# If your variables were "DOCUMENT, QUESTION", you would provide values here.
test_inputs = {
    # "DOCUMENT": "This is the text of the document...",
    # "QUESTION": "What is the main topic?"
}
```

### Step 7: Format the Final Test Prompt

Replace the variable placeholders in the instructions with the actual values from your `test_inputs` dictionary.

```python
# Create a copy of the instructions to format
formatted_instructions = instructions_text

# Replace each variable placeholder with its test value
for var_name, value in test_inputs.items():
    placeholder = "{$" + var_name + "}"
    formatted_instructions = formatted_instructions.replace(placeholder, value)

print("=== FINAL PROMPT TO SEND TO CLAUDE ===")
print(formatted_instructions)
print("=== END PROMPT ===\n")
```

### Step 8: Execute the Test

Finally, send the fully formatted prompt to Claude to see how it performs with your example.

```python
test_response = CLIENT.messages.create(
    model=MODEL_NAME,
    max_tokens=4000,
    temperature=0,
    messages=[
        {"role": "user", "content": formatted_instructions}
    ]
)

print("=== CLAUDE'S RESPONSE ===")
print(test_response.content[0].text)
print("=== END RESPONSE ===")
```

## Summary and Next Steps

You have successfully used the Metaprompt to:
1.  **Set up** the Anthropic API client.
2.  **Generate** a custom prompt template tailored to your task.
3.  **Test** the template with specific input values.

Remember, the generated prompt is a starting point. Use the output from your test to iterate and refine the instructions, variables, or examples until the AI's performance meets your needs. This process helps you move quickly from a blank page to a functional, testable prompt.
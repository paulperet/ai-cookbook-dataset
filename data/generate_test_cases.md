# Guide: Generating Synthetic Test Data for Prompt Templates

## Introduction
When developing a prompt template with variables (e.g., `{{DOCUMENTS}}`, `{{QUESTION}}`), you need diverse, realistic test data to evaluate and improve your prompt's performance. This guide shows you how to use Claude and the Anthropic API to generate synthetic test cases—ideal when real data is unavailable or restricted.

You will learn to:
1. Extract variables from a prompt template.
2. Generate realistic synthetic values for those variables.
3. Iteratively refine test cases using planning and examples.
4. Use the generated data for prompt evaluation or as few-shot examples.

---

## Prerequisites

Ensure you have the required libraries installed and your API key configured.

```bash
pip install anthropic IPython
```

```python
import re
import anthropic

# Replace with your actual API key
api_key = "your-api-key-here"
CLIENT = anthropic.Anthropic(api_key=api_key)
MODEL_NAME = "claude-3-5-sonnet-20241022"
```

---

## Step 1: Define Helper Functions

These functions handle variable extraction and formatting.

```python
def extract_variables(prompt_template):
    """Extract all {{variable}} placeholders from a template."""
    pattern = r"{{([^}]+)}}"
    variables = re.findall(pattern, prompt_template)
    return set(variables)

def construct_variables_names(prompt_template):
    """Return a newline-separated list of variable names."""
    variables = extract_variables(prompt_template)
    return "\n".join(variables)

def construct_variables_block(prompt_template):
    """Create an XML-style block describing each variable."""
    variables = extract_variables(prompt_template)
    output = ""
    for v in variables:
        output += f"<{v}>\n"
        output += f'[a full, complete, value for the variable "{v}". (You do not need to repeat the variable name inside the tags.)]\n'
        output += f"</{v}>\n"
    return output.strip()

def construct_example_block(variable_dict):
    """Format a dictionary of variable:value pairs into an XML example block."""
    output = "<example>\n<variables>\n"
    for k, v in variable_dict.items():
        output += f"<{k}>\n{v}\n</{k}>\n"
    output = output.strip()
    output += "\n</variables>\n</example>"
    return output
```

---

## Step 2: Prepare the Prompt for Generating Test Data

The core of this method is a meta-prompt that instructs Claude to analyze your template and produce realistic variable values. We provide two versions: one that incorporates existing examples and one that starts from scratch.

```python
def format_prompt_template_for_synth_evals(prompt_template, examples=None):
    """Format the prompt template for synthetic test data generation."""
    synth_test_data_prompt_template_with_example = """<Prompt Template>
{{PROMPT_TEMPLATE}}
</Prompt Template>

Your job is to construct a test case for the prompt template above. This template contains "variables", which are placeholders to be filled in later. In this case, the variables are:

<variables>
{{CONSTRUCT_VARIABLES_NAMES}}
</variables>

Here are the example test cases provided by the user.
<examples>
{{EXAMPLES}}
</examples>

First, in <planning> tags, do the following:

1. Summarize the prompt template. What is the goal of the user who created it?
2. For each variable in <variables>, carefully consider what a paradigmatic, realistic example of that variable would look like. You'll want to note who will be responsible "in prod" for supplying values. Written by a human "end user"? Downloaded from a website? Extracted from a database? Think about things like length, format, and tone in addition to semantic content. Use the examples provided by the user to guide this exercise. The goal is to acquire a sense of the statistical distribution the examples are being drawn from. The example you write should be drawn from that same distribution, but sufficiently different from the examples that it provides additional signal. A tricky balancing act, but I have faith in you.

Once you're done, output a test case for this prompt template with a full, complete, value for each variable. The output format should consist of a tagged block for each variable, with the value inside the block, like the below:

<variables>
{{CONSTRUCT_VARIABLES_BLOCK}}
</variables>"""

    synth_test_data_prompt_template_without_example = """<Prompt Template>
{{PROMPT_TEMPLATE}}
</Prompt Template>

Your job is to construct a test case for the prompt template above. This template contains "variables", which are placeholders to be filled in later. In this case, the variables are:

<variables>
{{CONSTRUCT_VARIABLES_NAMES}}
</variables>

First, in <planning> tags, do the following:

1. Summarize the prompt template. What is the goal of the user who created it?
2. For each variable in <variables>, carefully consider what a paradigmatic, realistic example of that variable would look like. You'll want to note who will be responsible "in prod" for supplying values. Written by a human "end user"? Downloaded from a website? Extracted from a database? Think about things like length, format, and tone in addition to semantic content.

Then, output a test case for this prompt template with a full, complete, value for each variable. The output format should consist of a tagged block for each variable, with the value inside the block, like the below:
<variables>
{{CONSTRUCT_VARIABLES_BLOCK}}
</variables>"""

    if examples:
        examples_block = "\n".join([construct_example_block(example) for example in examples])
        return (
            synth_test_data_prompt_template_with_example.replace(
                "{{PROMPT_TEMPLATE}}", prompt_template
            )
            .replace("{{CONSTRUCT_VARIABLES_NAMES}}", construct_variables_names(prompt_template))
            .replace("{{CONSTRUCT_VARIABLES_BLOCK}}", construct_variables_block(prompt_template))
            .replace("{{EXAMPLES}}", examples_block)
        )
    else:
        return (
            synth_test_data_prompt_template_without_example.replace(
                "{{PROMPT_TEMPLATE}}", prompt_template
            )
            .replace("{{CONSTRUCT_VARIABLES_NAMES}}", construct_variables_names(prompt_template))
            .replace("{{CONSTRUCT_VARIABLES_BLOCK}}", construct_variables_block(prompt_template))
        )
```

---

## Step 3: Generate a Test Case

Now, define your prompt template and generate the first synthetic test case.

```python
# Define your prompt template with {{variable}} placeholders
prompt_template = """You are a customer support bot for Acme Corporation.
Here is an FAQ with Acme's relevant policies:

<documents>
{{DOCUMENTS}}
</documents>

Please respond to this customer support question using details from the policies:

<question>
{{QUESTION}}
</question>"""

# Extract and display variables
variables = extract_variables(prompt_template)
print("Identified variables:")
for var in variables:
    print(f"- {var}")
```

Initialize an empty list for user-provided examples. You can optionally add golden examples here.

```python
planning_text = None
USER_EXAMPLES = []

# Optional: Manually add an example
# example = {
#     "DOCUMENTS": "Q1: What is your return policy?\nA1: Returns are accepted within 30 days.",
#     "QUESTION": "How long do I have to return an item?"
# }
# USER_EXAMPLES.append(example)
```

Create a function to call the Claude API with the formatted prompt.

```python
def get_test_data(prompt_template, examples, custom_planning=None):
    """Generate test data using the Claude API."""
    synth_eval_prompt_ready = format_prompt_template_for_synth_evals(prompt_template, examples)

    messages = [
        {
            "role": "user",
            "content": synth_eval_prompt_ready,
        }
    ]
    if custom_planning:
        messages.append(
            {
                "role": "assistant",
                "content": custom_planning,
            }
        )

    message = (
        CLIENT.messages.create(
            max_tokens=4000,
            messages=messages,
            model=MODEL_NAME,
            temperature=1,
        )
        .content[0]
        .text
    )

    return message
```

Generate your first test case.

```python
result = get_test_data(prompt_template, USER_EXAMPLES, planning_text)
```

---

## Step 4: Parse and Review the Results

Extract Claude's planning reasoning and the generated variable values from the response.

```python
# Extract the planning section
planning_match = re.search(r"<planning>(.*?)</planning>", result, re.DOTALL)
if planning_match and not planning_text:
    planning_text = "<planning>\n" + planning_match.group(1).strip() + "\n</planning>"

# Extract variable values
extracted_variables = {}
for var in variables:
    var_match = re.search(f"<{var}>(.*?)</{var}>", result[result.index("<variables>") :], re.DOTALL)
    if var_match:
        extracted_variables[var] = var_match.group(1).strip()

# Store this example for future use
USER_EXAMPLES.append(extracted_variables)

print("~~~~~~~~~~~\nGenerated test case:\n~~~~~~~~~~~")
for var, value in extracted_variables.items():
    print(f"{var}:\n{value}\n")

print("~~~~~~~~~~~\nPlanning:\n~~~~~~~~~~~")
print(planning_text)
```

You should see output similar to:

```
Generated test case:
~~~~~~~~~~~
DOCUMENTS:
[Generated FAQ content...]

QUESTION:
[Generated customer question...]

Planning:
~~~~~~~~~~~
<planning>
[Claude's analysis of the template and variable considerations...]
</planning>
```

---

## Step 5: Iteratively Refine the Test Cases

You can guide Claude's generation by modifying its planning text. For example, you might want the FAQ documents to use numbered Q&A format.

```python
# Edit the planning text to specify a numbered format
planning_text = planning_text.replace(
    "each with a question and answer format",
    "each with a question and answer format and associated number.",
)
```

Now, generate a new test case using the updated planning as a prefilled context. This saves tokens and steers the output.

```python
# Clear previous examples if desired, or keep them for context
USER_EXAMPLES = []
result = get_test_data(prompt_template, USER_EXAMPLES, planning_text)
```

Parse and review the new result.

```python
planning_match = re.search(r"<planning>(.*?)</planning>", result, re.DOTALL)
if planning_match and not planning_text:
    planning_text = "<planning>\n" + planning_match.group(1).strip() + "\n</planning>"

extracted_variables = {}
for var in variables:
    var_match = re.search(f"<{var}>(.*?)</{var}>", result[result.index("<variables>") :], re.DOTALL)
    if var_match:
        extracted_variables[var] = var_match.group(1).strip()

USER_EXAMPLES.append(extracted_variables)

print("~~~~~~~~~~~\nGenerated test case:\n~~~~~~~~~~~")
for var, value in extracted_variables.items():
    print(f"{var}:\n{value}\n")
```

The new FAQ documents should now include numbered items.

---

## Step 6: Generate Diverse Examples with Context

To create a second, distinct test case, provide the first example as context. This encourages diversity while maintaining realism.

```python
result = get_test_data(prompt_template, USER_EXAMPLES, planning_text)

# Parse and store as before
planning_match = re.search(r"<planning>(.*?)</planning>", result, re.DOTALL)
if planning_match:
    planning_text = "<planning>\n" + planning_match.group(1).strip() + "\n</planning>"

extracted_variables = {}
for var in variables:
    var_match = re.search(f"<{var}>(.*?)</{var}>", result[result.index("<variables>") :], re.DOTALL)
    if var_match:
        extracted_variables[var] = var_match.group(1).strip()

USER_EXAMPLES.append(extracted_variables)

print("~~~~~~~~~~~\nSecond generated test case:\n~~~~~~~~~~~")
for var, value in extracted_variables.items():
    print(f"{var}:\n{value}\n")
```

You now have multiple synthetic test cases that are realistic, varied, and tailored to your prompt template.

---

## Step 7: Use the Generated Data

You can use these synthetic test cases in two primary ways:

### 1. Prompt Evaluation
Test your final prompt by filling in the template with the generated values and evaluating Claude's response.

```python
def call_claude_with_template(prompt_template, variables):
    """Call Claude with a filled prompt template."""
    filled_template = prompt_template
    for var, value in variables.items():
        filled_template = filled_template.replace(f"{{{{{var}}}}}", value)

    message = (
        CLIENT.messages.create(
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": filled_template,
                }
            ],
            model=MODEL_NAME,
            temperature=0.7,
        )
        .content[0]
        .text
    )

    return message

# Example evaluation
test_response = call_claude_with_template(prompt_template, USER_EXAMPLES[0])
print("Claude's response to the first test case:")
print(test_response)
```

### 2. Create Few-Shot Examples
Add the generated input-output pairs to your prompt as examples to improve performance via few-shot learning. With prompt caching, adding many examples is efficient.

To create a golden answer, you can:
- Write it manually.
- Have Claude generate a response, then refine it.

---

## Conclusion

You have successfully built a pipeline to generate synthetic test data for any prompt template. By iterating on planning and using prior examples, you can produce diverse, realistic test cases suitable for evaluation or enriching your prompts with few-shot examples.

**Next Steps:**
- Automate generation in a loop to create dozens of test cases.
- Integrate this pipeline into a prompt testing framework.
- Use the synthetic data to fine-tune or benchmark your AI workflows.
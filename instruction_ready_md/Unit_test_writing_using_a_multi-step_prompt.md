# Guide: Writing Unit Tests with a Multi-Step AI Prompt

This guide demonstrates how to use a structured, multi-step prompt with an LLM to generate comprehensive unit tests for a Python function. Breaking the task into distinct phases—Explain, Plan, and Execute—helps the AI reason more effectively and produce higher-quality, more reliable test code.

## Prerequisites

Ensure you have the required Python packages installed and your OpenAI API key configured.

```bash
pip install openai
```

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Step 1: Import Libraries and Define Helper Functions

First, import the necessary modules and set up a helper function for printing messages. The `ast` module is used later to validate the generated Python code.

```python
import ast
import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def print_messages(messages):
    """Prints messages sent to or from the LLM."""
    for message in messages:
        role = message["role"]
        content = message["content"]
        print(f"\n[{role}]\n{content}")
```

## Step 2: Define the Core Multi-Step Function

The main function orchestrates the three-step process. It takes a Python function as a string and returns a generated unit test suite.

```python
def unit_tests_from_function(
    function_to_test: str,
    unit_test_package: str = "pytest",
    approx_min_cases_to_cover: int = 7,
    print_text: bool = False,
    explain_model: str = "gpt-3.5-turbo",
    plan_model: str = "gpt-3.5-turbo",
    execute_model: str = "gpt-3.5-turbo",
    temperature: float = 0.4,
    reruns_if_fail: int = 1,
) -> str:
    """Returns a unit test for a given Python function, using a 3-step GPT prompt."""
```

### Step 2.1: Explain the Function

Ask the LLM to explain the provided function in detail. This step ensures the model understands the code's intent and structure before planning tests.

```python
    # Step 1: Generate an explanation of the function
    explain_system_message = {
        "role": "system",
        "content": "You are a world-class Python developer with an eagle eye for unintended bugs and edge cases. You carefully explain code with great detail and accuracy. You organize your explanations in markdown-formatted, bulleted lists.",
    }
    explain_user_message = {
        "role": "user",
        "content": f"""Please explain the following Python function. Review what each element of the function is doing precisely and what the author's intentions may have been. Organize your explanation as a markdown-formatted, bulleted list.

```python
{function_to_test}
```""",
    }
    explain_messages = [explain_system_message, explain_user_message]

    if print_text:
        print_messages(explain_messages)

    explanation_response = client.chat.completions.create(
        model=explain_model,
        messages=explain_messages,
        temperature=temperature,
        stream=True
    )
    explanation = ""
    for chunk in explanation_response:
        if chunk.choices[0].delta.content:
            explanation += chunk.choices[0].delta.content

    explain_assistant_message = {"role": "assistant", "content": explanation}
```

### Step 2.2: Plan the Test Suite

Next, instruct the LLM to plan a diverse set of test scenarios. The plan should cover a wide range of inputs and edge cases.

```python
    # Step 2: Generate a plan to write a unit test
    plan_user_message = {
        "role": "user",
        "content": f"""A good unit test suite should aim to:
- Test the function's behavior for a wide range of possible inputs
- Test edge cases that the author may not have foreseen
- Take advantage of the features of `{unit_test_package}` to make the tests easy to write and maintain
- Be easy to read and understand, with clean code and descriptive names
- Be deterministic, so that the tests always pass or fail in the same way

To help unit test the function above, list diverse scenarios that the function should be able to handle (and under each scenario, include a few examples as sub-bullets).""",
    }
    plan_messages = [
        explain_system_message,
        explain_user_message,
        explain_assistant_message,
        plan_user_message,
    ]

    if print_text:
        print_messages([plan_user_message])

    plan_response = client.chat.completions.create(
        model=plan_model,
        messages=plan_messages,
        temperature=temperature,
        stream=True
    )
    plan = ""
    for chunk in plan_response:
        if chunk.choices[0].delta.content:
            plan += chunk.choices[0].delta.content

    plan_assistant_message = {"role": "assistant", "content": plan}
```

### Step 2.3: Conditionally Elaborate the Plan

If the initial plan is too brief (based on a count of top-level bullet points), ask the model to elaborate with additional edge cases.

```python
    # Step 2b: If the plan is short, ask GPT to elaborate further
    num_bullets = max(plan.count("\n-"), plan.count("\n*"))
    elaboration_needed = num_bullets < approx_min_cases_to_cover

    if elaboration_needed:
        elaboration_user_message = {
            "role": "user",
            "content": """In addition to those scenarios above, list a few rare or unexpected edge cases (and as before, under each edge case, include a few examples as sub-bullets).""",
        }
        elaboration_messages = [
            explain_system_message,
            explain_user_message,
            explain_assistant_message,
            plan_user_message,
            plan_assistant_message,
            elaboration_user_message,
        ]

        if print_text:
            print_messages([elaboration_user_message])

        elaboration_response = client.chat.completions.create(
            model=plan_model,
            messages=elaboration_messages,
            temperature=temperature,
            stream=True
        )
        elaboration = ""
        for chunk in elaboration_response:
            if chunk.choices[0].delta.content:
                elaboration += chunk.choices[0].delta.content

        elaboration_assistant_message = {"role": "assistant", "content": elaboration}
```

### Step 2.4: Execute and Generate the Unit Test Code

Finally, prompt the model to write the actual unit test code based on the accumulated explanations and plans.

```python
    # Step 3: Generate the unit test
    package_comment = ""
    if unit_test_package == "pytest":
        package_comment = "# below, each test case is represented by a tuple passed to the @pytest.mark.parametrize decorator"

    execute_system_message = {
        "role": "system",
        "content": "You are a world-class Python developer with an eagle eye for unintended bugs and edge cases. You write careful, accurate unit tests. When asked to reply only with code, you write all of your code in a single block.",
    }
    execute_user_message = {
        "role": "user",
        "content": f"""Using Python and the `{unit_test_package}` package, write a suite of unit tests for the function, following the cases above. Include helpful comments to explain each line. Reply only with code, formatted as follows:

```python
# imports
import {unit_test_package}  # used for our unit tests
{{insert other imports as needed}}

# function to test
{function_to_test}

# unit tests
{package_comment}
{{insert unit test code here}}
```""",
    }

    execute_messages = [
        execute_system_message,
        explain_user_message,
        explain_assistant_message,
        plan_user_message,
        plan_assistant_message,
    ]
    if elaboration_needed:
        execute_messages += [elaboration_user_message, elaboration_assistant_message]
    execute_messages += [execute_user_message]

    if print_text:
        print_messages([execute_system_message, execute_user_message])

    execute_response = client.chat.completions.create(
        model=execute_model,
        messages=execute_messages,
        temperature=temperature,
        stream=True
    )
    execution = ""
    for chunk in execute_response:
        if chunk.choices[0].delta.content:
            execution += chunk.choices[0].delta.content
```

### Step 2.5: Validate and Return the Code

Extract the Python code from the model's response, validate its syntax using `ast.parse`, and implement a re-run mechanism if parsing fails.

```python
    # Extract code from the markdown code block
    code = execution.split("```python")[1].split("```")[0].strip()

    # Validate the generated code syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax error in generated code: {e}")
        if reruns_if_fail > 0:
            print("Rerunning...")
            return unit_tests_from_function(
                function_to_test=function_to_test,
                unit_test_package=unit_test_package,
                approx_min_cases_to_cover=approx_min_cases_to_cover,
                print_text=print_text,
                explain_model=explain_model,
                plan_model=plan_model,
                execute_model=execute_model,
                temperature=temperature,
                reruns_if_fail=reruns_if_fail - 1,
            )

    return code
```

## Step 3: Run the Function with an Example

Now, let's test the function with a concrete example: a Pig Latin translator.

```python
example_function = """def pig_latin(text):
    def translate(word):
        vowels = 'aeiou'
        if word[0] in vowels:
            return word + 'way'
        else:
            consonants = ''
            for letter in word:
                if letter not in vowels:
                    consonants += letter
                else:
                    break
            return word[len(consonants):] + consonants + 'ay'

    words = text.lower().split()
    translated_words = [translate(word) for word in words]
    return ' '.join(translated_words)
"""

unit_tests = unit_tests_from_function(
    example_function,
    approx_min_cases_to_cover=10,
    print_text=False  # Set to True to see intermediate prompts and responses
)

print(unit_tests)
```

## Example Output

When executed, the function generates a `pytest`-compatible unit test suite:

```python
# imports
import pytest

# function to test
def pig_latin(text):
    def translate(word):
        vowels = 'aeiou'
        if word[0] in vowels:
            return word + 'way'
        else:
            consonants = ''
            for letter in word:
                if letter not in vowels:
                    consonants += letter
                else:
                    break
            return word[len(consonants):] + consonants + 'ay'

    words = text.lower().split()
    translated_words = [translate(word) for word in words]
    return ' '.join(translated_words)


# unit tests
@pytest.mark.parametrize('text, expected', [
    ('hello world', 'ellohay orldway'),  # basic test case
    ('Python is awesome', 'ythonPay isway awesomeway'),  # test case with multiple words
    ('apple', 'appleway'),  # test case with a word starting with a vowel
    ('', ''),  # test case with an empty string
    ('123', '123'),  # test case with non-alphabetic characters
    ('Hello World!', 'elloHay orldWay!'),  # test case with punctuation
    ('The quick brown fox', 'ethay ickquay ownbray oxfay'),  # test case with mixed case words
    ('a e i o u', 'away eway iway oway uway'),  # test case with all vowels
    ('bcd fgh jkl mnp', 'bcday fghay jklway mnpay'),  # test case with all consonants
])
def test_pig_latin(text, expected):
    assert pig_latin(text) == expected
```

## Best Practices and Final Notes

1.  **Model Selection:** For complex tasks, using a more capable model like `gpt-4` often yields better results, especially for character-based transformations.
2.  **Code Review:** Always review the generated code. LLMs can make mistakes, particularly with intricate logic or edge cases.
3.  **Parameter Tuning:** Adjust `temperature` and `approx_min_cases_to_cover` based on the complexity of your function.
4.  **Validation:** The built-in syntax check (`ast.parse`) catches glaring errors, but it does not verify the test logic. Ensure the generated tests are semantically correct.

This multi-step approach provides a structured framework for generating reliable unit tests, turning a complex coding task into a manageable, automated process.
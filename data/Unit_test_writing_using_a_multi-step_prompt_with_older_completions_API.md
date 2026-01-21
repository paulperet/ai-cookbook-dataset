# Guide: Generating Unit Tests with a Multi-Step GPT Prompt

This guide demonstrates how to automate the creation of comprehensive unit tests for a Python function using a structured, multi-step prompt with the OpenAI API. Instead of a single prompt, this method uses a three-step chain that explains the function, plans test scenarios, and finally generates the test code. This approach can produce higher-quality, more thoughtful test suites.

## Prerequisites

Ensure you have the `openai` Python package installed and your API key configured.

```bash
pip install openai
```

## Core Function: `unit_test_from_function`

The main function orchestrates the three-step process. Let's examine it step by step.

### Step 1: Import Required Modules

```python
import ast  # Used to validate generated Python code
import openai
```

### Step 2: Define the Main Function

We'll define the `unit_test_from_function` function with configurable parameters.

```python
def unit_test_from_function(
    function_to_test: str,  # Python function to test, as a string
    unit_test_package: str = "pytest",  # e.g., "pytest" or "unittest"
    approx_min_cases_to_cover: int = 7,  # Minimum number of test case categories
    print_text: bool = False,  # Print the generation process for debugging
    text_model: str = "gpt-3.5-turbo-instruct",  # Model for text/planning steps
    code_model: str = "gpt-3.5-turbo-instruct",  # Model for code generation step
    max_tokens: int = 1000,
    temperature: float = 0.4,  # Avoids repetitive loops; 0.4 is a good balance
    reruns_if_fail: int = 1,  # Number of retries if generated code has syntax errors
) -> str:
    """Outputs a unit test for a given Python function, using a 3-step GPT prompt."""
```

### Step 3: Generate an Explanation of the Function

The first prompt asks the model to explain the function's logic and intent.

```python
    # Step 1: Generate an explanation of the function
    prompt_to_explain_the_function = f"""# How to write great unit tests with {unit_test_package}

In this advanced tutorial for experts, we'll use Python 3.9 and `{unit_test_package}` to write a suite of unit tests to verify the behavior of the following function.
```python
{function_to_test}
```

Before writing any unit tests, let's review what each element of the function is doing exactly and what the author's intentions may have been.
- First,"""

    if print_text:
        text_color_prefix = "\033[30m"  # Black text
        print(text_color_prefix + prompt_to_explain_the_function, end="")

    explanation_response = openai.Completion.create(
        model=text_model,
        prompt=prompt_to_explain_the_function,
        stop=["\n\n", "\n\t\n", "\n    \n"],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    explanation_completion = ""
    if print_text:
        completion_color_prefix = "\033[92m"  # Green text
        print(completion_color_prefix, end="")
    for event in explanation_response:
        event_text = event["choices"][0]["text"]
        explanation_completion += event_text
        if print_text:
            print(event_text, end="")
```

### Step 4: Generate a Test Plan

Using the explanation, we now prompt the model to plan a set of test scenarios.

```python
    # Step 2: Generate a plan to write a unit test
    prompt_to_explain_a_plan = f"""
    
A good unit test suite should aim to:
- Test the function's behavior for a wide range of possible inputs
- Test edge cases that the author may not have foreseen
- Take advantage of the features of `{unit_test_package}` to make the tests easy to write and maintain
- Be easy to read and understand, with clean code and descriptive names
- Be deterministic, so that the tests always pass or fail in the same way

`{unit_test_package}` has many convenient features that make it easy to write and maintain unit tests. We'll use them to write unit tests for the function above.

For this particular function, we'll want our unit tests to handle the following diverse scenarios (and under each scenario, we include a few examples as sub-bullets):
-"""

    if print_text:
        print(text_color_prefix + prompt_to_explain_a_plan, end="")

    prior_text = prompt_to_explain_the_function + explanation_completion
    full_plan_prompt = prior_text + prompt_to_explain_a_plan

    plan_response = openai.Completion.create(
        model=text_model,
        prompt=full_plan_prompt,
        stop=["\n\n", "\n\t\n", "\n    \n"],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    plan_completion = ""
    if print_text:
        print(completion_color_prefix, end="")
    for event in plan_response:
        event_text = event["choices"][0]["text"]
        plan_completion += event_text
        if print_text:
            print(event_text, end="")
```

### Step 5: Conditionally Elaborate the Plan

If the initial plan is too brief, we ask the model to brainstorm additional edge cases.

```python
    # Step 2b: If the plan is short, ask GPT-3 to elaborate further
    elaboration_needed = plan_completion.count("\n-") + 1 < approx_min_cases_to_cover
    if elaboration_needed:
        prompt_to_elaborate_on_the_plan = f"""

In addition to the scenarios above, we'll also want to make sure we don't forget to test rare or unexpected edge cases (and under each edge case, we include a few examples as sub-bullets):
-"""

        if print_text:
            print(text_color_prefix + prompt_to_elaborate_on_the_plan, end="")

        prior_text = full_plan_prompt + plan_completion
        full_elaboration_prompt = prior_text + prompt_to_elaborate_on_the_plan

        elaboration_response = openai.Completion.create(
            model=text_model,
            prompt=full_elaboration_prompt,
            stop=["\n\n", "\n\t\n", "\n    \n"],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        elaboration_completion = ""
        if print_text:
            print(completion_color_prefix, end="")
        for event in elaboration_response:
            event_text = event["choices"][0]["text"]
            elaboration_completion += event_text
            if print_text:
                print(event_text, end="")
```

### Step 6: Generate the Unit Test Code

Finally, we prompt the model to write the actual unit test code based on the accumulated context.

```python
    # Step 3: Generate the unit test
    starter_comment = ""
    if unit_test_package == "pytest":
        starter_comment = "Below, each test case is represented by a tuple passed to the @pytest.mark.parametrize decorator"

    prompt_to_generate_the_unit_test = f"""

Before going into the individual tests, let's first look at the complete suite of unit tests as a cohesive whole. We've added helpful comments to explain what each line does.
```python
import {unit_test_package}  # used for our unit tests

{function_to_test}

#{starter_comment}"""

    if print_text:
        print(text_color_prefix + prompt_to_generate_the_unit_test, end="")

    if elaboration_needed:
        prior_text = full_elaboration_prompt + elaboration_completion
    else:
        prior_text = full_plan_prompt + plan_completion
    full_unit_test_prompt = prior_text + prompt_to_generate_the_unit_test

    unit_test_response = openai.Completion.create(
        model=code_model,
        prompt=full_unit_test_prompt,
        stop="```",
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )
    unit_test_completion = ""
    if print_text:
        print(completion_color_prefix, end="")
    for event in unit_test_response:
        event_text = event["choices"][0]["text"]
        unit_test_completion += event_text
        if print_text:
            print(event_text, end="")
```

### Step 7: Validate and Return the Output

The function validates the generated Python code using `ast.parse`. If a syntax error is found and retries remain, it recursively calls itself.

```python
    # Check the output for errors
    code_start_index = prompt_to_generate_the_unit_test.find("```python\n") + len("```python\n")
    code_output = prompt_to_generate_the_unit_test[code_start_index:] + unit_test_completion
    try:
        ast.parse(code_output)
    except SyntaxError as e:
        print(f"Syntax error in generated code: {e}")
        if reruns_if_fail > 0:
            print("Rerunning...")
            return unit_test_from_function(
                function_to_test=function_to_test,
                unit_test_package=unit_test_package,
                approx_min_cases_to_cover=approx_min_cases_to_cover,
                print_text=print_text,
                text_model=text_model,
                code_model=code_model,
                max_tokens=max_tokens,
                temperature=temperature,
                reruns_if_fail=reruns_if_fail - 1,
            )

    # Return the unit test as a string
    return unit_test_completion
```

## Example Usage

Let's test the function with a simple `is_palindrome` function.

```python
example_function = """def is_palindrome(s):
    return s == s[::-1]"""

generated_test = unit_test_from_function(example_function, print_text=True)
print("\n\nGenerated Test Code:")
print(generated_test)
```

When you run this, the function will print the three-step generation process in color (if `print_text=True`) and output the final unit test code. The output will be a `pytest` test suite covering various scenarios and edge cases for the `is_palindrome` function.

## Key Features of This Approach

1.  **Multi-Step Reasoning:** The model explains and plans before writing code, leading to more robust tests.
2.  **Conditional Branching:** It automatically requests more edge cases if the initial plan is insufficient.
3.  **Model Flexibility:** You can use different models for planning (`text_model`) and code generation (`code_model`).
4.  **Output Validation:** The code is parsed to catch syntax errors, with an option to retry.
5.  **Streaming:** Output is streamed, so you can see partial results immediately, which is useful for long generations.

You can adapt this function for other testing frameworks (like `unittest`) by adjusting the `unit_test_package` parameter and the `starter_comment` logic.
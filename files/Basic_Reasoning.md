##### Copyright 2025 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Basic reasoning

This notebook demonstrates how to use prompting to perform reasoning tasks using the Gemini API's Python SDK. In this example, you will work through a mathematical word problem using prompting.

The Gemini API can handle many tasks that involve indirect reasoning, such as solving mathematical or logical proofs.

In this example, you will see how the LLM explains given problems step by step.


```
%pip install -U -q "google-genai>=1.0.0"
```

    [Installing dependencies..., ..., Installation complete.]

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google import genai
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

Additionally, select the model you want to use from the available options below:


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Examples

Begin by defining some system instructions that will be include when you define and choose the model.


```
from google.genai import types

system_prompt = """
  You are a teacher solving mathematical and logical problems. Your task:
  1. Summarize given conditions.
  2. Identify the problem.
  3. Provide a clear, step-by-step solution.
  4. Provide an explanation for each step.

  Ensure simplicity, clarity, and correctness in all steps of your explanation.
  Each of your task should be done in order and seperately.
"""

config = types.GenerateContentConfig(
    system_instruction=system_prompt
)
```

Next, you can define a logical problem such as the one below.


```
from IPython.display import Markdown

logical_problem = """
  Assume a world where 1 in 5 dice are weighted and have 100% to roll a 6.
  A person rolled a dice and rolled a 6.
  Is it more likely that the die was weighted or not?
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=logical_problem,
    config=config,
)
Markdown(response.text)
```

Okay, I can help with that. Here's how we can break down this probability problem:

1.  **Summarize given conditions.**

    *   1 in 5 dice are weighted to always roll a 6. (20% of dice are weighted)
    *   4 in 5 dice are fair. (80% of dice are fair)
    *   A 6 was rolled.

2.  **Identify the problem.**

    *   We need to determine the probability that the die rolled was weighted, given that a 6 was rolled. This is a conditional probability problem.

3.  **Provide a clear, step-by-step solution.**

    Here's how we can use Bayes' Theorem to solve this:

    *   **Define Events:**
        *   `W` = The die is weighted.
        *   `F` = The die is fair.
        *   `6` = A 6 is rolled.

    *   **State Known Probabilities:**
        *   `P(W) = 1/5 = 0.2` (Probability of a die being weighted)
        *   `P(F) = 4/5 = 0.8` (Probability of a die being fair)
        *   `P(6|W) = 1` (Probability of rolling a 6 given the die is weighted)
        *   `P(6|F) = 1/6` (Probability of rolling a 6 given the die is fair)

    *   **Apply Bayes' Theorem:**
        We want to find `P(W|6)` (Probability the die is weighted given a 6 was rolled).
        Bayes' Theorem states:

        `P(W|6) = [P(6|W) * P(W)] / P(6)`

    *   **Calculate P(6):**
        We need to find the overall probability of rolling a 6.  This can happen in two ways: rolling a 6 with a weighted die or rolling a 6 with a fair die.

        `P(6) = P(6|W) * P(W) + P(6|F) * P(F)`
        `P(6) = (1 * 0.2) + (1/6 * 0.8)`
        `P(6) = 0.2 + 0.1333 = 0.3333` (approximately)

    *   **Calculate P(W|6):**
        Now we can plug the values into Bayes' Theorem:

        `P(W|6) = (1 * 0.2) / 0.3333`
        `P(W|6) = 0.2 / 0.3333`
        `P(W|6) = 0.6` (approximately)

    *   **Calculate P(F|6):**
        We can also calculate `P(F|6)` (Probability the die is fair given a 6 was rolled).

        `P(F|6) = [P(6|F) * P(F)] / P(6)`
        `P(F|6) = [(1/6) * 0.8] / 0.3333`
        `P(F|6) = 0.1333 / 0.3333`
        `P(F|6) = 0.4` (approximately)

4.  **Provide an explanation for each step.**

    *   **Step 1: Define Events:**  Clearly defining the events helps to organize the problem and understand what probabilities we are working with.
    *   **Step 2: State Known Probabilities:**  Listing the known probabilities is crucial for applying Bayes' Theorem correctly. We are given the probabilities of a die being weighted or fair, and we know the probabilities of rolling a 6 given each type of die.
    *   **Step 3: Apply Bayes' Theorem:**  Bayes' Theorem is the fundamental tool for solving this type of conditional probability problem.  It allows us to update our belief about the die being weighted after observing the evidence (rolling a 6).
    *   **Step 4: Calculate P(6):**  This step calculates the overall probability of rolling a 6, considering both weighted and fair dice. This is a necessary component of Bayes' Theorem.  We use the law of total probability to find P(6).
    *   **Step 5: Calculate P(W|6):**  Finally, we plug the calculated values into Bayes' Theorem to find the probability that the die is weighted given that a 6 was rolled.

    **Conclusion:**
    `P(W|6) ≈ 0.6`
    `P(F|6) ≈ 0.4`

    It is more likely (60% probability) that the die was weighted than it was fair (40% probability), given that a 6 was rolled.


```
math_problem = """
  Given a triangle with base b=6 and height h=8, calculate its area
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=math_problem,
    config=config,
)
Markdown(response.text)
```

Okay, I will help you calculate the area of the triangle.

1.  **Summarize given conditions:**
    *   The base of the triangle, b = 6
    *   The height of the triangle, h = 8

2.  **Identify the problem:**
    *   We need to find the area of the triangle given its base and height.

3.  **Provide a step-by-step solution:**
    *   **Step 1: Recall the formula for the area of a triangle.**
        *   The area of a triangle is given by the formula: Area = (1/2) * base * height
    *   **Step 2: Substitute the given values into the formula.**
        *   Area = (1/2) * 6 * 8
    *   **Step 3: Perform the calculation.**
        *   Area = (1/2) * 48
        *   Area = 24

4.  **Provide an explanation for each step:**
    *   **Step 1:** The formula Area = (1/2) \* base \* height is a fundamental formula in geometry for calculating the area of any triangle, given its base and corresponding height.
    *   **Step 2:** We substitute the given values of base (b=6) and height (h=8) into the area formula.
    *   **Step 3:** We first multiply the base and height (6\*8=48), and then multiply the result by 1/2 (or divide by 2) to get the final area, which is 24.

**Answer:** The area of the triangle is 24.

## Next steps

Be sure to explore other examples of prompting in the repository. Try creating your own prompts that include instructions on how to solve basic reasoning problems, or use the prompt given in this notebook as a template.
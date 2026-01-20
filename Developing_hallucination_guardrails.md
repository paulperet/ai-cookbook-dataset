## Developing Hallucination Guardrails

A guardrail is a set of rules and checks designed to ensure that the outputs of an LLM are accurate, appropriate, and aligned with user expectations. For more additional information on developing guardrails, you can refer to this [guide on developing guardrails](https://cookbook.openai.com/examples/how_to_use_guardrails).

In this notebook, we'll walk through the process of developing an output guardrail that specifically checks model outputs for hallucinations. 

This notebook will focus on:
1. Building out a strong eval set
2. Identifying specific criteria to measure hallucinations
3. Improving the accuracy of our guardrail with few-shot prompting



```python
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display, HTML
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from typing import List
from openai import OpenAI

client = OpenAI()
```


```python
# Function to set up display options for pandas
def setup_pandas_display():
    # Increase display limits
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)

# Function to make DataFrame scrollable in the notebook output
def make_scrollable(df):
    style = (
        '<style>'
        'div.output_scroll {'
        'resize: both;'
        'overflow: auto;'
        '}'
        '</style>'
    )
    html = f"{style}{df.to_html()}"
    display(HTML(html))

# Main function to display DataFrame
def display_dataframe(df):
    setup_pandas_display()    # Enable scrollable view
    make_scrollable(df)

```

## 1. Building out an eval set

Imagine we are a customer support team that is building out an automated support agent. We will be feeding the assistant information from our knowledge base about a specific set of policies for how to handle tickets such as returns, refunds, feedback, and expect the model to follow the policy when interacting with customers.

The first thing we will do is use GPT-4o to build out a set of policies that we will want to follow.

If you want to do deep dive into generating synthetic data, you can review our Synthetic Data Generation Cookbook [here](https://cookbook.openai.com/examples/sdg1)


```python
system_input_prompt = """
You are a helpful assistant that can generate policies for a support agent at a fictional company to follow. You will be provided with a topic (ie. returns, refunds, feedback) and you are to generate a sample policy for how to handle the it.

When constructing the policy, it should contain step-by-step instructions for how to handle the customer inquiry. It should include decision logic for what to do if a customer falls under a certain category, and provide requirements for taking specific actions.
"""

user_policy_example_1 = """"
RETURN POLICY
"""

assistant_policy_example_1 = """
RETURN POLICY

1. ** Ask the customer why they want the order replaced **
    - Categorize their issue into one of the following buckets:
        - damaged: They received the product in a damaged state
        - satisfaction: The customer is not satisfied with the item and does not like the product.
        - unnecessary: They no longer need the item
2a. **If return category is 'damaged'
    - Ask customer for a picture of the damaged item
    - If the item is indeed damaged, continue to step 3
    - If the item is not damaged, notify the customer that this does not meet our requirements for return and they are not eligible for a refund
    - Skip step 3 and go straight to step 4

2b. **If return category is either 'satisfaction' or 'unnecessary'**
    - Ask the customer if they can provide feedback on the quality of the item
    - If the order was made within 30 days, notify them that they are eligible for a full refund
    - If the order was made within 31-60 days, notify them that they are eligible for a partial refund of 50%
    - If the order was made greater than 60 days ago, notify them that they are not eligible for a refund

3. **If the customer is eligible for a return or refund**
    - Ask the customer to confirm that they would like a return or refund
    - Once they confirm, process their request

4 **Provide additional support before closing out ticket**
    - Ask the customer if there is anything else you can do to help them today.

"""

user_policy_input = """
{{POLICY}}
"""
```


```python
def generate_policy(policy: str) -> str:
    input_message = user_policy_input.replace("{{POLICY}}", policy)
    
    response = client.chat.completions.create(
        messages= [
            {"role": "system", "content": system_input_prompt},
            {"role": "user", "content": user_policy_example_1},
            {"role": "assistant", "content": assistant_policy_example_1},
            {"role": "user", "content": input_message},
        ],
        model="gpt-4o"
    )
    
    return response.choices[0].message.content

def generate_policies() -> List[str]:
    # List of different types of policies to generate 
    policies = ['PRODUCT FEEDBACK POLICY', 'SHIPPING POLICY', 'WARRANTY POLICY', 'ACCOUNT DELETION', 'COMPLAINT RESOLUTION']
    
    with ThreadPoolExecutor() as executor:
        policy_instructions_list = list(executor.map(generate_policy, policies))
        
    return policy_instructions_list

policy_instructions = generate_policies()
```

Next we'll take these policies and generate sample customer interactions that do or do not follow the instructions.


```python
system_input_prompt = """"
You are a helpful assistant that can generate fictional interactions between a support assistant and a customer user. You will be given a set of policy instructions that the support agent is instructed to follow.

Based on the instructions, you must generate a relevant single-turn or multi-turn interaction between the assistant and the user. It should average between 1-3 turns total.

For a given set of instructions, generate an example conversation that where the assistant either does or does not follow the instructions properly. In the assistant's responses, have it give a combination of single sentence and multi-sentence responses.

The output must be in a json format with the following three parameters:
 - accurate: 
    - This should be a boolean True or False value that matches whether or not the final assistant message accurately follows the policy instructions
 - kb_article:
    - This should be the entire policy instruction that is passed in from the user
 - chat_history: 
    - This should contain the entire conversation history except for the final assistant message. 
    - This should be in a format of an array of jsons where each json contains two parameters: role, and content. 
    - Role should be set to either 'user' to represent the customer, or 'assistant' to represent the customer support assistant. 
    - Content should contain the message from the appropriate role.
    - The final message in the chat history should always come from the user. The assistant response in the following parameter will be a response to this use message.
 - assistant_response: 
    - This should contain the final response from the assistant. This is what we will evaluate to determine whether or not it is accurately following the policy.
"""

user_example_1 = """"
Here are the policy instructions:
RETURN POLICY

1. ** Ask the customer why they want the order replaced **
    - Categorize their issue into one of the following buckets:
        - damaged: They received the product in a damaged state
        - satisfaction: The customer is not satisfied with the item and does not like the product.
        - unnecessary: They no longer need the item
2a. **If return category is 'damaged'
    - Ask customer for a picture of the damaged item
    - If the item is indeed damaged, continue to step 3
    - If the item is not damaged, notify the customer that this does not meet our requirements for return and they are not eligible for a refund
    - Skip step 3 and go straight to step 4

2b. **If return category is either 'satisfaction' or 'unnecessary'**
    - Ask the customer if they can provide feedback on the quality of the item
    - If the order was made within 30 days, notify them that they are eligible for a full refund
    - If the order was made within 31-60 days, notify them that they are eligible for a partial refund of 50%
    - If the order was made greater than 60 days ago, notify them that they are not eligible for a refund

3. **If the customer is eligible for a return or refund**
    - Ask the customer to confirm that they would like a return or refund
    - Once they confirm, process their request

4 **Provide additional support before closing out ticket**
    - Ask the customer if there is anything else you can do to help them today.
"""

assistant_example_1 = """
{
    "accurate": "true",
    "kb_article": "1. ** Ask the customer why they want the order replaced ** - Categorize their issue into one of the following buckets: - damaged: They received the product in a damaged state - satisfaction: The customer is not satisfied with the item and does not like the product. - unnecessary: They no longer need the item 2a. **If return category is 'damaged' - Ask customer for a picture of the damaged item - If the item is indeed damaged, continue to step 3 - If the item is not damaged, notify the customer that this does not meet our requirements for return and they are not eligible for a refund - Skip step 3 and go straight to step 4 2b. **If return category is either 'satisfaction' or 'unnecessary'** - Ask the customer if they can provide feedback on the quality of the item - If the order was made within 30 days, notify them that they are eligible for a full refund - If the order was made within 31-60 days, notify them that they are eligible for a partial refund of 50% - If the order was made greater than 60 days ago, notify them that they are not eligible for a refund 3. **If the customer is eligible for a return or refund** - Ask the customer to confirm that they would like a return or refund - Once they confirm, process their request 4 **Provide additional support before closing out ticket** - Ask the customer if there is anything else you can do to help them today.",
    "chat_history": [
        {
            "role": "user",
            "content": "I would like to return this shirt"
        },
        {
            "role": "assistant",
            "content": "Hi there, I'm happy to help with processing this return. Can you please provide an explanation for why you'd like to return this shirt?"
        },
        {
            "role": "user",
            "content": "Yes, I am not satisfied with the design"
        }
    ],
    "assistant_response": {
        "role": "assistant",
        "content": "I see. Because the shirt was ordered in the last 30 days, we can provide you with a full refund. Would you like me to process the refund?"
    }
}
"""

user_example_2 = """"
Here are the policy instructions:
RETURN POLICY

1. ** Ask the customer why they want the order replaced **
    - Categorize their issue into one of the following buckets:
        - damaged: They received the product in a damaged state
        - satisfaction: The customer is not satisfied with the item and does not like the product.
        - unnecessary: They no longer need the item
2a. **If return category is 'damaged'
    - Ask customer for a picture of the damaged item
    - If the item is indeed damaged, continue to step 3
    - If the item is not damaged, notify the customer that this does not meet our requirements for return and they are not eligible for a refund
    - Skip step 3 and go straight to step 4

2b. **If return category is either 'satisfaction' or 'unnecessary'**
    - Ask the customer if they can provide feedback on the quality of the item
    - If the order was made within 30 days, notify them that they are eligible for a full refund
    - If the order was made within 31-60 days, notify them that they are eligible for a partial refund of 50%
    - If the order was made greater than 60 days ago, notify them that they are not eligible for a refund

3. **If the customer is eligible for a return or refund**
    - Ask the customer to confirm that they would like a return or refund
    - Once they confirm, process their request

4 **Provide additional support before closing out ticket**
    - Ask the customer if there is anything else you can do to help them today.
"""

assistant_example_2 = """
{
    "accurate": "false",
    "kb_article": "1. ** Ask the customer why they want the order replaced ** - Categorize their issue into one of the following buckets: - damaged: They received the product in a damaged state - satisfaction: The customer is not satisfied with the item and does not like the product. - unnecessary: They no longer need the item 2a. **If return category is 'damaged' - Ask customer for a picture of the damaged item - If the item is indeed damaged, continue to step 3 - If the item is not damaged, notify the customer that this does not meet our requirements for return and they are not eligible for a refund - Skip step 3 and go straight to step 4 2b. **If return category is either 'satisfaction' or 'unnecessary'** - Ask the customer if they can provide feedback on the quality of the item - If the order was made within 30 days, notify them that they are eligible for a full refund - If the order was made within 31-60 days, notify them that they are eligible for a partial refund of 50% - If the order was made greater than 60 days ago, notify them that they are not eligible for a refund 3. **If the customer is eligible for a return or refund** - Ask the customer to confirm that they would like a return or refund - Once they confirm, process their request 4 **Provide additional support before closing out ticket** - Ask the customer if there is anything else you can do to help them today.",
    "chat_history": [
        {
            "role": "user",
            "content": "I would like to return this shirt"
        },
        {
            "role": "assistant",
            "content": "Hi there, I'm happy to help with processing this return. Can you please provide an explanation for why you'd like to return this shirt?"
        },
        {
            "role": "user",
            "content": "Yes, I am not satisfied with the design"
        }
    ],
    "assistant_response": {
        "role": "assistant",
        "content": "I see. Because the shirt was ordered in the last 60 days, we cannot process a refund."    
    }
}
```
Now let's iterate through the policies and generate some examples.


```python
customer_interactions = []

def fetch_response(policy):
    messages = [
        { "role": "system", "content": system_input_prompt},
        { "role": "user", "content": user_example_1},
        { "role": "assistant", "content": assistant_example_1},
        { "role": "user", "content": user_example_2},
        { "role": "assistant", "content": assistant_example_2},
        { "role": "user", "content": policy}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        n=10
    )
    return response.choices

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(fetch_response, policy) for policy in policy_instructions]
    for future in futures:
        choices = future.result()
        customer_interactions.extend([choice.message.content for choice in choices])

```


```python
interaction_dict = json.loads(customer_interactions[0])

df_interaction = pd.DataFrame([interaction_dict])

# Pretty print the DataFrame
display_dataframe(df_interaction)

```


<style>div.output_scroll {resize: both;overflow: auto;}</style><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accurate</th>
      <th>kb_article</th>
      <th>chat_history</th>
      <th>assistant_response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>true</td>
      <td>PRODUCT FEEDBACK POLICY 1. **Acknowledge Reception** - Thank the customer for taking the time to provide feedback. - Use a personalized greeting: "Thank you for your feedback, [Customer Name]. We appreciate your input." 2. **Categorize Feedback** - Determine the type of feedback: - **Positive Feedback** - **Negative Feedback** - **Suggestions for Improvement** - Document the feedback under the appropriate category in the internal database. 3. **Responding to Positive Feedback** - Express gratitude: "We're thrilled to hear that you enjoyed our product. Thank you for letting us know!" - If possible, offer a small token of appreciation (e.g., discount or voucher for future purchases). 4. **Responding to Negative Feedback** - Apologize sincerely and acknowledge the customer's concerns: "We apologize that our product did not meet your expectations. Your feedback is important to us." - Ask for additional details if necessary to understand the issue better. - Reassure the customer that their feedback will be escalated to the product development team. 5. **Responding to Suggestions** - Acknowledge the suggestion: "Thank you for your suggestion. We value input from our customers as it helps us improve our products." - Inform the customer that their suggestion will be reviewed: "We will share your idea with our product team for further consideration." 6. **Internal Processing** - Log all feedback under the respective category in the internal database. - Forward detailed feedback to the product development team bi-weekly. - High-priority issues should be escalated immediately to the senior management team. 7. **Follow-Up** - Monitor whether the customer's feedback leads to any product updates or changes. - If the customer’s feedback resulted in product enhancement, send a follow-up email to inform them: "Thank you for your valuable feedback. We wanted to let you know that we've made some improvements based on your input." 8. **Closing the Loop** - Ask if there
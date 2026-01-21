# AI-Powered Customer Service: Converting Knowledge Base Articles into Executable Routines

## Overview
This guide demonstrates how to transform complex customer service knowledge base articles into structured, executable routines for Large Language Models (LLMs). By breaking down human-oriented documentation into step-by-step instructions with clear actions and conditions, you enable LLMs to handle customer inquiries systematically and accurately.

## Prerequisites

Ensure you have the required Python packages installed:

```bash
pip install openai pandas
```

## Step 1: Import Required Libraries

```python
from openai import OpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import csv

# Initialize the OpenAI client
client = OpenAI()
MODEL = 'o1-preview'
```

## Step 2: Load Knowledge Base Articles

We'll use publicly available Help Center articles from OpenAI as our source material. These articles are stored in a CSV file.

```python
articles = []

with open('../data/helpcenter_articles.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        articles.append({
            "policy": row["policy"],
            "content": row["content"]
        })
```

The CSV contains these four articles:
- How do I delete my payment method
- How can I get a Business Associate Agreement (BAA) with OpenAI?
- How can I set up prepaid billing?
- How do I submit a VAT exemption request

## Step 3: Define the Conversion Prompt

Create a detailed prompt that instructs the LLM how to convert help center articles into executable routines. This prompt ensures consistent formatting and comprehensive coverage.

```python
CONVERSION_PROMPT = """
You are a helpful assistant tasked with taking an external facing help center article and converting it into a internal-facing programmatically executable routine optimized for an LLM. 
The LLM using this routine will be tasked with reading the policy, answering incoming questions from customers, and helping drive the case toward resolution.

Please follow these instructions:
1. **Review the customer service policy carefully** to ensure every step is accounted for. It is crucial not to skip any steps or policies.
2. **Organize the instructions into a logical, step-by-step order**, using the specified format.
3. **Use the following format**:
   - **Main actions are numbered** (e.g., 1, 2, 3).
   - **Sub-actions are lettered** under their relevant main actions (e.g., 1a, 1b).
      **Sub-actions should start on new lines**
   - **Specify conditions using clear 'if...then...else' statements** (e.g., 'If the product was purchased within 30 days, then...').
   - **For instructions that require more information from the customer**, provide polite and professional prompts to ask for additional information.
   - **For actions that require data from external systems**, write a step to call a function using backticks for the function name (e.g., `call the check_delivery_date function`).
      - **If a step requires the customer service agent to take an action** (e.g., process a refund), generate a function call for this action (e.g., `call the process_refund function`).
      - **Define any new functions** by providing a brief description of their purpose and required parameters.
   - **If there is an action an assistant can perform on behalf of the user**, include a function call for this action (e.g., `call the change_email_address function`), and ensure the function is defined with its purpose and required parameters.
      - This action may not be explicitly defined in the help center article, but can be done to help the user resolve their inquiry faster
   - **The step prior to case resolution should always be to ask if there is anything more you can assist with**.
   - **End with a final action for case resolution**: calling the `case_resolution` function should always be the final step.
4. **Ensure compliance** by making sure all steps adhere to company policies, privacy regulations, and legal requirements.
5. **Handle exceptions or escalations** by specifying steps for scenarios that fall outside the standard policy.

**Important**: If at any point you are uncertain, respond with "I don't know."

Please convert the customer service policy into the formatted routine, ensuring it is easy to follow and execute programmatically.
"""
```

## Step 4: Create the Routine Generation Function

Define a function that sends the article content to the LLM with the conversion prompt and returns the generated routine.

```python
def generate_routine(policy):
    try:
        messages = [
            {
                "role": "user",
                "content": f"""
                    {CONVERSION_PROMPT}

                    POLICY:
                    {policy}
                """
            }
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        
        return response.choices[0].message.content 
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
```

## Step 5: Process Articles in Parallel

To speed up the conversion process, we'll process multiple articles simultaneously using thread pooling.

```python
def process_article(article):
    routine = generate_routine(article['content'])
    return {
        "policy": article['policy'], 
        "content": article['content'], 
        "routine": routine
    }

# Process all articles in parallel
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_article, articles))
```

## Step 6: Review the Generated Routines

Store the results in a DataFrame for easy examination and analysis.

```python
# Create a DataFrame from the results
df = pd.DataFrame(results)

# Configure pandas to display full text content
pd.set_option('display.max_colwidth', None)

# Display the results
print("Generated Routines:")
print("=" * 80)
for index, row in df.iterrows():
    print(f"\nPolicy: {row['policy']}")
    print("-" * 40)
    print(f"Routine:\n{row['routine']}")
    print("=" * 80)
```

## Analysis of Generated Routines

After generating routines for all articles, you'll notice several key patterns:

### 1. Sample Responses
The model generates sample responses that the LLM can use when executing the policy, such as:
- "Instruct the user: 'Confirm and purchase your initial amount of credits.'"

### 2. Discrete Steps
Each routine breaks down complex processes into discrete, actionable steps that are easy for an LLM to interpret and execute.

### 3. Function Definitions
The routines include clearly defined functions for external interactions, such as:
- `review_and_apply_tax_exemption`
- `get_billing_plan`
- `update_payment_method`

These function definitions are crucial because they enable the LLM to interact with external systems and perform actions on behalf of users.

### 4. IFTTT Logic
The model effectively uses "If This, Then That" logic throughout the routines, which is ideal for LLM comprehension. For example:
- "If the customer requests assistance, proceed to step 3f."

This logical structure is particularly valuable when converting complex workflows and diagrams from original knowledge base articles.

## Next Steps

### 1. Integration into Agentic Systems
These generated routines can now be integrated into customer service systems. When a customer requests assistance, you can:
- Use a classifier to determine which routine applies to their inquiry
- Provide the appropriate routine to the LLM
- Allow the LLM to guide the customer through the process or perform actions on their behalf

### 2. Evaluation and Validation
Before deploying to production, develop comprehensive evaluations to:
- Test the quality of the model's responses
- Ensure compliance with company policies and regulations
- Validate effectiveness in real-world scenarios
- Adjust routines based on testing results

### 3. Scaling the Approach
This methodology can be extended to:
- Additional knowledge base articles
- Different types of documentation (FAQs, troubleshooting guides, etc.)
- Various domains beyond customer service

## Key Benefits

1. **Reduced Ambiguity**: Breaking complex articles into routines minimizes ambiguity for the LLM
2. **Methodical Processing**: The step-by-step approach allows systematic information processing
3. **Reduced Hallucination Risk**: Clear instructions reduce the likelihood of the LLM deviating from expected paths
4. **Scalability**: This approach can be automated and scaled across large knowledge bases
5. **Actionable Outputs**: The inclusion of function calls enables the LLM to perform actual tasks, not just provide information

By following this guide, you can transform human-oriented documentation into executable routines that enable LLMs to provide effective, accurate customer service at scale.
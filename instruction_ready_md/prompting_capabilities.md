# Prompting Capabilities with Mistral AI: A Practical Guide

This guide demonstrates four core prompting capabilities of Mistral AI models: Classification, Summarization, Personalization, and Evaluation. You'll learn practical strategies for crafting effective prompts through hands-on examples.

## Prerequisites

First, install the required library and set up your environment.

```bash
pip install mistralai requests
```

```python
from mistralai import Mistral
import requests

# Replace with your actual API key
api_key = "YOUR_API_KEY_HERE"
```

## Helper Function

We'll use a helper function to interact with the Mistral API throughout this guide.

```python
def run_mistral(user_message, model="mistral-large-latest"):
    """Send a prompt to the Mistral API and return the response."""
    client = Mistral(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]
    
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content
```

---

## 1. Classification

Mistral models can categorize text into predefined classes. This is useful for routing customer inquiries, tagging content, or organizing data.

### Step 1: Define the Classification Prompt

Create a function that constructs a classification prompt using few-shot learning, delimiters, and role-playing.

```python
def create_classification_prompt(inquiry):
    """
    Creates a prompt to classify a customer inquiry into predefined categories.
    
    Strategies used:
    - Few-shot learning: Providing examples improves model performance
    - Delimiters: Using ### and <<<>>> to separate sections
    - Role-playing: Giving the model a specific role (customer service bot)
    """
    user_message = f"""
    You are a bank customer service bot. Your task is to assess customer intent
    and categorize customer inquiry after <<<>>> into one of the following predefined categories:

    card arrival
    change pin
    exchange rate
    country support
    cancel transfer
    charge dispute

    If the text doesn't fit into any of the above categories, classify it as:
    customer service

    You will only respond with the predefined category. Do not include the word "Category". 
    Do not provide explanations or notes.

    ####
    Here are some examples:

    Inquiry: How do I know if I will get my card, or if it is lost? I am concerned about the delivery process and would like to ensure that I will receive my card as expected. Could you please provide information about the tracking process for my card, or confirm if there are any indicators to identify if the card has been lost during delivery?
    Category: card arrival
    Inquiry: I am planning an international trip to Paris and would like to inquire about the current exchange rates for Euros as well as any associated fees for foreign transactions.
    Category: exchange rate
    Inquiry: What countries are getting support? I will be traveling and living abroad for an extended period of time, specifically in France and Germany, and would appreciate any information regarding compatibility and functionality in these regions.
    Category: country support
    Inquiry: Can I get help starting my computer? I am having difficulty starting my computer, and would appreciate your expertise in helping me troubleshoot the issue.
    Category: customer service
    ###

    <<<
    Inquiry: {inquiry}
    >>>
    """
    return user_message
```

### Step 2: Test the Classifier

Now let's test our classifier with different inquiries.

```python
# Test with a relevant inquiry
inquiry1 = "I am inquiring about the availability of your cards in the EU, as I am a resident of France and am interested in using your cards."
classification1 = run_mistral(create_classification_prompt(inquiry1))
print(f"Inquiry: {inquiry1}")
print(f"Classification: {classification1}")
```

```
country support
```

```python
# Test with an unrelated inquiry
inquiry2 = "What's the weather today?"
classification2 = run_mistral(create_classification_prompt(inquiry2))
print(f"\nInquiry: {inquiry2}")
print(f"Classification: {classification2}")
```

```
customer service
```

The classifier correctly identifies banking-related inquiries and defaults to "customer service" for unrelated questions.

---

## 2. Summarization

LLMs excel at summarizing text while maintaining key information. Let's create a comprehensive summarization workflow.

### Step 1: Load the Source Material

We'll use Paul Graham's essay as our source text.

```python
# Download the essay
response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
essay = response.text[:2000]  # Use first 2000 characters for demonstration
```

### Step 2: Create a Multi-Step Summarization Prompt

This prompt uses step-by-step instructions to guide the model through a complex summarization task.

```python
summarization_prompt = f"""
You are a commentator. Your task is to write a report on an essay.
When presented with the essay, come up with interesting questions to ask, and answer each question.
Afterward, combine all the information and write a report in the markdown format.

# Essay:
{essay}

# Instructions:
## Summarize:
In clear and concise language, summarize the key points and themes presented in the essay.

## Interesting Questions:
Generate three distinct and thought-provoking questions that can be asked about the content of the essay. For each question:
- After "Q: ", describe the problem
- After "A: ", provide a detailed explanation of the problem addressed in the question.
- Enclose the ultimate answer in <>.

## Write a report
Using the essay summary and the answers to the interesting questions, create a comprehensive report in Markdown format.
"""
```

### Step 3: Execute the Summarization

```python
summary_report = run_mistral(summarization_prompt)
print(summary_report[:500])  # Print first 500 characters of the output
```

The model will generate a structured report including:
1. A concise summary of the essay
2. Three interesting questions with detailed answers
3. A comprehensive Markdown-formatted report

**Strategies used:**
- **Step-by-step instructions**: Breaking complex tasks into smaller steps improves reasoning
- **Example generation**: Asking the model to generate questions with explanations
- **Output formatting**: Specifying Markdown format for structured output

---

## 3. Personalization

LLMs can personalize responses based on provided facts and context. This is ideal for customer service applications.

### Step 1: Define the Customer Email and Facts

```python
customer_email = """
Dear mortgage lender,

What's your 30-year fixed-rate APR, how is it compared to the 15-year fixed rate?

Regards,
Anna
"""

mortgage_facts = """
30-year fixed-rate: interest rate 6.403%, APR 6.484%
20-year fixed-rate: interest rate 6.329%, APR 6.429%
15-year fixed-rate: interest rate 5.705%, APR 5.848%
10-year fixed-rate: interest rate 5.500%, APR 5.720%
7-year ARM: interest rate 7.011%, APR 7.660%
5-year ARM: interest rate 6.880%, APR 7.754%
3-year ARM: interest rate 6.125%, APR 7.204%
30-year fixed-rate FHA: interest rate 5.527%, APR 6.316%
30-year fixed-rate VA: interest rate 5.684%, APR 6.062%
"""
```

### Step 2: Create the Personalization Prompt

```python
personalization_prompt = f"""
You are a mortgage lender customer service bot, and your task is to create personalized email responses to address customer questions.
Answer the customer's inquiry using the provided facts below. Ensure that your response is clear, concise, and
directly addresses the customer's question. Address the customer in a friendly and professional manner. Sign the email with
"Lender Customer Support."

# Facts
{mortgage_facts}

# Email
{customer_email}
"""
```

### Step 3: Generate the Personalized Response

```python
personalized_response = run_mistral(personalization_prompt)
print(personalized_response)
```

The model will generate a professional email that:
- Addresses Anna by name
- Provides the specific APR rates she asked about
- Compares the 30-year and 15-year rates
- Maintains a friendly, professional tone
- Includes the requested signature

**Strategy used:**
- **Providing facts**: Giving the model specific data ensures accurate, consistent responses

---

## 4. Evaluation

Evaluating LLM outputs is crucial for production systems. Here are three approaches:

### Approach 1: Include Confidence Scores

Add confidence scores to model outputs by requesting JSON format.

```python
def run_mistral_json(user_message, model="mistral-large-latest"):
    """Run Mistral with JSON response format enabled."""
    client = Mistral(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]
    
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        temperature=1,  # Higher temperature for more creative outputs
        response_format={"type": "json_object"}
    )
    return chat_response.choices[0].message.content

# Create evaluation prompt with confidence scores
evaluation_prompt = f"""
You are a summarization system that can provide summaries with associated confidence scores.
In clear and concise language, provide three short summaries of the following essay, along with their confidence scores.
You will only respond with a JSON object with the keys "Summary" and "Confidence". Do not provide explanations.

# Essay:
{essay[:1000]}
"""

json_output = run_mistral_json(evaluation_prompt)
print(json_output)
```

**Strategies used:**
- **JSON output**: Structured format for downstream processing
- **Higher temperature**: Encourages diverse outputs for comparison

### Approach 2: Introduce an Evaluation Step

Build evaluation directly into the prompt.

```python
self_evaluation_prompt = f"""
You are given an essay text and need to provide summaries and evaluate them.

# Essay:
{essay[:1000]}

Step 1: In this step, provide three short summaries of the given essay. Each summary should be clear, concise, and capture the key points of the speech. Aim for around 2-3 sentences for each summary.
Step 2: Evaluate the three summaries from Step 1 and rate which one you believe is the best. Explain your choice by pointing out specific reasons such as clarity, completeness, and relevance to the speech content.
"""

evaluation_result = run_mistral(self_evaluation_prompt)
print(evaluation_result[:300])  # Print first part of the evaluation
```

### Approach 3: Employ Another LLM for Evaluation

For production systems, use separate LLMs for generation and evaluation.

```python
# Step 1: Generate summaries with first LLM
generation_prompt = f"""
Provide three short summaries of the given essay. Each summary should be clear, concise, and capture the key points of the essay.
Aim for around 2-3 sentences for each summary.

# essay:
{essay[:1000]}
"""
summaries = run_mistral(generation_prompt)

# Step 2: Evaluate summaries with second LLM (could be same or different model)
evaluation_prompt = f"""
You are given an essay and three summaries of the essay. Evaluate the three summaries and rate which one you believe is the best.
Explain your choice by pointing out specific reasons such as clarity, completeness, and relevance to the essay content.

# Essay:
{essay[:1000]}

# Summaries
{summaries}
"""
final_evaluation = run_mistral(evaluation_prompt)
print(final_evaluation[:200])  # Print beginning of evaluation
```

## Conclusion

You've learned four key prompting strategies with Mistral AI:

1. **Classification**: Using few-shot learning and delimiters for accurate categorization
2. **Summarization**: Employing step-by-step instructions for complex analysis
3. **Personalization**: Incorporating specific facts for tailored responses
4. **Evaluation**: Implementing confidence scores and multi-step evaluation

These techniques form the foundation for building robust AI applications with Mistral models. Experiment with combining these strategies for even more powerful workflows.
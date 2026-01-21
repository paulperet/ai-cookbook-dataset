# Guide: Using Chained Calls for Structured Outputs with OpenAI o1 Models

## Introduction

The OpenAI o1 reasoning models (released September 2024) offer advanced capabilities but currently lack native support for structured outputs. This means responses from o1 models aren't guaranteed to follow a specific JSON schema, making them less reliable for type-safe applications.

In this guide, you'll learn two methods to work with the `o1-preview` model to obtain structured JSON outputs:
1. **Direct prompting** - Using explicit instructions to request JSON
2. **Chained calls** - Combining o1 with `gpt-4o-mini` for reliable structured outputs

## Prerequisites

First, ensure you have the required packages installed:

```bash
pip install openai requests pydantic
```

Then import the necessary modules:

```python
import requests
from openai import OpenAI
from pydantic import BaseModel
```

Initialize the OpenAI client:

```python
client = OpenAI()
```

## Method 1: Direct Prompting for JSON Output

### Step 1: Fetch Data for Analysis

You'll start by creating a helper function to fetch HTML content and retrieve data about large US companies:

```python
def fetch_html(url):
    """Fetch HTML content from a given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

# Fetch data about large US companies
url = "https://en.wikipedia.org/wiki/List_of_largest_companies_in_the_United_States_by_revenue"
html_content = fetch_html(url)
```

### Step 2: Define Your Desired JSON Format

Create a clear JSON template that shows the model exactly what structure you expect:

```python
json_format = """
{
    companies: [
        {
            "company_name": "OpenAI",
            "page_link": "https://en.wikipedia.org/wiki/OpenAI",
            "reason": "OpenAI would benefit because they are an AI company..."
        }
    ]
}
"""
```

### Step 3: Make the API Call with Explicit Instructions

Now, craft a prompt that explicitly requests JSON output with your specified format:

```python
o1_response = client.chat.completions.create(
    model="o1-preview",
    messages=[
        {
            "role": "user", 
            "content": f"""
You are a business analyst designed to understand how AI technology could be used across large corporations.

- Read the following html and return which companies would benefit from using AI technology: {html_content}.
- Rank these prospects by opportunity by comparing them and show me the top 3. Return only as a JSON with the following format: {json_format}
"""
        }
    ]
)

print(o1_response.choices[0].message.content)
```

**How This Works:**
- The prompt explicitly instructs the model to return JSON
- You provide a clear template showing the expected structure
- The model typically returns well-formatted JSON based on these instructions

**Limitations of This Approach:**
- You must manually validate and parse the JSON response
- No built-in type safety or schema validation
- Model refusals aren't explicitly structured in the API response

## Method 2: Chained Calls for Reliable Structured Outputs

For production applications where type safety is critical, you can chain two API calls together. The first call uses `o1-preview` for reasoning, and the second uses `gpt-4o-mini` with structured outputs support.

### Step 1: Define Your Data Models

First, create Pydantic models to define your expected data structure:

```python
class CompanyData(BaseModel):
    """Data model for individual company information."""
    company_name: str
    page_link: str
    reason: str

class CompaniesData(BaseModel):
    """Container model for a list of companies."""
    companies: list[CompanyData]
```

### Step 2: Get Initial Analysis from o1-preview

Make the first API call to `o1-preview` to perform the analysis:

```python
o1_response = client.chat.completions.create(
    model="o1-preview",
    messages=[
        {
            "role": "user", 
            "content": f"""
You are a business analyst designed to understand how AI technology could be used across large corporations.

- Read the following html and return which companies would benefit from using AI technology: {html_content}.
- Rank these prospects by opportunity by comparing them and show me the top 3. Return each with {CompanyData.__fields__.keys()}
"""
        }
    ]
)

o1_response_content = o1_response.choices[0].message.content
```

### Step 3: Parse and Structure with gpt-4o-mini

Now, use `gpt-4o-mini` with the structured outputs feature to convert the o1 response into your defined schema:

```python
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user", 
            "content": f"""
Given the following data, format it with the given response format: {o1_response_content}
"""
        }
    ],
    response_format=CompaniesData,
)

# The parsed response is now a validated Pydantic model
structured_data = response.choices[0].message.parsed
print(structured_data)
```

**Benefits of This Approach:**
- **Type Safety:** The response is automatically validated against your Pydantic schema
- **Reliable Formatting:** Guaranteed JSON structure that matches your models
- **Refusal Handling:** Structured outputs API provides explicit refusal responses
- **Reusable Schemas:** Use the same data models across your application

## Cost Considerations

The chained approach requires two API calls, but the cost impact is minimal:
- The `o1-preview` call handles the complex reasoning
- The `gpt-4o-mini` call only processes the already-generated content
- The second call is significantly cheaper than the first

## Best Practices

1. **Schema Design:** Create comprehensive Pydantic models that match your application needs
2. **Error Handling:** Implement proper error handling for both API calls
3. **Caching:** Consider caching the o1-preview response if you need to experiment with different output formats
4. **Prompt Optimization:** Refine your initial prompt to get the most relevant analysis from o1-preview

## Conclusion

While `o1-preview` doesn't natively support structured outputs, you can achieve reliable type-safe responses by chaining it with `gpt-4o-mini`. This approach gives you the best of both worlds: advanced reasoning capabilities from o1 models combined with the reliability and type safety of structured outputs.

The chained method is particularly valuable for production applications where data consistency and validation are critical. As OpenAI continues to develop their models, this pattern provides a robust interim solution for working with o1 models in structured workflows.
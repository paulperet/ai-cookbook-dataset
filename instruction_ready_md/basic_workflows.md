# Multi-LLM Workflow Cookbook: Prompt-Chaining, Parallelization & Routing

This guide demonstrates three fundamental multi-LLM workflow patterns. Each pattern offers a different trade-off between cost, latency, and potential task performance improvement.

**Patterns Covered:**
1. **Prompt-Chaining:** Decomposes a complex task into sequential subtasks, where each step builds upon the previous result.
2. **Parallelization:** Distributes independent subtasks across multiple LLMs for concurrent processing, reducing overall latency.
3. **Routing:** Dynamically selects a specialized LLM path based on an analysis of the input characteristics.

> **Note:** These are sample implementations designed to illustrate core concepts and are not production-ready code.

## Prerequisites & Setup

First, ensure you have the necessary helper functions. The `llm_call` function simulates an LLM API call, and `extract_xml` is a utility for parsing XML tags from text.

```python
from concurrent.futures import ThreadPoolExecutor

# Assume these helper functions are defined elsewhere
from util import extract_xml, llm_call
```

## 1. The Chain Workflow

The `chain` function executes a series of LLM calls sequentially. The output from one step becomes the input for the next, allowing for progressive refinement of a task.

```python
def chain(input: str, prompts: list[str]) -> str:
    """Chain multiple LLM calls sequentially, passing results between steps."""
    result = input
    for i, prompt in enumerate(prompts, 1):
        print(f"\nStep {i}:")
        result = llm_call(f"{prompt}\nInput: {result}")
        print(result)
    return result
```

### Example: Structured Data Extraction & Formatting

This example transforms a raw performance report into a sorted, formatted Markdown table through four distinct chained steps.

```python
# Define the sequential processing steps
data_processing_steps = [
    """Extract only the numerical values and their associated metrics from the text.
    Format each as 'value: metric' on a new line.
    Example format:
    92: customer satisfaction
    45%: revenue growth""",
    """Convert all numerical values to percentages where possible.
    If not a percentage or points, convert to decimal (e.g., 92 points -> 92%).
    Keep one number per line.
    Example format:
    92%: customer satisfaction
    45%: revenue growth""",
    """Sort all lines in descending order by numerical value.
    Keep the format 'value: metric' on each line.
    Example:
    92%: customer satisfaction
    87%: employee satisfaction""",
    """Format the sorted data as a markdown table with columns:
    | Metric | Value |
    |:--|--:|
    | Customer Satisfaction | 92% |""",
]

# The raw input text
report = """
Q3 Performance Summary:
Our customer satisfaction score rose to 92 points this quarter.
Revenue grew by 45% compared to last year.
Market share is now at 23% in our primary market.
Customer churn decreased to 5% from 8%.
New user acquisition cost is $43 per user.
Product adoption rate increased to 78%.
Employee satisfaction is at 87 points.
Operating margin improved to 34%.
"""

print("\nInput text:")
print(report)

# Execute the chain
formatted_result = chain(report, data_processing_steps)
```

**Output Summary:**
The chain successfully processes the text. It first extracts key metrics, converts values to percentages, sorts them, and finally outputs a clean Markdown table.

```
| Metric | Value |
|:--|--:|
| Customer Satisfaction | 92% |
| Employee Satisfaction | 87% |
| Product Adoption Rate | 78% |
| Revenue Growth | 45% |
| User Acquisition Cost | 43.0 |
| Operating Margin | 34% |
| Market Share | 23% |
| Previous Customer Churn | 8% |
| Customer Churn | 5% |
```

## 2. The Parallel Workflow

The `parallel` function processes a list of independent inputs concurrently using the same prompt, leveraging a thread pool to speed up execution.

```python
def parallel(prompt: str, inputs: list[str], n_workers: int = 3) -> list[str]:
    """Process multiple inputs concurrently with the same prompt."""
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(llm_call, f"{prompt}\nInput: {x}") for x in inputs]
        return [f.result() for f in futures]
```

### Example: Stakeholder Impact Analysis

This example analyzes how market changes might impact four different stakeholder groups—Customers, Employees, Investors, and Suppliers—all at the same time.

```python
# Define the stakeholder groups to analyze
stakeholders = [
    """Customers:
    - Price sensitive
    - Want better tech
    - Environmental concerns""",
    """Employees:
    - Job security worries
    - Need new skills
    - Want clear direction""",
    """Investors:
    - Expect growth
    - Want cost control
    - Risk concerns""",
    """Suppliers:
    - Capacity constraints
    - Price pressures
    - Tech transitions""",
]

# Define the analysis prompt
analysis_prompt = """Analyze how market changes will impact this stakeholder group.
Provide specific impacts and recommended actions.
Format with clear sections and priorities."""

# Run the parallel analysis
impact_results = parallel(analysis_prompt, stakeholders)

# Print the results
for result in impact_results:
    print(result)
    print("+" * 80)
```

**Output Summary:**
Each stakeholder group receives a detailed, tailored analysis. For instance, the analysis for **Customers** highlights impacts on price sensitivity and technology demands, while the analysis for **Suppliers** focuses on capacity constraints and price pressures. Running these analyses in parallel significantly reduces the total wait time compared to running them sequentially.

## 3. The Route Workflow

The `route` function first uses an LLM to classify the input and select the most appropriate processing path from a predefined set of specialized options.

```python
def route(input: str, routes: dict[str, str]) -> str:
    """Route input to specialized prompt using content classification."""
    # First determine appropriate route using LLM with chain-of-thought
    print(f"\nAvailable routes: {list(routes.keys())}")
    selector_prompt = f"""
    Analyze the input and select the most appropriate support team from these options: {list(routes.keys())}
    First explain your reasoning, then provide your selection in this XML format:

    <reasoning>
    Brief explanation of why this ticket should be routed to a specific team.
    Consider key terms, user intent, and urgency level.
    </reasoning>

    <selection>
    The chosen team name
    </selection>

    Input: {input}""".strip()

    route_response = llm_call(selector_prompt)
    reasoning = extract_xml(route_response, "reasoning")
    route_key = extract_xml(route_response, "selection").strip().lower()

    print("Routing Analysis:")
    print(reasoning)
    print(f"\nSelected route: {route_key}")

    # Process input with selected specialized prompt
    selected_prompt = routes[route_key]
    return llm_call(f"{selected_prompt}\nInput: {input}")
```

### Example: Customer Support Ticket Routing

This example sets up specialized handlers for different types of support tickets (Billing, Technical, Account, Product) and routes incoming tickets accordingly.

```python
# Define the specialized support routes and their prompts
support_routes = {
    "billing": """You are a billing support specialist. Follow these guidelines:
    1. Always start with "Billing Support Response:"
    2. First acknowledge the specific billing issue
    3. Explain any charges or discrepancies clearly
    4. List concrete next steps with timeline
    5. End with payment options if relevant

    Keep responses professional but friendly.

    Input: """,
    "technical": """You are a technical support engineer. Follow these guidelines:
    1. Always start with "Technical Support Response:"
    2. List exact steps to resolve the issue
    3. Include system requirements if relevant
    4. Provide workarounds for common problems
    5. End with escalation path if needed

    Use clear, numbered steps and technical details.

    Input: """,
    "account": """You are an account security specialist. Follow these guidelines:
    1. Always start with "Account Support Response:"
    2. Prioritize account security and verification
    3. Provide clear steps for account recovery/changes
    4. Include security tips and warnings
    5. Set clear expectations for resolution time

    Maintain a serious, security-focused tone.

    Input: """,
    "product": """You are a product specialist. Follow these guidelines:
    1. Always start with "Product Support Response:"
    2. Focus on feature education and best practices
    3. Include specific examples of usage
    4. Link to relevant documentation sections
    5. Suggest related features that might help

    Be educational and encouraging in tone.

    Input: """,
}

# Test tickets
tickets = [
    """Subject: Can't access my account
    Message: Hi, I've been trying to log in for the past hour but keep getting an 'invalid password' error.
    I'm sure I'm using the right password. Can you help me regain access? This is urgent as I need to
    submit a report by end of day.
    - John""",
    """Subject: Unexpected charge on my card
    Message: Hello, I just noticed a charge of $49.99 on my credit card from your company, but I thought
    I was on the $29.99 plan. Can you explain this charge and adjust it if it's a mistake?
    Thanks,
    Sarah""",
    """Subject: How to export data?
    Message: I need to export all my project data to Excel. I've looked through the docs but can't
    figure out how to do a bulk export. Is this possible? If so, could you walk me through the steps?
    Best regards,
    Mike""",
]

# Process each ticket
print("Processing support tickets...\n")
for i, ticket in enumerate(tickets, 1):
    print(f"\nTicket {i}:")
    print("-" * 40)
    print(ticket)
    print("\nResponse:")
    print("-" * 40)
    response = route(ticket, support_routes)
    print(response)
    print("+" * 80)
```

**Output Summary:**
The router correctly classifies each ticket:
1.  **Login Issue:** Routed to **Account Support**. The LLM's reasoning notes the "account access and authentication problems."
2.  **Unexpected Charge:** Routed to **Billing Support** for handling payment discrepancies.
3.  **Data Export Question:** Routed to **Product Support** for feature education.

Each ticket then receives a response formatted with the appropriate tone and guidelines for its specialized team, demonstrating effective dynamic routing.

## Summary

You have now implemented three core multi-LLM workflow patterns:

*   **Chain:** Best for complex, multi-stage tasks that require sequential refinement.
*   **Parallel:** Ideal for processing batches of independent items where latency is a concern.
*   **Route:** Effective for directing inputs to specialized handlers based on content, improving response quality and efficiency.

You can combine these patterns to build even more sophisticated AI pipelines tailored to your specific application needs.
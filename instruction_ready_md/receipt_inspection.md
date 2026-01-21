# Eval-Driven System Design: From Prototype to Production

## Overview

### Purpose of This Guide
This guide provides a practical, end-to-end tutorial on using evaluations (evals) as the core process for building a production-grade autonomous system. It focuses on real-world challenges like limited labeled data and evolving problem understanding, moving beyond theoretical tutorials to deliver engineering rigor. By making evals central, you can replace guesswork with principled decisions about cost, performance, and investment.

### Target Audience
This tutorial is designed for ML/AI engineers and Solution Architects seeking actionable, modular code and strategies beyond introductory material. The content is fully executable and structured for easy adaptation into your own projects.

### Narrative: From Seed to System
We'll follow a realistic storyline: automating a manual receipt-analysis service for expense validation. Our journey will be:
1. **Start Small:** Begin with minimal labeled data (retail receipts).
2. **Build Incrementally:** Develop a minimal viable system and establish initial evals.
3. **Align with Business:** Map eval performance to business KPIs and dollar impact.
4. **Iterate with Evals:** Use eval scores to drive model improvements, then expand evals with better data to identify further enhancements.

### How to Use This Guide
- **For Concepts:** Read the text and skim the code.
- **For Code Reuse:** Jump to specific sections, copy the code, and adapt it.
- **For Deep Understanding:** Download and run the notebook, editing code to test your hypotheses.

> **Note:** If your OpenAI organization has a Zero Data Retention (ZDR) policy, Evals will still be available but will retain data to maintain application state.

## Use Case: Receipt Parsing
We'll use a small, hypothetical receipt-parsing problem that's complex enough to require detailed, multi-faceted evals. This mirrors scenarios with limited initial data.

### Problem Definition
Assume an existing workflow where:
1. Users upload receipt photos.
2. An accounting team reviews each receipt to categorize and approve/audit expenses.

The team's decisions are based on:
- Merchant
- Geographic location
- Expense amount
- Items/services purchased
- Handwritten notes/annotations

Our system must:
- Handle most receipts autonomously.
- Escalate low-confidence decisions for human QA.
- Reduce the total cost of the accounting process, which depends on:
  - Previous/current system cost per receipt.
  - Number of receipts sent to QA.
  - New system's run cost per receipt plus fixed costs.
  - Business impact of mistakes (escalated or missed).
  - Engineering development and integration costs.

### Dataset Overview
We'll use receipt images from the CC BY 4.0 licensed [Receipt Handwriting Detection dataset](https://universe.roboflow.com/newreceipts/receipt-handwriting-detection) by Roboflow, augmented with custom labels for narrative purposes.

## Project Lifecycle
While projects vary, they share common components. The lifecycle involves primary steps (solid arrows) and ongoing problem understanding (dotted line), which influences every stage. We'll explore several iterative refinement cycles.

1. **Understand the Problem:** Interview domain experts (e.g., accounting team) to grasp task details. This understanding evolves as we build.
2. **Assemble Examples (Gather Data):** Start with system inputs (receipt images) but no fully annotated data. Incrementally expand test/training sets with domain experts.
3. **Build an End-to-End V0 System:** Quickly create a skeleton system that accepts correct inputs and outputs the right type, often via a simple prompt and single model.
4. **Label Data and Build Initial Evals:** Use the V0 system to generate 'draft' truth data for expert annotation. Use corrections and expert insights to design evals and embed expertise.
5. **Map Evals to Business Metrics:** Assess costs/benefits, link eval measurements to key objectives, and create a dollar-value model balancing performance with development/run costs.
6. **Progressively Improve System and Evals:** Iterate on improvements guided by evals, ensuring objective progress and regression avoidance.
7. **Integrate QA Process and Ongoing Improvements:** Instrument production services to surface more test/training samples over time, ensuring long-term performance.

## V0 System Construction
We'll distill the system into two functions for this tutorial:
- `extract_receipt_details`: Takes an image and outputs structured receipt details.
- `evaluate_receipt_for_audit`: Takes the structured details and decides if the receipt should be audited.

> **Note:** Breaking the process into steps simplifies examination and development but can lead to information loss (like "telephone"). We separate steps here for instructional clarity in evals.

We'll start with data extraction—an intermediate step where information is examined but often unrecorded, so labeled data is scarce.

### Step 1: Setup and Dependencies
First, install required packages and set up your environment.

```bash
pip install --upgrade openai pydantic python-dotenv rich persist-cache -qqq
```

Create a `.env` file in your project root with your OpenAI API key:

```plaintext
OPENAI_API_KEY=sk-...
```

Then, load the environment variables in your script.

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Loads from .env file
```

### Step 2: Define Structured Output Models
We'll use Pydantic to define structured models for receipt details, location, and line items.

```python
from pydantic import BaseModel


class Location(BaseModel):
    city: str | None
    state: str | None
    zipcode: str | None


class LineItem(BaseModel):
    description: str | None
    product_code: str | None
    category: str | None
    item_price: str | None
    sale_price: str | None
    quantity: str | None
    total: str | None


class ReceiptDetails(BaseModel):
    merchant: str | None
    location: Location
    time: str | None
    items: list[LineItem]
    subtotal: str | None
    tax: str | None
    total: str | None
    handwritten_notes: list[str]
```

> **Note:** We use strings for numbers and time fields for simplicity in serialization. In production, you'd add validation to convert to `decimal.Decimal` and `datetime.datetime`.

### Step 3: Build the Basic Info Extraction Function
Now, create the `extract_receipt_details` function. We'll start with a naive prompt generated by an LLM based on our problem description—no need for extensive prompt engineering before establishing a benchmark.

Define the prompt:

```python
BASIC_PROMPT = """
Given an image of a retail receipt, extract all relevant information and format it as a structured response.

# Task Description

Carefully examine the receipt image and identify the following key information:

1. Merchant name and any relevant store identification
2. Location information (city, state, ZIP code)
3. Date and time of purchase
4. All purchased items with their:
   * Item description/name
   * Item code/SKU (if present)
   * Category (infer from context if not explicit)
   * Regular price per item (if available)
   * Sale price per item (if discounted)
   * Quantity purchased
   * Total price for the line item
5. Financial summary:
   * Subtotal before tax
   * Tax amount
   * Final total
6. Any handwritten notes or annotations on the receipt (list each separately)

## Important Guidelines

* If information is unclear or missing, return null for that field
* Format dates as ISO format (YYYY-MM-DDTHH:MM:SS)
* Format all monetary values as decimal numbers
* Distinguish between printed text and handwritten notes
* Be precise with amounts and totals
* For ambiguous items, use your best judgment based on context

Your response should be structured and complete, capturing all available information
from the receipt.
"""
```

Implement the extraction function using the OpenAI API:

```python
import base64
import mimetypes
from pathlib import Path
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def extract_receipt_details(
    image_path: str, model: str = "o4-mini"
) -> ReceiptDetails:
    """Extract structured details from a receipt image."""
    # Determine image MIME type
    mime_type, _ = mimetypes.guess_type(image_path)

    # Read and base64 encode the image
    b64_image = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    image_data_url = f"data:{mime_type};base64,{b64_image}"

    # Call the OpenAI API with structured output
    response = await client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": BASIC_PROMPT},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        text_format=ReceiptDetails,
    )

    return response.output_parsed
```

### Step 4: Test on a Single Receipt
Let's evaluate the function on one receipt to see how a smart model with a naive prompt performs. We'll manually review the output.

First, set up paths and run the extraction:

```python
from pathlib import Path

example_receipt = Path(
    "data/train/Supplies_20240322_220858_Raven_Scan_3_jpeg.rf.50852940734939c8838819d7795e1756.jpg"
)
result = await extract_receipt_details(example_receipt)
```

Outputs will vary per run, but typically, the model gets most things correct with some errors. For instance, a sample output might look like:

```python
from rich import print

# Example structured output (simulated)
walmart_receipt = ReceiptDetails(
    merchant="Walmart",
    location=Location(city="Vista", state="CA", zipcode="92083"),
    time="2023-06-30T16:40:45",
    items=[
        LineItem(
            description="SPRAY 90",
            product_code="001920056201",
            category=None,
            item_price=None,
            sale_price=None,
            quantity="2",
            total="28.28",
        ),
        LineItem(
            description="LINT ROLLER 70",
            product_code="007098200355",
            category=None,
            item_price=None,
            sale_price=None,
            quantity="1",
            total="6.67",
        ),
        LineItem(
            description="SCRUBBER",
            product_code="003444193232",
            category=None,
            item_price=None,
            sale_price=None,
            quantity="2",
            total="12.70",
        ),
        LineItem(
            description="FLOUR SACK 10",
            product_code="003444194263",
            category=None,
            item_price=None,
            sale_price=None,
            quantity="1",
            total="0.77",
        ),
    ],
    subtotal="50.77",
    tax="4.19",
    total="54.96",
    handwritten_notes=[],
)

print(walmart_receipt)
```

In this example, the model extracted many details correctly but renamed some line items incorrectly, got some prices wrong, and didn't categorize items. That's expected at this stage—our goal is to build a basic system we can evaluate, not perfection.

### Step 5: Add Action Decision
Next, we need to close the loop by deciding whether a receipt should be audited. We'll create an `evaluate_receipt_for_audit` function. Typically, you'd start with the most capable model (e.g., `o3`) for correctness, then experiment with trade-offs. For this tutorial, we'll use `o4-mini` to illustrate handling reasoning errors.

First, define the audit criteria and output model:

```python
from pydantic import BaseModel, Field

class AuditDecision(BaseModel):
    """Structured output for audit evaluation."""
    NOT_TRAVEL_RELATED: bool = Field(
        description="True if the receipt is NOT for travel-related expenses (e.g., office supplies). False if it IS travel-related (e.g., gas, hotel)."
    )
    AMOUNT_OVER_LIMIT: bool = Field(
        description="True if the total amount exceeds $100. False otherwise."
    )
    SUSPICIOUS_MERCHANT: bool = Field(
        description="True if the merchant is on a watchlist or appears suspicious. False otherwise."
    )
    HANDWRITTEN_NOTES_PRESENT: bool = Field(
        description="True if there are handwritten notes on the receipt. False otherwise."
    )
    RECOMMEND_AUDIT: bool = Field(
        description="True if the receipt should be audited based on the criteria above. False otherwise."
    )
```

Define the audit prompt:

```python
audit_prompt = """
Evaluate this receipt data to determine if it need to be audited based on the following criteria:

1. NOT_TRAVEL_RELATED:
   - IMPORTANT: For this criterion, travel-related expenses include but are not limited to: gas, hotel, airfare, or car rental.
   - If the receipt IS for a travel-related expense, set this to FALSE.
   - If the receipt is NOT for a travel-related expense (like office supplies), set this to TRUE.
   - In other words, if the receipt shows FUEL/GAS, this would be FALSE because gas IS travel-related.

2. AMOUNT_OVER_LIMIT: The total amount exceeds $100.

3. SUSPICIOUS_MERCHANT: The merchant is on a watchlist or appears suspicious (use your judgment based on merchant name).

4. HANDWRITTEN_NOTES_PRESENT: There are handwritten notes or annotations on the receipt.

Based on these criteria, output a structured decision including whether to recommend an audit.
"""
```

Implement the evaluation function:

```python
async def evaluate_receipt_for_audit(
    receipt_details: ReceiptDetails, model: str = "o4-mini"
) -> AuditDecision:
    """Evaluate receipt details and decide if audit is needed."""
    # Convert receipt details to a string representation for the prompt
    details_str = receipt_details.json(indent=2)

    response = await client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": audit_prompt},
                    {"type": "input_text", "text": f"Receipt Details:\n{details_str}"},
                ],
            }
        ],
        text_format=AuditDecision,
    )

    return response.output_parsed
```

### Step 6: Test the End-to-End V0 System
Now, combine both functions to test the full V0 system on our example receipt.

```python
# Extract details
extracted_details = await extract_receipt_details(example_receipt)

# Evaluate for audit
audit_decision = await evaluate_receipt_for_audit(extracted_details)

print("Extracted Details:")
print(extracted_details)
print("\nAudit Decision:")
print(audit_decision)
```

The output will show the structured details and the audit decision (e.g., `RECOMMEND_AUDIT: True` if criteria are met). This completes our V0 system—a basic, functional pipeline from image to decision.

## Next Steps
With the V0 system built, you're ready to:
1. **Label Data and Build Initial Evals:** Use the system to generate draft outputs for expert annotation, then design evals based on corrections.
2. **Map Evals to Business Metrics:** Link eval scores to costs/benefits for informed prioritization.
3. **Iterate:** Use evals to guide improvements, such as prompt engineering, model selection, or architecture changes.

This foundation ensures your development is driven by objective metrics, not intuition, setting the stage for a robust production system.
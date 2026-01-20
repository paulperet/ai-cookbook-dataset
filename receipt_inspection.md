# Eval-Driven System Design: From Prototype to Production

## Overview

### Purpose of This Cookbook

This cookbook provides a **practical**, end-to-end guide on how to effectively use 
evals as the core process in creating a production-grade autonomous system to 
replace a labor-intensive human workflow. It's a direct product of collaborative 
experience dealing with projects where users may not have started with pristine 
labeled data or a perfect understanding of the problem - two issues that most tutorials gloss 
over but are in practice almost always serious challenges.

Making evals the core process prevents poke-and-hope guesswork and impressionistic
judgments of accuracy, instead demanding engineering rigor. This means we can make
principled decisions about cost trade-offs and investment. 

### Target Audience

This guide is designed for ML/AI engineers and Solution Architects who are
looking for practical guidance beyond introductory tutorials. This notebook is fully
executable and organized to be as modular as possible to support using code
samples directly in your own applications.

### Guiding Narrative: From Tiny Seed to Production System

We'll follow a realistic storyline: replacing a manual receipt-analysis service for validating expenses.

* **Start Small:** Begin with a very small set of labeled data (retail receipts). Many businesses don't have good ground truth data sets. 
* **Build Incrementally:** Develop a minimal viable system and establish initial evals. 
* **Business Alignment:** Evaluate eval performance in the context of business KPIs and
  dollar impact, and target efforts to avoid working on low-impact improvements.
* **Eval-Driven Iteration:** Iteratively improve by using eval scores to power model
  improvements, then by using better models on more data to expand evals and identify more
  areas for improvement.

### How to Use This Cookbook

This cookbook is structured as an eval-centric guide through the lifecycle of building
an LLM application.

1. If you're primarily interested in the ideas presented, read through the text and skim over
   the code.
2. If you're here because of something else you're working on, you can go ahead and jump to that
   section and dig into the code there, copy it, and adapt it to your needs.
3. If you want to really understand how this all works, download this notebook and run
   the cells as you read through it; edit the code to make your own changes, test your
   hypotheses, and make sure you actually understand how it all works together.

> Note: If your OpenAI organization has a Zero Data Retention (ZDR) policy, Evals will still be available, but will retain data to maintain application state.

## Use Case: Receipt Parsing

In order to condense this guide we'll be using a small hypothetical problem that's still complex
enough to merit detailed and multi-faceted evals. In particular, we'll be focused on how
to solve a problem given a limited amount of data to work with, so we're working with a
dataset that's quite small.

### Problem Definition

For this guide, we assume that we are starting with a workflow for reviewing and filing 
receipts. While in general, this is a problem that already has a lot of established 
solutions, it's analogous to other problems that don't have nearly so much prior work; 
further, even when good enterprise solutions exist there is often still a 
"last mile" problem that still requires human time.

In our case, we'll assume we have a pipeline where:

* People upload photos of receipts
* An accounting team reviews each receipt to categorize and approve or audit the expense

Based on interviews with the accounting team, they make their decisions based on

1. Merchant
2. Geographic location
3. Expense amount
4. Items or services purchased
5. Handwritten notes or annotations

Our system will be expected to handle most receipts without any human intervention, but
escalate low-confidence decisions for human QA. We'll be focused on reducing the total
cost of the accounting process, which is dependent on

1. How much the previous / current system cost to run per-receipt
2. How many receipts the new system sends to QA
3. How much the system costs to run per-receipt, plus any fixed costs
4. What the business impact is of mistakes, either receipts kicked out for review or mistakes missed
5. The cost of engineering to develop and integrate the system

### Dataset Overview

The receipt images come from the CC by 4.0 licensed
[Receipt Handwriting Detection Computer Vision Project](https://universe.roboflow.com/newreceipts/receipt-handwriting-detection)
dataset published by Roboflow. We've added our own labels and narrative spin in order to
tell a story with a small number of examples.

## Project Lifecycle

Not every project will proceed in the same way, but projects generally have some 
important components in common.

The solid arrows show the primary progressions or steps, while the dotted line 
represents the ongoing nature of problem understanding - uncovering more about
the customer domain will influence every step of the process. We wil examine 
several of these iterative cycles of refinement in detail below. 
Not every project will proceed in the same way, but projects generally have some common
important components.

### 1. Understand the Problem

Usually, the decision to start an engineering process is made by leadership who
understand the business impact but don't need to know the process details. In our
example, we're building a system designed to replace a non-AI workflow. In a sense this
is ideal: we have a set of domain experts, *the people currently doing the task* who we
can interview to understand the task details and who we can lean upon to help develop
appropriate evals.

This step doesn't end before we start building our system; invariably, our initial
assessments are an incomplete understanding of the problem space and we will continue to
refine our understanding as we get closer to a solution.

### 2. Assemble Examples (Gather Data)

It's very rare for a real-world project to begin with all the data necessary to achieve a satisfactory solution, let alone establish confidence.

In our case, we'll assume we have a decent sample of system *inputs*, in the form of but receipt images, but start without any fully annotated data. We find this is a not-unusual situation when automating an existing process. We'll walk through the process of incrementally expanding our test and training sets in collaboration with domain experts as we go along and make our evals progressively more comprehensive.

### 3. Build an End-to-End V0 System

We want to get the skeleton of a system built as quickly as possible. We don't need a
system that performs well - we just need something that accepts the right inputs and
provides outputs of the correct type. Usually this is almost as simple as describing the
task in a prompt, adding the inputs, and using a single model (usually with structured
outputs) to make an initial best-effort attempt.

### 4. Label Data and Build Initial Evals

We've found that in the absence of an established ground truth, it's not uncommon to 
use an early version of a system to generate 'draft' truth data which can be annotated 
or corrected by domain experts.

Once we have an end-to-end system constructed, we can start processing the inputs we
have to generate plausible outputs. We'll send these to our domain experts to grade 
and correct. We will use these corrections and conversations about how the experts 
are making their decisions to design further evals and to embed expertise in the system.

### 5. Map Evals to Business Metrics

Before we jump into correcting every error, we need to make sure that we're investing
time effectively. The most critical task at this stage is to review our evals and
gain an understanding of how they connect to our key objectives.

- Step back and assess the potential costs and benefits of the system
- Identify which eval measurements speak directly to those costs and benefits
- For example, what does "failure" on a particular eval cost? Are we measuring
  something worthwhile?
- Create a (non-LLM) model that uses eval metrics to provide a dollar value
- Balance performance (accuracy, or speed) with cost to develop and run

### 6. Progressively Improve System and Evals

Having identified which efforts are most worth making, we can begin iterating on 
improvements to the system. The evals act as an objective guide so we know when we've
made the system good enough, and ensure we avoid or identify regression. 

### 7. Integrate QA Process and Ongoing Improvements

Evals aren't just for development. Instrumenting all or a portion of a production
service will surface more useful test and training samples over time, identifying
incorrect assumptions or finding areas with insufficient coverage. This is also the only
way you can ensure that your models continue performing well long after your initial
development process is complete.

## V0 System Construction

In practice, we would probably be building a system that operates via a REST API,
possibly with some web frontend that would have access to some set of components and
resources. For the purposes of this cookbook, we'll distill that down to a pair of
functions, `extract_receipt_details` and `evaluate_receipt_for_audit` that collectively
decide what we should do with a given receipt.

- `extract_receipt_details` will take an image as input and produce structured output
  containing important details about the receipt.
- `evaluate_receipt_for_audit` will take that structure as input and decide whether or
  not the receipt should be audited.

> Breaking up a process into steps like this has both pros and cons; it is easier to
> examine and develop if the process is made up of small isolated steps. But you can
> progressively lose information, effectively letting your agents play "telephone". In
> this notebook we break up the steps and don't let the auditor see the actual receipt
> because it's more instructive for the evals we want to discuss.

We'll start with the first step, the literal data extraction. This is *intermediate*
data: it's information that people would examine implicitly, but often isn't recorded.
And for this reason, we often don't have labeled data to work from.


```python
%pip install --upgrade openai pydantic python-dotenv rich persist-cache -qqq
%load_ext dotenv
%dotenv

# Place your API key in a file called .env
# OPENAI_API_KEY=sk-...
```

### Structured Output Model

Capture the meaningful information in a structured output.


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

> *Note*: Normally we would use `decimal.Decimal` objects for the numbers above and `datetime.datetime` objects for `time` field, but neither of those deserialize well. For the purposes of this cookbook, we'll work with strings, but in practice you'd want to have another level of translation to get the correct output validated.

### Basic Info Extraction

Let's build our `extract_receipt_details` function.

Usually, for the very first stab at something that might work, we'll simply feed ChatGPT
the available documents we've assembled so far and ask it to generate a prompt. It's not
worth spending too much time on prompt engineering before you have a benchmark to grade
yourself against! This is a prompt produced by o4-mini based on the problem description
above.


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
```
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
    # Determine image type for data URI.
    mime_type, _ = mimetypes.guess_type(image_path)

    # Read and base64 encode the image.
    b64_image = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    image_data_url = f"data:{mime_type};base64,{b64_image}"

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

### Test on one receipt

Let's evaluate just a single receipt and review it manually to see how well a smart model with a naive prompt can do.


```python
from rich import print

receipt_image_dir = Path("data/test")
ground_truth_dir = Path("data/ground_truth")

example_receipt = Path(
    "data/train/Supplies_20240322_220858_Raven_Scan_3_jpeg.rf.50852940734939c8838819d7795e1756.jpg"
)
result = await extract_receipt_details(example_receipt)
```

We'll get different answers if we re-run it, but it usually gets most things correct
with a few errors. Here's a specific example:


```python
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
```

The model extracted a lot of things correctly, but renamed some of the line
items - incorrectly, in fact. More importantly, it got some of the prices wrong, and it
decided not to categorize any of the line items.

That's okay, we don't expect to have perfect answers at this point! Instead, our
objective is to build a basic system we can evaluate. Then, when we start iterating, we
won't be 'vibing' our way to something that *looks* better -- we'll be engineering a
reliable solution. But first, we'll add an action decision to complete our draft system.

### Action Decision

Next, we need to close the loop and get to an actual decision based on receipts. This
looks pretty similar, so we'll present the code without comment.

Ordinarily one would start with the most capable model - `o3`, at this time - for a 
first pass, and then once correctness is established experiment with different models
to analyze any tradeoffs for their business impact, and potentially consider whether 
they are remediable with iteration. A client may be willing to take a certain accuracy 
hit for lower latency or cost, or it may be more effective to change the architecture
to hit cost, latency, and accuracy goals. We'll get into how to make these tradeoffs
explicitly and objectively later on. 

For this cookbook, `o3` might be too good. We'll use `o4-mini` for our first pass, so 
that we get a few reasoning errors we can use to illustrate the means of addressing
them when they occur.

Next, we need to close the loop and get to an actual decision based on receipts. This
looks pretty similar, so we'll present the code without comment.


```python
from pydantic import BaseModel, Field

audit_prompt = """
Evaluate this receipt data to determine if it need to be audited based on the following
criteria:

1. NOT_TRAVEL_RELATED:
   - IMPORTANT: For this criterion, travel-related expenses include but are not limited
   to: gas, hotel, airfare, or car rental.
   - If the receipt IS for a travel-related expense, set this to FALSE.
   - If the receipt is NOT for a travel-related expense (like office supplies), set this
   to TRUE.
   - In other words, if the receipt shows FUEL/GAS, this would be FALSE because gas IS
   travel-related.

2. AMOUNT_OVER_LIMIT: The total amount exceeds $
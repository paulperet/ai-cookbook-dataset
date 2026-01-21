# Guide: Using GPT-4o Vision with Function Calling

This guide demonstrates how to leverage the multimodal capabilities of GPT-4o (`gpt-4o-2024-11-20`) by combining vision with function calling. You will learn to build two practical applications:
1. A customer service assistant that analyzes package images to automate delivery exception handling.
2. An organizational chart analyzer that extracts structured employee data from an image.

## Prerequisites & Setup

First, install the required libraries.

```bash
pip install pymupdf openai matplotlib instructor pandas
```

Now, import the necessary modules.

```python
import base64
import os
from enum import Enum
from io import BytesIO
from typing import Iterable, List, Literal, Optional

import fitz
import instructor
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel, Field
```

Set the model name as a constant for reuse.

```python
MODEL = "gpt-4o-2024-11-20"
```

---

## Part 1: Customer Service Assistant for Delivery Exceptions

In this section, you will build an assistant that analyzes images of packages to decide on an action: refund, replace, or escalate to a human agent.

### Step 1: Prepare and Encode Sample Images

Create a helper function to encode images as base64 strings, which the OpenAI API can process.

```python
def encode_image(image_path: str):
    """Encode an image file to a base64 string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
```

Assume your sample images are stored in a directory named `images`. Encode all images and store them in a dictionary.

```python
image_dir = "images"
image_files = os.listdir(image_dir)
image_data = {}

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    # Use the filename (without extension) as the key
    key = image_file.split('.')[0]
    image_data[key] = encode_image(image_path)
    print(f"Encoded image: {image_file}")
```

### Step 2: Define the Order Data Model and Placeholder Functions

Define a Pydantic model to represent an order and create placeholder functions for order processing actions.

```python
class Order(BaseModel):
    """Represents an order with its details."""
    order_id: str = Field(..., description="The unique identifier of the order")
    product_name: str = Field(..., description="The name of the product")
    price: float = Field(..., description="The price of the product")
    status: str = Field(..., description="The status of the order")
    delivery_date: str = Field(..., description="The delivery date of the order")

# Placeholder functions
def get_order_details(order_id):
    """Simulate fetching order details from a database."""
    return Order(
        order_id=order_id,
        product_name="Product X",
        price=100.0,
        status="Delivered",
        delivery_date="2024-04-10",
    )

def escalate_to_agent(order: Order, message: str):
    return f"Order {order.order_id} has been escalated to an agent with message: `{message}`"

def refund_order(order: Order):
    return f"Order {order.order_id} has been refunded successfully."

def replace_order(order: Order):
    return f"Order {order.order_id} has been replaced with a new order."
```

### Step 3: Define the Function Call Structure

Create a base model for function calls and specific models for each action. The `__call__` method executes the corresponding placeholder function.

```python
class FunctionCallBase(BaseModel):
    rationale: Optional[str] = Field(..., description="The reason for the action.")
    image_description: Optional[str] = Field(
        ..., description="The detailed description of the package image."
    )
    action: Literal["escalate_to_agent", "replace_order", "refund_order"]
    message: Optional[str] = Field(
        ...,
        description="The message to be escalated to the agent if action is escalate_to_agent",
    )

    def __call__(self, order_id):
        """Execute the action based on the order ID."""
        order: Order = get_order_details(order_id=order_id)
        if self.action == "escalate_to_agent":
            return escalate_to_agent(order, self.message)
        if self.action == "replace_order":
            return replace_order(order)
        if self.action == "refund_order":
            return refund_order(order)

class EscalateToAgent(FunctionCallBase):
    """Escalate to an agent for further assistance."""
    pass

class OrderActionBase(FunctionCallBase):
    pass

class ReplaceOrder(OrderActionBase):
    """Tool call to replace an order."""
    pass

class RefundOrder(OrderActionBase):
    """Tool call to refund an order."""
    pass
```

### Step 4: Implement the Assistant Handler

Define the system prompt and a function that sends the image and prompt to GPT-4o, instructing it to return the appropriate function call.

```python
ORDER_ID = "12345"  # Placeholder order ID for testing
INSTRUCTION_PROMPT = """You are a customer service assistant for a delivery service, equipped to analyze images of packages.
If a package appears damaged in the image, automatically process a refund according to policy.
If the package looks wet, initiate a replacement.
If the package appears normal and not damaged, escalate to agent.
For any other issues or unclear images, escalate to agent.
You must always use tools!"""

def delivery_exception_support_handler(test_image: str):
    """Send the image to GPT-4o and execute the returned function call."""
    payload = {
        "model": MODEL,
        "response_model": Iterable[RefundOrder | ReplaceOrder | EscalateToAgent],
        "tool_choice": "auto",  # Model chooses the tool
        "temperature": 0.0,
        "seed": 123,  # For reproducibility
        "messages": [
            {"role": "user", "content": INSTRUCTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data[test_image]}"
                        }
                    },
                ],
            }
        ]
    }

    # Use Instructor for structured, parallel tool extraction
    function_calls = instructor.from_openai(
        OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS
    ).chat.completions.create(**payload)

    for tool in function_calls:
        print(f"- Tool call: {tool.action} for provided img: {test_image}")
        print(f"- Parameters: {tool}")
        print(f">> Action result: {tool(ORDER_ID)}")
        return tool
```

### Step 5: Test the Assistant

Run the handler on your sample images to see the automated decisions in action.

```python
print("Processing delivery exception support for different package images...")

print("\n===================== Simulating user message 1 =====================")
assert delivery_exception_support_handler("damaged_package").action == "refund_order"

print("\n===================== Simulating user message 2 =====================")
assert delivery_exception_support_handler("normal_package").action == "escalate_to_agent"

print("\n===================== Simulating user message 3 =====================")
assert delivery_exception_support_handler("wet_package").action == "replace_order"
```

**Expected Output:**
The assistant will analyze each image, select the correct tool (`refund_order`, `escalate_to_agent`, or `replace_order`), and print the action result.

---

## Part 2: Analyzing an Organizational Chart

Now, you will extract structured employee information from an organizational chart image using GPT-4o's vision and structured output capabilities.

### Step 1: Convert a PDF Page to an Image

If your organizational chart is in PDF format, convert a specific page to a JPEG image for analysis.

```python
def convert_pdf_page_to_jpg(pdf_path: str, output_path: str, page_number=0):
    """Convert a single PDF page to a JPEG image."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap()
    pix.save(output_path)

pdf_path = 'data/org-chart-sample.pdf'
output_path = 'org-chart-sample.jpg'
convert_pdf_page_to_jpg(pdf_path, output_path)

# Display the image (optional)
img = Image.open(output_path)
display(img)
```

### Step 2: Define the Employee Data Models

Create Pydantic models to structure the extracted data. An `Enum` defines possible roles.

```python
class RoleEnum(str, Enum):
    """Defines possible roles within an organization."""
    CEO = "CEO"
    CTO = "CTO"
    CFO = "CFO"
    COO = "COO"
    EMPLOYEE = "Employee"
    MANAGER = "Manager"
    INTERN = "Intern"
    OTHER = "Other"

class Employee(BaseModel):
    """Represents an employee, including their name, role, and optional manager information."""
    employee_name: str = Field(..., description="The name of the employee")
    role: RoleEnum = Field(..., description="The role of the employee")
    manager_name: Optional[str] = Field(None, description="The manager's name, if applicable")
    manager_role: Optional[RoleEnum] = Field(None, description="The manager's role, if applicable")

class EmployeeList(BaseModel):
    """A list of employees within the organizational structure."""
    employees: List[Employee] = Field(..., description="A list of employees")
```

### Step 3: Create the Chart Parser Function

This function sends the encoded image to GPT-4o with instructions to extract and structure the employee data.

```python
def parse_orgchart(base64_img: str) -> EmployeeList:
    """Analyze an organizational chart image and return a structured list of employees."""
    response = instructor.from_openai(OpenAI()).chat.completions.create(
        model=MODEL,
        response_model=EmployeeList,
        messages=[
            {
                "role": "user",
                "content": 'Analyze the given organizational chart and very carefully extract the information.',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    },
                ],
            }
        ],
    )
    return response
```

### Step 4: Execute the Parser and Display Results

Encode the generated image and run the parser. The results are displayed in a pandas DataFrame for easy review.

```python
# Encode the image
base64_img = encode_image(output_path)

# Call the parser
result = parse_orgchart(base64_img)

# Convert to DataFrame for a clean display
df = pd.DataFrame([{
    'employee_name': employee.employee_name,
    'role': employee.role.value,
    'manager_name': employee.manager_name,
    'manager_role': employee.manager_role.value if employee.manager_role else None
} for employee in result.employees])

display(df)
```

**Expected Output:**
A table listing each extracted employee with their name, role, and manager details (if applicable). The accuracy depends on the clarity and complexity of the input chart.

---

## Summary

You have successfully built two applications using GPT-4o Vision with function calling:

1.  **Delivery Exception Assistant:** An automated system that analyzes package images and triggers business logic (refund, replace, escalate) based on visual cues.
2.  **Organizational Chart Parser:** A tool that extracts hierarchical employee data from an image into a structured format ready for analysis.

This pattern—combining vision for understanding with function calling for structured action—unlocks powerful multimodal workflows that go beyond simple image description. You can adapt these examples to automate document processing, quality control, inventory management, and more.
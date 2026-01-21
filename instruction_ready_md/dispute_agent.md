# Building a Dispute Management System with the OpenAI Agents SDK

This guide demonstrates how to build an automated dispute management workflow using the OpenAI Agents SDK and the Stripe API. You will create a system that intelligently triages and processes customer disputes, handling both clear-cut cases and complex investigations.

## Prerequisites

Before you begin, ensure you have the following:

1.  **OpenAI Account:** Required for API access. [Sign up here](https://openai.com) and create an API key on the [API Keys page](https://platform.openai.com/api-keys).
2.  **Stripe Account:** Required to simulate payment disputes. [Create a free account](https://dashboard.stripe.com/register). Use your **Test Secret Key** from **Developers > API keys** in the dashboard.
3.  **Environment File:** Create a `.env` file in your project root with your API keys:
    ```bash
    OPENAI_API_KEY=your_openai_api_key_here
    STRIPE_SECRET_KEY=your_stripe_test_secret_key_here
    ```

## Step 1: Environment Setup

Install the required Python packages and configure your environment.

```bash
pip install python-dotenv openai-agents stripe typing_extensions
```

```python
import os
import logging
import json
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
import stripe
from typing_extensions import TypedDict, Any

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Stripe API key from environment variables
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
```

## Step 2: Define Helper Function Tools

You will create several utility functions that your agents will use as tools to fetch data and interact with Stripe.

```python
@function_tool
def get_phone_logs(phone_number: str) -> list:
    """
    Return a list of phone call records for the given phone number.
    Each record might include call timestamps, durations, notes,
    and an associated order_id if applicable.
    """
    phone_logs = [
        {
            "phone_number": "+15551234567",
            "timestamp": "2023-03-14 15:24:00",
            "duration_minutes": 5,
            "notes": "Asked about status of order #1121",
            "order_id": 1121
        },
        {
            "phone_number": "+15551234567",
            "timestamp": "2023-02-28 10:10:00",
            "duration_minutes": 7,
            "notes": "Requested refund for order #1121, I told him we were unable to refund the order because it was final sale",
            "order_id": 1121
        },
        {
            "phone_number": "+15559876543",
            "timestamp": "2023-01-05 09:00:00",
            "duration_minutes": 2,
            "notes": "General inquiry; no specific order mentioned",
            "order_id": None
        },
    ]
    return [
        log for log in phone_logs if log["phone_number"] == phone_number
    ]


@function_tool
def get_order(order_id: int) -> str:
    """
    Retrieve an order by ID from a predefined list of orders.
    Returns the corresponding order object or 'No order found'.
    """
    orders = [
        {
            "order_id": 1234,
            "fulfillment_details": "not_shipped"
        },
        {
            "order_id": 9101,
            "fulfillment_details": "shipped",
            "tracking_info": {
                "carrier": "FedEx",
                "tracking_number": "123456789012"
            },
            "delivery_status": "out for delivery"
        },
        {
            "order_id": 1121,
            "fulfillment_details": "delivered",
            "customer_id": "cus_PZ1234567890",
            "customer_phone": "+15551234567",
            "order_date": "2023-01-01",
            "customer_email": "customer1@example.com",
            "tracking_info": {
                "carrier": "UPS",
                "tracking_number": "1Z999AA10123456784",
                "delivery_status": "delivered"
            },
            "shipping_address": {
                "zip": "10001"
            },
            "tos_acceptance": {
                "date": "2023-01-01",
                "ip": "192.168.1.1"
            }
        }
    ]
    for order in orders:
        if order["order_id"] == order_id:
            return order
    return "No order found"


@function_tool
def get_emails(email: str) -> list:
    """
    Return a list of email records for the given email address.
    """
    emails = [
        {
            "email": "customer1@example.com",
            "subject": "Order #1121",
            "body": "Hey, I know you don't accept refunds but the sneakers don't fit and I'd like a refund"
        },
        {
            "email": "customer2@example.com",
            "subject": "Inquiry about product availability",
            "body": "Hello, I wanted to check if the new model of the smartphone is available in stock."
        },
        {
            "email": "customer3@example.com",
            "subject": "Feedback on recent purchase",
            "body": "Hi, I recently purchased a laptop from your store and I am very satisfied with the product. Keep up the good work!"
        }
    ]
    return [email_data for email_data in emails if email_data["email"] == email]


@function_tool
async def retrieve_payment_intent(payment_intent_id: str) -> dict:
    """
    Retrieve a Stripe payment intent by ID.
    Returns the payment intent object on success or an empty dictionary on failure.
    """
    try:
        return stripe.PaymentIntent.retrieve(payment_intent_id)
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error occurred while retrieving payment intent: {e}")
        return {}

@function_tool
async def close_dispute(dispute_id: str) -> dict:
    """
    Close a Stripe dispute by ID.
    Returns the dispute object on success or an empty dictionary on failure.
    """
    try:
        return stripe.Dispute.close(dispute_id)
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error occurred while closing dispute: {e}")
        return {}
```

## Step 3: Define the Specialized Agents

Now, you will create three distinct agents, each with a specific role in the dispute resolution workflow.

### Investigator Agent
This agent gathers comprehensive evidence for complex disputes.

```python
investigator_agent = Agent(
    name="Dispute Intake Agent",
    instructions=(
        "As a dispute investigator, please compile the following details in your final output:\n\n"
        "Dispute Details:\n"
        "- Dispute ID\n"
        "- Amount\n"
        "- Reason for Dispute\n"
        "- Card Brand\n\n"
        "Payment & Order Details:\n"
        "- Fulfillment status of the order\n"
        "- Shipping carrier and tracking number\n"
        "- Confirmation of TOS acceptance\n\n"
        "Email and Phone Records:\n"
        "- Any relevant email threads (include the full body text)\n"
        "- Any relevant phone logs\n"
    ),
    model="o3-mini",
    tools=[get_emails, get_phone_logs]
)
```

### Acceptance Agent
This agent handles straightforward cases by closing the dispute.

```python
accept_dispute_agent = Agent(
    name="Accept Dispute Agent",
    instructions=(
        "You are an agent responsible for accepting disputes. Please do the following:\n"
        "1. Use the provided dispute ID to close the dispute.\n"
        "2. Provide a short explanation of why the dispute is being accepted.\n"
        "3. Reference any relevant order details (e.g., unfulfilled order, etc.) retrieved from the database.\n\n"
        "Then, produce your final output in this exact format:\n\n"
        "Dispute Details:\n"
        "- Dispute ID\n"
        "- Amount\n"
        "- Reason for Dispute\n\n"
        "Order Details:\n"
        "- Fulfillment status of the order\n\n"
        "Reasoning for closing the dispute\n"
    ),
    model="gpt-4o",
    tools=[close_dispute]
)
```

### Triage Agent
This is the central decision-maker. It examines the order status and decides which specialized agent should handle the dispute.

```python
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "Please do the following:\n"
        "1. Find the order ID from the payment intent's metadata.\n"
        "2. Retrieve detailed information about the order (e.g., shipping status).\n"
        "3. If the order has shipped, escalate this dispute to the investigator agent.\n"
        "4. If the order has not shipped, accept the dispute.\n"
    ),
    model="gpt-4o",
    tools=[retrieve_payment_intent, get_order],
    handoffs=[accept_dispute_agent, investigator_agent],
)
```

## Step 4: Create the Core Workflow Function

This function retrieves dispute data from Stripe and initiates the agentic workflow by passing the data to the triage agent.

```python
async def process_dispute(payment_intent_id, triage_agent):
    """Retrieve and process dispute data for a given PaymentIntent."""
    disputes_list = stripe.Dispute.list(payment_intent=payment_intent_id)
    if not disputes_list.data:
        logger.warning("No dispute data found for PaymentIntent: %s", payment_intent_id)
        return None

    dispute_data = disputes_list.data[0]

    relevant_data = {
        "dispute_id": dispute_data.get("id"),
        "amount": dispute_data.get("amount"),
        "due_by": dispute_data.get("evidence_details", {}).get("due_by"),
        "payment_intent": dispute_data.get("payment_intent"),
        "reason": dispute_data.get("reason"),
        "status": dispute_data.get("status"),
        "card_brand": dispute_data.get("payment_method_details", {}).get("card", {}).get("brand")
    }

    event_str = json.dumps(relevant_data)
    # Pass the dispute data to the triage agent
    result = await Runner.run(triage_agent, input=event_str)
    logger.info("WORKFLOW RESULT: %s", result.final_output)

    return relevant_data, result.final_output
```

## Step 5: Test the Workflow with Scenarios

### Scenario 1: Company Mistake (Product Not Received)
This scenario simulates a case where the company failed to ship an order. The system should automatically accept the dispute.

```python
# Create a test payment intent that will trigger a "product not received" dispute
payment = stripe.PaymentIntent.create(
  amount=2000,
  currency="usd",
  payment_method = "pm_card_createDisputeProductNotReceived",
  confirm=True,
  metadata={"order_id": "1234"}, # This order ID corresponds to an unshipped order
  off_session=True,
  automatic_payment_methods={"enabled": True},
)

# Process the dispute
relevant_data, triage_result = await process_dispute(payment.id, triage_agent)
```

**Expected Workflow:** The triage agent will find that order `1234` has `fulfillment_details: "not_shipped"`. It will then hand off to the `accept_dispute_agent`, which will close the dispute and provide reasoning.

### Scenario 2: Customer Dispute (Final Sale)
This scenario simulates a complex case where a customer disputes a delivered, final-sale item. The system should escalate to the investigator for evidence gathering.

```python
# Create a test payment intent that will trigger a standard dispute
payment = stripe.PaymentIntent.create(
  amount=2000,
  currency="usd",
  payment_method = "pm_card_createDispute",
  confirm=True,
  metadata={"order_id": "1121"}, # This order ID corresponds to a delivered order
  off_session=True,
  automatic_payment_methods={"enabled": True},
)

# Process the dispute
relevant_data, triage_result = await process_dispute(payment.id, triage_agent)
```

**Expected Workflow:** The triage agent will find that order `1121` has `fulfillment_details: "delivered"`. It will then hand off to the `investigator_agent`, which will gather email and phone logs to compile an evidence report.

## Conclusion

You have successfully built an automated dispute management system using the OpenAI Agents SDK. This tutorial demonstrated key SDK features:

*   **Agent Loop:** The SDK manages the iterative process of tool calling and LLM communication.
*   **Handoffs:** Specialized agents (`accept_dispute_agent`, `investigator_agent`) are seamlessly invoked by the central `triage_agent` based on logic.
*   **Function Tools:** Python functions are easily converted into tools with automatic schema generation.

For production applications, consider implementing **Guardrails** to validate inputs and monitor for errors. The SDK also provides built-in **Tracing** via the OpenAI dashboard, which is invaluable for debugging and optimizing agent workflows.

This foundation enables you to extend the system with more agents, integrate additional data sources, and handle more complex business logic.
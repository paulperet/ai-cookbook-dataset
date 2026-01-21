# Guide: Batch Processing with the Messages Batches API

This guide demonstrates how to use the Anthropic Messages Batches API to process large volumes of requests asynchronously and cost-effectively. You will learn to create, monitor, and retrieve results from message batches, which can reduce costs by up to 50% compared to real-time requests.

## Prerequisites

First, install the required library and set up your environment.

```bash
pip install anthropic
```

```python
import time
import anthropic

# Initialize the client
client = anthropic.Anthropic()
MODEL_NAME = "claude-3-5-sonnet-20241022"
```

## Step 1: Create and Submit a Basic Message Batch

Let's start by creating a batch of simple message requests. We'll define a list of questions and format them into the required batch structure.

```python
# Prepare a list of questions for batch processing
questions = [
    "How do solar panels convert sunlight into electricity?",
    "What's the difference between mutual funds and ETFs?",
    "What is a pick and roll in basketball?",
    "Why do leaves change color in autumn?",
]

# Create batch requests with unique custom IDs
batch_requests = [
    {
        "custom_id": f"question-{i}",
        "params": {
            "model": MODEL_NAME,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": question}],
        },
    }
    for i, question in enumerate(questions)
]

# Submit the batch to the API
response = client.beta.messages.batches.create(requests=batch_requests)

print(f"Batch ID: {response.id}")
print(f"Status: {response.processing_status}")
print(f"Created at: {response.created_at}")
```

The API returns a batch object with a unique ID and an initial status (typically `in_progress`).

## Step 2: Monitor Batch Processing Status

Batch processing happens asynchronously. Use a polling function to monitor the batch until it completes.

```python
def monitor_batch(batch_id, polling_interval=5):
    """
    Poll the batch status until processing ends.
    """
    while True:
        batch_update = client.beta.messages.batches.retrieve(batch_id)
        batch_status = batch_update.processing_status
        print(f"Status: {batch_status}")

        if batch_status == "ended":
            return batch_update

        time.sleep(polling_interval)

# Monitor the batch we just created
batch_result = monitor_batch(response.id)
print("\nBatch processing complete!")

# Print summary counts
print("\nRequest counts:")
print(f"  Succeeded: {batch_result.request_counts.succeeded}")
print(f"  Errored: {batch_result.request_counts.errored}")
print(f"  Processing: {batch_result.request_counts.processing}")
print(f"  Canceled: {batch_result.request_counts.canceled}")
print(f"  Expired: {batch_result.request_counts.expired}")
```

## Step 3: Retrieve and Process Batch Results

Once the batch status is `ended`, you can retrieve the individual results.

```python
def process_results(batch_id):
    """
    Retrieve and display results for a completed batch.
    """
    # Get the final batch status
    batch = client.beta.messages.batches.retrieve(batch_id)

    print(f"\nBatch {batch.id} Summary:")
    print(f"Status: {batch.processing_status}")
    print(f"Created: {batch.created_at}")
    print(f"Ended: {batch.ended_at}")
    print(f"Expires: {batch.expires_at}")

    if batch.processing_status == "ended":
        print("\nIndividual Results:")
        # Iterate through all results in the batch
        for result in client.beta.messages.batches.results(batch_id):
            print(f"\nResult for {result.custom_id}:")
            print(f"Status: {result.result.type}")

            if result.result.type == "succeeded":
                # Print a preview of the response content
                content_preview = result.result.message.content[0].text[:200]
                print(f"Content: {content_preview}...")
            elif result.result.type == "errored":
                print("Request errored")
            elif result.result.type == "canceled":
                print("Request was canceled")
            elif result.result.type == "expired":
                print("Request expired")

# Process the results of our completed batch
process_results(batch_result.id)
```

## Step 4: Advanced Batch with Diverse Message Types

The batch API supports various request types. This example creates a batch containing a simple message, an image analysis request, a message with a system prompt, and a multi-turn conversation.

```python
import base64

def get_base64_encoded_image(image_path):
    """Helper function to encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base64_string = base64.b64encode(binary_data).decode("utf-8")
        return base64_string

def create_complex_batch():
    """
    Create a batch with mixed request types.
    """
    # Mix of different request types
    batch_requests = [
        {
            "custom_id": "simple-question",
            "params": {
                "model": MODEL_NAME,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "What is quantum computing?"}],
            },
        },
        {
            "custom_id": "image-analysis",
            "params": {
                "model": MODEL_NAME,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": get_base64_encoded_image(
                                        "../images/sunset-dawn-nature-mountain-preview.jpg"
                                    ),
                                },
                            },
                            {
                                "type": "text",
                                "text": "Describe this mountain landscape. What time of day does it appear to be, and what weather conditions do you observe?",
                            },
                        ],
                    }
                ],
            },
        },
        {
            "custom_id": "system-prompt",
            "params": {
                "model": MODEL_NAME,
                "max_tokens": 1024,
                "system": "You are a helpful science teacher.",
                "messages": [{"role": "user", "content": "Explain gravity to a 5-year-old."}],
            },
        },
        {
            "custom_id": "multi-turn",
            "params": {
                "model": MODEL_NAME,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "What is DNA?"},
                    {
                        "role": "assistant",
                        "content": "DNA is like a blueprint for living things...",
                    },
                    {"role": "user", "content": "How is DNA copied?"},
                ],
            },
        },
    ]

    try:
        response = client.beta.messages.batches.create(requests=batch_requests)
        return response.id
    except Exception as e:
        print(f"Error creating batch: {e}")
        return None

# Create the advanced batch
complex_batch_id = create_complex_batch()
print(f"Complex batch ID: {complex_batch_id}")
```

## Step 5: Monitor and Process the Advanced Batch

Use the same monitoring and result processing functions to handle the complex batch.

```python
# Monitor the batch until completion
batch_status = monitor_batch(complex_batch_id)

# Process and display the results
if batch_status.processing_status == "ended":
    process_results(batch_status.id)
```

## Summary

You have now learned how to:
1. **Create a batch** of message requests with unique custom IDs.
2. **Monitor batch status** using a polling function.
3. **Retrieve and process results** for individual requests within a batch.
4. **Build advanced batches** containing diverse message types like images, system prompts, and multi-turn conversations.

The Messages Batches API is ideal for processing high volumes of non-real-time requests efficiently and at a lower cost. Remember to handle potential errors (errored, canceled, or expired requests) in your production code.
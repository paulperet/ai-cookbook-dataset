# Fine-Tuning Claude 3 Haiku on Amazon Bedrock: A Step-by-Step Guide

This guide walks you through the process of fine-tuning Anthropic's Claude 3 Haiku model on Amazon Bedrock. You will prepare a dataset, launch a fine-tuning job, and deploy the custom model for inference.

## Prerequisites

Before you begin, ensure you have the following:

1.  An AWS account with access to Amazon Bedrock.
2.  An S3 bucket for storing your training data and job outputs.
3.  An IAM service role with the necessary permissions for Bedrock model customization and S3 access. [Review the IAM role requirements here.](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-iam-role.html)

## Step 1: Environment Setup

First, install the required Python library and import the necessary modules.

```bash
pip install boto3
```

```python
import boto3
import json
```

## Step 2: Prepare Your Dataset

Your dataset must be a JSON Lines file (`.jsonl`), where each line is a valid JSON object following a specific conversation format.

### Dataset Format Requirements

Each JSON object must have this structure:
```json
{
  "system": "<optional_system_message>",
  "messages": [
    {"role": "user", "content": "user message"},
    {"role": "assistant", "content": "assistant response"}
  ]
}
```

**Key Rules:**
*   The `system` field is optional.
*   The `messages` array must contain at least two messages.
*   The first message must have `"role": "user"`.
*   The last message must have `"role": "assistant"`.
*   User and assistant messages must alternate.
*   No extra keys are allowed in the object.

### Load a Sample Dataset

For this tutorial, we'll use a sample dataset that teaches the model to respond exclusively with JSON. Let's inspect the first entry to understand the format.

```python
dataset_path = "datasets/json_mode_dataset.jsonl"
sample_dataset = []

with open(dataset_path) as f:
    for line in f:
        sample_dataset.append(json.loads(line))

print(json.dumps(sample_dataset[0], indent=2))
```

## Step 3: Upload Dataset to Amazon S3

Amazon Bedrock requires your training data to be stored in an S3 bucket. Upload your prepared `.jsonl` file.

```python
# Replace with your bucket name and desired S3 key
bucket_name = "YOUR_BUCKET_NAME"
s3_path = "json_mode_dataset.jsonl"

s3 = boto3.client("s3")
s3.upload_file(dataset_path, bucket_name, s3_path)

print(f"Dataset uploaded to: s3://{bucket_name}/{s3_path}")
```

## Step 4: Launch the Fine-Tuning Job

Now, configure and launch the model customization job using the Bedrock client.

### 4.1 Configure Job Parameters

Set the job name, model identifier, IAM role, and hyperparameters.

```python
# Job Configuration
job_name = "anthropic-finetuning-cookbook-training"
custom_model_name = "anthropic_finetuning_cookbook"
role = "YOUR_AWS_SERVICE_ROLE_ARN"  # e.g., 'arn:aws:iam::123456789012:role/MyBedrockRole'
output_path = f"s3://{bucket_name}/finetuning_example_results/"
base_model_id = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0:200k"

# Hyperparameters
epoch_count = 5
batch_size = 4
learning_rate_multiplier = 1.0
```

### 4.2 Create the Fine-Tuning Job

Use the `create_model_customization_job` API to start the training process.

```python
bedrock = boto3.client(service_name="bedrock")

response = bedrock.create_model_customization_job(
    customizationType="FINE_TUNING",
    jobName=job_name,
    customModelName=custom_model_name,
    roleArn=role,
    baseModelIdentifier=base_model_id,
    hyperParameters={
        "epochCount": f"{epoch_count}",
        "batchSize": f"{batch_size}",
        "learningRateMultiplier": f"{learning_rate_multiplier}",
    },
    trainingDataConfig={"s3Uri": f"s3://{bucket_name}/{s3_path}"},
    outputDataConfig={"s3Uri": output_path},
)

print(f"Job created: {job_name}")
print(f"Job ARN: {response['jobArn']}")
```

### 4.3 Monitor Job Status

You can check the status of your job while it's running.

```python
status_response = bedrock.get_model_customization_job(jobIdentifier=job_name)
current_status = status_response["status"]
print(f"Job Status: {current_status}")
```

## Step 5: Deploy and Use Your Fine-Tuned Model

Once the fine-tuning job completes successfully, you must deploy your custom model using **Provisioned Throughput** in Amazon Bedrock before you can invoke it.

[Follow the official guide to host your model with Provisioned Throughput.](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-use.html)

### 5.1 Invoke the Custom Model

After your model is provisioned, you can invoke it using the Bedrock Runtime client. You will need the ARN of your provisioned throughput.

```python
# Replace with your Provisioned Throughput ARN
provisioned_throughput_arn = "YOUR_PROVISIONED_THROUGHPUT_ARN"

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# Prepare the request body following the Claude message format
request_body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "system": "JSON Mode: Enabled",  # Optional: Use a system prompt relevant to your fine-tuning
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is a large language model?"}],
        }
    ],
}

# Invoke the model
response = bedrock_runtime.invoke_model(
    modelId=provisioned_throughput_arn,
    body=json.dumps(request_body)
)

# Parse and print the response
response_body = json.loads(response["body"].read().decode("utf-8"))
print(response_body["content"][0]["text"])
```

Your fine-tuned Claude 3 Haiku model is now ready to generate responses based on your custom dataset.
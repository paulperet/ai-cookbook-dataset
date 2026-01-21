# Automating AWS S3 Tasks with OpenAI Function Calling

This guide demonstrates how to build an intelligent assistant that interacts with Amazon S3 using OpenAI's function calling capabilities. You'll create a system that understands natural language requests (e.g., "list my buckets" or "download report.pdf") and executes the corresponding S3 operations.

## Prerequisites

Before you begin, ensure you have the following:

1.  **AWS Credentials:** An AWS Access Key ID and Secret Access Key with permissions to read from and write to your S3 buckets.
2.  **OpenAI API Key:** An active API key from OpenAI.
3.  **Python Environment:** A Python environment where you can install packages.

Store your credentials in a `.env` file in your project directory:

```bash
AWS_ACCESS_KEY_ID=<your-aws-access-key>
AWS_SECRET_ACCESS_KEY=<your-aws-secret-key>
OPENAI_API_KEY=<your-openai-api-key>
```

## Setup and Installation

First, install the required Python libraries.

```bash
pip install openai boto3 tenacity python-dotenv
```

Now, let's set up the environment and initialize the clients.

```python
import json
import os
from urllib.request import urlretrieve
import datetime

import boto3
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
GPT_MODEL = "gpt-3.5-turbo-1106" # or "gpt-4"

# Initialize the AWS S3 client using credentials from the environment
s3_client = boto3.client('s3')
```

## Step 1: Define the Tool Schema for OpenAI

To enable the AI model to call our functions, we must describe them in a format it understands. We define a list of tools (formerly called "functions") that detail each S3 operation's purpose, parameters, and requirements.

```python
# Define the tools (functions) that the AI model can call
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_buckets",
            "description": "List all available S3 buckets in the AWS account.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_objects",
            "description": "List the objects or files inside a given S3 bucket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket": {
                        "type": "string",
                        "description": "The name of the S3 bucket."
                    },
                    "prefix": {
                        "type": "string",
                        "description": "The folder path prefix to filter objects in the S3 bucket."
                    }
                },
                "required": ["bucket"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "download_file",
            "description": "Download a specific file from an S3 bucket to a local directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket": {
                        "type": "string",
                        "description": "The name of the S3 bucket."
                    },
                    "key": {
                        "type": "string",
                        "description": "The full path to the file inside the bucket."
                    },
                    "directory": {
                        "type": "string",
                        "description": "The absolute or relative local destination directory path."
                    }
                },
                "required": ["bucket", "key", "directory"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "upload_file",
            "description": "Upload a file from a local path or a remote URL to an S3 bucket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "The local file path or remote URL of the file to upload."
                    },
                    "bucket": {
                        "type": "string",
                        "description": "The name of the destination S3 bucket."
                    },
                    "key": {
                        "type": "string",
                        "description": "The desired path/name for the file inside the bucket."
                    },
                    "is_remote_url": {
                        "type": "boolean",
                        "description": "True if the source is a URL, False if it's a local file path."
                    }
                },
                "required": ["source", "bucket", "key", "is_remote_url"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_s3_objects",
            "description": "Search for files by name across one or all S3 buckets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_name": {
                        "type": "string",
                        "description": "The full or partial name of the file to search for."
                    },
                    "bucket": {
                        "type": "string",
                        "description": "The name of a specific S3 bucket to search in. Omit to search all buckets."
                    },
                    "prefix": {
                        "type": "string",
                        "description": "A folder path prefix to limit the search within a bucket."
                    },
                    "exact_match": {
                        "type": "boolean",
                        "description": "True to match the exact filename, False for a partial (contains) match."
                    }
                },
                "required": ["search_name"],
            },
        }
    }
]
```

## Step 2: Implement the S3 Helper Functions

Next, we write the actual Python functions that will be called by the AI model. Each function corresponds to a tool defined above.

We need a helper to serialize datetime objects returned by Boto3.

```python
def datetime_converter(obj):
    """Helper function to convert datetime objects to ISO format strings for JSON serialization."""
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
```

Now, implement the core S3 operations.

```python
def list_buckets():
    """Lists all S3 buckets in the account."""
    response = s3_client.list_buckets()
    # Serialize the response, handling datetime objects
    return json.dumps(response['Buckets'], default=datetime_converter)

def list_objects(bucket, prefix=''):
    """Lists objects within a specified bucket and optional prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    # Return an empty list if the bucket is empty
    return json.dumps(response.get('Contents', []), default=datetime_converter)

def download_file(bucket, key, directory):
    """Downloads a file from S3 to a local directory."""
    # Extract the filename from the S3 key
    filename = os.path.basename(key)
    # Create the full local destination path
    destination = os.path.join(directory, filename)

    # Perform the download
    s3_client.download_file(bucket, key, destination)
    return json.dumps({
        "status": "success",
        "bucket": bucket,
        "key": key,
        "destination": destination
    })

def upload_file(source, bucket, key, is_remote_url=False):
    """Uploads a file to S3 from a local path or a remote URL."""
    if is_remote_url:
        # If source is a URL, download it to a temporary local file first
        file_name = os.path.basename(source)
        urlretrieve(source, file_name)
        source = file_name

    # Perform the upload
    s3_client.upload_file(source, bucket, key)
    return json.dumps({
        "status": "success",
        "source": source,
        "bucket": bucket,
        "key": key
    })

def search_s3_objects(search_name, bucket=None, prefix='', exact_match=True):
    """Searches for objects by name across one or all buckets."""
    search_name = search_name.lower()
    results = []

    # Determine which buckets to search
    if bucket is None:
        # Search all buckets
        buckets_response = json.loads(list_buckets())
        buckets_to_search = [b["Name"] for b in buckets_response]
    else:
        # Search only the specified bucket
        buckets_to_search = [bucket]

    # Iterate through buckets and search for the file
    for bucket_name in buckets_to_search:
        objects_response = json.loads(list_objects(bucket_name, prefix))

        if exact_match:
            bucket_results = [obj for obj in objects_response if search_name == obj['Key'].lower()]
        else:
            bucket_results = [obj for obj in objects_response if search_name in obj['Key'].lower()]

        if bucket_results:
            # Add bucket context to each found object
            results.extend([{"Bucket": bucket_name, "Object": obj} for obj in bucket_results])

    return json.dumps(results)
```

Finally, create a mapping so our conversation engine can call the correct function by name.

```python
# Map function names from the AI model to our Python functions
available_functions = {
    "list_buckets": list_buckets,
    "list_objects": list_objects,
    "download_file": download_file,
    "upload_file": upload_file,
    "search_s3_objects": search_s3_objects
}
```

## Step 3: Build the Conversation Engine

This is the core logic that connects the user's natural language request to the execution of S3 functions.

First, a helper function to call the OpenAI API.

```python
def chat_completion_request(messages, tools=None, model_name=GPT_MODEL):
    """Sends a request to the OpenAI Chat Completions API."""
    if tools is not None:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # Let the model decide if/when to call a function
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
    return response
```

Now, the main function that orchestrates the entire interaction.

```python
def run_conversation(user_input, topic="S3 bucket operations.", is_log=False):
    """
    Processes a user's natural language request, calls the AI model,
    executes any required S3 functions, and returns a final response.
    """
    # System message sets the context and rules for the AI assistant
    system_message = f"""You are a helpful assistant for AWS S3 operations.
    Your scope is strictly limited to {topic}.
    Do not make assumptions about parameter values. Ask for clarification if a request is ambiguous.
    If a user asks about something outside your scope, politely decline."""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    
    # 1. Get the initial response from the AI model
    response = chat_completion_request(messages, tools=tools)
    response_message = response.choices[0].message
    
    # Optional: Log the raw response for debugging
    if is_log:
        print("Initial API Response:", response_message)
    
    # 2. Check if the model wants to call a function
    if response_message.tool_calls:
        # We'll handle the first function call in the response
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"[Assistant is calling function: {function_name} with args: {function_args}]")
        
        # 3. Execute the actual S3 function
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)
        
        # 4. Append the function call and its result back to the conversation history
        messages.append(response_message)  # Add the assistant's function request
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": function_response,
        })
        
        # 5. Send the conversation history back to the model for a final, summarized response
        second_response = chat_completion_request(messages)
        final_message = second_response.choices[0].message.content
    else:
        # If no function was called, use the model's direct response
        final_message = response_message.content

    return final_message
```

## Step 4: Test Your S3 Assistant

Replace the placeholders (like `<bucket_name>`, `<file_name>`, and `<directory_path>`) with your actual values before running these examples.

### Example 1: Listing and Searching

Start by asking your assistant to list all your S3 buckets.

```python
print(run_conversation('list my S3 buckets'))
```

Search for a specific file across all your buckets.

```python
search_file = 'quarterly-report.pdf'
print(run_conversation(f'search for the file {search_file} in all buckets'))
```

Search for files containing a specific word within a particular bucket.

```python
search_word = 'invoice'
bucket_name = 'my-documents-bucket'
print(run_conversation(f'find files containing the word {search_word} in my {bucket_name} bucket'))
```

### Example 2: Handling Ambiguity

The assistant should ask for clarification when a request is missing key information, as instructed in the system prompt.

```python
print(run_conversation('search for a file'))
# Expected: The assistant should ask which file you want to search for.
```

### Example 3: Staying Within Scope

The assistant should politely refuse to answer questions outside its defined scope of S3 operations.

```python
print(run_conversation('what is the weather today?'))
# Expected: A response indicating it can only help with S3-related tasks.
```

### Example 4: Downloading a File

Instruct the assistant to download a file. It will need the bucket name, file key, and a local directory path from your request.

```python
file_to_download = 'projects/data.csv'
target_bucket = 'company-data-archive'
local_folder = './downloads'
print(run_conversation(f'download {file_to_download} from the {target_bucket} bucket to the {local_folder} folder'))
```

### Example 5: Uploading a File

Ask the assistant to upload a file from your local machine.

```python
local_file = './reports/final_analysis.pdf'
destination_bucket = 'team-share-bucket'
print(run_conversation(f'upload the local file {local_file} to the {destination_bucket} bucket'))
```

## Conclusion

You've successfully built an intelligent agent that uses OpenAI's function calling to perform real-world AWS S3 operations based on natural language commands. This pattern can be extended to integrate with countless other APIs and services, creating powerful, conversational interfaces for complex systems.

**Key Takeaways:**
1.  Define your tools clearly so the AI model understands when and how to use them.
2.  Implement robust Python functions that safely execute the desired actions.
3.  The conversation engine manages the flow between user input, AI reasoning, function execution, and final response.
4.  System prompts are crucial for guiding the model's behavior and keeping it within safe boundaries.
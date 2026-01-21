# Guide: Extracting Nutrition Data from Images with Vision and Tools

This tutorial demonstrates how to combine Claude's vision capabilities with custom tools to analyze an image of a nutrition label and extract structured data. You'll create a tool that Claude can call to output nutrition information in a clean JSON format.

## Prerequisites

First, install the required packages and set up your environment.

```bash
pip install anthropic IPython
```

```python
import base64
from anthropic import Anthropic
from IPython.display import Image

# Initialize the Anthropic client
client = Anthropic()
MODEL_NAME = "claude-3-opus-20240229"  # Or your preferred Claude model
```

## Step 1: Define a Custom Nutrition Extraction Tool

You'll create a tool that Claude can use to output structured nutrition data. This tool defines the schema for the information you want to extract.

```python
nutrition_tool = {
    "name": "print_nutrition_info",
    "description": "Extracts nutrition information from an image of a nutrition label",
    "input_schema": {
        "type": "object",
        "properties": {
            "calories": {
                "type": "integer",
                "description": "The number of calories per serving"
            },
            "total_fat": {
                "type": "integer",
                "description": "The amount of total fat in grams per serving"
            },
            "cholesterol": {
                "type": "integer",
                "description": "The amount of cholesterol in milligrams per serving"
            },
            "total_carbs": {
                "type": "integer",
                "description": "The amount of total carbohydrates in grams per serving"
            },
            "protein": {
                "type": "integer",
                "description": "The amount of protein in grams per serving"
            }
        },
        "required": ["calories", "total_fat", "cholesterol", "total_carbs", "protein"]
    }
}
```

This tool definition specifies that Claude should output an object with five required integer fields when it extracts nutrition information.

## Step 2: Prepare the Image for Analysis

Create a helper function to encode your image in base64 format, which is required for the Anthropic API.

```python
def get_base64_encoded_image(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base64_encoded_data = base64.b64encode(binary_data)
        return base64_encoded_data.decode("utf-8")
```

## Step 3: Construct the Message with Image and Prompt

Now, build the message that will be sent to Claude. This includes both the image and your text instruction.

```python
# Encode your nutrition label image
image_path = "../images/tool_use/nutrition_label.png"  # Update with your actual path
base64_image = get_base64_encoded_image(image_path)

# Create the message list with image and text
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image
                }
            },
            {
                "type": "text",
                "text": "Please print the nutrition information from this nutrition label image."
            }
        ]
    }
]
```

## Step 4: Send the Request and Handle Tool Use

Make the API call, passing both your messages and the tool definition. Claude will analyze the image and decide whether to use your tool.

```python
response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=4096,
    messages=message_list,
    tools=[nutrition_tool]
)

# Check if Claude decided to use the tool
if response.stop_reason == "tool_use":
    last_content_block = response.content[-1]
    if last_content_block.type == "tool_use":
        tool_name = last_content_block.name
        tool_inputs = last_content_block.input
        
        print(f"Claude called the '{tool_name}' tool with the following data:")
        print(tool_inputs)
else:
    print("No tool was called. Claude may have responded with text instead.")
```

## Expected Output

When successful, you'll see output similar to:

```
Claude called the 'print_nutrition_info' tool with the following data:
{'calories': 200, 'total_fat': 15, 'cholesterol': 30, 'total_carbs': 30, 'protein': 5}
```

## How It Works

1. **Vision Analysis**: Claude analyzes the nutrition label image to identify the relevant nutritional values.
2. **Tool Decision**: Based on your prompt and the tool definition, Claude determines that the `print_nutrition_info` tool is appropriate for this task.
3. **Structured Output**: Claude extracts the numerical values from the image and formats them according to your tool's schema.

## Next Steps

You can extend this approach to:
- Extract different types of data from various document images
- Chain multiple tools together for more complex workflows
- Integrate the extracted data into databases or other applications
- Add validation logic to verify extracted values fall within expected ranges

This pattern demonstrates how vision models can be combined with structured output tools to create reliable data extraction pipelines from visual content.
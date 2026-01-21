# Image-to-Code Generation with OpenVINO and Phi-3-Vision

This guide demonstrates how to use a pretrained vision-language model to generate Python code based on an image and a text prompt. You'll learn how to process image inputs, format prompts correctly, and generate functional Python code using the OpenVINO-optimized Phi-3-Vision model.

## Prerequisites

First, install the required dependencies:

```bash
pip install openvino transformers pillow requests matplotlib
```

## Step 1: Import Required Libraries

Begin by importing the necessary modules for image processing, model handling, and text generation:

```python
import requests
from PIL import Image
from transformers import AutoProcessor, TextStreamer
import openvino as ov
```

## Step 2: Load and Display the Input Image

Load your target image using PIL. For this example, we'll use a sample image file:

```python
# Load the image
image = Image.open("demo.png")

# Display the image (optional)
image.show()
```

## Step 3: Define Your Prompt

Create a message that combines your image with a text instruction. The model will generate Python code based on this combined input:

```python
# Define the message with image and text prompt
message = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Write Python code to process this image and save it using plt."}
    ]}
]
```

## Step 4: Load the Model Processor

Load the processor that will handle both text and image inputs. The processor tokenizes text and prepares image tensors in the format the model expects:

```python
# Specify the model directory
out_dir = "phi3-vision-openvino"

# Load the processor
processor = AutoProcessor.from_pretrained(out_dir, trust_remote_code=True)
```

## Step 5: Format the Prompt

Use the processor's chat template method to format your message into a structured prompt. This ensures the input matches the model's training format:

```python
# Apply chat template to format the message
prompt = processor.apply_chat_template(message, add_generation_prompt=True)
```

## Step 6: Process Inputs for the Model

Convert both the text prompt and image into tensors that the model can process:

```python
# Process inputs (text prompt + image)
inputs = processor(prompt, image, return_tensors="pt")
```

## Step 7: Configure Generation Parameters

Set parameters that control how the model generates text. These parameters balance creativity with coherence:

```python
# Define generation arguments
gen_kwargs = {
    "max_new_tokens": 500,  # Maximum length of generated code
    "do_sample": True,      # Enable sampling for more diverse outputs
}
```

## Step 8: Generate Python Code

Now, generate the Python code using the model. The `TextStreamer` provides real-time output display:

```python
# Initialize text streamer for real-time output
streamer = TextStreamer(
    processor.tokenizer,
    skip_prompt=True,      # Don't show the input prompt
    skip_special_tokens=True  # Skip special tokens in output
)

# Generate code
generated_ids = model.generate(**inputs, streamer=streamer, **gen_kwargs)
```

## Step 9: Extract and Display the Generated Code

Extract the generated text from the model's output and display the complete Python code:

```python
# Decode the generated tokens to text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Print the generated code
print("Generated Python Code:")
print(generated_text)
```

## Example Output

The model will generate Python code similar to this:

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the image
img = Image.open('demo.png')
img_array = np.array(img)

# Process the image (example: convert to grayscale)
gray_img = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])

# Display and save the processed image
plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.savefig('processed_image.png', bbox_inches='tight', pad_inches=0)
plt.show()
```

## Key Takeaways

1. **Multimodal Input**: The model processes both images and text simultaneously to understand your request
2. **Code Generation**: The output is executable Python code that addresses your specific image processing task
3. **Real-time Streaming**: The `TextStreamer` lets you watch the code generation process unfold
4. **OpenVINO Optimization**: The model runs efficiently using OpenVINO's inference engine

## Next Steps

- Experiment with different image types and processing requests
- Modify the generation parameters (`temperature`, `top_p`) to adjust creativity
- Integrate the generated code into your own image processing pipelines
- Try the model with different programming languages by adjusting your prompt

This workflow demonstrates how vision-language models can bridge visual understanding with code generation, creating powerful tools for automating image processing tasks.
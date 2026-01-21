# Guide: Building a Multimodal AI Assistant with Phi-4

This guide walks you through setting up and using the Phi-4 multimodal model from Microsoft to generate professional content from both image and audio inputs. You will create a system that acts as a technology journalist, producing a press release based on visual and auditory data.

## Prerequisites & Setup

Before you begin, ensure you have the necessary libraries installed. The following commands will set up your environment.

### 1. Install Core Dependencies

First, install the Hugging Face Hub library and the `backoff` utility for robust API interactions.

```bash
pip install huggingface_hub backoff
```

### 2. Install Flash Attention (Optional, for Performance)

For optimal performance on compatible hardware (e.g., NVIDIA GPUs with CUDA 12.3), you can install the Flash Attention 2 library. This step is optional but recommended for speed.

```bash
# Download the pre-built wheel for your specific environment.
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install the wheel without its dependencies.
pip install --no-dependencies --upgrade flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### 3. Authenticate and Download the Model

You need to authenticate with Hugging Face to access the model and then clone the repository.

```python
from huggingface_hub import login

# This will prompt you for your Hugging Face token if not already cached.
login()
```

Now, clone the model repository using Git LFS.

```bash
git lfs install
git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct
```

### 4. Import Required Libraries

With the environment ready, import the necessary Python modules.

```python
import requests
import torch
from PIL import Image
import soundfile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, pipeline, AutoTokenizer
```

## Step 1: Load the Model and Processor

Define the path to your downloaded model and initialize the processor and language model.

```python
# Set the path to the cloned model directory
model_path = './Phi-4-multimodal-instruct'

# Initialize the processor. The `trust_remote_code=True` is required for this model.
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Load the model with Flash Attention 2 for efficiency and move it to the GPU.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto', # Automatically uses bfloat16 if supported
    _attn_implementation='flash_attention_2',
).cuda()
```

## Step 2: Configure Generation and Prompts

Set up the generation parameters and define the special tokens used by the Phi-4 model for structuring conversations.

```python
# Load the generation configuration from the model's files.
generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

# Define the special tokens for the model's chat template.
system_prompt = '<|system|>'
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
```

## Step 3: Construct the Multimodal Prompt

Now, you will craft a prompt that instructs the model to act as a journalist. The prompt will reference an image and an audio file that you will provide.

```python
prompt = f'''{system_prompt}You are a technology journalist who writes professional content based on audio and picture content{prompt_suffix}
{user_prompt}Reorganize the content provided by audio <|audio_1|> and image <|image_1|> from a professional perspective and write a press release based on the current development of AI. Output is markdown format and including title and content {prompt_suffix}
{assistant_prompt}'''
```

**Explanation:** This prompt has three parts:
1.  **System Instruction:** Defines the AI's role.
2.  **User Request:** Asks the AI to process specific audio and image placeholders (`<|audio_1|>`, `<|image_1|>`) and output a markdown press release.
3.  **Assistant Starter:** The `<|assistant|>` token signals the model to begin its response.

## Step 4: Prepare the Input Media

Load the image and audio files that will be fed into the model alongside the text prompt.

```python
# Load an image file (replace './copilot.png' with your image path)
image = Image.open("./copilot.png")

# Load an audio file (replace './satya1.mp3' with your audio path)
# soundfile.read returns the audio data and the sample rate.
audio, sample_rate = soundfile.read('./satya1.mp3')
```

## Step 5: Process Inputs and Generate a Response

The processor handles the tokenization of text and the encoding of image/audio data into a format the model understands.

```python
# Prepare all inputs for the model.
inputs = processor(text=prompt, images=[image], audios=[audio], return_tensors='pt').to('cuda:0')

# Generate the response from the model.
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1500, # Limit the length of the generated text
    generation_config=generation_config,
)

# The generated IDs include the input prompt. Slice them to get only the new tokens.
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]

# Decode the token IDs back into human-readable text.
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
```

## Step 6: View the Output

Finally, print or use the AI-generated press release.

```python
print(response)
```

**Example Output:**

The model will generate a structured markdown document. Here is an example of what the output might look like:

```
# Copilot: The Future of AI-Driven Business Growth

In the rapidly evolving world of technology, AI is transforming the way businesses operate, offering unprecedented efficiency and growth opportunities. Today, we delve into the latest developments in AI, focusing on the innovative Copilot platform, which is revolutionizing the business landscape.

## Copilot: A New Era of AI Integration

Copilot is at the forefront of AI technology, designed to enhance business processes and productivity. This platform is not just an AI tool; it's a comprehensive ecosystem that includes Copilot devices and a Copilot & AI stack, all underpinned by robust security measures.

### Key Features of Copilot

1.  **Copilot**: The core AI platform that serves as the brain of the operation...
2.  **Copilot Devices**: These are the physical or virtual tools that interact with the Copilot platform...
3.  **Copilot & AI Stack**: This component ensures that all elements of the Copilot system are securely connected...

### Security: The Backbone of Copilot

Security is paramount in the Copilot ecosystem. Each layer of the platform is designed with advanced security protocols...

## The Impact of Copilot on Business

The introduction of Copilot is set to transform business operations by automating routine tasks, enhancing decision-making processes, and fostering innovation...

### Benefits of Copilot

-   **Enhanced Efficiency**: Automating repetitive tasks frees up time for strategic activities.
-   **Improved Decision Making**: AI-driven insights provide data-backed decisions...
-   **Innovation**: The flexibility of Copilot encourages experimentation...

## Conclusion

As businesses continue to navigate the complexities of the digital age, Copilot stands out as a pivotal solution, driving growth and efficiency through AI...
```

## Summary

You have successfully built a pipeline that uses the Phi-4 multimodal model to synthesize information from text, image, and audio inputs to generate professional written content. This workflow can be adapted for various use cases, such as creating reports from meetings (audio + slides) or generating product descriptions from demos.
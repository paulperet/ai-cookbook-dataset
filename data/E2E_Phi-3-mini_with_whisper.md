# Building an Interactive Chatbot with Phi-3 Mini and Whisper

## Overview

This guide walks you through creating an interactive chatbot that uses the Microsoft Phi-3 Mini 4K Instruct model for text generation and OpenAI's Whisper for speech recognition. The application supports both text and voice input, making it versatile for various conversational AI tasks.

## Prerequisites

Before starting, ensure you have:

- A GPU with at least 12GB of memory
- Python 3.8 or higher
- Access to Hugging Face models (requires authentication token)

## Setup and Installation

### Step 1: Install Required Packages

First, install the necessary Python libraries:

```bash
pip install torch transformers accelerate gradio openai-whisper edge-tts
```

### Step 2: Configure GPU and CUDA

If you're using a Linux system, ensure your GPU drivers are properly configured:

```bash
# Update system packages
sudo apt update

# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Register CUDA driver location
echo /usr/lib64-nvidia/ > /etc/ld.so.conf.d/libcuda.conf
sudo ldconfig

# Verify GPU memory (should show at least 12GB)
nvidia-smi

# Check CUDA installation
nvcc --version
```

### Step 3: Set Up Hugging Face Authentication

To access the Phi-3 model, you need a Hugging Face token:

1. Navigate to [Hugging Face Token Settings](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Enter a name for your project
4. Select **Write** as the token type
5. Copy the generated token

## Building the Chatbot Application

### Step 4: Import Required Libraries

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr
import whisper
import edge_tts
import os
import uuid
from pathlib import Path
```

### Step 5: Initialize Models

```python
# Clear GPU cache to free memory
torch.cuda.empty_cache()

# Load Phi-3 Mini model
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load Whisper model for speech recognition
whisper_model = whisper.load_model("base")
```

### Step 6: Create Text-to-Speech Functions

```python
def tts_file_name(text):
    """Generate a unique filename based on input text."""
    # Create a short hash from the text for the filename
    text_hash = str(uuid.uuid4())[:8]
    return f"audio_{text_hash}.mp3"

def edge_free_tts(chunks_list, speed=1.0, voice_name="en-US-JennyNeural", save_path="."):
    """Convert text chunks to speech using Edge TTS."""
    audio_files = []
    
    for i, chunk in enumerate(chunks_list):
        output_file = os.path.join(save_path, f"chunk_{i}.mp3")
        communicate = edge_tts.Communicate(chunk, voice_name, rate=speed)
        communicate.save(output_file)
        audio_files.append(output_file)
    
    return audio_files

def talk(input_text):
    """Generate speech from text and return the audio file path."""
    # Create audio directory if it doesn't exist
    audio_dir = Path("/content/audio")
    audio_dir.mkdir(exist_ok=True)
    
    # Generate filename and save audio
    filename = tts_file_name(input_text)
    output_path = audio_dir / filename
    
    # Split long text into chunks if needed
    chunks = [input_text[i:i+200] for i in range(0, len(input_text), 200)]
    audio_files = edge_free_tts(chunks, save_path=str(audio_dir))
    
    # For simplicity, return the first chunk's audio
    return str(audio_files[0]) if audio_files else None
```

### Step 7: Implement Chat Functions

```python
def run_text_prompt(message, chat_history):
    """Process text input and generate response using Phi-3."""
    
    # Format the conversation for the model
    conversation = [
        {"role": "user", "content": message}
    ]
    
    # Tokenize input
    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Generate speech from response
    audio_file = talk(response)
    
    # Update chat history
    chat_history.append((message, response))
    
    return chat_history, audio_file

def run_audio_prompt(audio, chat_history):
    """Convert audio to text, then process as text prompt."""
    
    # Transcribe audio using Whisper
    result = whisper_model.transcribe(audio)
    transcribed_text = result["text"]
    
    # Process the transcribed text
    return run_text_prompt(transcribed_text, chat_history)
```

### Step 8: Create the Gradio Interface

```python
def create_chatbot_interface():
    """Build the Gradio web interface."""
    
    with gr.Blocks(title="Phi-3 Mini Chatbot with Whisper") as demo:
        gr.Markdown("# Interactive Phi-3 Mini Chatbot")
        gr.Markdown("Chat with the Phi-3 Mini model using text or voice input.")
        
        # Chat history display
        chatbot = gr.Chatbot(height=400)
        
        # State to maintain conversation history
        state = gr.State([])
        
        with gr.Row():
            # Text input
            text_input = gr.Textbox(
                label="Type your message",
                placeholder="Ask me anything...",
                scale=4
            )
            
            # Audio input
            audio_input = gr.Audio(
                label="Or record audio",
                type="filepath",
                scale=1
            )
        
        with gr.Row():
            # Submit buttons
            text_button = gr.Button("Send Text", variant="primary")
            audio_button = gr.Button("Send Audio", variant="secondary")
            clear_button = gr.Button("Clear Chat")
        
        # Audio output player
        audio_output = gr.Audio(label="Response Audio", type="filepath")
        
        # Connect event handlers
        text_button.click(
            fn=run_text_prompt,
            inputs=[text_input, state],
            outputs=[chatbot, audio_output]
        ).then(lambda: "", None, text_input)  # Clear text input after sending
        
        audio_button.click(
            fn=run_audio_prompt,
            inputs=[audio_input, state],
            outputs=[chatbot, audio_output]
        )
        
        clear_button.click(
            fn=lambda: ([], []),
            inputs=None,
            outputs=[chatbot, state]
        )
    
    return demo

# Launch the application
if __name__ == "__main__":
    demo = create_chatbot_interface()
    demo.launch(share=True)
```

## Running the Application

### Step 9: Start the Chatbot

Execute the script to launch the web interface:

```python
# In your Python environment or notebook
demo = create_chatbot_interface()
demo.launch()
```

The application will start a local web server and provide a URL (typically `http://localhost:7860`). Open this URL in your browser to interact with the chatbot.

## Usage Instructions

1. **Text Input**: Type your message in the text box and click "Send Text"
2. **Voice Input**: Click the microphone icon to record audio, then click "Send Audio"
3. **Clear Chat**: Use the "Clear Chat" button to reset the conversation
4. **Audio Response**: The chatbot's response will be played automatically and available for download

## Troubleshooting

### Memory Management

If you encounter memory issues:

```python
# Clear GPU cache periodically
torch.cuda.empty_cache()

# Use smaller batch sizes
# Consider using model quantization for lower memory footprint
```

### Common Issues

1. **Permission Errors**: If you see permission denied errors with `ldconfig`:
   ```bash
   sudo ldconfig
   ```

2. **Model Loading Failures**: Ensure your Hugging Face token is properly set:
   ```python
   from huggingface_hub import login
   login(token="YOUR_HF_TOKEN")
   ```

3. **Audio Processing Errors**: Check that `edge-tts` is properly installed:
   ```bash
   pip install --upgrade edge-tts
   ```

## Conclusion

You've successfully built an interactive chatbot that combines the power of Phi-3 Mini for text generation with Whisper for speech recognition. This application demonstrates:

- Loading and running large language models on GPU
- Integrating speech-to-text and text-to-speech capabilities
- Creating user-friendly interfaces with Gradio
- Managing conversation state and history

The chatbot can be extended with additional features like:
- Support for multiple languages
- Custom voice options
- Conversation memory persistence
- Integration with external APIs for real-time information

Remember to monitor GPU memory usage and optimize batch sizes based on your specific hardware configuration.
# Fine-tuning LLMs for Function Calling with xLAM Dataset

_Authored by: [Behrooz Azarkhalili](https://github.com/behroozazarkhalili)_

This guide demonstrates how to fine-tune language models for function calling capabilities using the **xLAM dataset** from Salesforce and the **QLoRA** (Quantized Low-Rank Adaptation) technique. You will learn to work with popular models like Llama 3, Qwen2, Mistral, and others.

**What is Function Calling?**
Function calling enables language models to interact with external tools and APIs by generating structured function invocations. Instead of just generating text, the model learns to call specific functions with the right parameters based on user requests.

**What You'll Learn:**
- **Data Processing**: How to format the xLAM dataset for function calling training
- **Model Fine-tuning**: Using QLoRA for memory-efficient training on consumer GPUs
- **Evaluation**: Testing the fine-tuned models with example prompts
- **Multi-model Support**: Working with different model architectures

**Key Benefits:**
- **Memory Efficient**: QLoRA enables training on 16-24GB GPUs
- **Production Ready**: Modular code with proper error handling
- **Flexible Architecture**: Easy to adapt for different models and datasets
- **Universal Support**: Works with Llama, Qwen, Mistral, Gemma, Phi, and more

**Hardware Requirements:**
- **GPU**: 16GB+ VRAM (24GB recommended for larger models)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space for models and datasets

**Software Dependencies:**
This guide will install required packages automatically, including:
- `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, `accelerate`

*For detailed methodology and results, see: [Function Calling: Fine-tuning Llama 3 and Qwen2 on xLAM](https://newsletter.kaitchup.com/p/function-calling-fine-tuning-llama)*

## Setup and Installation

First, install the required packages for function calling fine-tuning.

```bash
pip install --upgrade bitsandbytes peft trl python-dotenv
```

## 1. Basic Setup and Imports

Let's start with the essential imports and basic setup.

```python
import torch
import os
import warnings
from typing import Dict, Any, Optional, Tuple

# Set up GPU and suppress warnings for cleaner output
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 2. Hugging Face Authentication Setup

Next, set up authentication with HuggingFace Hub. This allows you to download models and datasets, and optionally upload your fine-tuned models.

```python
# Set up HuggingFace authentication
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file (optional)
load_dotenv()

# Authenticate with HuggingFace using token from .env file
hf_token = os.getenv('hf_api_key')
if hf_token:
    login(token=hf_token)
    print("✅ Successfully authenticated with HuggingFace!")
else:
    print("⚠️  Warning: HF_API_KEY not found in .env file")
    print("   You can still run the notebook, but won't be able to upload models")
```

## 3. Model Configuration Classes

Create two configuration classes to organize your settings:

1. **ModelConfig**: Stores model-specific settings like tokenizer configuration
2. **TrainingConfig**: Stores training parameters like learning rate and batch size

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model-specific settings."""
    model_name: str           # HuggingFace model identifier
    pad_token: str           # Padding token for the tokenizer
    pad_token_id: int        # Numerical ID for the padding token
    padding_side: str        # Side to add padding ('left' or 'right')
    eos_token: str          # End of sequence token
    eos_token_id: int       # End of sequence token ID
    vocab_size: int         # Vocabulary size
    model_type: str         # Model architecture type

@dataclass 
class TrainingConfig:
    """Configuration for training hyperparameters."""
    output_dir: str                    # Directory to save model checkpoints
    batch_size: int = 16              # Training batch size per device
    gradient_accumulation_steps: int = 8  # Steps to accumulate gradients
    learning_rate: float = 1e-4       # Learning rate for optimization
    max_steps: int = 1000             # Maximum training steps
    max_seq_length: int = 2048        # Maximum sequence length
    lora_r: int = 16                  # LoRA rank parameter
    lora_alpha: int = 16              # LoRA alpha scaling parameter
    lora_dropout: float = 0.05        # LoRA dropout rate
    save_steps: int = 250             # Steps between checkpoint saves
    logging_steps: int = 10           # Steps between log outputs
    warmup_ratio: float = 0.1         # Warmup ratio for learning rate
```

## 4. Automatic Model Configuration

Now, create a function that automatically detects the model's tokenizer settings and creates a proper configuration. It handles different model architectures (Llama, Qwen, Mistral, etc.) and their specific token requirements.

```python
from transformers import AutoTokenizer, AutoConfig

def auto_configure_model(model_name: str, custom_pad_token: str = None) -> ModelConfig:
    """
    Automatically configure any model by extracting information from its tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        custom_pad_token: Custom pad token if model doesn't have one
        
    Returns:
        ModelConfig: Complete model configuration
    """
    
    print(f"🔍 Loading model configuration: {model_name}")
    
    # Load tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_config = AutoConfig.from_pretrained(model_name)
    
    # Extract basic model info
    model_type = getattr(model_config, 'model_type', 'unknown')
    vocab_size = getattr(model_config, 'vocab_size', len(tokenizer.get_vocab()))
    
    print(f"📊 Model: {model_type}, vocab_size: {vocab_size:,}")
    
    # Get EOS token (required)
    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    
    if eos_token is None:
        raise ValueError(f"Model '{model_name}' missing EOS token")
    
    # Get or set pad token
    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.pad_token_id
    
    if pad_token is None:
        if custom_pad_token is None:
            raise ValueError(f"Model needs custom_pad_token. Use '<|eot_id|>' for Llama, '<|im_end|>' for Qwen")
        
        pad_token = custom_pad_token
        if pad_token in tokenizer.get_vocab():
            pad_token_id = tokenizer.get_vocab()[pad_token]
        else:
            tokenizer.add_special_tokens({'pad_token': pad_token})
            pad_token_id = tokenizer.pad_token_id
    
    print(f"✅ Configured - pad: '{pad_token}' (ID: {pad_token_id}), eos: '{eos_token}' (ID: {eos_token_id})")
    
    return ModelConfig(
        model_name=model_name,
        pad_token=pad_token,
        pad_token_id=pad_token_id,
        padding_side='left',  # Standard for causal LMs
        eos_token=eos_token,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        model_type=model_type
    )
```

Create a helper function to generate a training configuration with an automatic output directory.

```python
def create_training_config(model_name: str, **kwargs) -> TrainingConfig:
    """Create training configuration with automatic output directory."""
    # Create clean directory name from model name
    model_clean = model_name.split('/')[-1].replace('-', '_').replace('.', '_')
    default_output_dir = f"./{model_clean}_xLAM"
    
    config_dict = {'output_dir': default_output_dir, **kwargs}
    return TrainingConfig(**config_dict)

print("✅ Configuration system ready!")
print("💡 Supports Llama, Qwen, Mistral, Gemma, Phi, and more")
```

## 5. Hardware Detection and Setup

Detect your hardware capabilities and configure optimal settings. This function checks for bfloat16 support and sets up the best attention mechanism for your GPU.

```python
def setup_hardware_config() -> Tuple[torch.dtype, str]:
    """
    Automatically detect and configure hardware-specific settings.
    
    Returns:
        Tuple[torch.dtype, str]: compute_dtype and attention_implementation
    """
    print("🔍 Detecting hardware capabilities...")
    
    if torch.cuda.is_bf16_supported():
        print("✅ bfloat16 supported - using optimal precision")
        print("📦 Installing FlashAttention for better performance...")
        
        # Install FlashAttention for supported hardware
        os.system('pip install flash_attn --no-build-isolation')
        
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
        
        print("🚀 Configuration: bfloat16 + FlashAttention 2")
    else:
        print("⚠️  bfloat16 not supported - using float16 fallback")
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'  # Scaled Dot Product Attention
        
        print("🔄 Configuration: float16 + SDPA")
    
    return compute_dtype, attn_implementation

# Configure hardware settings
compute_dtype, attn_implementation = setup_hardware_config()
```

## 6. Tokenizer Setup Function

Now create a function to set up your tokenizer with the right configuration from your model settings.

```python
from transformers import AutoTokenizer

def setup_tokenizer(model_config: ModelConfig) -> AutoTokenizer:
    """
    Initialize and configure the tokenizer using model configuration.
    
    Args:
        model_config: Model configuration with all token information
        
    Returns:
        AutoTokenizer: Configured tokenizer with proper pad token settings
    """
    print(f"🔤 Loading tokenizer for {model_config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, use_fast=True)
    
    # Configure padding token using values from model_config
    tokenizer.pad_token = model_config.pad_token
    tokenizer.pad_token_id = model_config.pad_token_id
    tokenizer.padding_side = model_config.padding_side
    
    print(f"✅ Tokenizer configured - pad: '{model_config.pad_token}' (ID: {model_config.pad_token_id})")
    
    return tokenizer

print(f"📊 Hardware Configuration Complete:")
print(f"   • Compute dtype: {compute_dtype}")
print(f"   • Attention implementation: {attn_implementation}")
print(f"   • Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## 7. Dataset Processing

Now you'll work with the xLAM dataset from Salesforce. This dataset contains about 60,000 examples of function calling conversations that you'll use to train your model.

**Key Functions:**
- **`process_xlam_sample()`**: Converts a single dataset example into the training format with special tags (`<user>`, `<tools>`, `<calls>`) and EOS token
- **`load_and_process_xlam_dataset()`**: Loads the complete xLAM dataset (60K samples) from Hugging Face and processes all samples using multiprocessing for efficiency
- **`preview_dataset_sample()`**: Displays a formatted preview of a processed dataset sample for inspection with statistics

First, define the function to process a single sample.

```python
import json
import multiprocessing
from datasets import load_dataset, Dataset

def process_xlam_sample(row: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    Process a single xLAM dataset sample into training format.
    
    The format we create is:
    <user>[user query]</user>
    
    <tools>
    [tool definitions]
    </tools>
    
    <calls>
    [expected function calls]
    </calls>[EOS_TOKEN]
    """
    # Format user query
    formatted_query = f"<user>{row['query']}</user>\n\n"

    # Parse and format available tools
    try:
        parsed_tools = json.loads(row["tools"])
        tools_text = '\n'.join(str(tool) for tool in parsed_tools)
    except json.JSONDecodeError:
        tools_text = str(row["tools"])  # Fallback to raw string
    
    formatted_tools = f"<tools>{tools_text}</tools>\n\n"

    # Parse and format expected function calls
    try:
        parsed_answers = json.loads(row["answers"])
        answers_text = '\n'.join(str(answer) for answer in parsed_answers)
    except json.JSONDecodeError:
        answers_text = str(row["answers"])  # Fallback to raw string

    formatted_answers = f"<calls>{answers_text}</calls>"

    # Combine all parts with EOS token
    complete_text = formatted_query + formatted_tools + formatted_answers + tokenizer.eos_token

    # Update row with processed data
    row["query"] = formatted_query
    row["tools"] = formatted_tools
    row["answers"] = formatted_answers
    row["text"] = complete_text

    return row
```

Next, create the function to load and process the entire dataset.

```python
def load_and_process_xlam_dataset(tokenizer: AutoTokenizer, sample_size: Optional[int] = None) -> Dataset:
    """
    Load and process the complete xLAM dataset for function calling training.
    
    Args:
        tokenizer: Configured tokenizer for the model
        sample_size: Optional number of samples to use (None for full dataset)
        
    Returns:
        Dataset: Processed dataset ready for training
    """
    print("📊 Loading xLAM function calling dataset...")
    
    # Load the Salesforce xLAM dataset from Hugging Face
    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    
    print(f"📋 Original dataset size: {len(dataset):,} samples")
    
    # Sample dataset if requested (useful for testing)
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))
        print(f"🔬 Using sample size: {sample_size:,} samples")
    
    # Process all samples using multiprocessing for efficiency
    print("⚙️ Processing dataset samples into training format...")
    
    def process_batch(batch):
        """Process a batch of samples with the tokenizer."""
        processed_batch = []
        for i in range(len(batch['query'])):
            row = {
                'query': batch['query'][i],
                'tools': batch['tools'][i], 
                'answers': batch['answers'][i]
            }
            processed_row = process_xlam_sample(row, tokenizer)
            processed_batch.append(processed_row)
        
        # Convert to batch format
        return {
            'text': [item['text'] for item in processed_batch],
            'query': [item['query'] for item in processed_batch],
            'tools': [item['tools'] for item in processed_batch],
            'answers': [item['answers'] for item in processed_batch]
        }
    
    # Process the dataset
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=1000,  # Process in batches for efficiency
        num_proc=min(4, multiprocessing.cpu_count()),  # Use multiple cores
        desc="Processing xLAM samples"
    )
    
    print("✅ Dataset processing complete!")
    print(f"📊 Final dataset size: {len(processed_dataset):,} samples")
    print(f"🔤 Average text length: {sum(len(text) for text in processed_dataset['text']) / len(processed_dataset):,.0f} characters")
    
    return processed_dataset
```

Finally, create a function to preview a sample from the processed dataset.

```python
def preview_dataset_sample(dataset: Dataset, index: int = 0) -> None:
    """
    Display a formatted preview of a dataset sample for inspection.
    
    Args:
        dataset: The processed dataset
        index: Index of the sample to preview (default: 0)
    """
    if index >= len(dataset):
        print(f"❌ Index {index} is out of range. Dataset has {len(dataset)} samples.")
        return
    
    sample = dataset[index]
    
    print(f"📋 Dataset Sample Preview (Index: {index})")
    print("=" * 80)
    
    print(f"\n🔍 Raw Components:")
    print(f"Query: {sample['query'][:200]}{'...' if len(sample['query']) > 200 else ''}")
    print(f"Tools: {sample['tools'][:200]}{'...' if len(sample['tools']) > 200 else ''}")
    print(f"Answers: {sample['answers'][:200]}{'...' if len(sample['answers']) > 200 else ''}")
    
    print(f"\n📝 Complete Training Text:")
    print("-" * 40)
    print(sample['text'])
    print("-" * 40)
    
    print(f"\n📊 Sample Statistics:")
    print(f"   • Text length: {len(sample['text']):,} characters")
    print(f"   • Estimated tokens: ~{len(sample['text']) // 4:,} tokens")
    
    print("\n✅ Preview complete")
```

You have now set up the foundational components for fine-tuning LLMs for function calling. The next steps would involve loading a specific model, applying QLoRA configuration, and starting the training process.
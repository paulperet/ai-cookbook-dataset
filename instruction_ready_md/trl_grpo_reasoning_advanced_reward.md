# Advanced GRPO Fine-tuning for Mathematical Reasoning with Multi-Reward Training

_Authored by: [Behrooz Azarkhalili](https://github.com/behroozazarkhalili)_

This guide demonstrates **advanced GRPO (Group Relative Policy Optimization)** for mathematical reasoning using a comprehensive multi-reward training system. You will fine-tune a model on the GSM8K dataset with four specialized reward functions.

**Key Features:**
- **4 Reward Functions**: Format compliance, approximate matching, answer correctness, and number extraction.
- **Memory Efficient**: 4-bit quantization + LoRA for consumer GPUs.
- **Interactive Monitoring**: Real-time training metrics with trackio dashboard.
- **Structured Output**: Enforces step-by-step reasoning format.

The model learns to generate structured mathematical solutions with clear reasoning steps and accurate numerical answers.

## Prerequisites and Setup

First, install the required packages for GRPO training with memory-efficient techniques.

```bash
pip install transformers datasets trl bitsandbytes peft trackio
```

## 1. Verify GPU Environment

Check GPU availability and display hardware specifications for optimal training configuration.

```python
import torch

# Verify CUDA availability and display GPU specifications
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    # Display current GPU details for training optimization
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    # Provide guidance for enabling GPU in Colab
    print("⚠️  No GPU available. This notebook requires a GPU for efficient training.")
    print("In Colab: Runtime → Change runtime type → Hardware accelerator → GPU")
```

## 2. Import Core Libraries

Import essential libraries for GRPO training, model configuration, and experiment tracking.

```python
import trackio  # Experiment tracking dashboard
import re       # Regex patterns for reward functions

# GRPO training components
from trl import GRPOConfig, GRPOTrainer

# Model and tokenization
from transformers import (
    AutoModelForCausalLM,   # Causal language model loading
    AutoTokenizer,          # Text tokenization
    BitsAndBytesConfig,     # Quantization configuration
)

# Parameter-efficient fine-tuning
from peft import LoraConfig, get_peft_model, TaskType

# Dataset handling
from datasets import load_dataset

# Logging configuration
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx request logs that appear during trackio usage
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio_client").setLevel(logging.WARNING)
```

## 3. Select and Configure the Model

Choose a compact but capable model suitable for mathematical reasoning with memory constraints.

```python
# Select model optimized for instruction-following and reasoning
model_name = "Qwen/Qwen2.5-3B-Instruct"  # 3B parameter model balances capability and memory usage
max_seq_length = 2048                     # Token limit for mathematical problems (reduce if OOM)

print(f"Loading model: {model_name}")
print(f"Max sequence length: {max_seq_length}")
```

### 3.1 Configure 4-bit Quantization

Set up 4-bit quantization for significant memory reduction.

```python
# Configure 4-bit quantization for ~75% memory reduction
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit precision (vs 16-bit default)
    bnb_4bit_quant_type="nf4",           # NormalFloat4: optimal for neural network weights
    bnb_4bit_compute_dtype=torch.float16, # Use FP16 for forward/backward passes
    bnb_4bit_use_double_quant=True,      # Further quantize quantization constants
)

print("✅ 4-bit quantization configured")
print("   Memory reduction: ~75% vs FP16")
```

### 3.2 Load the Model and Tokenizer

Load the model with quantization and its corresponding tokenizer.

```python
# Load model with quantization and automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,      # Apply 4-bit quantization
    device_map="auto",                   # Auto-distribute across available GPUs/CPU
    trust_remote_code=True,              # Allow custom model code execution
    torch_dtype=torch.float16,           # Use FP16 for non-quantized operations
)

# Load corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True               # Allow custom tokenizer code
)

# Ensure tokenizer has proper padding token for batch processing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model loaded successfully!")
print(f"📊 Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"🧮 Quantized parameters: ~{sum(p.numel() for p in model.parameters() if hasattr(p, 'quant_type')) / 1e6:.1f}M")
```

## 4. Apply LoRA Configuration

Apply Low-Rank Adaptation to train only a small fraction of parameters while maintaining performance.

```python
# Configure LoRA for mathematical reasoning adaptation
lora_config = LoraConfig(
    r=16,                              # Rank: adaptation capacity (16 good for reasoning tasks)
    lora_alpha=32,                     # Scaling factor (typically 2x rank)
    target_modules=["q_proj", "v_proj"], # Focus on attention query/value for reasoning
    lora_dropout=0.1,                  # Regularization to prevent overfitting
    bias="none",                       # Skip bias adaptation for simplicity
    task_type=TaskType.CAUSAL_LM,      # Causal language modeling task
)

print("🔧 Applying LoRA adaptation to model...")

# Apply LoRA configuration to create trainable adapter
model = get_peft_model(model, lora_config)

# Display parameter efficiency
print("📊 LoRA Training Parameters Summary:")
model.print_trainable_parameters()  # Shows trainable vs total parameters
```

## 5. Prepare the GSM8K Dataset

Configure the GSM8K mathematical reasoning dataset with a structured output format for step-by-step solutions.

### 5.1 Define the Output Format

Define the special tokens and system prompt that will teach the model the desired reasoning structure.

```python
# Define structured output format for mathematical reasoning
reasoning_start = "<start_working_out>"   # Begin reasoning section
reasoning_end = "<end_working_out>"       # End reasoning section
solution_start = "<SOLUTION>"            # Begin final answer
solution_end = "</SOLUTION>"              # End final answer

# System prompt that teaches the model our desired reasoning structure
system_prompt = f"""You are a mathematical reasoning assistant.
When given a math problem:
1. Show your step-by-step work between {reasoning_start} and {reasoning_end}
2. Provide your final numerical answer between {solution_start} and {solution_end}
3. Be precise and show all calculation steps clearly."""

print("✅ Format tokens and system prompt defined")
print(f"   Reasoning format: {reasoning_start} ... {reasoning_end}")
print(f"   Solution format: {solution_start} ... {solution_end}")
```

### 5.2 Create Dataset Processing Functions

Create helper functions to convert the raw GSM8K dataset into the conversation format needed for GRPO training.

```python
# Dataset processing utilities
def extract_hash_answer(text):
    """Extract numerical answer from GSM8K format (#### marker)"""
    if "####" not in text:
        return None
    # GSM8K uses format: "Explanation... #### 42"
    return text.split("####")[1].strip()

def process_dataset_example(example):
    """Convert GSM8K example to conversation format for GRPO training"""
    question = example["question"]
    answer = extract_hash_answer(example["answer"])
    
    # Create conversation with system prompt for structured reasoning
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    return {
        "prompt": prompt,           # Input conversation
        "answer": answer,          # Ground truth for reward functions
    }

print("✅ Dataset processing functions defined")
```

### 5.3 Load and Process the Dataset

Load the GSM8K dataset and apply the formatting function to all examples.

```python
# Load and preprocess GSM8K training dataset
print("🔄 Loading GSM8K mathematical reasoning dataset...")
dataset = load_dataset("openai/gsm8k", "main", split="train")

# Apply conversation formatting to all examples
dataset = dataset.map(process_dataset_example)

print(f"✅ Dataset loaded and processed!")
print(f"📊 Training examples: {len(dataset):,}")
print(f"🎯 Sample question: {dataset[0]['prompt'][1]['content']}...")
print(f"🎯 Sample answer: {dataset[0]['answer']}")

# Show structure of first example for verification
print(f"\n📋 Example structure:")
print(f"   Prompt: {len(dataset[0]['prompt'])} messages (system + user)")
print(f"   Answer: {dataset[0]['answer']} (ground truth for rewards)")
```

## 6. Design the Multi-Reward System

Implement four complementary reward functions to evaluate different aspects of mathematical reasoning:
1. **Exact Format Matching**: Perfect structure compliance.
2. **Approximate Matching**: Partial credit for format elements.
3. **Answer Correctness**: Mathematical accuracy with graduated scoring.
4. **Number Extraction**: Ability to parse and output numerical results.

### 6.1 Compile Regex Patterns

First, compile regex patterns for efficient reward computation.

```python
# Compiled regex patterns for efficient reward computation
match_format = re.compile(
    rf"^[\s]{{0,}}"                      # Optional whitespace at start
    rf"{reasoning_start}.+?{reasoning_end}.*?"  # Reasoning section (non-greedy)
    rf"{solution_start}(.+?){solution_end}"     # Solution section with capture group
    rf"[\s]{{0,}}$",                     # Optional whitespace at end
    flags=re.MULTILINE | re.DOTALL       # Multi-line matching with . matching newlines
)

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", # Extract numbers from solution section
    flags=re.MULTILINE | re.DOTALL        # Flexible pattern matching
)
```

### 6.2 Reward Function 1: Exact Format Compliance

Award a high reward for perfect adherence to the complete structured output pattern.

```python
# Reward Function 1: Exact Format Compliance
def match_format_exactly(completions, **kwargs):
    """
    High reward (3.0) for perfect format adherence
    Ensures model learns the complete structured output pattern
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        # Check if response matches complete format pattern
        score = 3.0 if match_format.search(response) is not None else 0.0
        scores.append(score)
    return scores
```

### 6.3 Reward Function 2: Partial Format Credit

Provide graduated scoring for individual format elements to encourage learning components even if the output isn't perfect.

```python
# Reward Function 2: Partial Format Credit
def match_format_approximately(completions, **kwargs):
    """
    Graduated scoring for format elements
    Encourages learning individual components even if not perfect
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0
        
        # Award +0.5 for correct token count, -0.5 for wrong count
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5
        
        scores.append(score)
    return scores
```

### 6.4 Reward Function 3: Mathematical Accuracy

Score mathematical accuracy with graduated partial credit for answers that are close to the correct value.

```python
# Reward Function 3: Mathematical Accuracy
def check_answer_correctness(prompts, completions, answer, **kwargs):
    """
    Graduated scoring for mathematical accuracy:
    - 3.0: Exact match
    - 1.5: Within 10% (close answer)
    - 0.5: Within 20% (reasonable attempt)
    - -0.5: Wrong answer (penalty for incorrect math)
    """
    responses = [completion[0]["content"] for completion in completions]
    
    # Extract answers using format pattern
    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]
    
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:  # No extractable answer
            scores.append(0)
            continue
            
        # Exact string match gets full points
        if guess.strip() == true_answer.strip():
            scores.append(3.0)
        else:
            # Try numerical comparison for partial credit
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:      # Within 10%
                    scores.append(1.5)
                elif 0.8 <= ratio <= 1.2:    # Within 20%
                    scores.append(0.5)
                else:                         # Wrong answer
                    scores.append(-0.5)
            except (ValueError, ZeroDivisionError):
                scores.append(-0.5)           # Invalid numerical format
    
    return scores
```

### 6.5 Reward Function 4: Number Extraction Ability

Test the model's ability to extract numerical values from the solution section, focusing on parsing capability.

```python
# Reward Function 4: Number Extraction Ability  
def check_numbers_extraction(prompts, completions, answer, **kwargs):
    """
    Tests the model's ability to extract numerical values from solution sections
    Complementary to exact format matching - focuses on parsing capability
    """
    responses = [completion[0]["content"] for completion in completions]
    
    # Extract numbers from solution sections using number pattern
    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]
    
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:  # No extractable number
            scores.append(0)
            continue
            
        try:
            # Simple numerical equality check
            true_val = float(true_answer.strip())
            guess_val = float(guess.strip())
            # Binary scoring: correct (1.5) or incorrect (0)
            scores.append(1.5 if guess_val == true_val else 0.0)
        except (ValueError, TypeError):
            scores.append(0)  # Invalid number format
    
    return scores
```

## 7. Configure GRPO Training

Set up the training parameters optimized for mathematical reasoning with memory constraints.

```python
# Configure GRPO training parameters for mathematical reasoning
training_args = GRPOConfig(
    # Learning parameters optimized for reasoning tasks
    learning_rate=5e-6,              # Conservative LR to prevent destabilizing reasoning
    
    # Memory-efficient batch configuration
    per_device_train_batch_size=2,   # Small batch for GPU memory constraints
    gradient_accumulation_steps=8,   # Effective batch size = 2 * 8 = 16
    
    # Sequence length limits for mathematical problems
    max_prompt_length=1024,          # Sufficient for complex word problems
    max_completion_length=1024,      # Room for detailed step-by-step reasoning
    
    # Training duration and monitoring
    max_steps=10,                    # Short demo run (increase to 500+ for production)
    logging_steps=1,                 # Log metrics every step for close monitoring
    
    # Stability and output configuration
    output_dir="./trl_grpo_outputs",
    max_grad_norm=0.1,               # Aggressive gradient clipping for stable training
    report_to="trackio",                # use trackio for experiment tracking (instead of wandb/tensorboard)
)
```

## 8. Initialize Experiment Tracking with Trackio

Create a unique run name and initialize the trackio experiment tracker to monitor training.

```python
# Create unique run name with timestamp to ensure fresh tracking
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"qwen2.5-3b-gsm8k-grpo-{timestamp}"

# Initialize trackio experiment tracking with unique run name
trackio.init(
    project="GRPO-Mathematical-Reasoning",  # Project name for organization
    name=run_name,                         # Unique run identifier with timestamp
    config={
        # Model and dataset configuration
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "dataset": "GSM8K", 
        "technique": "GRPO + LoRA + 4-bit",
        
        # Training hyperparameters
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        "max_steps": training_args.max_steps,
        
        # LoRA configuration
        "lora_r": 16,
        "lora_alpha": 32,
        
        # GRPO-specific settings
        "num_generations": training_args.num_generations,  # Default: 8 generations per step
        "max_prompt_length": training_args.max_prompt_length,
        "max_completion_length": training_args.max_completion_length,
        
        # Reward
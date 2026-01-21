# Fine-Tuning Phi-4-Mini-Reasoning for Medical Reasoning with Microsoft Olive

This guide walks you through fine-tuning the `Phi-4-mini-reasoning` model to enhance its reasoning capabilities for medical scenarios. We'll use a specialized medical reasoning dataset and the Microsoft Olive framework for efficient fine-tuning and conversion, culminating in inference using ONNX Runtime GenAI.

## Prerequisites

This tutorial assumes you are using a GPU-equipped environment (an A100 is recommended). Ensure you have sufficient disk space for the model and datasets.

## Step 1: Environment Setup

Begin by installing the required Python packages.

```bash
pip install datasets
pip install git+https://github.com/microsoft/Olive.git
pip install transformers==4.49.0
pip install protobuf==3.20.3 -U
pip install onnxruntime-genai-cuda
pip install bitsandbytes
pip install optimum
```

## Step 2: Download the Base Model

Download the `Phi-4-mini-reasoning` model from Hugging Face to your local directory. Replace the placeholders with your specific model ID and desired local path.

```bash
huggingface-cli download {Phi-4-mini-reasoning hugging face id} --local-dir {Your Phi-4-mini-reasoning model location}
```

## Step 3: Prepare the Training Dataset

We will use the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset to teach the model medical Chain-of-Thought reasoning.

First, import the necessary library and define a formatting function.

```python
from datasets import load_dataset

prompt_template = """<|user|>{}<|end|><|assistant|><think>{}</think>{}"""

def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = prompt_template.format(input, cot, output) + "<|end|>"
        texts.append(text)
    return {"text": texts}
```

Now, load the English split of the dataset, apply the formatting function, and save it as a JSON Lines file.

```python
# Load the dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train", trust_remote_code=True)

# Format the prompts
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=["Question", "Complex_CoT", "Response"])

# Save the processed dataset
dataset.to_json("en_dataset.jsonl")
```

## Step 4: Fine-Tune the Model with LoRA

We'll use Microsoft Olive to perform parameter-efficient fine-tuning via LoRA (Low-Rank Adaptation). This command fine-tunes the model for 100 steps, using the first 16,000 samples for training and the next 3,700 for evaluation.

```bash
olive finetune \
    --method lora \
    --model_name_or_path "./phi-4-mini-reasoning" \
    --trust_remote_code \
    --data_name json \
    --data_files ./en_dataset.jsonl \
    --train_split "train[:16000]" \
    --eval_split "train[16000:19700]" \
    --text_field "text" \
    --max_steps 100 \
    --logging_steps 10 \
    --output_path models/phi-4-mini-reasoning/ft \
    --log_level 1
```

Upon successful completion, the fine-tuned adapter weights will be saved in the specified `output_path`.

## Step 5: Convert the Model to ONNX Format

To enable efficient inference with ONNX Runtime, we need to convert the Hugging Face model (with its LoRA adapter) into an ONNX graph.

First, capture the model's computation graph.

```bash
olive capture-onnx-graph \
    --model_name_or_path "./phi-4-mini-reasoning" \
    --adapter_path models/phi-4-mini-reasoning/ft/adapter \
    --use_model_builder \
    --output_path models/phi-4-mini-reasoning/onnx \
    --log_level 1
```

Next, generate the ONNX adapter file which packages the model and its weights for runtime.

```bash
olive generate-adapter \
    --model_name_or_path models/phi-4-mini-reasoning/onnx \
    --output_path models/phi-4-mini-reasoning/adapter-onnx \
    --log_level 1
```

The final, runnable ONNX model will be located in the `adapter-onnx` directory.

## Step 6: Run Inference with the Fine-Tuned Model

Now, let's test our fine-tuned model using ONNX Runtime GenAI. We'll load the model, activate the medical reasoning adapter, and generate a response to a sample medical question.

First, import the necessary libraries and load the model.

```python
import onnxruntime_genai as og

# Define the path to the ONNX model
model_folder = "./models/phi-4-mini-reasoning/adapter-onnx/model/"

# Load the model
model = og.Model(model_folder)
```

Load the specific LoRA adapter we trained for medical reasoning.

```python
adapters = og.Adapters(model)
adapters.load('./models/phi-4-mini-reasoning/adapter-onnx/model/adapter_weights.onnx_adapter', "en_medical_reasoning")
```

Prepare the tokenizer and set up the generation parameters.

```python
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Configure text generation options
search_options = {
    'max_length': 200,
    'past_present_share_buffer': False,
    'temperature': 1,
    'top_k': 1
}
```

Define your medical question and format it into the prompt template the model expects.

```python
prompt_template = """<|user|>{}<|end|><|assistant|><think>"""

question = """A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. Given her vital signs of pulse 110/min, respirations 22/min, and blood pressure 90/65 mm Hg, along with the presence of a 5-cm deep stab wound at the upper border of the 8th rib in the left midaxillary line, which anatomical structure in her chest is most likely to be injured?"""

prompt = prompt_template.format(question)
```

Finally, tokenize the input, configure the generator, activate the adapter, and produce the output.

```python
# Tokenize the input prompt
input_tokens = tokenizer.encode(prompt)

# Set up the generator
params = og.GeneratorParams(model)
params.set_search_options(**search_options)
generator = og.Generator(model, params)

# Activate our fine-tuned adapter
generator.set_active_adapter(adapters, "en_medical_reasoning")

# Feed the tokens to the generator
generator.append_tokens(input_tokens)

# Generate the response token by token
while not generator.is_done():
    generator.generate_next_token()
    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)
```

The model will generate a reasoned response, demonstrating its enhanced capability to analyze the medical scenario.

## Conclusion

You have successfully fine-tuned the `Phi-4-mini-reasoning` model for medical reasoning tasks. The process involved:
1.  Setting up the environment and downloading the base model.
2.  Preparing a specialized medical reasoning dataset.
3.  Performing efficient fine-tuning using LoRA via Microsoft Olive.
4.  Converting the model to the optimized ONNX format.
5.  Running inference to see the model's improved medical reasoning.

This fine-tuned model can now be integrated into applications requiring advanced medical question-answering and reasoning.
# Fine-tuning Phi-3 with Apple MLX Framework

This guide walks you through fine-tuning Microsoft's Phi-3-mini-4k-instruct model using the Apple MLX framework. You'll use LoRA (Low-Rank Adaptation) for efficient fine-tuning, test the adapted model, merge the adapters, and finally deploy a quantized version with Ollama.

## Prerequisites

Ensure you have the MLX framework installed. If not, you can install it via pip:

```bash
pip install mlx-lm
```

You will also need the example dataset. Download it from the provided [link](../../code/04.Finetuning/mlx/) and place all `.jsonl` files in a `data` directory within your project folder.

## Step 1: Prepare Your Data

The MLX framework expects training data in `jsonl` format, where each line is a JSON object with a `"text"` key. The text must follow the Phi-3 chat template.

**Example `jsonl` entry:**
```json
{"text": "<|user|>\nWhen were iron maidens commonly used? <|end|>\n<|assistant|> \nIron maidens were never commonly used <|end|>"}
```

**Important Notes:**
1.  The example uses data from TruthfulQA, which is limited. For best results, prepare a larger, high-quality dataset relevant to your specific task.
2.  Ensure your data is formatted with the correct `<|user|>`, `<|assistant|>`, and `<|end|>` tokens.

Your project structure should look like this:
```
your_project/
├── data/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── (your scripts and configs)
```

## Step 2: Run LoRA Fine-tuning

You can start fine-tuning directly from the command line with default parameters.

```bash
python -m mlx_lm.lora --model microsoft/Phi-3-mini-4k-instruct --train --data ./data --iters 1000
```

This command will:
*   Load the base `Phi-3-mini-4k-instruct` model.
*   Apply LoRA to the model's layers.
*   Train for 1000 iterations using the data in your `./data` folder.
*   Save the trained adapter weights to a default `adapters` directory.

### Advanced Configuration with YAML

For more control, create a configuration file (e.g., `lora_config.yaml`). This allows you to adjust hyperparameters like learning rate, batch size, and which layers LoRA is applied to.

```yaml
model: "microsoft/Phi-3-mini-4k-instruct"
train: true
data: "data"
seed: 0
lora_layers: 32
batch_size: 1
iters: 1000
val_batches: 25
learning_rate: 1e-6
steps_per_report: 10
steps_per_eval: 200
adapter_path: "adapters"
save_every: 1000
test: false
test_batches: 100
max_seq_length: 2048
grad_checkpoint: true
lora_parameters:
  keys: ["o_proj","qkv_proj"]
  rank: 64
  scale: 1
  dropout: 0.1
```

To run fine-tuning with your custom configuration:

```bash
python -m mlx_lm.lora --config lora_config.yaml
```

## Step 3: Test Your Fine-tuned Adapter

After training, you can generate text using your fine-tuned model by specifying the `--adapter-path`.

**Run the fine-tuned model:**
```bash
python -m mlx_lm.generate \
  --model microsoft/Phi-3-mini-4k-instruct \
  --adapter-path ./adapters \
  --max-token 2048 \
  --prompt "Why do chameleons change colors? " \
  --eos-token "<|end|>"
```

**Compare with the original base model:**
```bash
python -m mlx_lm.generate \
  --model microsoft/Phi-3-mini-4k-instruct \
  --max-token 2048 \
  --prompt "Why do chameleons change colors? " \
  --eos-token "<|end|>"
```

Compare the outputs to evaluate the impact of your fine-tuning.

## Step 4: Merge Adapters into a New Model

To create a standalone model that includes your fine-tuned weights, merge the LoRA adapters with the base model. This creates a new model directory (default: `./mlx_model`).

```bash
python -m mlx_lm.fuse --model microsoft/Phi-3-mini-4k-instruct
```

## Step 5: Deploy a Quantized Model with Ollama

To run your model efficiently, you can convert it to the GGUF format and serve it via Ollama.

### 1. Convert to GGUF Format

First, set up `llama.cpp` for conversion.

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```

Now, convert your merged MLX model. You'll need to download the `tokenizer.model` file from the [Phi-3-mini-4k-instruct Hugging Face page](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) and place it in your merged model directory.

```bash
# Replace 'Your_merged_model_path' with the path to your fused model (e.g., ./mlx_model)
python convert.py 'Your_merged_model_path' --outfile phi-3-mini-ft.gguf --outtype f16
```
**Note:** The converter supports `f32`, `f16`, and `q8_0` (INT8) quantization. Use `--outtype q8_0` for an 8-bit quantized model.

### 2. Create an Ollama Model File

Create a file named `Modelfile` with the following content:

```txt
FROM ./phi-3-mini-ft.gguf
PARAMETER stop "<|end|>"
```

### 3. Create and Run the Ollama Model

Finally, create a new model in Ollama and run it.

```bash
ollama create phi3ft -f Modelfile
ollama run phi3ft "Why do chameleons change colors?"
```

Congratulations! You have successfully fine-tuned a Phi-3 model using the MLX framework, tested it, and deployed it as a quantized model with Ollama.
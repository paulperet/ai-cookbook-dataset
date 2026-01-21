# Fine-Tuning Phi-3.5-Vision: A Practical Guide

This guide provides a step-by-step tutorial for fine-tuning the Phi-3.5-Vision multimodal model using Hugging Face libraries. You will learn how to set up your environment, prepare your data, and execute training across different hardware configurations.

## Prerequisites & Setup

Before you begin, ensure you have the correct directory and environment.

### 1. Navigate to the Code Directory
All scripts referenced in this guide are located in the `vision_finetuning` folder.
```bash
cd code/03.Finetuning/vision_finetuning
```

### 2. Create and Activate a Conda Environment
Set up a dedicated Python environment to manage dependencies.
```bash
conda create -n phi3v python=3.10
conda activate phi3v
```

### 3. Install Core Dependencies
Install PyTorch with CUDA support and other required libraries.
```bash
# Install PyTorch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other required packages
pip install -r requirements.txt
```

### 4. Install Optional Performance Libraries
Depending on your GPU architecture, you can install optional libraries to accelerate training.

*   **Flash Attention (for Ampere+ GPUs like A100, H100):** Significantly speeds up attention computation.
    ```bash
    pip install ninja
    MAX_JOBS=32 pip install flash-attn==2.4.2 --no-build-isolation
    ```
*   **Bitsandbytes (for Turing+ GPUs like RTX 8000):** Enables 4-bit quantization for QLoRA, reducing memory footprint.
    ```bash
    pip install bitsandbytes==0.43.1
    ```

---

## Quick Start: Running Example Scripts

We provide example scripts for different vision-language tasks. The minimal tested hardware is **4x RTX 8000 (48GB VRAM per GPU)**.

1.  **Document VQA (DocVQA):** Fine-tune on a question-answering task for documents.
    ```bash
    torchrun --nproc_per_node=4 finetune_hf_trainer_docvqa.py
    ```

2.  **Natural Language for Visual Reasoning (NLVR2):** Demonstrates Phi-3.5-Vision's support for **multi-image inputs**.
    ```bash
    torchrun --nproc_per_node=8 finetune_hf_trainer_nlvr2.py
    ```

---

## Preparing Your Custom Dataset

To fine-tune on your own data, you must convert it into the required format. We use a subset of the UCF-101 video classification dataset as a working example.

### Step 1: Convert Your Data
Run the conversion script. It will process your raw data and output images and annotation files.
```bash
python convert_ucf101.py --out_dir /path/to/converted_ucf101
```

### Step 2: Understand the Output Structure
The script creates a standardized directory and file structure.
```bash
/path/to/converted_ucf101
├── images
│   ├── test
│   ├── train
│   └── val
├── ucf101_test.jsonl
├── ucf101_train.jsonl
└── ucf101_val.jsonl
```
*   The `images/` subdirectories contain the image files for each split.
*   The `.jsonl` files contain the annotations, with one JSON object per line.

### Step 3: Format Your Annotations
Each line in the `.jsonl` file must be a dictionary with a specific structure. The `conversations` field is a list, allowing for multi-turn dialogue if your data supports it.

**Example Annotation (Single Turn):**
```json
{
  "id": "val-0000000300",
  "source": "ucf101",
  "conversations": [
    {
      "images": [
        "val/BabyCrawling/v_BabyCrawling_g21_c04.0.jpg",
        "val/BabyCrawling/v_BabyCrawling_g21_c04.1.jpg"
      ],
      "user": "Classify the video into one of the following classes: ApplyEyeMakeup, ApplyLipstick, Archery, BabyCrawling, BalanceBeam, BandMarching, BaseballPitch, Basketball, BasketballDunk, BenchPress.",
      "assistant": "BabyCrawling"
    }
  ]
}
```

### Step 4: Launch Training on Your Data
Once your data is converted, you can start the fine-tuning process.
```bash
torchrun --nproc_per_node=4 finetune_hf_trainer_ucf101.py --data_dir /path/to/converted_ucf101
```

---

## Hardware-Specific Configuration Guide

Your choice of fine-tuning strategy (full fine-tuning, LoRA, QLoRA) depends largely on your available GPU hardware.

### Scenario 1: High-End Data Center GPUs (A100/H100)
For best performance, use **full fine-tuning** with Flash Attention and BF16 precision.
```bash
torchrun --nproc_per_node=8 \
  finetune_hf_trainer_hateful_memes.py \
  --output_dir <output_dir> \
  --batch_size 64 \
  --use_flash_attention \
  --bf16
```

### Scenario 2: V100 GPUs (e.g., Azure `Standard_ND40rs_v2`)
Full fine-tuning is possible but will be slower due to the lack of Flash Attention support. Use FP16 mixed precision.
```bash
torchrun --nproc_per_node=8 \
  finetune_hf_trainer_hateful_memes.py \
  --output_dir <output_dir> \
  --batch_size 64
```

### Scenario 3: Consumer or Limited GPUs
When memory is constrained, **LoRA** (Low-Rank Adaptation) is the recommended approach.

*   **Standard LoRA:**
    ```bash
    torchrun --nproc_per_node=2 \
      finetune_hf_trainer_hateful_memes.py \
      --output_dir <output_dir> \
      --batch_size 64 \
      --use_lora
    ```
*   **QLoRA (4-bit Quantized LoRA for Turing+ GPUs):** Further reduces memory usage.
    ```bash
    torchrun --nproc_per_node=2 \
      finetune_hf_trainer_hateful_memes.py \
      --output_dir <output_dir> \
      --batch_size 64 \
      --use_lora \
      --use_qlora
    ```

---

## Hyperparameter Tuning & Expected Results

The following tables provide suggested hyperparameters and resulting accuracy for reference. **Note:** Results for DocVQA and Hateful Memes are based on the previous Phi-3-Vision model and will be updated for Phi-3.5-Vision.

### NLVR2 Fine-Tuning
**Command Template:**
```bash
torchrun --nproc_per_node=4 \
  finetune_hf_trainer_nlvr2.py \
  --bf16 --use_flash_attention \
  --batch_size 64 \
  --output_dir <output_dir> \
  --learning_rate <lr> \
  --num_train_epochs <epochs>
```

| Training Method      | Frozen Vision Model | Data Type | Batch Size | Learning Rate | Epochs | Accuracy |
| :------------------- | :-----------------: | :-------: | :--------: | :-----------: | :----: | :------: |
| Full Fine-Tuning     |                     |   BF16    |     64     |     1e-5      |   3    |  89.40   |
| Full Fine-Tuning     |         ✓           |   BF16    |     64     |     2e-5      |   2    |  89.20   |
| LoRA                 |         *Results Coming Soon*         |           |            |               |        |          |

### DocVQA Fine-Tuning (Phi-3-Vision Results)
**Command Template:**
```bash
torchrun --nproc_per_node=4 \
  finetune_hf_trainer_docvqa.py \
  --full_train \
  --bf16 --use_flash_attention \
  --batch_size 64 \
  --output_dir <output_dir> \
  --learning_rate <lr> \
  --num_train_epochs <epochs>
```

| Training Method      | Data Type | LoRA Rank | LoRA Alpha | Batch Size | Learning Rate | Epochs |  ANLS  |
| :------------------- | :-------: | :-------: | :--------: | :--------: | :-----------: | :----: | :----: |
| Full Fine-Tuning     |   BF16    |     -     |     -      |     64     |     5e-6      |   2    | 83.65  |
| Full Fine-Tuning     |   FP16    |     -     |     -      |     64     |     5e-6      |   2    | 82.60  |
| Frozen Image Model   |   BF16    |     -     |     -      |     64     |     1e-4      |   2    | 79.19  |
| Frozen Image Model   |   FP16    |     -     |     -      |     64     |     1e-4      |   2    | 78.74  |
| LoRA                 |   BF16    |    32     |     16     |     64     |     2e-4      |   2    | 82.46  |
| LoRA                 |   FP16    |    32     |     16     |     64     |     2e-4      |   2    | 82.34  |
| QLoRA                |   BF16    |    32     |     16     |     64     |     2e-4      |   2    | 81.85  |
| QLoRA                |   FP16    |    32     |     16     |     64     |     2e-4      |   2    | 81.85  |

### Hateful Memes Fine-Tuning (Phi-3-Vision Results)
**Command Template:**
```bash
torchrun --nproc_per_node=4 \
  finetune_hf_trainer_hateful_memes.py \
  --bf16 --use_flash_attention \
  --batch_size 64 \
  --output_dir <output_dir> \
  --learning_rate <lr> \
  --num_train_epochs <epochs>
```

| Training Method      | Data Type | LoRA Rank | LoRA Alpha | Batch Size | Learning Rate | Epochs | Accuracy |
| :------------------- | :-------: | :-------: | :--------: | :--------: | :-----------: | :----: | :------: |
| Full Fine-Tuning     |   BF16    |     -     |     -      |     64     |     5e-5      |   2    |  86.4    |
| Full Fine-Tuning     |   FP16    |     -     |     -      |     64     |     5e-5      |   2    |  85.4    |
| Frozen Image Model   |   BF16    |     -     |     -      |     64     |     1e-4      |   3    |  79.4    |
| Frozen Image Model   |   FP16    |     -     |     -      |     64     |     1e-4      |   3    |  78.6    |
| LoRA                 |   BF16    |    128    |    256     |     64     |     2e-4      |   2    |  86.6    |
| LoRA                 |   FP16    |    128    |    256     |     64     |     2e-4      |   2    |  85.2    |
| QLoRA                |   BF16    |    128    |    256     |     64     |     2e-4      |   2    |  84.0    |
| QLoRA                |   FP16    |    128    |    256     |     64     |     2e-4      |   2    |  83.8    |

---

## Performance Benchmarks (Phi-3-Vision)

Benchmarks were performed on the DocVQA dataset. Throughput (images/second) and memory usage are key metrics for planning your training.

### 8x A100-80GB (Ampere) Performance

| Training Method      | Nodes | GPUs | Flash Attn | Effective Batch Size | Throughput (img/s) | Speedup | Peak GPU Mem (GB) |
| :------------------- | :---: | :--: | :--------: | :------------------: | :----------------: | :-----: | :---------------: |
| Full Fine-Tuning     |   1   |  8   |            |          64          |       5.041        |   1x    |        ~42        |
| Full Fine-Tuning     |   1   |  8   |     ✓      |          64          |       8.657        |  1.72x  |        ~36        |
| Full Fine-Tuning     |   2   |  16  |     ✓      |          64          |       16.903       |  3.35x  |        ~29        |
| Full Fine-Tuning     |   4   |  32  |     ✓      |          64          |       33.433       |  6.63x  |        ~26        |
| Frozen Image Model   |   1   |  8   |            |          64          |       17.578       |  3.49x  |        ~29        |
| Frozen Image Model   |   1   |  8   |     ✓      |          64          |       31.736       |  6.30x  |        ~27        |
| LoRA                 |   1   |  8   |            |          64          |       5.591        |  1.11x  |        ~50        |
| LoRA                 |   1   |  8   |     ✓      |          64          |       12.127       |  2.41x  |        ~16        |
| QLoRA                |   1   |  8   |            |          64          |       4.831        |  0.96x  |        ~32        |
| QLoRA                |   1   |  8   |     ✓      |          64          |       10.545       |  2.09x  |        ~10        |

### 8x V100-32GB (Volta) Performance

| Training Method      | Nodes | GPUs | Flash Attn | Effective Batch Size | Throughput (img/s) | Speedup | Peak GPU Mem (GB) |
| :------------------- | :---: | :--: | :--------: | :------------------: | :----------------: | :-----: | :---------------: |
| Full Fine-Tuning     |   1   |  8   |            |          64          |       2.462        |   1x    |        ~32        |
| Full Fine-Tuning     |   2   |  16  |            |          64          |       4.182        |  1.70x  |        ~32        |
| Full Fine-Tuning     |   4   |  32  |            |          64          |       5.465        |  2.22x  |        ~32        |
| Frozen Image Model   |   1   |  8   |            |          64          |       8.942        |  3.63x  |        ~27        |
| LoRA                 |   1   |  8   |            |          64          |       2.807        |  1.14x  |        ~30        |

---

## Known Issues & Limitations

1.  **Flash Attention & Precision:** Flash Attention cannot be used with FP16 precision. BF16 is required and is recommended for all GPUs that support it (Ampere architecture and newer).
2.  **Training Interruption:** The current scripts do not support saving intermediate checkpoints or resuming training from a checkpoint. Ensure you have a stable environment for long-running jobs.

---
**Next Steps:** With your environment set up and data prepared, you are ready to launch your first fine-tuning job. Start with the Quick Start examples to validate your setup before moving to your custom dataset.
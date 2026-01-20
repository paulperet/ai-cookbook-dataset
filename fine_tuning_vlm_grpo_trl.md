# Post training a VLM for reasoning with GRPO using TRL

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

🚨 **WARNING**: This notebook is resource-intensive and requires substantial computational power. If you're running this in Colab, it will utilize an A100 GPU.

In this recipe, we'll demonstrate how to post-train a [Vision Language Model (VLM)](https://huggingface.co/blog/vlms-2025) using [GRPO](https://huggingface.co/docs/trl/grpo_trainer) for adding reasoning capabilities to a VLM using the Hugging Face ecosystem, specifically with the [Transformer Reinforcement Learning library (trl)](https://huggingface.co/docs/trl/index).


We'll be fine-tuning [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) using a subset of the [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) dataset. This dataset includes images with problem descriptions along with their solution and thinking trace to reach that solution. We'll leverage this data format, along with the GRPO reward functions, to teach the model how to reason to reach the solution.



## 1. Install Dependencies

Let's start by installing the essential libraries we'll need for fine-tuning.
We'll install `trl` from source, as the VLM GRPO trainer hasn't been included in an official release at the time of writing.



```python
!pip install -U -q git+https://github.com/huggingface/trl.git peft math_verify qwen-vl-utils[decord]
```

Authenticate using your Hugging Face 🤗 account to save and share the trained model.


```python
from huggingface_hub import login

login()
```

## 2. Load Dataset 📁

We leverage [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) for this recipe. This dataset contains 8k multimodal RL training examples focused on math reasoning. This data was created using GPT4o and includes `image`, `problem`, `solution`, `original question` and `original answer` for each sample. It was created in [this project](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal).

For our particular case where we want the model to learn to reason using images, we use `image` and  `problem` as input and `solution` as output.

For this educational resource, we'll only use 5% of the dataset and divide it into train and test sets to make it faster to train. In a real training, we'd use the full dataset.

We'll load the dataset and divide it.


```python
from datasets import load_dataset

dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
dataset = load_dataset(dataset_id, split='train[:5%]')

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']
```

Let's check the structure of the dataset.


```python
print(train_dataset)
```

    Dataset({
        features: ['image', 'problem', 'solution', 'original_question', 'original_answer'],
        num_rows: 307
    })


Let's check one sample:


```python
print(train_dataset[0])
```

In addition to the `problem` and `image` columns, we also include a custom system prompt to tell the model how we'd like the generation.

The system prompt is extracted from DeepSeek R1. Refer to [this previous recipe](https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl) for more details.

We convert the dataset samples into conversation samples, including the system prompt and one image and problem description per sample, since this is how the GRPO trainer expects them.

We also set `padding_side="left"` to ensure that generated completions during training are concatenated directly after the prompt, which is essential for GRPO to correctly compare token-level probabilities between preferred and rejected responses.


```python
from transformers import AutoProcessor

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["problem"]},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {
        "prompt": prompt,
        "image": example["image"],
    }

train_dataset = train_dataset.map(make_conversation)
```

    You have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it automatically. Loading from `preprocessor.json` will be removed in v5.0.


Let's take a look at a converted example:


```python
print(train_dataset[0]['prompt'])
```

    <|im_start|>system
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer><|im_end|>
    <|im_start|>user
    <|vision_start|><|image_pad|><|vision_end|>Based on the image, determine the constant term after combining all the polynomial expressions representing the side lengths of the triangle. Choose the correct answer from the options provided.
    
    Choices:
    A. 3
    B. 5
    C. 8
    D. 13<|im_end|>
    <|im_start|>assistant
    


We'll remove the the columns that we don't need for training.


```python
train_dataset
```




    Dataset({
        features: ['image', 'problem', 'solution', 'original_question', 'original_answer', 'prompt'],
        num_rows: 307
    })



We can check that the columns are now gone.


```python
train_dataset = train_dataset.remove_columns(['problem', 'original_question', 'original_answer'])
print(train_dataset)
```

    Dataset({
        features: ['image', 'solution', 'prompt'],
        num_rows: 307
    })


## 3. Post-Training the VLM Using GRPO

The diagram below highlights the main differences between **PPO** (Proximal Policy Optimization) and **GRPO** (Group Relative Policy Optimization), specifically the removal of the value model in GRPO. For more detailed information on the key differences, you can refer to this [further explanation](https://www.philschmid.de/deepseek-r1).

To implement the training pipeline, we leverage trl, Hugging Face's library for reinforcement learning, which provides a streamlined interface and built-in support for key training algorithms. In our case, we use the `GRPOConfig` and `GRPOTrainer` classes. A crucial step in this process is defining custom reward functions that guide the model's behavior and help it align with our specific objectives.

But first, let's load the model. In this case, we use [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen/Qwen2.5-VL-3B-Instruct), a powerful VLM developed by [Qwen](https://huggingface.co/Qwen). For better results, it would be important to consider models with a larger number of parameters.

Others examples of VLM projects that include reasoning capabilities are:


* [GLM-4.1V-9B-Thinking](https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking)

* [VLM-R1 models](https://huggingface.co/collections/omlab/vlm-r1-models-67b7352db15c19d57157c348)

* [R1-V](https://github.com/StarsfieldAI/R1-V)



### 3.1 Loading the Baseline Model

Let's load the baseline model first. As previously introduced, `Qwen/Qwen2.5-VL-3B-Instruct`.



```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


### 3.2 Configuring LoRA

We'll leverage LoRA for training the model, so let's configure it.



```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()
```

    trainable params: 1,843,200 || all params: 3,756,466,176 || trainable%: 0.0491


### 3.3 Loading Reward Functions



For the reward component of the system, we can use either pretrained reward models or reward functions defined directly in code. For training, the DeepSeek-R1 authors used an accuracy-based reward model that evaluates whether the response is correct, alongside a format-based reward that ensures the model places its reasoning process between `<think> </think>` tags. You can find more details [here](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py). We can simply define and implement these reward functions as generic Python functions.

In this case, we will utilize the following reward functions, directly extracted from the Open R1 [implementation](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py):

1. **Format Enforcement:** Ensures that the generation follows a specific format using `<think> </think> <answer> </answer>` tags for reasoning.  


```python
import re
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards
```

2. **Solution Accuracy:** Verifies whether the solution to the problem is correct, comparing it to the `solution` column in the dataset.


```python
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from typing import Optional

def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.
    """
    rewards = []

    for completion, sol in zip(completions, solution):
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(completion.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards
```

### 3.4 Configuring GRPO Training Parameters

Next, let's configure the training parameters for GRPO. We recommend experimenting with the `max_completion_length`, `num_generations`, and `max_prompt_length` parameters.

It'd be interesting to play with the `max_completion_length`, `num_generations`, and `max_prompt_length` params in order to find the best training combination.

The parameter selection has been adjusted to fit within the hardware limitations of a Google Colab session. To observe the full potential of reward improvements, especially in the second objective function, and to further improve the model's reasoning capabilities in a real-world scenario, a more ambitious setup would be required. This would involve larger models, an increased number of generations, and a high-quality, diverse dataset.


```python
from trl import GRPOConfig

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2.5-VL-3B-Instruct-Thinking",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    num_train_epochs=1,
    bf16=True,

    # Parameters that control the data preprocessing
    per_device_train_batch_size=2,
    max_completion_length=1024, # default: 256
    num_generations=2, # default: 8
    max_prompt_length=2048,

    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
)
```

### 3.5 Training the Model 🏃

Now, let's configure the trainer and start training the model!

In this case, we pass the two reward functions we previously defined to the trainer, in addition with the model, trainings arguments and dataset.

Below, you'll find a diagram of the training procedure we'll be reproducing, which is extracted from the [Open-R1 project](https://github.com/huggingface/open-r1).




```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)
```

    No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.


Time to train the model!


```python
trainer.train()
```



    [307/307 1:50:01, Epoch 1/1]
    
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>10</td>
      <td>0.012900</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>30</td>
      <td>-0.039200</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.003400</td>
    </tr>
    <tr>
      <td>50</td>
      <td>-0.056600</td>
    </tr>
    <tr>
      <td>60</td>
      <td>-0.036600</td>
    </tr>
    <tr>
      <td>70</td>
      <td>0.025500</td>
    </tr>
    <tr>
      <td>80</td>
      <td>-0.006600</td>
    </tr>
    <tr>
      <td>90</td>
      <td>0.100600</td>
    </tr>
    <tr>
      <td>100</td>
      <td>-0.002800</td>
    </tr>
    <tr>
      <td>110</td>
      <td>-0.017700</td>
    </tr>
    <tr>
      <td>120</td>
      <td>0.009000</td>
    </tr>
    <tr>
      <td>130</td>
      <td>0.013600</td>
    </tr>
    <tr>
      <td>140</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>150</td>
      <td>0.033800</td>
    </tr>
    <tr>
      <td>160</td>

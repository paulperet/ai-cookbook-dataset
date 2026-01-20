## **Sample : fine-tuning Phi-4-mini-reasoning with Apple MLX Framewrok**

This is based on the data from FreedomIntelligence/medical-o1-reasoning-SFT, which enhances the ability of Phi-4-mini-reasoning to reason about medical events. If you want to learn more about the data, you can use this link to learn more about it: https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT

*Note： Please use 64gb memory Apple silicon devices to run this sample*

```python
! pip install -U mlx-lm
```

```python
! pip install datasets
```

    Requirement already satisfied: datasets in /Users/lokinfey/miniforge3/envs/mlxdev/lib/python3.11/site-packages (3.5.1)
    [..., Requirement already satisfied: six>=1.5 in /Users/lokinfey/miniforge3/envs/mlxdev/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.17.0)]

### **Data**

We need to divide the data into train and valid and place them in the data folder. Note that apple mlx names the training data by default. train is named train.jsonl and valid is named valid.json

```python
from datasets import load_dataset
```

    /Users/lokinfey/miniforge3/envs/mlxdev/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm

```python
prompt_template = """<|user|>{}<|end|><|assistant|><think>{}</think>{}<|end|>"""


def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        # text = prompt_template.format(input, cot, output) + "<|end|>"
        text = prompt_template.format(input, cot, output) + "<|endoftext|>"
        texts.append(text)
    return {
        "text": texts,
    }
```

```python
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en", trust_remote_code=True)
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=200)
```

```python
train_dataset = split_dataset['train']
validation_dataset = split_dataset['test']
```

```python
train_dataset = train_dataset.map(formatting_prompts_func, batched = True,remove_columns=["Question", "Complex_CoT", "Response"])
train_dataset.to_json("./data/train.jsonl")
```

    Map: 100%|██████████| 15763/15763 [00:00<00:00, 100186.71 examples/s]
    Creating json from Arrow format: 100%|██████████| 16/16 [00:00<00:00, 93.83ba/s]





    46545696




```python
validation_dataset = validation_dataset.map(formatting_prompts_func, batched = True,remove_columns=["Question", "Complex_CoT", "Response"])
validation_dataset.to_json("./data/valid.jsonl")
```

    Map: 100%|██████████| 3941/3941 [00:00<00:00, 88383.53 examples/s]
    Creating json from Arrow format: 100%|██████████| 4/4 [00:00<00:00, 95.53ba/s]





    11674511



### **Training**

```python
! python -m mlx_lm.lora --model ./phi-4-mini-reasoning --train --data ./data  --iters 100
```

    Calling `python -m mlx_lm.lora...` directly is deprecated. Use `mlx_lm.lora...` or `python -m mlx_lm lora ...` instead.
    Loading pretrained model
    Loading datasets
    Training
    Trainable parameters: 0.027% (1.049M/3836.022M)
    Starting training..., iters: 100
    [Iter 1: Val loss 2.827, Val took 89.684s, ..., Iter 100: Train loss 1.702, Learning Rate 1.000e-05, It/sec 0.175, Tokens/sec 418.543, Trained Tokens 238626, Peak mem 28.763 GB]
    Iter 100: Saved adapter weights to adapters/adapters.safetensors and adapters/0000100_adapters.safetensors.
    Saved final weights to adapters/adapters.safetensors.

### **Testing**

```python
! python -m mlx_lm.generate --model ./phi-4-mini-reasoning --adapter-path ./adapters --max-token 4096 --prompt "A 54-year-old construction worker with a long history of smoking presents with swelling in his upper extremity and face, along with dilated veins in this region. After conducting a CT scan and venogram of the neck, what is the most likely diagnosis for the cause of these symptoms?" --extra-eos-token "<|end|>"
```

    <frozen runpy>:128: RuntimeWarning: 'mlx_lm.generate' found in sys.modules after import of package 'mlx_lm', but prior to execution of 'mlx_lm.generate'; this may result in unpredictable behaviour
    Calling `python -m mlx_lm.generate...` directly is deprecated. Use `mlx_lm.generate...` or `python -m mlx_lm generate ...` instead.
    ==========
    <think>Okay, let's see. The patient is a 54-year-old construction worker with a long history of smoking. He has swelling in his upper extremity and face, and dilated veins in this region. The CT scan and venogram of the neck are done. Hmm, what could be causing these symptoms?
    
    First, let's think about the symptoms. Swelling in the upper extremity and face, along with dilated veins. That sounds like edema, which is swelling due to fluid accumulation. The dilated veins might be related to increased pressure or venous insufficiency. 
    
    Now, considering the patient's history of smoking, that might be a clue. Smoking can contribute to vascular issues, like atherosclerosis or even something like a varicitis? Wait, no, varicitis is more about varicose veins. But the swelling in the upper extremity and face, along with dilated veins, could be pointing towards something like a venous thrombosis. But wait, venous thrombosis would cause swelling and dilated veins, right? But then, the CT scan and venogram would show a thrombus in the veins. 
    
    Alternatively, could it be something like a varicoceles? No, that usually involves varicose veins, which are more about veins in the legs. 
    
    Wait, the upper extremity and face swelling. Hmm, maybe it's a case of a venous obstruction. But the venogram would show dilated veins around the area, which is typical of venous obstruction. 
    
    Another possibility is a venous thrombosis, like deep vein Thrombosis (DVT) in the upper extremities. But DVT is more common in the legs. However, the upper extremity could be involved, especially if there's some kind of venous obstruction. 
    
    Wait, but the CT scan and venogram of the neck. The venogram would show dilated veins in the neck region, which is unusual. So, the venous obstruction in the neck area could be due to a thrombus in the jugular vein. 
    
    But why would the jugular vein be dilated? That could be due to a venous obstruction, like a thrombosis. 
    
    So, putting it all together, the patient has swelling in the upper extremity and face, which is likely due to increased venous pressure. The dilated veins in the neck region, as seen on the venogram, suggest a venous obstruction. 
    
    Given the patient's history of smoking, which is a risk factor for thrombosis, this could be a case of jugular vein thrombosis. But wait, the jugular vein is part of the venous system in the neck, and if there's a thrombus there, it could cause swelling in the upper extremity and face due to increased pressure. 
    
    Alternatively, could it be something like a varicoceles? No, that doesn't seem right. 
    
    Another possibility is a venous obstruction in the subclavian vein. If there's a thrombus in the subclavian vein, it could lead to swelling in the upper extremity and face, and the venogram would show dilated veins in the neck region. 
    
    So, the most likely diagnosis is a venous thrombosis in the subclavian vein. That would explain the swelling and the dilated veins seen on the venogram. 
    
    But wait, the CT scan and venogram of the neck. The venogram would show dilated veins around the area, which is typical of venous obstruction. 
    
    So, yeah, the diagnosis is a venous thrombosis in the subclavian vein. That makes sense given the patient's symptoms and history. Smoking is a risk factor for thrombosis, and the venous obstruction would cause the swelling and dilated veins. 
    
    I think that's the most likely diagnosis.</think>The patient’s symptoms—swelling in the upper extremity and face along with dilated veins in the neck—are consistent with increased venous pressure and venous obstruction. Given his history of smoking, which is a known risk factor for thrombosis, the most likely diagnosis is a venous thrombosis in the subclavian vein. This condition can lead to venous obstruction, resulting in the observed swelling and dilated veins seen on the venogram.
    ==========
    Prompt: 76 tokens, 329.797 tokens-per-sec
    Generation: 871 tokens, 29.596 tokens-per-sec
    Peak memory: 7.846 GB
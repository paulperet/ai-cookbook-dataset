## **Sample : fine-tuning Phi-4-mini-reasoning with Apple MLX Framewrok**

This is based on the data from FreedomIntelligence/medical-o1-reasoning-SFT, which enhances the ability of Phi-4-mini-reasoning to reason about medical events. If you want to learn more about the data, you can use this link to learn more about it: https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT 

*Note： Please use A100 as your GPU*


```python
! pip install datasets
```

[Collecting datasets, ..., Successfully installed datasets-3.5.1 dill-0.3.8 fsspec-2025.3.0 multiprocess-0.70.16 xxhash-3.5.0]



```python
! pip install git+https://github.com/microsoft/Olive.git
```

[Collecting git+https://github.com/microsoft/Olive.git, ..., Successfully installed alembic-1.15.2 colorlog-6.9.0 lightning-utilities-0.14.3 nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-nvjitlink-cu12-12.4.127 olive-ai-0.9.0.dev0 onnx-1.17.0 onnxscript-0.2.5 optuna-4.3.0 torchmetrics-1.7.1]



```python
! pip install transformers==4.49.0
```

[Collecting transformers==4.49.0, ..., Successfully installed transformers-4.49.0]



```python
! pip install protobuf==3.20.3 -U
```

[Collecting protobuf==3.20.3, ..., Successfully installed protobuf-3.20.3]



```python
! pip install onnxruntime-genai-cuda
```

[Collecting onnxruntime-genai-cuda, ..., Successfully installed onnxruntime-genai-cuda-0.7.1 onnxruntime-gpu-1.21.1]



```python
! pip list
```

[Package, ..., zstandard 0.23.0]



```python
! pip install bitsandbytes
```

[Collecting bitsandbytes, ..., Successfully installed bitsandbytes-0.45.5]



```python
! huggingface-cli download {Phi-4-mini-reasoning hugging face id} --local-dir {Your Phi-4-mini-reasoning model location}
```

[Fetching 16 files: 0% 0/16, ..., /content/phi-4-mini-reasoning]

### **Data**


```python
from datasets import load_dataset
```


```python
prompt_template = """<|user|>{}<|end|><|assistant|><think>{}</think>{}"""


def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = prompt_template.format(input, cot, output) + "<|end|>"
        # text = prompt_template.format(input, cot, output) + "<|endoftext|>"
        texts.append(text)
    return {
        "text": texts,
    }

# Create the English dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en", split = "train",trust_remote_code=True)
dataset = dataset.map(formatting_prompts_func, batched = True,remove_columns=["Question", "Complex_CoT", "Response"])
dataset.to_json("en_dataset.jsonl")
```

57964055

### **Fine-tuning with Microsoft Olive**


```python
!olive finetune \
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

[Loading HuggingFace model from ./phi-4-mini-reasoning, ..., Model is saved at /content/models/phi-4-mini-reasoning/ft]



```python
! pip install optimum
```

[Collecting optimum, ..., Successfully installed optimum-1.24.0]

### **Convert with Microsoft Olive**


```python
! olive capture-onnx-graph \
    --model_name_or_path "./phi-4-mini-reasoning" \
    --adapter_path models/phi-4-mini-reasoning/ft/adapter \
    --use_model_builder  \
    --output_path models/phi-4-mini-reasoning/onnx \
    --log_level 1
```

[Loading HuggingFace model from ./phi-4-mini-reasoning, ..., Model is saved at /content/models/phi-4-mini-reasoning/onnx]



```python
! olive generate-adapter \
    --model_name_or_path models/phi-4-mini-reasoning/onnx \
    --output_path models/phi-4-mini-reasoning/adapter-onnx \
    --log_level 1
```

[Loaded previous command output of type onnxmodel from models/phi-4-mini-reasoning/onnx, ..., Model is saved at /content/models/phi-4-mini-reasoning/adapter-onnx]

### **Inference Fine-tuning model with ONNXRuntime-GenAI**


```python
import onnxruntime_genai as og
import numpy as np
import os
```


```python
model_folder = "./models/phi-4-mini-reasoning/adapter-onnx/model/"
```


```python
model = og.Model(model_folder)
```


```python
adapters = og.Adapters(model)
adapters.load('./models/phi-4-mini-reasoning/adapter-onnx/model/adapter_weights.onnx_adapter', "en_medical_reasoning")
```


```python
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
```


```python
search_options = {}
search_options['max_length'] = 200
search_options['past_present_share_buffer'] = False
search_options['temperature'] = 1
search_options['top_k'] = 1
```


```python
prompt_template = """<|user|>{}<|end|><|assistant|><think>"""

question = """ A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. Given her vital signs of pulse 110\/min, respirations 22\/min, and blood pressure 90\/65 mm Hg, along with the presence of a 5-cm deep stab wound at the upper border of the 8th rib in the left midaxillary line, which anatomical structure in her chest is most likely to be injured? """
```


```python
prompt = prompt_template.format(question, "")
```


```python
prompt
```




    '<|user|> A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. Given her vital signs of pulse 110\\/min, respirations 22\\/min, and blood pressure 90\\/65 mm Hg, along with the presence of a 5-cm deep stab wound at the upper border of the 8th rib in the left midaxillary line, which anatomical structure in her chest is most likely to be injured? <|end|><|assistant|><think>'




```python
input_tokens = tokenizer.encode(prompt)
```


```python
params = og.GeneratorParams(model)
params.set_search_options(**search_options)
generator = og.Generator(model, params)
```


```python
generator.set_active_adapter(adapters, "en_medical_reasoning")
```


```python
generator.append_tokens(input_tokens)
```


```python
while not generator.is_done():
    generator.generate_next_token()
    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)
```

    Okay, let's dig deeper into figuring out what's going on internally with Ms.A's chest wound. She's experiencing severe symptoms—a fast-tickling pulse, rapid breathing, dropping blood pressure—and she's visibly struggling internally. Fast breathing signals hyperventilating hyperactive lungs—a symptom tied tightly to hyperactive breathing triggered internally—not externally triggered hyperventilations—which wouldn't trigger hyperactive breathing externally—or hyperactive breathing triggered externally—which relies exclusively on external stimuli/hypersensitisation/hyp
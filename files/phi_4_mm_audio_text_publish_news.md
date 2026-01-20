```python
!pip install huggingface_hub
```

    Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.11/dist-packages (0.27.1)
    [..., Successfully installed backoff-2.2.1]



```python
! pip install backoff
```

    Collecting backoff
    [..., Successfully installed backoff-2.2.1]



```python
! wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

    --2025-01-22 00:57:39--  https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    [..., 2025-01-22 00:57:41 (130 MB/s) - ‘flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl’ saved [187328293/187328293]]



```python
! pip install --no-dependencies --upgrade flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

    Processing ./flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    [..., Successfully installed flash-attn-2.6.3]



```python
from huggingface_hub import login

login()
```



```python
! git lfs install
! git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct
```

    Git LFS initialized.
    Cloning into 'phi-4-omni'...
    [..., Encountered 2 file(s) that may not have been copied correctly on Windows:
    	model-00002-of-00003.safetensors
    	model-00001-of-00003.safetensors
    See: `git lfs help smudge` for more details.]



```python
import requests
import torch
from PIL import Image
import soundfile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig,pipeline,AutoTokenizer
```


```python
model_path = './Phi-4-multimodal'
```


```python
kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16
```


```python
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
```


```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    _attn_implementation='flash_attention_2',
).cuda()
```

    Loading checkpoint shards: [0%, ..., 100%]



```python
generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')
```


```python
system_prompt = '<|system|>'
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
```


```python
from PIL import Image
```


```python
prompt = f'{system_prompt}You are a technology journalist who writes professional content based on audio and picture content{prompt_suffix}{user_prompt}Reorganize the content provided by audio <|audio_1|> and image <|image_1|> from a professional perspective and write a press release based on the current development of AI ,output is markdown format and including title and content {prompt_suffix}{assistant_prompt}'
```


```python
image = Image.open("./copilot.png")
```


```python
audio = soundfile.read('./satya1.mp3')
```


```python
inputs = processor(text=prompt, images=[image], audios=[audio], return_tensors='pt').to('cuda:0')
```


```python
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1500,
    generation_config=generation_config,
)
```


```python
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
```


```python
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
```


```python
response
```




    "# Copilot: The Future of AI-Driven Business Growth\n\nIn the rapidly evolving world of technology, AI is transforming the way businesses operate, offering unprecedented efficiency and growth opportunities. Today, we delve into the latest developments in AI, focusing on the innovative Copilot platform, which is revolutionizing the business landscape.\n\n## Copilot: A New Era of AI Integration\n\nCopilot is at the forefront of AI technology, designed to enhance business processes and productivity. This platform is not just an AI tool; it's a comprehensive ecosystem that includes Copilot devices and a Copilot & AI stack, all underpinned by robust security measures.\n\n### Key Features of Copilot\n\n1. **Copilot**: The core AI platform that serves as the brain of the operation, providing intelligent insights and automation capabilities.\n2. **Copilot Devices**: These are the physical or virtual tools that interact with the Copilot platform, enabling seamless integration and execution of AI-driven tasks.\n3. **Copilot & AI Stack**: This component ensures that all elements of the Copilot system are securely connected and function in harmony, providing a stable and reliable AI environment.\n\n### Security: The Backbone of Copilot\n\nSecurity is paramount in the Copilot ecosystem. Each layer of the platform is designed with advanced security protocols to protect data and ensure the integrity of operations. From the Copilot devices to the Copilot & AI stack, every aspect is fortified to prevent unauthorized access and cyber threats.\n\n## The Impact of Copilot on Business\n\nThe introduction of Copilot is set to transform business operations by automating routine tasks, enhancing decision-making processes, and fostering innovation. Employees will have personalized AI assistants that understand their work styles and preferences, leading to increased productivity and creativity.\n\n### Benefits of Copilot\n\n- **Enhanced Efficiency**: Automating repetitive tasks frees up time for strategic activities.\n- **Improved Decision Making**: AI-driven insights provide data-backed decisions, reducing errors and increasing profitability.\n- **Innovation**: The flexibility of Copilot encourages experimentation and the development of new business models.\n\n## Conclusion\n\nAs businesses continue to navigate the complexities of the digital age, Copilot stands out as a pivotal solution, driving growth and efficiency through AI. With its comprehensive security measures and user-centric design, Copilot is not just a tool but a partner in the journey towards a smarter, more efficient future.\n\nStay tuned for more updates on how Copilot is shaping the future of business and technology."
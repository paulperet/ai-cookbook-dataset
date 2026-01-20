```python
# pip install transformers
```

```python
# pip install torch torchvision torchaudio -U
```

```python
# pip install flash-attn --no-build-isolation
```

```python
# ! pip install flash_attn -U
```

```python
from torch import bfloat16
import transformers
```

```python
model_id = "../Phi3MOE"
```

```python
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=bfloat16,
    device_map='auto'
)

```

[Loading checkpoint shards:   0%|          | 0/17 [00:00<?, ?it/s], WARNING:root:Some parameters are on the meta device device because they were offloaded to the cpu.]

```python
model.eval()
```

PhiMoEForCausalLM(
  (model): PhiMoEModel(
    (embed_tokens): Embedding(32064, 4096)
    (layers): ModuleList(
      (0-31): 32 x PhiMoEDecoderLayer(
        (self_attn): PhiMoESdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=True)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=True)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=True)
          (rotary_emb): Phi3LongRoPEScaledRotaryEmbedding()
        )
        (block_sparse_moe): PhiMoESparseMoeBlock(
          (gate): Linear(in_features=4096, out_features=16, bias=False)
          (experts): ModuleList(
            (0-15): 16 x PhiMoEBlockSparseTop2MLP(
              (w1): Linear(in_features=4096, out_features=6400, bias=False)
              (w2): Linear(in_features=6400, out_features=4096, bias=False)
              (w3): Linear(in_features=4096, out_features=6400, bias=False)
              (act_fn): SiLU()
            )
          )
        )
        (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
      )
    )
    (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4096, out_features=32064, bias=True)
)

```python
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
```

```python
pipe = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
```

[2024-08-20 04:51:54.431510: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory, 2024-08-20 04:51:54.431543: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine., The model 'PhiMoEForCausalLM' is not supported for text-generation. Supported models are ['BartForCausalLM', ..., 'XmodForCausalLM'].]

```python
generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "temperature": 0.3,
    "do_sample": False,
}
```

```python
sys_msg = """You are a helpful AI assistant, you are an agent capable of using a variety of tools to answer a question. Here are a few of the tools available to you:

- Blog: This tool helps you describe a certain knowledge point and content, and finally write it into Twitter or Facebook style content
- Translate: This is a tool that helps you translate into any language, using plain language as required

To use these tools you must always respond in JSON format containing `"tool_name"` and `"input"` key-value pairs. For example, to answer the question, "Build Muliti Agents with MOE models" you must use the calculator tool like so:

```json
{
    "tool_name": "Blog",
    "input": "Build Muliti Agents with MOE models"
}
```

Or to translate the question "can you introduce yourself in Chinese" you must respond:

```json
{
    "tool_name": "Search",
    "input": "can you introduce yourself in Chinese"
}
```

Remember just output the final result, ouput in JSON format containing `"agentid"`,`"tool_name"` , `"input"` and `"output"`  key-value pairs .:

```json
[
{   "agentid": "step1",
    "tool_name": "Blog",
    "input": "Build Muliti Agents with MOE models",
    "output": "........."
},
{   "agentid": "step2",
    "tool_name": "Search",
    "input": "can you introduce yourself in Chinese",
    "output": "........."
},
{
    "agentid": "final"
    "tool_name": "Result",
    "output": "........."
}
]
```

The users answer is as follows.
"""
```

```python
def instruction_format(sys_message: str, query: str):
    # note, don't "</s>" to the end
    return f'<|system|> {sys_message} <|end|>\n<|user|> {query} <|end|>\n<|assistant|>'
```

```python
query ='Write something about Generative AI with MOE , translate it to Chinese'
```

```python
input_prompt = instruction_format(sys_msg, query)
```

```python
input_prompt
```

'<|system|> You are a helpful AI assistant, you are an agent capable of using a variety of tools to answer a question. Here are a few of the tools available to you:

- Blog: This tool helps you describe a certain knowledge point and content, and finally write it into Twitter or Facebook style content
- Translate: This is a tool that helps you translate into any language, using plain language as required

To use these tools you must always respond in JSON format containing `"tool_name"` and `"input"` key-value pairs. For example, to answer the question, "Build Muliti Agents with MOE models" you must use the calculator tool like so:

```json
{
    "tool_name": "Blog",
    "input": "Build Muliti Agents with MOE models"
}
```

Or to translate the question "can you introduce yourself in Chinese" you must respond:

```json
{
    "tool_name": "Search",
    "input": "can you introduce yourself in Chinese"
}
```

Remember just output the final result, ouput in JSON format containing `"agentid"`,`"tool_name"` , `"input"` and `"output"`  key-value pairs .:

```json
[
{   "agentid": "step1",
    "tool_name": "Blog",
    "input": "Build Muliti Agents with MOE models",
    "output": "........."
},
{   "agentid": "step2",
    "tool_name": "Search",
    "input": "can you introduce yourself in Chinese",
    "output": "........."
},
{
    "agentid": "final"
    "tool_name": "Result",
    "output": "........."
}
]
```

The users answer is as follows.
 <|end|>
<|user|> Write something about Generative AI with MOE , translate it to Chinese <|end|>
<|assistant|>'

```python
import torch
torch.cuda.empty_cache() 
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True "
```

```python
output = pipe(input_prompt, **generation_args)
```

[/anaconda/envs/azureml_py38/lib/python3.9/site-packages/transformers/generation/configuration_utils.py:567: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.3` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
, The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` model input instead.]

```python
output[0]['generated_text']
```

' ```json
[
{   "agentid": "step1",
    "tool_name": "Blog",
    "input": "Generative AI with MOE",
    "output": "Generative AI with MOE (Mixture of Experts) is a powerful approach that combines the strengths of generative models and the flexibility of MOE architecture. This hybrid model can generate high-quality, diverse, and contextually relevant content, making it suitable for various applications such as content creation, data augmentation, and more."
},
{   "agentid": "step2",
    "tool_name": "Translate",
    "input": "Generative AI with MOE is a powerful approach that combines the strengths of generative models and the flexibility of MOE architecture. This hybrid model can generate high-quality, diverse, and contextually relevant content, making it suitable for various applications such as content creation, data augmentation, and more.",
    "output": "基于生成AI的MOE（Mixture of Experts）是一种强大的方法，它结合了生成模型的优势和MOE架构的灵活性。这种混合模型可以生成高质量、多样化且上下文相关的内容，使其适用于各种应用，如内容创建、数据增强等。"
},
{
    "agentid": "final",
    "tool_name": "Result",
    "output": "基于生成AI的MOE（Mixture of Experts）是一种强大的方法，它结合了生成模型的优势和MOE架构的灵活性。这种混合模型可以生成高质量、多样化且上下文相关的内容，使其适用于各种应用，如内容创建、数据增强等。"
}
]
```'
# AWQ
Similar to GPTQ, AWQ is optimized for GPU inference. It is based on the fact that ~1% of weights actually contribute significantly to the model's accuracy, and hence these must be treated delicately by using a dataset to analyze the activation distributions during inference and identify those important and critical weights.

### Quantizing with [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

Let's do a short demo and quantize Mistral 7B!

First, we install `autoawq`. It will allow us to easily quantize and perform inference on AWQ models! AutoAWQ also provides, by default, a `pile-val` dataset that will be used for the quantization process!


```python
!pip install autoawq
```

[Collecting autoawq, ..., Successfully installed autoawq-0.2.6 autoawq-kernels-0.0.7 datasets-2.21.0 dill-0.3.8 multiprocess-0.70.16 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.20.5 nvidia-nvjitlink-cu12-12.6.20 nvidia-nvtx-cu12-12.1.105 pyarrow-17.0.0 xxhash-3.5.0 zstandard-0.23.0]

Once we're done, we can download the model we want to quantize. First, let's log in with a read access token so we have access to the models.

Note: You need to first accept the terms in the repo.


```python
from huggingface_hub import login

login("read_token")
```

The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `huggingface-cli` if you want to set the git credential as well.
Token is valid (permission: read).
Your token has been saved to /root/.cache/huggingface/token
Login successful

Now everything is ready, so we can load the model and quantize it! Here, we will quantize the model to 4-bit!


```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

pretrained_model_dir = "mistralai/Mistral-7B-Instruct-v0.3"
quantized_model_dir = "mistral_awq_quant"

model = AutoAWQForCausalLM.from_pretrained(
    pretrained_model_dir, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, trust_remote_code=True)

# quantize the model
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
model.quantize(tokenizer, quant_config=quant_config)
```

[config.json: 0%| | 0.00/601 [00:00<?, ?B/s], ..., AWQ: 100%|██████████| 32/32 [17:47<00:00, 33.37s/it]]

Now that the model is quantized, we can save it so we can share it or load it later! Since quantizing with AWQ takes a while and some resources, it's advised to always save them.


```python
model.save_quantized(quantized_model_dir)

tokenizer.save_pretrained(quantized_model_dir)
```

Note that `shard_checkpoint` is deprecated and will be removed in v4.44. We recommend you using split_torch_state_dict_into_shards from huggingface_hub library

('mistral_awq_quant/tokenizer_config.json',
 'mistral_awq_quant/special_tokens_map.json',
 'mistral_awq_quant/tokenizer.model',
 'mistral_awq_quant/added_tokens.json',
 'mistral_awq_quant/tokenizer.json')

Model quantized and saved to AWQ 4-bit precision!

You can also load it for inference using `autoawq` as follows:


```python
model = AutoAWQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0") # loads quantized model to the first GPU
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)

conversation = [{"role": "user", "content": "How are you today?"}]

prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt")
inputs.to("cuda:0") # loads tensors to the first GPU

outputs = model.generate(**inputs, max_new_tokens=32)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

[Replacing layers...: 100%|██████████| 32/32 [00:07<00:00,  4.23it/s], Fusing layers...: 100%|██████████| 32/32 [00:00<00:00, 130.00it/s], Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.]

How are you today? I'm an AI and don't have feelings, but I'm here and ready to help you with your questions! How can I assist you today
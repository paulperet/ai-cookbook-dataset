# Documentation Chatbot with Meta Synthetic Data Kit

_Authored by: [Alan Ponnachan](https://huggingface.co/AlanPonnachan)_

This notebook demonstrates a practical approach to building a domain-specific Question & Answering chatbot. We'll focus on creating a chatbot that can answer questions about a specific piece of documentation – in this case, LangChain's documentation on Chat Models.

**Goal:** To fine-tune a small, efficient Language Model (LLM) to understand and answer questions about the LangChain Chat Models documentation.

**Approach:**
1.  **Data Acquisition:** Obtain the text content from the target LangChain documentation page.
2.  **Synthetic Data Generation:** Use Meta's `synthetic-data-kit` to automatically generate Question/Answer pairs from this documentation.
3.  **Efficient Fine-tuning:** Employ Unsloth and `Hugging Face's TRL SFTTrainer` to efficiently fine-tune a Llama-3.2-3B model on the generated synthetic data.
4.  **Evaluation:** Test the fine-tuned model with specific questions about the documentation.

This method allows us to adapt an LLM to a niche domain without requiring a large, manually curated dataset.


**Hardware Used:**

This notebook was run on Google Colab (Free Tier) with an NVIDIA T4 GPU

## 1. Setup and Installation

First, we need to install the necessary libraries. We'll use `unsloth` for efficient model handling and training, and `synthetic-data-kit` for generating our training data.


```python
%%capture
# In Colab, we skip dependency installation to avoid conflicts with preinstalled packages.
# On local machines, we include dependencies for completeness.

import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm==0.8.2
else:

    !pip install --no-deps unsloth vllm==0.8.2

# Get https://github.com/meta-llama/synthetic-data-kit
!pip install synthetic-data-kit
```


```python

%%capture
import os
if "COLAB_" in "".join(os.environ.keys()):

    import sys, re, requests; modules = list(sys.modules.keys())
    for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft "trl==0.15.2" triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub[hf_xet] hf_transfer

    # vLLM requirements - vLLM breaks Colab due to reinstalling numpy
    f = requests.get("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/requirements/common.txt").content
    with open("vllm_requirements.txt", "wb") as file:
        file.write(re.sub(rb"(transformers|numpy|xformers|importlib_metadata)[^\n]{0,}\n", b"", f))
    !pip install -r vllm_requirements.txt
```

## 2. Synthetic Data Generation

We'll use `SyntheticDataKit` from Unsloth (which wraps Meta's `synthetic-data-kit`) to create Question/Answer pairs from our chosen documentation.




```python
from unsloth.dataprep import SyntheticDataKit

generator = SyntheticDataKit.from_pretrained(

    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 2048,
)
```

    🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
    🦥 Unsloth Zoo will now patch everything to make training faster!
    INFO 05-05 15:14:48 [__init__.py:239] Automatically detected platform cuda.

    Unsloth: Using dtype = torch.float16 for vLLM.
    Unsloth: vLLM loading unsloth/Llama-3.2-3B-Instruct with actual GPU utilization = 89.39%
    Unsloth: Your GPU has CUDA compute capability 7.5 with VRAM = 14.74 GB.
    Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 2048. Num Sequences = 192.
    Unsloth: vLLM's KV Cache can use up to 7.19 GB. Also swap space = 2 GB.
    vLLM STDOUT: INFO 05-05 15:15:07 [__init__.py:239] Automatically detected platform cuda.
    vLLM STDOUT: INFO 05-05 15:15:08 [api_server.py:981] vLLM API server version 0.8.2
    vLLM STDOUT: INFO 05-05 15:15:08 [api_server.py:982] args: Namespace(subparser='serve', model_tag='unsloth/Llama-3.2-3B-Instruct', config='', host=None, port=8000, uvicorn_log_level='info', disable_uvicorn_access_log=False, allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, chat_template_content_format='auto', response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, enable_ssl_refresh=False, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_request_id_headers=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='unsloth/Llama-3.2-3B-Instruct', task='auto', tokenizer=None, hf_config_path=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, allowed_local_media_path=None, download_dir=None, load_format='auto', config_format=<ConfigFormat.AUTO: 'auto'>, dtype='float16', kv_cache_dtype='auto', max_model_len=2048, guided_decoding_backend='xgrammar', logits_processor_pattern=None, model_impl='auto', distributed_executor_backend=None, pipeline_parallel_size=1, tensor_parallel_size=1, enable_expert_parallel=False, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=None, enable_prefix_caching=True, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, seed=0, swap_space=2.0, cpu_offload_gb=0, gpu_memory_utilization=0.8938626454842437, num_gpu_blocks_override=None, max_num_batched_tokens=2048, max_num_partial_prefills=1, max_long_partial_prefills=1, long_prefill_token_threshold=0, max_num_seqs=192, max_logprobs=0, disable_log_stats=True, quantization=None, rope_scaling=None, rope_theta=None, hf_overrides=None, enforce_eager=False, max_seq_len_to_capture=2304, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, disable_mm_preprocessor_cache=False, enable_lora=False, enable_lora_bias=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, use_tqdm_on_load=True, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_config=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, scheduling_policy='fcfs', scheduler_cls='vllm.core.scheduler.Scheduler', override_neuron_config=None, override_pooler_config=None, compilation_config={"level":3,"splitting_ops":[]}, kv_transfer_config=None, worker_cls='auto', worker_extension_cls='', generation_config='auto', override_generation_config=None, enable_sleep_mode=False, calculate_kv_scales=False, additional_config=None, enable_reasoning=False, reasoning_parser=None, disable_cascade_attn=False, disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False, enable_prompt_tokens_details=False, enable_server_load_tracking=False, dispatch_function=<function ServeSubcommand.cmd at 0x7f225baa9a80>)
    vLLM STDOUT: WARNING 05-05 15:15:09 [config.py:2614] Casting torch.bfloat16 to torch.float16.
    vLLM STDOUT: INFO 05-05 15:15:24 [config.py:585] This model supports multiple tasks: {'classify', 'embed', 'score', 'generate', 'reward'}. Defaulting to 'generate'.
    vLLM STDOUT: WARNING 05-05 15:15:24 [arg_utils.py:1854] Compute Capability < 8.0 is not supported by the V1 Engine. Falling back to V0.
    vLLM STDOUT: INFO 05-05 15:15:24 [api_server.py:241] Started engine process with PID 1401
    vLLM STDOUT: INFO 05-05 15:15:33 [__init__.py:239] Automatically detected platform cuda.
    vLLM STDOUT: INFO 05-05 15:15:34 [llm_engine.py:241] Initializing a V0 LLM engine (v0.8.2) with config: model='unsloth/Llama-3.2-3B-Instruct', speculative_config=None, tokenizer='unsloth/Llama-3.2-3B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar', reasoning_backend=None), observability_config=ObservabilityConfig(show_hidden_metrics=False, otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=unsloth/Llama-3.2-3B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"level":3,"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":192}, use_cached_outputs=True,
    vLLM STDOUT: INFO 05-05 15:15:37 [cuda.py:239] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
    vLLM STDOUT: INFO 05-05 15:15:37 [cuda.py:288] Using XFormers backend.
    vLLM STDOUT: INFO 05-05 15:15:38 [parallel_state.py:954] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0
    vLLM STDOUT: INFO 05-05 15:15:38 [model_runner.py:1110] Starting to load model unsloth/Llama-3.2-3B-Instruct...
    vLLM STDOUT: INFO 05-05 15:15:39 [weight_utils.py:265] Using model weights format ['*.safetensors']
    vLLM STDOUT: INFO 05-05 15:23:55 [weight_utils.py:281] Time spent downloading weights for unsloth/Llama-3.2-3B-Instruct: 496.088307 seconds
    vLLM STDOUT: INFO 05-05 15:24:32 [loader.py:447] Loading weights took 36.15 seconds
    vLLM STDOUT: INFO 05-05 15:24:32 [model_runner.py:1146] Model loading took 6.0160 GB and 533.851791 seconds
    vLLM STDOUT: INFO 05-05 15:24:45 [backends.py:415] Using cache directory: /root/.cache/vllm/torch_compile_cache/3fbc507988/rank_0_0 for vLLM's torch.compile
    vLLM STDOUT: INFO 05-05 15:24:45 [backends.py:425] Dynamo bytecode transform time: 12.18 s
    vLLM STDOUT: INFO 05-05 15:25:28 [backends.py:132] Cache the graph of shape None for later use
    vLLM STDOUT: INFO 05-05 15:25:28 [backends.py:144] Compiling a graph for general shape takes 38.17 s
    vLLM STDOUT: INFO 05-05 15:25:29 [monitor.py:33] torch.compile takes 50.34 s in total
    vLLM STDOUT: INFO 05-05 15:25:32 [worker.py:267] Memory profiling takes 58.92 seconds
    vLLM STDOUT: INFO 05-05 15:25:32 [worker.py:267] the current vLLM instance can use total_gpu_memory (14.74GiB) x gpu_memory_utilization (0.89) = 13.18GiB
    vLLM STDOUT: INFO 05-05 15:25:32 [worker.py:267] model weights take 6.02GiB; non_torch_memory takes 0.05GiB; PyTorch activation peak memory takes 0.90GiB; the rest of the memory reserved for KV Cache is 6.22GiB.
    vLLM STDOUT: INFO 05-05 15:25:32 [executor_base.py:111] # cuda blocks: 3637, # CPU blocks: 1170
    vLLM STDOUT: INFO 05-05 15:25:32 [executor_base.py:116] Maximum concurrency for 2048 tokens per request: 28.41x
    vLLM STDOUT: INFO 05-05 15:25:35 [model_runner.py:1442] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
    vLLM STDOUT: INFO 05-05 15:26:16 [model_runner.py:1570] Graph capturing finished in 41 secs, took 0.17 GiB
    vLLM STDOUT: INFO 05-05 15:26:16 [llm_engine.py:447] init engine (profile, create kv cache, warmup model) took 104.05 seconds
    vLLM STDOUT: WARNING 05-05 15:26:17 [config.py:1028] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
    vLLM STDOUT: INFO 05-05 15:26:17 [serving_chat.py:115] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
    vLLM STDOUT: INFO 05-05 15:26:17 [serving_completion.py:61] Using default completion sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
    v
# Introduction
The purpose of this cookbook is to show you how to properly benchmark TGI. For more background details and explanation, please check out this [popular blog](https://huggingface.co/blog/tgi-benchmarking) first.

## Setup
Make sure you have an environment with TGI installed; docker is a great choice.The commands here can be easily copied/pasted into a terminal, which might be even easier. Don't feel compelled to use Jupyter. If you just want to test this out, you can duplicate and use [derek-thomas/tgi-benchmark-space](https://huggingface.co/spaces/derek-thomas/tgi-benchmark-space). 

# TGI Launcher


```python
!text-generation-launcher --version
```

    text-generation-launcher 2.2.1-dev0


Below we can see the different settings for TGI. Be sure to read through them and decide which settings are most 
important for your use-case.

Here are some of the most important ones:
- `--model-id`
- `--quantize` Quantization saves memory, but does not always improve speed
- `--max-input-tokens` This allows TGI to optimize the prefilling operation
- `--max-total-tokens` In combination with the above TGI now knows what the max input and output tokens are
- `--max-batch-size` This lets TGI know how many requests it can process at once.

The last 3 together provide the necessary restrictions to optimize for your use-case. You can find a lot of performance improvements by setting these as appropriately as possible.


```python
!text-generation-launcher -h
```

    Text Generation Launcher
    
    Usage: text-generation-launcher [OPTIONS]
    
    Options:
          --model-id <MODEL_ID>
              The name of the model to load. Can be a MODEL_ID as listed on <https://hf.co/models> like `gpt2` or `OpenAssistant/oasst-sft-1-pythia-12b`. Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of transformers [env: MODEL_ID=] [default: bigscience/bloom-560m]
          --revision <REVISION>
              The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2` [env: REVISION=]
          --validation-workers <VALIDATION_WORKERS>
              The number of tokenizer workers used for payload validation and truncation inside the router [env: VALIDATION_WORKERS=] [default: 2]
          --sharded <SHARDED>
              Whether to shard the model across multiple GPUs By default text-generation-inference will use all available GPUs to run the model. Setting it to `false` deactivates `num_shard` [env: SHARDED=] [possible values: true, false]
          --num-shard <NUM_SHARD>
              The number of shards to use if you don't want to use all GPUs on a given machine. You can use `CUDA_VISIBLE_DEVICES=0,1 text-generation-launcher... --num_shard 2` and `CUDA_VISIBLE_DEVICES=2,3 text-generation-launcher... --num_shard 2` to launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance [env: NUM_SHARD=]
          --quantize <QUANTIZE>
              Whether you want the model to be quantized [env: QUANTIZE=] [possible values: awq, eetq, exl2, gptq, marlin, bitsandbytes, bitsandbytes-nf4, bitsandbytes-fp4, fp8]
          --speculate <SPECULATE>
              The number of input_ids to speculate on If using a medusa model, the heads will be picked up automatically Other wise, it will use n-gram speculation which is relatively free in terms of compute, but the speedup heavily depends on the task [env: SPECULATE=]
          --dtype <DTYPE>
              The dtype to be forced upon the model. This option cannot be used with `--quantize` [env: DTYPE=] [possible values: float16, bfloat16]
          --trust-remote-code
              Whether you want to execute hub modelling code. Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision [env: TRUST_REMOTE_CODE=]
          --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
              The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly [env: MAX_CONCURRENT_REQUESTS=] [default: 128]
          --max-best-of <MAX_BEST_OF>
              This is the maximum allowed value for clients to set `best_of`. Best of makes `n` generations at the same time, and return the best in terms of overall log probability over the entire generated sequence [env: MAX_BEST_OF=] [default: 2]
          --max-stop-sequences <MAX_STOP_SEQUENCES>
              This is the maximum allowed value for clients to set `stop_sequences`. Stop sequences are used to allow the model to stop on more than just the EOS token, and enable more complex "prompting" where users can preprompt the model in a specific way and define their "own" stop token aligned with their prompt [env: MAX_STOP_SEQUENCES=] [default: 4]
          --max-top-n-tokens <MAX_TOP_N_TOKENS>
              This is the maximum allowed value for clients to set `top_n_tokens`. `top_n_tokens` is used to return information about the the `n` most likely tokens at each generation step, instead of just the sampled token. This information can be used for downstream tasks like for classification or ranking [env: MAX_TOP_N_TOKENS=] [default: 5]
          --max-input-tokens <MAX_INPUT_TOKENS>
              This is the maximum allowed input length (expressed in number of tokens) for users. The larger this value, the longer prompt users can send which can impact the overall memory required to handle the load. Please note that some models have a finite range of sequence they can handle. Default to min(max_position_embeddings - 1, 4095) [env: MAX_INPUT_TOKENS=]
          --max-input-length <MAX_INPUT_LENGTH>
              Legacy version of [`Args::max_input_tokens`] [env: MAX_INPUT_LENGTH=]
          --max-total-tokens <MAX_TOTAL_TOKENS>
              This is the most important value to set as it defines the "memory budget" of running clients requests. Clients will send input sequences and ask to generate `max_new_tokens` on top. with a value of `1512` users can send either a prompt of `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for `1511` max_new_tokens. The larger this value, the larger amount each request will be in your RAM and the less effective batching can be. Default to min(max_position_embeddings, 4096) [env: MAX_TOTAL_TOKENS=]
          --waiting-served-ratio <WAITING_SERVED_RATIO>
              This represents the ratio of waiting queries vs running queries where you want to start considering pausing the running queries to include the waiting ones into the same batch. `waiting_served_ratio=1.2` Means when 12 queries are waiting and there's only 10 queries left in the current batch we check if we can fit those 12 waiting queries into the batching strategy, and if yes, then batching happens delaying the 10 running queries by a `prefill` run [env: WAITING_SERVED_RATIO=] [default: 0.3]
          --max-batch-prefill-tokens <MAX_BATCH_PREFILL_TOKENS>
              Limits the number of tokens for the prefill operation. Since this operation take the most memory and is compute bound, it is interesting to limit the number of requests that can be sent. Default to `max_input_tokens + 50` to give a bit of room [env: MAX_BATCH_PREFILL_TOKENS=]
          --max-batch-total-tokens <MAX_BATCH_TOTAL_TOKENS>
              **IMPORTANT** This is one critical control to allow maximum usage of the available hardware [env: MAX_BATCH_TOTAL_TOKENS=]
          --max-waiting-tokens <MAX_WAITING_TOKENS>
              This setting defines how many tokens can be passed before forcing the waiting queries to be put on the batch (if the size of the batch allows for it). New queries require 1 `prefill` forward, which is different from `decode` and therefore you need to pause the running batch in order to run `prefill` to create the correct values for the waiting queries to be able to join the batch [env: MAX_WAITING_TOKENS=] [default: 20]
          --max-batch-size <MAX_BATCH_SIZE>
              Enforce a maximum number of requests per batch Specific flag for hardware targets that do not support unpadded inference [env: MAX_BATCH_SIZE=]
          --cuda-graphs <CUDA_GRAPHS>
              Specify the batch sizes to compute cuda graphs for. Use "0" to disable. Default = "1,2,4,8,16,32" [env: CUDA_GRAPHS=]
          --hostname <HOSTNAME>
              The IP address to listen on [env: HOSTNAME=r-derek-thomas-tgi-benchmark-space-geij6846-b385a-lont4] [default: 0.0.0.0]
      -p, --port <PORT>
              The port to listen on [env: PORT=80] [default: 3000]
          --shard-uds-path <SHARD_UDS_PATH>
              The name of the socket for gRPC communication between the webserver and the shards [env: SHARD_UDS_PATH=] [default: /tmp/text-generation-server]
          --master-addr <MASTER_ADDR>
              The address the master shard will listen on. (setting used by torch distributed) [env: MASTER_ADDR=] [default: localhost]
          --master-port <MASTER_PORT>
              The address the master port will listen on. (setting used by torch distributed) [env: MASTER_PORT=] [default: 29500]
          --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
              The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance [env: HUGGINGFACE_HUB_CACHE=]
          --weights-cache-override <WEIGHTS_CACHE_OVERRIDE>
              The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance [env: WEIGHTS_CACHE_OVERRIDE=]
          --disable-custom-kernels
              For some models (like bloom), text-generation-inference implemented custom cuda kernels to speed up inference. Those kernels were only tested on A100. Use this flag to disable them if you're running on different hardware and encounter issues [env: DISABLE_CUSTOM_KERNELS=]
          --cuda-memory-fraction <CUDA_MEMORY_FRACTION>
              Limit the CUDA available memory. The allowed value equals the total visible memory multiplied by cuda-memory-fraction [env: CUDA_MEMORY_FRACTION=] [default: 1.0]
          --rope-scaling <ROPE_SCALING>
              Rope scaling will only be used for RoPE models and allow rescaling the position rotary to accomodate for larger prompts [env: ROPE_SCALING=] [possible values: linear, dynamic]
          --rope-factor <ROPE_FACTOR>
              Rope scaling will only be used for RoPE models See `rope_scaling` [env: ROPE_FACTOR=]
          --json-output
              Outputs the logs in JSON format (useful for telemetry) [env: JSON_OUTPUT=]
          --otlp-endpoint <OTLP_ENDPOINT>
              [env: OTLP_ENDPOINT=]
          --otlp-service-name <OTLP_SERVICE_NAME>
              [env: OTLP_SERVICE_NAME=] [default: text-generation-inference.router]
          --cors-allow-origin <CORS_ALLOW_ORIGIN>
              [env: CORS_ALLOW_ORIGIN=]
          --api-key <API_KEY>
              [env: API_KEY=]
          --watermark-gamma <WATERMARK_GAMMA>
              [env: WATERMARK_GAMMA=]
          --watermark-delta <WATERMARK_DELTA>
              [env: WATERMARK_DELTA=]
          --ngrok
              Enable ngrok tunneling [env: NGROK=]
          --ngrok-authtoken <NGROK_AUTHTOKEN>
              ngrok authentication token [env: NGROK_AUTHTOKEN=]
          --ngrok-edge <NGROK_EDGE>
              ngrok edge [env: NGROK_EDGE=]
          --tokenizer-config-path <TOKENIZER_CONFIG_PATH>
              The path to the tokenizer config file. This path is used to load the tokenizer configuration which may include a `chat_template`. If not provided, the default config will be used from the model hub [env: TOKENIZER_CONFIG_PATH=]
          --disable-grammar-support
              Disable outlines grammar constrained generation. This is a feature that allows you to generate text that follows a specific grammar [env: DISABLE_GRAMMAR_SUPPORT=]
      -e, --env
              Display a lot of information about your runtime environment
          --max-client-batch-size <MAX_CLIENT_BATCH_SIZE>
              Control the maximum number of inputs that a client can send in a single request [env: MAX_CLIENT_BATCH_SIZE=] [default: 4]
          --lora-adapters <LORA_ADAPTERS>
              Lora Adapters a list of adapter ids i.e. `repo/adapter1,repo/adapter2` to load during startup that will be available to callers via the `adapter_id` field in a request [env: LORA_ADAPTERS=]
          --usage-stats <USAGE_STATS>
              Control if anonymous usage stats are collected. Options are "on", "off" and "no-stack" Defaul is on [env: USAGE_STATS=] [default: on] [possible values: on, off, no-stack]
      -h, --help
              Print help (see more with '--help')
      -V, --version
              Print version


We can launch directly from the cookbook since we dont need the command to be interactive.

We will just be using defaults in this cookbook as the intent is to understand the benchmark tool.

These parameters were changed if you're running on a Space because we don't want to conflict with the Spaces server:
- `--hostname`
- `--port`

Feel free to change or remove them based on your requirements.


```python
!RUST_BACKTRACE=1 \
text-generation-launcher \
--model-id astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit \
--quantize gptq \
--hostname 0.0.0.0 \
--port 1337
```

    [2024-08-16T12:07:56.411768Z INFO text_generation_launcher: Args { model_id: "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit", revision: None, validation_workers: 2, sharded: None, num_shard: None, quantize: Some( Gptq, ), speculate: None, dtype: None, trust_remote_code: false, max_concurrent_requests: 128, max_best_of: 2, max_stop_sequences: 4, max_top_n_tokens: 5, max_input_tokens: None, max_input_length: None, max_total_tokens: None, waiting_served_ratio: 0.3, max_batch_prefill_tokens: None, max_batch_total_tokens: None, max_waiting_tokens: 20, max_batch_size: None, cuda_graphs: None, hostname: "0.0.0.0", port: 1337, shard_uds_path: "/tmp/text-generation-server", master_addr: "localhost", master_port: 29500, huggingface_hub_cache: None, weights_cache_override: None, disable_custom_kernels: false, cuda_memory_fraction: 1.0, rope_scaling: None, rope_factor: None, json_output: false, otlp_endpoint: None, otlp_service_name: "text-generation-inference.router", cors_allow_origin: [], api_key: None, watermark_gamma: None, watermark_delta: None, ngrok: false, ngrok_authtoken: None, ngrok_edge: None, tokenizer_config_path: None, disable_grammar_support: false, env: false, max_client_batch_size: 4, lora_adapters: None, usage_stats: On, }, ..., 2024-08-16T12:08:09.102368Z INFO download: text_generation_launcher: Waiting for download to gracefully shutdown]

# TGI Benchmark
Now lets learn how to launch the benchmark tool!

Here we can see the different settings for TGI Benchmark.

Here are some of the more important TGI Benchmark settings:

- `--tokenizer-name` This is required so the tool knows what tokenizer to use
- `--batch-size` This is important for load testing. We should use enough values to see what happens to throughput and latency. Do note that batch-size in the context of the benchmarking tool is number of virtual users. 
- `--sequence-length` AKA input tokens, it is important to match your use-case needs
- `--decode-length` AKA output tokens, it is important to match your use-case needs
- `--runs` 10 is the default

💡 Tip: Use a low number for `--runs` when you are exploring but a higher number as you finalize to get more precise statistics



```python
!text-generation-benchmark -h
```

    Text Generation Benchmarking tool
    
   
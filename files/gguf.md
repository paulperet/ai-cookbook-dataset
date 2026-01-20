# GGUF
Previously nammed GGML, GGUF is favored by a lot of the community for its ability to run efficiently on CPU and Apple devices, offloading to a GPU if available! Making it a good choice for local testing and deployment as it can make good use of both RAM (and VRAM if available).
### Quantizing with [llama.cpp](https://github.com/ggerganov/llama.cpp)
Here is a list of possible quantizations with llama.cpp:
- `q2_k`
- `q3_k_l`
- `q3_k_m`
- `q3_k_s`
- `q4_0`: <- 4-bit
- `q4_1`
- `q4_k_s`
- `q4_k_m` <- Recommended
- `q5_0`: <- 5-bit
- `q5_1`
- `q5_k_s`
- `q5_k_m`: <- Recommended
- `q6_k`: <- Recommended
- `q8_0`: <- 8-bit, very close to lossless compared to the original weights

Lets do a short demo and quantize Mistral 7B!

First lets install all the dependencies required and `llama.cpp`, as well as downloading the model.


```python
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
model_name = model_id.split('/')[-1]
user_name = "huggingface_username"
hf_token = "read_token"

# install llama.cpp
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && git pull && make clean && LLAMA_CUBLAS=1 make
!pip install -r llama.cpp/requirements.txt
!(cd llama.cpp && make)

# download model
!git lfs install
!git clone https://{user_name}:{hf_token}@huggingface.co/{model_id}
```

[Cloning into 'llama.cpp'..., remote: Enumerating objects: 32489, done., ..., Filtering content: 100% (5/5), 3.00 GiB | 8.66 MiB/s, done., Encountered 4 file(s) that may not have been copied correctly on Windows: ...]

Once everything installed and downloaded, we can convert our model to fp16, required before quantizing to GGUF.


```python
fp16 = f"{model_name}/{model_name.lower()}.fp16.bin"
!python llama.cpp/convert_hf_to_gguf.py {model_name} --outtype f16 --outfile {fp16}
```

[INFO:hf-to-gguf:Loading model: Mistral-7B-Instruct-v0.3, ..., INFO:gguf.gguf_writer:Writing the following files:, INFO:gguf.gguf_writer:Mistral-7B-Instruct-v0.3/mistral-7b-instruct-v0.3.fp16.bin: n_tensors = 291, total_size = 14.5G, Writing: 100% 14.5G/14.5G [01:41<00:00, 143Mbyte/s], INFO:hf-to-gguf:Model successfully exported to Mistral-7B-Instruct-v0.3/mistral-7b-instruct-v0.3.fp16.bin]

Now our model is ready, we can quantize, feel free to change the method, in this example we will quantize to `q4_k_m`.


```python
method = "q4_k_m"
qtype = f"{model_name}/{model_name.lower()}.{method.upper()}.gguf"
!./llama.cpp/llama-quantize {fp16} {qtype} {method}
```

Perfect, we may now test it with `llama.cpp` using the folllowing:


```python
!./llama.cpp/llama-cli -m {qtype} -n 128 --color -ngl 35 -cnv --chat-template mistral
```

[warning: not compiled with GPU offload support, --gpu-layers option will be ignored, warning: see main README.md for information on enabling GPU BLAS support, Log start, main: build = 3613 (fc54ef0d), ..., main: interactive mode on., sampling: ..., generate: n_ctx = 32768, n_batch = 2048, n_predict = 128, n_keep = 1, == Running in interactive mode. ==, - Press Ctrl+C to interject at any time., - Press Return to return control to the AI., - To return control without starting a new line, end your input with '/'. , - If you want to submit another line, end your input with '\'., hello, Hello there! How can I help you today? If you have any questions or need assistance with something, feel free to ask. I'm here to help!, llama_print_timings: ..., llama_print_timings:       total time =   11538.73 ms /    41 tokens]
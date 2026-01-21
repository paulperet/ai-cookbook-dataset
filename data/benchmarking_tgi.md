# Benchmarking TGI: A Step-by-Step Guide

This guide walks you through benchmarking the Text Generation Inference (TGI) server using its official benchmarking tool. Proper benchmarking is crucial for understanding the performance characteristics of your model deployment under realistic loads.

## Prerequisites

Before you begin, ensure you have:
1.  An environment with TGI installed (Docker is recommended).
2.  Sufficient GPU resources for your target model.
3.  The `text-generation-launcher` and `text-generation-benchmark` CLI tools available.

> **Tip:** For a quick test, you can duplicate and use the [tgi-benchmark-space](https://huggingface.co/spaces/derek-thomas/tgi-benchmark-space) on Hugging Face Spaces.

## Step 1: Launch the TGI Server

First, you need to start a TGI server with your chosen model and configuration. The launcher has many options, but the most critical for performance are:

*   `--model-id`: The model to load (from Hugging Face Hub or a local path).
*   `--quantize`: Enables quantization (e.g., `gptq`, `awq`) to reduce memory usage, though it may not always improve speed.
*   `--max-input-tokens`: The maximum allowed prompt length. Setting this allows TGI to optimize memory allocation.
*   `--max-total-tokens`: The maximum combined length of input + output tokens. This defines the total "memory budget" per request.
*   `--max-batch-size`: The maximum number of concurrent requests TGI will process.

Providing the last three parameters allows TGI to optimize specifically for your expected workload.

### 1.1 Check the TGI Launcher Version

Let's first verify the installed version.

```bash
text-generation-launcher --version
```

### 1.2 Launch the Server

For this example, we'll launch a quantized Llama 3 8B Instruct model. We adjust the port to avoid conflicts (e.g., when running in a Space).

```bash
RUST_BACKTRACE=1 \
text-generation-launcher \
--model-id astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit \
--quantize gptq \
--hostname 0.0.0.0 \
--port 1337
```

The server will start, download the model if necessary, and begin listening for requests on port 1337. Keep this process running in a terminal or background.

## Step 2: Run the Benchmark

With the server running, you can now use the `text-generation-benchmark` tool to simulate load and measure performance.

Key benchmark parameters include:

*   `--tokenizer-name`: **Required.** The tokenizer to use for encoding prompts (often the same as the model ID).
*   `--batch-size`: The number of concurrent virtual users/requests to simulate. Test a range of values to see how throughput and latency scale.
*   `--sequence-length`: The length of the input prompt (in tokens).
*   `--decode-length`: The number of new tokens to generate for each request.
*   `--runs`: The number of benchmark iterations to run. Use a lower number (e.g., 3) for exploration and a higher number (e.g., 20) for final, stable results.

### 2.1 Explore Benchmark Options

View the full help menu to see all available parameters.

```bash
text-generation-benchmark -h
```

### 2.2 Execute a Benchmark Run

Now, execute a benchmark. This command simulates 4 concurrent users, each sending a 512-token prompt and requesting 128 new tokens, repeating the test 5 times.

```bash
text-generation-benchmark \
    --tokenizer-name astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit \
    --batch-size 4 \
    --sequence-length 512 \
    --decode-length 128 \
    --runs 5
```

The benchmark tool will connect to your running TGI server (defaulting to `localhost:3000`), so ensure your server's `--port` matches or use the benchmark's `--url` parameter to specify the correct address (e.g., `--url http://localhost:1337`).

## Step 3: Analyze the Results

After the benchmark completes, it will output a summary of key metrics. Focus on:

*   **Throughput (tokens/second)**: How many tokens are generated per second across all concurrent requests. Higher is better.
*   **Latency (ms/token)**: The average time to generate a single token per request. Lower is better.
*   **Latency Percentiles (P50, P90, P99)**: These show the distribution of request latencies. A large gap between P50 and P99 may indicate variability or instability under load.

Experiment by changing the `--batch-size`, `--sequence-length`, and `--decode-length` to match your application's expected traffic patterns and observe how the performance metrics change. This data is essential for right-sizing your deployment infrastructure.
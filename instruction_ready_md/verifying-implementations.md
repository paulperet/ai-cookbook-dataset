# Verifying gpt-OSS Implementations: A Developer's Guide

The new [OpenAI gpt-OSS models](https://openai.com/open-models) introduce several novel concepts to the open-model ecosystem. As with any new technology, ensuring your implementation performs as expected is crucial. This guide provides a step-by-step process for developers to verify their inference solutions or to test any provider's implementation.

## Why Implementing gpt-OSS Models is Different

These models behave more like other OpenAI models than existing open models. Key differences include:

1.  **The Harmony Response Format:** Models are trained on the [OpenAI Harmony format](https://cookbook.openai.com/articles/openai-harmony) to structure conversations. Inference providers must map inputs correctly to this format; incorrect formatting can degrade generation quality and function-calling performance.
2.  **Handling Chain of Thought (CoT) Between Tool Calls:** These models can perform tool calls as part of their reasoning (CoT). The raw CoT must be returned by APIs so developers can pass it back in subsequent turns, along with tool calls and outputs. [Learn more in the dedicated guide](https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot).
3.  **Inference Code Differences:** We published MoE weights in MXFP4 format and provided reference implementations in [PyTorch](https://github.com/openai/gpt-oss/tree/main/gpt_oss/torch) and [Triton](https://github.com/openai/gpt-oss/tree/main/gpt_oss/triton). The [vLLM implementation](https://github.com/vllm-project/vllm/blob/7e3a8dc90670fd312ce1e0d4eba9bf11c571e3ad/vllm/model_executor/models/gpt_oss.py) has also been verified. These serve as educational references for other implementations.

## Prerequisites

Before you begin, ensure you have:
*   A running inference endpoint for a gpt-OSS model (e.g., `gpt-oss-120b`).
*   Node.js (for API compatibility tests) or Python (for running evals) installed.

## Step 1: Understand the Correct API Design

For optimal performance, your API should correctly handle the raw Chain of Thought.

### Option A: Implementing the Responses API (Recommended)

The Responses API is designed for behaviors like outputting raw CoT. The key is to return the raw CoT within the `output` array, wrapped in a `reasoning_text` element.

**Example `output` item:**
```json
{
  "type": "reasoning",
  "id": "item_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
  "status": "completed",
  "summary": [ /* optional summary elements */ ],
  "content": [
    {
      "type": "reasoning_text",
      "text": "The user needs to know the weather, I will call the get_weather tool."
    }
  ]
}
```
Subsequent API calls must receive these items and insert them back into the Harmony-formatted prompt. [See the full Responses API specification](https://platform.openai.com/docs/api-reference/responses/create).

### Option B: Implementing a Chat Completions-Compatible API

Many providers offer a Chat Completions-compatible API. To ensure compatibility with clients like the OpenAI Agents SDK, **use a `reasoning` field as the primary property for the raw CoT in Chat Completions messages**.

## Step 2: Run Quick Compatibility Tests

We provide a Node.js test suite to verify basic tool-calling functionality and API shape compatibility.

1.  **Clone the repository and navigate to the test directory:**
    ```bash
    git clone https://github.com/openai/gpt-oss.git
    cd gpt-oss/compatibility-test/
    ```

2.  **Install the dependencies:**
    ```bash
    npm install
    ```

3.  **Configure your provider.** Edit the `providers.ts` file to add your provider's endpoint and authentication details.

4.  **Run the test suite against your provider:**
    ```bash
    npm start -- --provider <your-provider-name>
    ```
    *   Use `-n 1` to run only one test for easier debugging.
    *   Use `--streaming` to test streaming events.
    *   For verbose request/response logging, use: `DEBUG=openai-agents:openai npm start -- --provider <provider-name>`

### Interpreting the Results

A successful test will show:
*   `0` invalid requests.
*   Over `90%` on both `pass@k` and `pass^k` metrics.

This indicates your API is likely compatible and handles basic function calling correctly. Detailed responses are logged to a `.jsonl` file in your directory.

> **Note:** This is a smoke test. It does not guarantee full inference accuracy or complete API compatibility.

## Step 3: Verify Correctness with Evals

For thorough verification, run standardized evaluations. The team at [Artificial Analysis](https://artificialanalysis.ai/models/gpt-oss-120b/providers#evaluations) runs public evals (AIME, GPQA). You should also run them yourself.

The evaluation harness is in the same repository.

1.  **Navigate to the Python evals directory** (from the root of the cloned repo).

2.  **Test a Responses API-compatible endpoint:**
    ```bash
    python -m gpt_oss.evals --base-url http://localhost:8000/v1 --eval aime25 --sampler responses --model openai/gpt-oss-120b --reasoning-effort high
    ```

3.  **Test a Chat Completions API-compatible endpoint:**
    ```bash
    python -m gpt_oss.evals --base-url http://localhost:8000/v1 --eval aime25 --sampler chat_completions --model openai/gpt-oss-120b --reasoning-effort high
    ```
    *   You can change the `--eval` flag to `gpqa` or `healthbench`.
    *   Adjust the `--base-url` and `--model` arguments to match your deployment.

### Final Verification

If your implementation passes both the **compatibility tests (Step 2)** and achieves **eval scores similar to the published benchmarks (Step 3)**, you can be confident you have a correct implementation of gpt-OSS.
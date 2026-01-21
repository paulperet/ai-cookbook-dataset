# Cross-Platform Inference with Rust and Candle

This guide walks you through building a text generation application using Rust and the Candle ML framework. Rust provides high performance, memory safety, and true cross-platform compatibility, making it an excellent choice for deploying AI models on diverse systems—from servers to mobile devices.

## Prerequisites

1.  **Install Rust:** Ensure you have Rust and Cargo installed. Follow the instructions at [rust-lang.org](https://www.rust-lang.org/tools/install).
2.  **Internet Connection:** The model and tokenizer will be downloaded from Hugging Face Hub on first run.

## Step 1: Create a New Rust Project

Open your terminal and create a new Rust project named `phi-console-app`.

```bash
cargo new phi-console-app
cd phi-console-app
```

This command creates a basic project structure with a `Cargo.toml` manifest and a `src/main.rs` source file.

Next, open `Cargo.toml` and add the required dependencies. We'll use `candle` for tensor operations and model loading, `hf-hub` to download models, and `tokenizers` for text processing.

```toml
[package]
name = "phi-console-app"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = { version = "0.6.0" }
candle-transformers = { version = "0.6.0" }
hf-hub = { version = "0.3.2", features = ["tokio"] }
rand = "0.8"
tokenizers = "0.15.2"
```

## Step 2: Define the Application Parameters

Open `src/main.rs`. We'll start by defining the core parameters that control the text generation process. These are hardcoded for simplicity but can be easily parameterized later.

Add the following code inside the `main` function:

```rust
// 1. Configure basic parameters
let temperature: f64 = 1.0;
let sample_len: usize = 100;
let top_p: Option<f64> = None;
let repeat_last_n: usize = 64;
let repeat_penalty: f32 = 1.2;
let mut rng = rand::thread_rng();
let seed: u64 = rng.gen();
let prompt = "<|user|>\nWrite a haiku about ice hockey<|end|>\n<|assistant|>";
let device = Device::Cpu;
```

Let's break down these parameters:
*   **`temperature`**: Controls randomness in sampling. Higher values (e.g., 1.0) produce more creative, varied text.
*   **`sample_len`**: The maximum number of tokens to generate.
*   **`top_p`**: Implements nucleus sampling, limiting the token selection pool to a cumulative probability threshold.
*   **`repeat_last_n` & `repeat_penalty`**: Work together to discourage repetitive text by penalizing tokens that have appeared recently.
*   **`seed`**: A random seed for reproducible sampling.
*   **`prompt`**: The input text for the model. We use special tokens (`<|user|>`, `<|end|>`, `<|assistant|>`) to format the conversation for the Phi-3 model.
*   **`device`**: Specifies where to run computations. Here we use the CPU, but Candle also supports CUDA and Metal backends.

## Step 3: Download the Model and Tokenizer

We'll use the Hugging Face Hub API to download the quantized Phi-3 model and its tokenizer. The model is cached locally, so subsequent runs will be faster.

```rust
// 2. Download/prepare model and tokenizer
let api = hf_hub::api::sync::Api::new()?;
let model_path = api
    .repo(hf_hub::Repo::with_revision(
        "microsoft/Phi-3-mini-4k-instruct-gguf".to_string(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ))
    .get("Phi-3-mini-4k-instruct-q4.gguf")?;

let tokenizer_path = api
    .model("microsoft/Phi-3-mini-4k-instruct".to_string())
    .get("tokenizer.json")?;
let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
```

This code:
1.  Initializes a synchronous Hugging Face Hub API client.
2.  Downloads the 4-bit quantized GGUF model file (~2.4 GB on first run).
3.  Downloads the tokenizer configuration file.
4.  Loads the tokenizer from the downloaded file.

## Step 4: Load the Model into Memory

Now we load the quantized model weights from the GGUF file into a `Phi3` struct, ready for inference.

```rust
// 3. Load model
let mut file = std::fs::File::open(&model_path)?;
let model_content = gguf_file::Content::read(&mut file)?;
let mut model = Phi3::from_gguf(false, model_content, &mut file, &device)?;
```

The `from_gguf` function parses the GGUF file format and initializes the model's weights on the specified device (CPU in our case).

## Step 5: Prepare the Prompt for Inference

Before generating text, we need to tokenize the input prompt and set up the sampling logic.

```rust
// 4. Process prompt and prepare for inference
let tokens = tokenizer.encode(prompt, true).map_err(|e| e.to_string())?;
let tokens = tokens.get_ids();
let to_sample = sample_len.saturating_sub(1);
let mut all_tokens = vec![];

let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), top_p);

let mut next_token = *tokens.last().unwrap();
let eos_token = *tokenizer.get_vocab(true).get("<|end|>").unwrap();
let mut prev_text_len = 0;

for (pos, &token) in tokens.iter().enumerate() {
    let input = Tensor::new(&[token], &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, pos)?;
    let logits = logits.squeeze(0)?;

    // Sample next token only for the last token in the prompt
    if pos == tokens.len() - 1 {
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
    }
}
```

Here's what happens:
1.  The prompt is converted into a sequence of token IDs.
2.  A `LogitsProcessor` is created to handle temperature-based sampling.
3.  We identify the end-of-sequence (`<|end|>`) token ID to know when to stop generation.
4.  The prompt tokens are fed through the model one by one. For the final prompt token, we sample the first new token to begin the generation sequence.

## Step 6: Generate Text Token by Token

This is the core generation loop. We repeatedly sample the next token, apply penalties to avoid repetition, and decode the growing sequence.

```rust
// 5. Inference
for index in 0..to_sample {
    let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, tokens.len() + index)?;
    let logits = logits.squeeze(0)?;
    
    // Apply repetition penalty if configured
    let logits = if repeat_penalty == 1. {
        logits
    } else {
        let start_at = all_tokens.len().saturating_sub(repeat_last_n);
        candle_transformers::utils::apply_repeat_penalty(
            &logits,
            repeat_penalty,
            &all_tokens[start_at..],
        )?
    };

    next_token = logits_processor.sample(&logits)?;
    all_tokens.push(next_token);

    // Decode the current sequence of tokens
    let decoded_text = tokenizer.decode(&all_tokens, true).map_err(|e| e.to_string())?;

    // Only print the newly generated text (streaming output)
    if decoded_text.len() > prev_text_len {
        let new_text = &decoded_text[prev_text_len..];
        print!("{new_text}");
        std::io::stdout().flush()?;
        prev_text_len = decoded_text.len();
    }

    // Stop if we generate the end-of-sequence token
    if next_token == eos_token {
        break;
    }
}
```

The loop continues until we either:
*   Reach the maximum token count (`sample_len`).
*   Generate the end-of-sequence token.
*   The streaming output prints tokens as they're generated, providing real-time feedback.

## Step 7: Build and Run the Application

Compile and run your application with the following command:

```bash
cargo run --release
```

The `--release` flag enables optimizations for faster execution. On first run, the model will be downloaded, which may take several minutes depending on your connection.

You should see output similar to:

```
Puck glides swiftly,  
Blades on ice dance and clash—peace found 
in the cold battle.
```

Each run will produce a different haiku due to the random sampling.

## Conclusion

You've successfully built a cross-platform text generation application in under 100 lines of Rust. This application leverages:
*   **Candle** for efficient tensor operations and model loading.
*   **Quantized models** (GGUF) for reduced memory footprint.
*   **Streaming generation** for real-time output.

The same codebase can run on Windows, macOS, Linux, and can be adapted as a library for mobile applications thanks to Rust's excellent cross-compilation support.

## Complete Code Reference

Here's the full `src/main.rs` for reference:

```rust
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::{
    generation::LogitsProcessor, models::quantized_phi3::ModelWeights as Phi3,
};
use rand::Rng;
use std::io::Write;
use tokenizers::Tokenizer;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Configure basic parameters
    let temperature: f64 = 1.0;
    let sample_len: usize = 100;
    let top_p: Option<f64> = None;
    let repeat_last_n: usize = 64;
    let repeat_penalty: f32 = 1.2;
    let mut rng = rand::thread_rng();
    let seed: u64 = rng.gen();
    let prompt = "<|user|>\nWrite a haiku about ice hockey<|end|>\n<|assistant|>";
    let device = Device::Cpu;

    // 2. Download/prepare model and tokenizer
    let api = hf_hub::api::sync::Api::new()?;
    let model_path = api
        .repo(hf_hub::Repo::with_revision(
            "microsoft/Phi-3-mini-4k-instruct-gguf".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ))
        .get("Phi-3-mini-4k-instruct-q4.gguf")?;

    let tokenizer_path = api
        .model("microsoft/Phi-3-mini-4k-instruct".to_string())
        .get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;

    // 3. Load model
    let mut file = std::fs::File::open(&model_path)?;
    let model_content = gguf_file::Content::read(&mut file)?;
    let mut model = Phi3::from_gguf(false, model_content, &mut file, &device)?;

    // 4. Process prompt and prepare for inference
    let tokens = tokenizer.encode(prompt, true).map_err(|e| e.to_string())?;
    let tokens = tokens.get_ids();
    let to_sample = sample_len.saturating_sub(1);
    let mut all_tokens = vec![];

    let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), top_p);

    let mut next_token = *tokens.last().unwrap();
    let eos_token = *tokenizer.get_vocab(true).get("<|end|>").unwrap();
    let mut prev_text_len = 0;

    for (pos, &token) in tokens.iter().enumerate() {
        let input = Tensor::new(&[token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos)?;
        let logits = logits.squeeze(0)?;

        if pos == tokens.len() - 1 {
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
        }
    }

    // 5. Inference
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &all_tokens[start_at..],
            )?
        };

        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        let decoded_text = tokenizer.decode(&all_tokens, true).map_err(|e| e.to_string())?;

        if decoded_text.len() > prev_text_len {
            let new_text = &decoded_text[prev_text_len..];
            print!("{new_text}");
            std::io::stdout().flush()?;
            prev_text_len = decoded_text.len();
        }

        if next_token == eos_token {
            break;
        }
    }

    Ok(())
}
```

## Platform-Specific Notes

For **aarch64 Linux** or **aarch64 Windows**, create a `.cargo/config` file in your project root with the following content to enable FP16 optimizations:

```toml
[target.aarch64-pc-windows-msvc]
rustflags = [
    "-C", "target-feature=+fp16"
]

[target.aarch64-unknown-linux-gnu]
rustflags = [
    "-C", "target-feature=+fp16"
]
```

## Next Steps

Explore the [Candle examples repository](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-phi/main.rs) for more advanced use cases, including GPU acceleration, different sampling strategies, and alternative model architectures.
# On-Device AI: Building an Android App with Microsoft Phi-3.5

This guide walks you through the process of converting the Microsoft Phi-3.5-mini-instruct model into a format that runs efficiently on Android devices using TensorFlow Lite and MediaPipe. By the end, you'll have a bundled model ready for integration into your Android application.

## Prerequisites

Before you begin, ensure you have the following setup:

*   **Operating System:** Ubuntu 20.04 or 22.04 (A cloud VM like Azure is recommended).
*   **Python:** Version 3.10.12. Using Conda to manage your environment is suggested.
*   **Target Device:** An Android device or emulator running Android 14+.
*   **Android Debug Bridge (ADB):** Installed and configured to communicate with your target device.

## Step 1: Set Up the Conversion Environment

You will use Google's AI Edge Torch library to convert the PyTorch model. First, clone the repository and install its dependencies.

1.  Clone the `ai-edge-torch` repository and navigate into it.
    ```bash
    git clone https://github.com/google-ai-edge/ai-edge-torch.git
    cd ai-edge-torch
    ```

2.  Install the required Python libraries.
    ```bash
    pip install -r requirements.txt -U
    pip install tensorflow-cpu -U
    pip install -e .
    ```

## Step 2: Download the Phi-3.5 Model

Download the `Phi-3.5-mini-instruct` model from Hugging Face. This requires Git LFS.

1.  Install and initialize Git LFS.
    ```bash
    git lfs install
    ```

2.  Clone the model repository.
    ```bash
    git clone https://huggingface.co/microsoft/Phi-3.5-mini-instruct
    ```

## Step 3: Convert the Model to TensorFlow Lite

Now, use the conversion script provided by AI Edge Torch. This step quantizes the model (reduces its size and improves inference speed) and prepares it with specific sequence length parameters.

Run the following command, replacing the placeholder paths with your actual directory paths.

```bash
python ai-edge-torch/ai_edge_torch/generative/examples/phi/convert_phi3_to_tflite.py \
  --checkpoint_path /path/to/your/Phi-3.5-mini-instruct \
  --tflite_path /path/to/save/phi3.5_model.tflite \
  --prefill_seq_len 1024 \
  --kv_cache_max_len 1280 \
  --quantize True
```

**Parameters Explained:**
*   `--checkpoint_path`: The directory where you cloned the Phi-3.5 model.
*   `--tflite_path`: The desired output path for the `.tflite` file.
*   `--prefill_seq_len`: The sequence length for the initial prompt processing.
*   `--kv_cache_max_len`: The maximum length for the Key-Value cache during generation.
*   `--quantize`: Enables quantization for a smaller, faster model.

## Step 4: Bundle the Model for MediaPipe

To use the model with MediaPipe's on-device LLM Inference API on Android, you must bundle the `.tflite` model with its tokenizer into a single `.task` file.

1.  Install the MediaPipe Python package.
    ```bash
    pip install mediapipe
    ```

2.  Create a Python script (e.g., `bundle_model.py`) with the following code. You will need to fill in the correct paths and tokens.

    ```python
    import mediapipe as mp
    from mediapipe.tasks.python.genai import bundler

    # Define your configuration
    config = bundler.BundleConfig(
        tflite_model='/path/to/your/phi3.5_model.tflite',
        tokenizer_model='/path/to/your/Phi-3.5-mini-instruct/tokenizer.model',
        start_token='<|endoftext|>', # Consult the model's documentation for the correct start token
        stop_tokens=['<|endoftext|>', '<|endoftext|>'], # Consult the model's documentation for stop tokens
        output_filename='/path/to/save/phi3.5.task',
        enable_bytes_to_unicode_mapping=True, # Set to False if your tokenizer doesn't require it
    )

    # Create the bundled .task file
    bundler.create_bundle(config)
    print("Model bundling complete.")
    ```

    **Important:** You must find the correct `start_token` and `stop_tokens` for the Phi-3.5 model. These are specific to the model's training and tokenization process. Check the model card or source files in the Hugging Face repository.

3.  Run the script to generate the `phi3.5.task` file.
    ```bash
    python bundle_model.py
    ```

## Step 5: Deploy the Model to Your Android Device

With the `.task` file created, push it to your connected Android device's local storage.

1.  Connect your Android device via USB and ensure `adb devices` lists it.
2.  Run the following commands to clear any old models and deploy the new one.

    ```bash
    adb shell rm -rf /data/local/tmp/llm/
    adb shell mkdir -p /data/local/tmp/llm/
    adb push /path/to/your/phi3.5.task /data/local/tmp/llm/phi3.task
    ```

## Next Steps: Running on Android

Your model is now ready on the device. To use it in your Android application:

1.  Integrate the **MediaPipe LLM Inference API** into your Android project.
2.  In your app code, initialize the `LlmInference` object, pointing it to the model path on the device: `/data/local/tmp/llm/phi3.task`.
3.  You can now use the inference object to generate text, answer questions, or summarize content completely on-device.

Refer to the official [MediaPipe LLM Inference documentation](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference) for detailed instructions on the Android API integration and usage.
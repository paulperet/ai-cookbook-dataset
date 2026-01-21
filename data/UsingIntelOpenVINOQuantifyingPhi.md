# Quantizing Phi-3.5 Models with Intel OpenVINO

## Introduction

Intel OpenVINO is an open-source toolkit designed to optimize and deploy deep learning models from the cloud to the edge. It accelerates inference for a wide range of AI workloads—including generative AI, language, and vision—across Intel hardware like CPUs, GPUs, and NPUs. This makes it a powerful tool for enabling AI capabilities on end-user devices, such as AI PCs and Copilot PCs.

In this guide, you will learn how to quantize two members of the Phi-3.5 family—the **Phi-3.5-mini-instruct** language model and the **Phi-3.5-vision-instruct** multimodal model—using OpenVINO. Quantization reduces model size and accelerates inference, which is crucial for efficient deployment on resource-constrained hardware.

## Prerequisites and Setup

Before you begin, ensure you have the necessary Python packages installed. Create a `requirements.txt` file with the following content:

```txt
--extra-index-url https://download.pytorch.org/whl/cpu
optimum-intel>=1.18.2
nncf>=2.11.0
openvino>=2024.3.0
transformers>=4.40
openvino-genai>=2024.3.0.0
```

Then, install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Step 1: Quantizing Phi-3.5-mini-instruct

The `optimum-intel` CLI provides a straightforward way to export and quantize Hugging Face models for OpenVINO. You will convert the model to the INT4 format with symmetric quantization, which balances performance and accuracy.

1.  Set environment variables for the model ID and your desired output directory.

    ```bash
    export llm_model_id="microsoft/Phi-3.5-mini-instruct"
    export llm_model_path="./quantized_phi3_instruct"
    ```

2.  Run the `optimum-cli export` command. This command:
    *   Exports the model to the OpenVINO Intermediate Representation (IR) format.
    *   Applies **INT4 symmetric quantization**.
    *   Uses a **group size of 128** and a **compression ratio of 0.6** to optimize the weight quantization.
    *   The `--trust-remote-code` flag is required for models with custom code, like Phi-3.

    ```bash
    optimum-cli export openvino \
      --model $llm_model_id \
      --task text-generation-with-past \
      --weight-format int4 \
      --group-size 128 \
      --ratio 0.6 \
      --sym \
      --trust-remote-code \
      $llm_model_path
    ```

Upon completion, your quantized model will be saved in the directory specified by `$llm_model_path`.

## Step 2: Quantizing Phi-3.5-vision-instruct

Quantizing the vision model requires a Python script, as the process involves handling both language and vision components. We'll use a helper script from the OpenVINO Notebooks repository.

1.  Create a new Python script or Jupyter notebook and begin by importing the necessary modules.

    ```python
    import requests
    from pathlib import Path
    import nncf
    ```

2.  Download the required helper scripts from the OpenVINO GitHub repository if they are not already present.

    ```python
    # Download the main conversion script
    if not Path("ov_phi3_vision.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/phi-3-vision/ov_phi3_vision.py")
        open("ov_phi3_vision.py", "w").write(r.text)

    # Download optional Gradio helper (for UI demos)
    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/phi-3-vision/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    # Download general notebook utilities
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)
    ```

3.  Now, import the conversion function and define your model and quantization parameters.

    ```python
    from ov_phi3_vision import convert_phi3_model

    model_id = "microsoft/Phi-3.5-vision-instruct"
    out_dir = Path("./quantized_phi3_vision")  # Specify your output directory

    # Define the compression configuration
    compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_SYM,  # Use INT4 symmetric quantization
        "group_size": 64,  # Group size for quantization
        "ratio": 0.6,      # Compression ratio
    }
    ```

4.  Finally, execute the conversion. The script will download the model, apply quantization, and save the OpenVINO-optimized model to the `out_dir`.

    ```python
    if not out_dir.exists():
        convert_phi3_model(model_id, out_dir, compression_configuration)
    ```

## Next Steps and Resources

You have successfully quantized both the Phi-3.5 Instruct and Vision models. These optimized models are now ready for deployment and inference using the OpenVINO runtime on your Intel AI PC.

To learn more about using these quantized models, you can explore the following sample labs:

*   **Lab: Introducing Phi-3.5 Instruct** - Learn how to run the quantized language model.
*   **Lab: Introducing Phi-3.5 Vision (Image)** - Learn how to use the vision model for image analysis.
*   **Lab: Introducing Phi-3.5 Vision (Video)** - Learn how to apply the vision model to video analysis.

### Further Reading
*   [Intel OpenVINO Official Website](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
*   [OpenVINO Generative AI GitHub Repository](https://github.com/openvinotoolkit/openvino.genai)
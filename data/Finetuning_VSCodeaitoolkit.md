# AI Toolkit for VS Code: A Guide to Local and Remote Fine-Tuning

The AI Toolkit for VS Code integrates models from Azure AI Studio and other catalogs like Hugging Face, streamlining the development of AI applications. This guide walks you through setting up your environment and performing model fine-tuning both locally and on remote Azure resources.

## Prerequisites & Setup

Before you begin, ensure you have the following:

1.  **Install the Extension:** Install the [AI Toolkit for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio) from the marketplace.
2.  **Local GPU (Optional):** For local fine-tuning, ensure your NVIDIA drivers are installed if you plan to use a GPU.
3.  **Hugging Face Account:** If you intend to use datasets from Hugging Face, run `huggingface-cli login` in your terminal to authenticate.
4.  **Conda Environment:** The toolkit uses Conda for environment management. Activate your project's Conda environment before running any commands.
    ```bash
    conda activate [your-conda-env-name]
    ```

## Part 1: Local Development

This section covers fine-tuning and running inference on your local machine.

### Step 1: Choose and Download a Base Model

First, select a model suitable for your platform and hardware from the catalog. The Phi-3-mini model is a common choice for getting started.

| Platform(s) | GPU Available | Recommended Model | Size |
| :---------- | :------------ | :---------------- | :--- |
| Windows     | Yes           | `Phi-3-mini-4k-**directml**-int4-awq-block-128-onnx` | ~2.1 GB |
| Linux       | Yes           | `Phi-3-mini-4k-**cuda**-int4-onnx` | ~2.3 GB |
| Windows/Linux | No          | `Phi-3-mini-4k-**cpu**-int4-rtn-block-32-acc-level-4-onnx` | ~2.7 GB |

You do not need an Azure account to download these models. The download will begin automatically when you configure your project in the toolkit.

### Step 2: Configure and Run Fine-Tuning

The toolkit uses **Microsoft Olive** to run QLoRA fine-tuning, which is memory-efficient and optimized for local execution.

1.  Open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and run `AI Toolkit: Focus on Resource View`.
2.  Navigate to **Model Fine-tuning**, give your project a name, and select a location.
3.  Click **Configure Project**. Ensure the **"Fine-tune locally"** option is enabled.
4.  Review the Olive configuration settings (they are pre-set with sensible defaults) and click **Generate Project**.
5.  Once the project is generated, open a terminal in the project root and run the fine-tuning script.
    ```bash
    python finetuning/invoke_olive.py
    ```
    The process will save checkpoints and the final fine-tuned model in the `./models` directory.

### Step 3: Run Inference with Your Fine-Tuned Model

After fine-tuning, you can interact with your model using different interfaces.

1.  Navigate to the inference directory:
    ```bash
    cd inference
    ```

2.  Choose your interface:
    *   **For a console chat:** Run the console interface script.
        ```bash
        python console_chat.py
        ```
    *   **For a web browser chat (Gradio):** Run the Gradio script and open the provided link (e.g., `http://127.0.0.1:7860`) in your browser.
        ```bash
        python gradio_chat.py
        ```
    *   **To test the base model (without fine-tuning):** Use the `--baseonly` flag with the Gradio script.
        ```bash
        python gradio_chat.py --baseonly
        ```

## Part 2: Remote Development (Private Preview)

This section guides you through fine-tuning and deploying models on Azure Container Apps, which is currently in Private Preview.

### Step 1: Enable the Feature and Check Prerequisites

1.  **Enable the Feature Flag:**
    *   Open VS Code Settings (`File > Preferences > Settings`).
    *   Go to `Extensions > AI Toolkit`.
    *   Check the option **"Enable Remote Fine-tuning And Inference"**.
    *   Reload VS Code.

2.  **Ensure Capacity:** Verify your Azure subscription has GPU quota for Azure Container Apps. If not, [submit a support ticket](https://azure.microsoft.com/support/create-ticket/).
3.  **Prepare Hugging Face Token:** If using private datasets, [generate a Hugging Face access token](https://huggingface.co/docs/hub/security-tokens).

### Step 2: Create a Remote Project

1.  Run `AI Toolkit: Focus on Resource View` from the command palette.
2.  Navigate to **Model Fine-tuning**, name your project, and select a location.
3.  Click **Configure Project**. **Crucially, do NOT enable "Fine-tune locally"**.
4.  Adjust the Olive configuration as needed and click **Generate Project**.
5.  Click **"Relaunch Window In Workspace"** to open your new remote project.

### Step 3: Provision Azure Resources for Fine-Tuning

Run the following command from the command palette to set up the required Azure infrastructure:
```
AI Toolkit: Provision Azure Container Apps job for fine-tuning
```
Monitor the provisioning progress via the link provided in the VS Code Output panel.

**Optional:** If using a private Hugging Face dataset, add your token as a secret to avoid manual login:
```
AI Toolkit: Add Azure Container Apps Job secret for fine-tuning
```
Set the secret name as `HF_TOKEN` and provide your token as the value.

### Step 4: Execute Remote Fine-Tuning

Start the fine-tuning job on Azure by running:
```
AI Toolkit: Run fine-tuning
```
You can monitor the logs:
*   **In Azure Portal:** Use the link from the output panel.
*   **In VS Code:** Run `AI Toolkit: Show the running fine-tuning job streaming logs`.

The fine-tuned model adapters will be saved to Azure Files.

### Step 5: Deploy a Remote Inference Endpoint

Once fine-tuning is complete, deploy an endpoint to interact with your model.

1.  **Provision Inference Resources:** Run the provisioning command for inference.
    ```
    AI Toolkit: Provision Azure Container Apps for inference
    ```
2.  **Deploy Your Code:** To deploy the inference application (or update it), run:
    ```
    AI Toolkit: Deploy for inference
    ```
3.  **Access Your Endpoint:** After successful deployment, click the **"Go to Inference Endpoint"** button in the VS Code notification. The endpoint URL is also stored in `./infra/inference.config.json` under `ACA_APP_ENDPOINT`.

## Next Steps and Resources

*   **Local Fine-Tuning Guides:**
    *   [Fine-tuning Getting Started Guide](https://learn.microsoft.com/windows/ai/toolkit/toolkit-fine-tune)
    *   [Fine-tuning with a HuggingFace Dataset](https://github.com/microsoft/vscode-ai-toolkit/blob/main/archive/walkthrough-hf-dataset.md)
*   **Remote Development Documentation:**
    *   [Fine-Tuning Models Remotely](https://aka.ms/ai-toolkit/remote-provision)
    *   [Inferencing with the Fine-Tuned Model](https://aka.ms/ai-toolkit/remote-inference)

You are now ready to build and deploy AI applications using the AI Toolkit for VS Code, leveraging both local compute and scalable Azure cloud resources.
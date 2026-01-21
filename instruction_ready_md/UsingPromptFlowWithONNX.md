# Building a Local AI Application with Prompt Flow and Phi-3.5 ONNX

This guide walks you through creating a local AI application using Microsoft's Prompt Flow development suite and a quantized Phi-3.5-instruct model in ONNX format. By the end, you will have a working pipeline that runs efficiently on a Windows machine with a GPU.

## Prerequisites

Before you begin, ensure you have:
*   A Windows machine with a compatible NVIDIA GPU.
*   Visual Studio Code (VS Code) installed.
*   A Python environment (e.g., Conda or venv) ready for use.

## Step 1: Set Up the ONNX Runtime GenAI Environment

The ONNX Runtime GenAI library provides optimized inference for generative AI models on Windows GPU.

1.  Follow the dedicated setup guide to install and configure ONNX Runtime GenAI for your Windows GPU environment. The instructions cover installing necessary drivers, the ONNX Runtime package, and any specific dependencies for the Phi-3.5 model.
    *   **Guide:** [ORTWindowGPUGuideline.md](./ORTWindowGPUGuideline.md)

## Step 2: Install the Prompt Flow VS Code Extension

Prompt Flow integrates directly into VS Code, providing a visual interface for building and testing AI workflows.

1.  Open VS Code.
2.  Navigate to the Extensions view (Ctrl+Shift+X).
3.  Search for "Prompt flow" and install the official Microsoft extension.
4.  Once installed, click on the Prompt Flow icon in the Activity Bar to open the extension pane.
5.  In the extension pane, select **"Installation dependencies"**. This will guide you through installing the Prompt Flow SDK into your active Python environment.

## Step 3: Configure the Sample Project

We'll use a provided sample project to get started quickly.

1.  Download the sample code: [onnx_inference_pf Sample](../../../../code/09.UpdateSamples/Aug/pf/onnx_inference_pf/)
2.  Open the downloaded folder in VS Code.
3.  **Configure the Python Environment:**
    *   Open the file `flow.dag.yaml`.
    *   At the top of the file, ensure the `environment.python` path points to your intended Python executable (e.g., from your Conda or venv environment).
4.  **Point to Your Phi-3.5 ONNX Model:**
    *   Open the file `chat_phi3_ort.py`.
    *   Locate the section where the model path is defined. Update the path variable to point to the location of your downloaded `Phi-3.5-instruct` ONNX model files on your local machine.

    ```python
    # Example: Update this path to your model's location
    model_path = "C:/models/phi-3.5-instruct-onnx"
    ```

## Step 4: Test the Prompt Flow

Now, let's run a test to ensure everything is connected correctly.

1.  In VS Code, open the `flow.dag.yaml` file.
2.  Click the **"Visual Editor"** button (usually at the top-right of the editor). This opens a diagram view of your AI flow.
3.  In the Visual Editor, click the **"Run"** button to execute the flow. This will use the default inputs defined in the flow to test the connection to your local Phi-3.5 ONNX model.
4.  Observe the output in the **"Output"** or **"Run"** tab that opens. You should see the model's generated response.

## Step 5: Run a Batch Evaluation (Optional)

For more thorough testing, you can run a batch evaluation using a set of predefined questions and answers.

1.  Open a terminal in VS Code (Terminal -> New Terminal).
2.  Navigate to the root of your sample project folder.
3.  Execute the following command, replacing `'Your eval qa name'` with a descriptive name for your run:

    ```bash
    pf run create --file batch_run.yaml --stream --name 'Your eval qa name'
    ```

4.  The `--stream` flag will show you the progress in the terminal. Once complete, Prompt Flow will automatically open your default web browser to display detailed results, including metrics like accuracy and latency for each test case in your batch.

You have now successfully set up a local AI application pipeline using Prompt Flow and a Phi-3.5 ONNX model. You can use this foundation to build more complex flows, integrate other tools, or deploy the solution for production use.
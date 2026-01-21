# Guide: Setting Up ONNX Runtime GenAI with GPU on Windows

This guide provides a step-by-step tutorial for setting up ONNX Runtime (ORT) with GPU acceleration on a Windows system. You will configure your environment, install necessary dependencies, and run inference using a pre-trained model to leverage GPU performance improvements.

## Prerequisites

Before starting, ensure you have:
*   A Windows machine with an NVIDIA GPU.
*   Administrator privileges for software installation.

## Step 1: Set Up the Python Environment

We recommend using Miniforge to manage your Python environment.

1.  Download and install [Miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe).
2.  Open a terminal (like Anaconda Prompt or Windows Terminal) and create a new Conda environment with Python 3.11.8.

    ```bash
    conda create -n pydev python==3.11.8
    conda activate pydev
    ```

    > **Note:** If you have any existing `onnx` or `onnxruntime` packages installed, uninstall them before proceeding.

## Step 2: Install Build Tools

### Install CMake
CMake is required for building native extensions. Install it using the Windows Package Manager (`winget`).

```bash
winget install -e --id Kitware.CMake
```

### Install Visual Studio 2022 (Optional)
The Visual Studio C++ build tools are necessary if you plan to compile ONNX Runtime from source. You can skip this step if you do not intend to compile.

1.  Download and run the [Visual Studio 2022 Installer](https://visualstudio.microsoft.com/downloads/).
2.  During installation, select the **"Desktop development with C++"** workload.

## Step 3: Install NVIDIA GPU Drivers and Libraries

To enable GPU acceleration, you must install the correct NVIDIA drivers and libraries.

1.  **NVIDIA GPU Driver**: Download and install the latest driver for your GPU from the [NVIDIA Driver Website](https://www.nvidia.com/en-us/drivers/).
2.  **NVIDIA CUDA 12.4**: Download and install CUDA 12.4 from the [CUDA 12.4 Archive](https://developer.nvidia.com/cuda-12-4-0-download-archive). Use the default installation settings.
3.  **NVIDIA cuDNN 9.4**: Download cuDNN 9.4 for CUDA 12.x from the [cuDNN Download Page](https://developer.nvidia.com/cudnn-downloads). You will need to create a free NVIDIA Developer account.

## Step 4: Configure the NVIDIA Environment

After installing CUDA and cuDNN, you need to copy the cuDNN files into your CUDA installation directory.

Assuming default installation paths, copy the files as follows:

*   Copy all files from `C:\Program Files\NVIDIA\CUDNN\v9.4\bin\12.6\` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\`.
*   Copy all files from `C:\Program Files\NVIDIA\CUDNN\v9.4\include\12.6\` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include\`.
*   Copy all files from `C:\Program Files\NVIDIA\CUDNN\v9.4\lib\12.6\` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64\`.

## Step 5: Download a Pre-Trained ONNX Model

We will use the `Phi-3.5-mini-instruct` model in ONNX format for testing. You'll need `git` and `git-lfs` to download it.

1.  Install Git and Git LFS:
    ```bash
    winget install -e --id Git.Git
    winget install -e --id GitHub.GitLFS
    git lfs install
    ```
2.  Clone the model repository:
    ```bash
    git clone https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx
    ```

## Step 6: Run Inference with the Model

You can now test the setup by running a provided inference notebook.

1.  Navigate to the directory containing the `ortgpu-phi35-instruct.ipynb` notebook.
2.  Open the notebook with Jupyter Lab or Jupyter Notebook.
3.  Execute the cells to perform inference. If your setup is correct, the model will run using GPU acceleration.

## Step 7: Compile ONNX Runtime GenAI from Source (Optional)

This step is only required if you need a custom build of ONNX Runtime GenAI. If you are using pre-built wheels, you can skip this section.

### Step 7.1: Clean Previous Installations
First, remove any existing ONNX Runtime packages from your Python environment.

```bash
pip uninstall onnxruntime onnxruntime-genai onnxruntime-genai-cuda -y
```

### Step 7.2: Verify Visual Studio Integration
Ensure the Visual Studio integration files for CUDA are present. Check for the folder:
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\visual_studio_integration`

If it's missing, locate it in another CUDA toolkit directory and copy it to the path above.

### Step 7.3: Build ONNX Runtime GenAI
1.  Clone the `onnxruntime-genai` repository and download a pre-built ONNX Runtime binary.
    ```bash
    git clone https://github.com/microsoft/onnxruntime-genai
    ```
2.  Download the `onnxruntime-win-x64-gpu-1.19.2.zip` file from the [ORT releases page](https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-gpu-1.19.2.zip).
3.  Extract the ZIP file, rename the extracted folder to `ort`, and copy this `ort` folder into your cloned `onnxruntime-genai` directory.
4.  Open the **"Developer Command Prompt for VS 2022"** and navigate to the `onnxruntime-genai` folder.
5.  Run the build script, specifying your CUDA path.
    ```bash
    cd onnxruntime-genai
    python build.py --use_cuda --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4" --config Release
    ```
6.  Once the build completes, install the generated wheel.
    ```bash
    cd build\Windows\Release\Wheel
    pip install *.whl
    ```

## Conclusion

You have successfully set up ONNX Runtime with GPU support on Windows. You can now run AI models with accelerated performance. For further optimization, refer to the official [ONNX Runtime documentation](https://onnxruntime.ai/).
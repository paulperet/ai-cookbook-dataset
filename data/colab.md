# Guide: Running PyTorch Tutorials in Google Colab

This guide walks you through the essential setup steps required to run PyTorch tutorials smoothly in Google Colab. You'll learn how to manage PyTorch versions, access tutorial data from Google Drive, and enable GPU acceleration.

## Prerequisites

Before starting, ensure you have:
- A Google account to access Google Colab and Google Drive.
- Basic familiarity with running code in a Jupyter-like environment.

---

## Step 1: Verify and Install the Correct PyTorch Version

Some tutorials require a specific, often newer, version of PyTorch. Colab's default installation may be outdated.

1. **Check the currently installed packages** by running:

```bash
!pip list
```

2. **If you need to update PyTorch and its associated libraries**, uninstall the current versions and install the latest ones. Run the following commands in a Colab cell:

```python
!pip3 uninstall --yes torch torchaudio torchvision torchtext torchdata
!pip3 install torch torchaudio torchvision torchtext torchdata
```

This ensures you have a compatible, up-to-date PyTorch ecosystem.

---

## Step 2: Access Tutorial Data from Google Drive

Many tutorials require external datasets. Since Colab provides temporary storage, you can store persistent data in your Google Drive.

Here, we'll use the **Chatbot Tutorial** as an example, which needs the Cornell Movie Dialogs Corpus.

### 2.1 Prepare the Data in Google Drive

1. Log into your [Google Drive](https://drive.google.com/).
2. Create a folder named `data` with a subfolder named `cornell` inside it. The path will be: `My Drive/data/cornell/`.
3. Visit the [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) page and download the `movie-corpus.zip` file.
4. Unzip the file on your computer.
5. Locate the file `utterances.jsonl` and upload it to the `My Drive/data/cornell/` folder you created.

### 2.2 Modify the Tutorial Notebook in Colab

1. Open the [Chatbot Tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html) in your browser.
2. Click the **"Run in Google Colab"** button at the top of the page. This opens the notebook in Colab.
3. In the first code cell of the notebook, **add the following lines at the top** to mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

4. **Locate the lines in the notebook that define the data path.** They typically look like this:
    ```python
    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)
    ```
5. **Modify these lines** to point to your Google Drive folder:
    ```python
    corpus_name = "cornell"
    corpus = os.path.join("/content/gdrive/My Drive/data", corpus_name)
    ```

### 2.3 Authorize and Run

1. Run the modified cell. You will be prompted with a link to authorize Google Drive access.
2. Click the link, log in if needed, copy the provided authorization code, and paste it back into the Colab prompt.
3. Once authorized, you can run the entire notebook by selecting **Runtime > Run All** from the menu.

> **Note:** Some tutorials, like the Chatbot one, are computationally intensive and may take a long time to complete.

---

## Step 3: Enable GPU Acceleration (CUDA)

Tutorials involving deep learning models often run significantly faster on a GPU.

1. In your Colab notebook, go to the top menu and select **Runtime**.
2. Click **Change runtime type**.
3. In the dialog that opens, under **"Hardware accelerator"**, select **T4 GPU** (or any available GPU option).
4. Click **Save**.

Your notebook session will restart with GPU access enabled. You can verify the GPU is available within a code cell:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Summary

You are now ready to run PyTorch tutorials in Google Colab. The key steps are:
1. Ensuring the correct PyTorch version is installed.
2. Mounting Google Drive and configuring data paths for tutorials that require external datasets.
3. Enabling GPU acceleration for performance-intensive tasks.

By following this guide, you can adapt most PyTorch tutorials to run successfully in the Colab environment.
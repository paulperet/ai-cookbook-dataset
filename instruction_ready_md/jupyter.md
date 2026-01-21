# Getting Started with Jupyter Notebooks

This guide will walk you through the basics of using Jupyter Notebooks to edit, run, and manage code, which is essential for working through the tutorials in this book.

## Prerequisites

Before you begin, ensure you have:
1.  Installed Jupyter Notebook.
2.  Downloaded the code repository for this book.

If you haven't completed these steps, please refer to the installation chapter first.

## Step 1: Launch Your Local Jupyter Notebook

1.  Open your terminal or command prompt.
2.  Navigate to the directory containing the book's code. For example, if your path is `xx/yy/d2l-en/`, run:
    ```bash
    cd xx/yy/d2l-en
    ```
3.  Start the Jupyter Notebook server by running:
    ```bash
    jupyter notebook
    ```

Your default web browser should automatically open to `http://localhost:8888`, displaying the Jupyter interface and the list of files in your directory.

## Step 2: Understand the Notebook Interface

When you open a notebook file (with the `.ipynb` extension), you'll see it is composed of individual **cells**. There are two primary types of cells:
*   **Markdown Cells:** Contain formatted text, headings, and instructions.
*   **Code Cells:** Contain executable code (e.g., Python).

## Step 3: Edit and Run a Markdown Cell

Let's practice editing text.

1.  Locate a markdown cell. Its content will be rendered as formatted text.
2.  Double-click on the cell to enter **edit mode**. You'll see the raw markdown syntax.
3.  Modify the text. For instance, add "Hello world." to the end of the cell's content.
4.  To render your changes, run the cell. You can do this by:
    *   Clicking `Cell` → `Run Cells` in the menu bar.
    *   Using the keyboard shortcut `Shift + Enter` (which also moves to the next cell) or `Ctrl + Enter` (which runs the cell and stays on it).

After running, the cell will exit edit mode and display the formatted markdown.

## Step 4: Edit and Run a Code Cell

Now, let's work with code.

1.  Click on a code cell to select it. You will see code, such as:
    ```python
    import numpy as np
    x = np.array([1, 2, 3])
    ```
2.  Edit the code. For example, add a line to multiply the array:
    ```python
    import numpy as np
    x = np.array([1, 2, 3])
    x * 2
    ```
3.  Run the cell using the same methods (`Shift + Enter` or `Ctrl + Enter`). The output of the last line will be displayed directly below the cell.

## Step 5: Run an Entire Notebook

If your notebook has multiple cells, you can execute them all in sequence.

1.  In the menu bar, click `Kernel`.
2.  Select `Restart & Run All`. This will restart the Python kernel (clearing any existing variables in memory) and then execute every cell from top to bottom.

## Advanced Configuration

### Editing Markdown Source Files in Jupyter

To contribute content directly, you may need to edit the markdown (`.md`) source files within Jupyter. This requires the `notedown` plugin.

1.  Install the specialized plugin:
    ```bash
    pip install d2l-notedown
    ```
2.  Launch Jupyter Notebook with the plugin enabled:
    ```bash
    jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
    ```

**To make this configuration permanent:**
1.  Generate a Jupyter config file (skip if you already have one):
    ```bash
    jupyter notebook --generate-config
    ```
2.  Open the config file (typically at `~/.jupyter/jupyter_notebook_config.py` on Linux/macOS).
3.  Add the following line to the end of the file:
    ```python
    c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
    ```
Now, simply running `jupyter notebook` will always enable markdown file editing.

### Running Jupyter on a Remote Server

You can run Jupyter on a powerful remote machine (like a cloud server) and access it from your local browser using **SSH port forwarding**.

Run this command on your local machine:
```bash
ssh myserver -L 8888:localhost:8888
```
Replace `myserver` with your remote server's address. Then, access the notebook by opening `http://localhost:8888` in your local browser.

### Timing Code Execution

To measure how long each code cell takes to run, install the `ExecuteTime` extension.

1.  Install the necessary packages:
    ```bash
    pip install jupyter_contrib_nbextensions
    jupyter contrib nbextension install --user
    ```
2.  Enable the timing extension:
    ```bash
    jupyter nbextension enable execute_time/ExecuteTime
    ```
After enabling, each code cell's output will show its execution time.

## Summary

*   Jupyter Notebooks allow you to interactively edit and run code and text in cells.
*   You can run notebooks locally or on remote servers via SSH port forwarding.
*   Plugins like `notedown` and `ExecuteTime` extend functionality for source editing and performance measurement.

## Exercises

1.  Open a notebook from this book's code on your local machine. Practice editing and running both markdown and code cells.
2.  Set up a Jupyter notebook on a remote server (if you have access to one) and connect to it from your local computer using the port forwarding method described above.
3.  **Code Timing Challenge:** In a new notebook, compare the execution time of the matrix operations $\mathbf{A}^\top \mathbf{B}$ and $\mathbf{A} \mathbf{B}$ for two square matrices `A` and `B` in $\mathbb{R}^{1024 \times 1024}$. Which operation is faster? Use the `ExecuteTime` plugin or Python's `time` module to measure.

---
*For questions and discussions, visit the [book's forum](https://discuss.d2l.ai/t/421).*
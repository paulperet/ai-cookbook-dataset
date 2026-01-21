# Guide: Running Deep Learning Code with Amazon SageMaker

Deep learning applications often require significant computational resources, such as GPUs, which may exceed the capabilities of a local machine. Cloud platforms like Amazon Web Services (AWS) provide scalable resources to run demanding code. This guide walks you through using Amazon SageMaker, a managed service, to run Jupyter notebooks for deep learning projects.

## Prerequisites

Before you begin, ensure you have:
* An active AWS account.
* Basic familiarity with Jupyter notebooks and Git.

## Step 1: Sign Up and Access the AWS Console

1.  If you don't have one, sign up for an account at [https://aws.amazon.com/](https://aws.amazon.com/). It's recommended to enable two-factor authentication for security.
2.  Configure billing alerts in your AWS account to monitor costs and avoid unexpected charges.
3.  Log into the [AWS Management Console](http://console.aws.amazon.com/).
4.  Use the search bar at the top to find "Amazon SageMaker" and open the SageMaker service panel.

## Step 2: Create a SageMaker Notebook Instance

A notebook instance is a managed Jupyter environment with attached compute resources.

1.  In the SageMaker panel, navigate to **Notebook** > **Notebook instances**.
2.  Click the **Create notebook instance** button.
3.  In the creation form:
    *   Provide a descriptive **Notebook instance name**.
    *   Choose an **Instance type**. For most deep learning tasks in this book, a GPU instance like `ml.p3.2xlarge` (with one Tesla V100 GPU) is recommended. Review the [SageMaker pricing page](https://aws.amazon.com/sagemaker/pricing/instance-types/) for details on other options.

4.  In the **Git repositories** section, configure the source for your notebooks. Choose the repository corresponding to your deep learning framework:
    *   **For PyTorch:** `https://github.com/d2l-ai/d2l-pytorch-sagemaker`
    *   **For TensorFlow:** `https://github.com/d2l-ai/d2l-tensorflow-sagemaker`
    *   **For MXNet:** `https://github.com/d2l-ai/d2l-en-sagemaker`

5.  Leave other settings at their defaults or adjust as needed, then click **Create notebook instance**.

The instance provisioning will take a few minutes. Its status will change from "Pending" to "InService" when ready.

## Step 3: Run Your Notebooks

Once the instance status is **InService**:

1.  Find your instance in the list and click **Open Jupyter**.
2.  This opens the familiar JupyterLab interface in your browser. Navigate to the cloned Git repository folder (e.g., `d2l-pytorch-sagemaker/`).
3.  You can now open, edit, and execute any `.ipynb` notebook file, leveraging the GPU power of the SageMaker instance.

## Step 4: Stop the Instance to Manage Costs

SageMaker charges for the time your instance is running. To avoid unnecessary costs:

1.  Return to the **Notebook instances** page in the SageMaker console.
2.  Select your running instance.
3.  Click **Actions** > **Stop**. This terminates the underlying compute resources while preserving your notebook files and configuration. You can restart it later by selecting **Actions** > **Start**.

**Important:** Always stop your instance when you are not actively using it.

## Step 5: Update the Notebook Repository

The source GitHub repositories are updated periodically. To sync your SageMaker instance with the latest changes:

1.  From your instance's JupyterLab interface, open a new **Terminal**.
2.  Navigate to your book's directory. The command depends on your framework:
    ```bash
    # For PyTorch
    cd SageMaker/d2l-pytorch-sagemaker/

    # For TensorFlow
    cd SageMaker/d2l-tensorflow-sagemaker/

    # For MXNet
    cd SageMaker/d2l-en-sagemaker/
    ```
3.  To discard any local changes and pull the latest updates from GitHub, run:
    ```bash
    git reset --hard
    git pull
    ```
    If you have local modifications you wish to keep, commit them before running `git pull`.

## Summary

*   Amazon SageMaker provides a managed, scalable environment to run GPU-intensive deep learning code via Jupyter notebooks.
*   You can launch a notebook instance, select a powerful GPU type, and clone a repository containing all necessary notebooks.
*   Remember to **stop your instance** when not in use to control costs.
*   You can update the notebook code by pulling the latest changes from GitHub via the terminal.

## Next Steps

1.  Launch a SageMaker instance and run a notebook chapter that requires a GPU.
2.  Explore the terminal to see the local file structure and experiment with Git commands.
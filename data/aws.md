# Guide: Setting Up an AWS EC2 Instance for Deep Learning

This guide walks you through launching a GPU-enabled AWS EC2 instance, installing the necessary drivers and libraries, and connecting to it to run Jupyter Notebooks remotely. This setup is ideal for running compute-intensive deep learning workloads.

## Prerequisites

Before you begin, ensure you have:
*   An active AWS account.
*   Basic familiarity with the Linux command line and SSH.

---

## Step 1: Launch Your EC2 Instance

### 1.1 Navigate to EC2
1.  Log into your AWS Management Console.
2.  Search for and select **EC2** to open the EC2 dashboard.

### 1.2 Configure the Instance
1.  Click the **"Launch Instance"** button.
2.  **Name your instance** (e.g., `d2l-gpu-server`).
3.  In the **"Application and OS Images"** section, select **Ubuntu** as your Amazon Machine Image (AMI). A recent LTS version like Ubuntu 22.04 is recommended.
4.  In the **"Instance type"** section, choose a GPU-enabled instance. For learning and experimentation, a `p2.xlarge` (1x NVIDIA K80 GPU) is a cost-effective starting point. For more performance, consider `g4dn` or `p3` instances.
5.  In the **"Key pair (login)"** section, create or select an existing key pair. **This is crucial for SSH access.**
    *   If creating a new key pair, download the `.pem` file immediately and store it securely.
6.  In the **"Network settings"** section, you can keep the default VPC and subnet. Ensure **"Allow SSH traffic from"** is checked to permit your connection.
7.  In the **"Configure storage"** section, increase the root volume size. **64 GB** is a good minimum, as CUDA and datasets require significant space.
8.  Review your settings and click **"Launch instance"**.

### 1.3 Locate Your Instance
1.  After launch, navigate to the **"Instances"** section of the EC2 console.
2.  Wait for the **"Instance state"** to change to `Running` and the **"Status check"** to show `2/2 checks passed`.

---

## Step 2: Connect to Your Instance via SSH

You will connect to your instance from your local machine's terminal.

### 2.1 Set Key Permissions
If your key pair file (e.g., `my-key.pem`) is new, you must set the correct permissions so that only you can read it.

```bash
# Navigate to the directory containing your .pem file
cd ~/path/to/your/key/
chmod 400 my-key.pem
```

### 2.2 Establish the SSH Connection
1.  In the EC2 console, select your running instance.
2.  Click the **"Connect"** button.
3.  Go to the **"SSH client"** tab. You will see a command example.
4.  Copy and run this command in your local terminal. It will look like this:

```bash
ssh -i "my-key.pem" ubuntu@ec2-12-345-678-901.compute-1.amazonaws.com
```

5.  If prompted to confirm the connection, type `yes` and press Enter.

You are now logged into your remote Ubuntu server.

---

## Step 3: Install System Dependencies and CUDA

### 3.1 Update the System
First, update the package list and install essential build tools.

```bash
sudo apt-get update
sudo apt-get install -y build-essential git libgfortran3
```

### 3.2 Install CUDA
We will install CUDA 12.1. Always check the [official NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) for the latest stable version and instructions tailored to your OS.

Run the following commands in sequence:

```bash
# Add the NVIDIA package repository and key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update

# Install CUDA
sudo apt-get -y install cuda-12-1
```

### 3.3 Verify CUDA Installation
After installation, verify the GPU and driver are recognized.

```bash
nvidia-smi
```

You should see a table displaying your GPU model, driver version, and CUDA version.

### 3.4 Update Your Shell Configuration
Add CUDA to your system's `PATH` and library paths by editing your shell configuration file.

```bash
echo 'export PATH="/usr/local/cuda-12.1/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

Activate the changes in your current session:

```bash
source ~/.bashrc
```

---

## Step 4: Install Miniconda and Python Libraries

### 4.1 Install Miniconda
Miniconda is a lightweight package and environment manager. We'll use it to create a clean Python environment.

```bash
# Download the latest Miniconda installer for Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installer (follow prompts, say 'yes' to the license agreement)
bash Miniconda3-latest-Linux-x86_64.sh

# Activate Conda in your current shell
source ~/.bashrc
```

### 4.2 Create and Activate a Conda Environment
Create a dedicated environment for your project (e.g., named `d2l`).

```bash
conda create --name d2l python=3.9 -y
conda activate d2l
```

### 4.3 Install Deep Learning Frameworks
Install PyTorch with CUDA support. Use the command from the [official PyTorch website](https://pytorch.org/get-started/locally/) for the most up-to-date version.

```bash
# Example for PyTorch 2.0+ with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Jupyter and other common data science libraries
pip install jupyter matplotlib pandas scikit-learn
```

---

## Step 5: Run Jupyter Notebook Remotely

To access Jupyter Lab/Notebook running on your EC2 instance from your local browser, you need to set up SSH port forwarding.

### 5.1 Set Up SSH Tunnel
**On your local machine**, open a **new terminal window** (do not close your existing SSH session). Run the following command, replacing the key path and hostname with your details:

```bash
ssh -i "/path/to/my-key.pem" ubuntu@ec2-12-345-678-901.compute-1.amazonaws.com -L 8889:localhost:8888
```

This command forwards your local machine's port `8889` to port `8888` on the EC2 instance.

### 5.2 Start Jupyter on the EC2 Instance
**In your original SSH session (on the EC2 instance)**, navigate to your project directory and start Jupyter.

```bash
# Ensure your Conda environment is activated
conda activate d2l

# Navigate to your code directory
cd ~/my-project

# Start Jupyter Notebook. The `--no-browser` flag is used since we don't have a browser on the server.
jupyter notebook --no-browser --port=8888
```

Jupyter will start and print a log to the terminal. Look for a line like:
```
http://localhost:8888/?token=long_hash_of_letters_and_numbers
```

### 5.3 Connect from Your Local Browser
1.  Copy the URL from the Jupyter output.
2.  In the URL, change the port from `8888` to `8889` (the local port you forwarded).
3.  Paste the modified URL (e.g., `http://localhost:8889/?token=...`) into your **local machine's web browser**.

You should now see the Jupyter interface and can work with your notebooks.

---

## Step 6: Managing Your Instance and Costs

Cloud instances incur costs while they are running. Practice good cost management.

*   **Stop vs. Terminate:**
    *   **Stopping** (`Instance State -> Stop`) shuts down the instance but preserves the root EBS volume. You are billed only for storage, not compute. You can restart it later.
    *   **Terminating** (`Instance State -> Terminate`) shuts down the instance **and deletes the root EBS volume**. All data is lost. You are only billed for the time it ran.
*   **Create an AMI (Image):** Before terminating an instance you've fully configured, you can create an Amazon Machine Image (AMI). Right-click the instance, select **"Image and templates" -> "Create image"**. This snapshot allows you to launch identical pre-configured instances in the future, saving setup time.
*   **Use Spot Instances:** For fault-tolerant, flexible workloads (like training jobs), consider [Spot Instances](https://aws.amazon.com/ec2/spot/), which can reduce costs by 60-90%.

---

## Summary

You have successfully:
1.  Launched a GPU-powered AWS EC2 instance.
2.  Connected to it securely via SSH.
3.  Installed CUDA drivers and verified the GPU.
4.  Set up a Python deep learning environment using Conda.
5.  Configured SSH port forwarding to run and access Jupyter Notebooks remotely from your local machine.

This setup provides a powerful, scalable platform for deep learning development. Remember to stop or terminate your instances when not in use to control costs.
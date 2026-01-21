# Real-Time Image Classification on Raspberry Pi with PyTorch

**Author**: [Tristan Rice](https://github.com/d4l3k)

This guide will walk you through setting up a Raspberry Pi 4 or 5 to run a PyTorch-based image classification model in real time. You'll use a quantized MobileNetV2 model to achieve 30-40 frames per second (FPS) on the CPU.

## Prerequisites

You will need the following hardware:
*   A Raspberry Pi 4 Model B (2GB+) or Raspberry Pi 5.
*   A Raspberry Pi Camera Module (v2 or compatible).
*   A 5V 3A USB-C power supply.
*   A microSD card (at least 8GB) and a card reader.
*   (Recommended) Heat sinks and a fan for the Raspberry Pi.

## Step 1: Raspberry Pi OS Setup

PyTorch provides official packages only for the 64-bit ARM (aarch64) architecture. Therefore, you must install a 64-bit version of the operating system.

1.  Download and install the official [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2.  Use the imager to install **Raspberry Pi OS (64-bit)** onto your microSD card. **The 32-bit version will not work.**
3.  Once the imaging process is complete, insert the SD card into your Raspberry Pi, connect the camera module, and power it on.
4.  Complete the initial OS setup wizard.

### Additional Configuration for Raspberry Pi 4

If you are using a Raspberry Pi 4, you must enable the camera in the boot configuration. This step is **not required** for Raspberry Pi 5.

1.  Open a terminal and edit the boot configuration file:
    ```bash
    sudo nano /boot/config.txt
    ```
2.  Add or ensure the following lines are present:
    ```toml
    # Enables extended features like the camera
    start_x=1
    # Allocates memory for the GPU (at least 128MB is needed for camera processing)
    gpu_mem=128
    ```
3.  Save the file (`Ctrl+O`, then `Enter`) and exit the editor (`Ctrl+X`).
4.  Reboot your Raspberry Pi for the changes to take effect:
    ```bash
    sudo reboot
    ```

## Step 2: Install Required Software

With the 64-bit OS running, you can install PyTorch and the camera libraries directly via `pip`.

1.  First, install the system packages for camera access:
    ```bash
    sudo apt update
    sudo apt install -y python3-picamera2 python3-libcamera
    ```
2.  Install PyTorch and TorchVision. The `--break-system-packages` flag is often necessary on newer Debian-based systems like Raspberry Pi OS.
    ```bash
    pip install torch torchvision --break-system-packages
    ```
3.  Verify the installation was successful:
    ```bash
    python3 -c "import torch; print(torch.__version__)"
    ```
    You should see the PyTorch version number printed.

## Step 3: Test the Camera

Before writing code, test that your camera hardware is working correctly.
```bash
libcamera-hello
```
A camera preview window should open for a few seconds.

## Step 4: Capture Video Frames with `picamera2`

We'll use the `picamera2` library to capture frames. Our model expects images sized `224x224`, so we'll configure the camera to output that resolution at 36 FPS to ensure we always have a frame ready for processing at our target of 30 FPS.

Create a new Python script, for example `real_time_inference.py`, and start with the following imports and camera setup:

```python
import time
import torch
from torchvision import models, transforms
from picamera2 import Picamera2

# Set PyTorch to use the optimized QNNPACK quantization engine for ARM CPUs
torch.backends.quantized.engine = 'qnnpack'

# Initialize the camera
picam2 = Picamera2()

# Configure the camera stream
config = picam2.create_still_configuration(
    main={"size": (224, 224), "format": "BGR888"},
    display="main"
)
picam2.configure(config)
picam2.set_controls({"FrameRate": 36})
picam2.start()

print("Camera started successfully.")
```

## Step 5: Define Image Preprocessing

The model requires input images to be normalized. We'll use standard TorchVision transforms.

```python
# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## Step 6: Load and Optimize the Model

We'll use a pre-trained, quantized, and fused version of MobileNetV2 for optimal performance on the Raspberry Pi. We then convert it using TorchScript (JIT) to minimize Python overhead and fuse operations, which significantly boosts FPS.

```python
# Load a pre-quantized MobileNetV2 model
net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
# Convert the model to TorchScript for performance gains (~20 FPS -> ~30 FPS)
net = torch.jit.script(net)
# Set the model to evaluation mode (important for inference)
net.eval()
```

## Step 7: Create the Main Inference Loop

Now, we combine all components into a loop that captures frames, processes them, and runs inference.

```python
# Performance logging variables
started = time.time()
last_logged = time.time()
frame_count = 0

print("Starting inference loop. Press Ctrl+C to stop.")

with torch.no_grad():  # Disable gradient calculation for inference
    while True:
        # 1. Capture a frame from the camera
        image = picam2.capture_image("main")

        # 2. Preprocess the image for the model
        input_tensor = preprocess(image)
        # Add a batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
        input_batch = input_tensor.unsqueeze(0)

        # 3. Run the model inference
        output = net(input_batch)

        # 4. Get the predicted class (index with the highest score)
        top_class_id = output.argmax().item()
        # For now, we just print the class ID. We'll add labels later.
        # print(f"Predicted class index: {top_class_id}")

        # 5. Log frames per second (FPS) every second
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            fps = frame_count / (now - last_logged)
            print(f"Current rate: {fps:.2f} FPS")
            last_logged = now
            frame_count = 0
```

Run your script:
```bash
python3 real_time_inference.py
```

You should see FPS output hovering around **30 FPS on a Raspberry Pi 4** and **over 40 FPS on a Raspberry Pi 5**.

## Step 8: Add Human-Readable Labels

To see what the model is actually predicting, we need to map the class index to a label. Download the ImageNet class labels.

1.  You can save the labels from this [gist](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) as a file named `imagenet_classes.txt` in your project directory.
2.  Modify the inference section of your loop:

```python
# ... inside the inference loop, after 'output = net(input_batch)' ...

# Load the class labels (do this once before the loop)
with open("imagenet_classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ... inside the loop ...
# 3. Run the model inference
output = net(input_batch)

# 4. Get the top 5 predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Print the top 5 predictions
for i in range(top5_prob.size(0)):
    class_id = top5_catid[i].item()
    probability = top5_prob[i].item()
    print(f"{classes[class_id]:<30}: {probability * 100:5.2f}%")
print("-" * 50)
```

## Performance Tuning and Troubleshooting

*   **Background Processes:** The default Raspberry Pi OS runs several services. For the most stable performance, consider disabling the desktop GUI or unnecessary background tasks.
*   **Thread Contention:** PyTorch uses all CPU cores by default. If you experience latency spikes, you can limit the number of threads, which trades a small amount of peak performance for more consistent latency.
    ```python
    # Add this near the top of your script, after imports
    torch.set_num_threads(2)
    ```

## Next Steps and Further Reading

You can extend this project by:
*   **Fine-Tuning:** Use transfer learning to adapt the quantized MobileNetV2 to your own dataset. The [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) is an excellent resource.
*   **Custom Models:** Quantize and deploy your own models. Refer to the [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html) for details.
*   **Exploring Models:** The table in the original guide compares the performance of various models (`mobilenet_v3_large`, `shufflenet`, `resnet`, etc.) on the Raspberry Pi 4. You can experiment with other models from `torchvision.models.quantized`.

You now have a real-time image classification system running on your Raspberry Pi!
# PyTorch C++ Frontend Tutorial: Building and Training a DCGAN

## Overview

This tutorial will guide you through building a complete C++ application that uses the PyTorch C++ frontend to train a Deep Convolutional Generative Adversarial Network (DCGAN) on the MNIST dataset. You'll learn how to define neural network modules, load data, implement training loops, and manage GPU acceleration—all in C++.

### What You'll Learn
- How to build a C++ application using the PyTorch C++ frontend
- How to define and train neural networks from C++ using PyTorch abstractions
- How to implement a complete GAN training pipeline

### Prerequisites
- PyTorch 1.5 or later
- Basic understanding of C++ programming
- Basic Ubuntu Linux environment with CMake ≥ 3.5 (similar commands work on macOS/Windows)
- (Optional) A CUDA-based GPU for GPU training sections

## Why Use the C++ Frontend?

The PyTorch C++ frontend is a pure C++17 interface to PyTorch's underlying C++ codebase. While Python is the primary interface, the C++ frontend enables research in environments where Python isn't suitable:

- **Low Latency Systems**: Reinforcement learning in game engines with high FPS requirements
- **Highly Multithreaded Environments**: Bypassing Python's Global Interpreter Lock (GIL)
- **Existing C++ Codebases**: Integrating ML into C++ applications without Python-C++ binding overhead

The C++ frontend maintains the same intuitive API design as Python PyTorch, making it easy to transition between the two.

## Setting Up Your Environment

### Step 1: Download LibTorch

First, download the LibTorch distribution—a ready-built package containing all necessary headers, libraries, and CMake files:

```bash
# For CPU-only version (replace "cpu" with "cu90" for CUDA 9.0 support)
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

### Step 2: Create a Basic Application

Let's verify our setup with a minimal C++ application. Create a file called `dcgan.cpp`:

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
}
```

### Step 3: Configure CMake

Create a `CMakeLists.txt` file in the same directory:

```cmake
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(dcgan)

find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 17)
```

### Step 4: Build and Run

Set up your directory structure:
```
dcgan/
  CMakeLists.txt
  dcgan.cpp
```

Then build and run the application:

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
./dcgan
```

You should see a 3x3 identity matrix printed to the console.

## Understanding the Module API

### Basic Module Structure

In C++, modules are derived from `torch::nn::Module`. Like in Python, they contain parameters, buffers, and submodules. Here's a simple module definition:

```cpp
#include <torch/torch.h>

struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M) {
    W = register_parameter("W", torch::randn({N, M}));
    b = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return torch::addmm(b, input, W);
  }
  torch::Tensor W, b;
};
```

### Registering Submodules

To create more complex architectures, you can register submodules:

```cpp
struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};
```

### Module Ownership Model

The C++ frontend uses a reference semantics model with smart pointers. For convenience, use the `TORCH_MODULE` macro:

```cpp
struct NetImpl : torch::nn::Module {
  // Implementation details
};
TORCH_MODULE(Net);  // Creates Net as a wrapper around std::shared_ptr<NetImpl>
```

This approach maintains Python-like ergonomics while ensuring proper memory management.

## Defining the DCGAN Architecture

### Understanding GANs

A Generative Adversarial Network consists of two models:
- **Generator**: Transforms noise samples into realistic-looking images
- **Discriminator**: Distinguishes between real images and generated fakes

The two models compete: the generator tries to fool the discriminator, while the discriminator tries to correctly identify real vs. fake images.

### Implementing the Generator

The generator uses transposed convolutions to upsample noise into images:

```cpp
struct DCGANGeneratorImpl : nn::Module {
  DCGANGeneratorImpl(int kNoiseSize)
      : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)),
        batch_norm1(256),
        conv2(nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
        batch_norm2(128),
        conv3(nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
        batch_norm3(64),
        conv4(nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false)) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(batch_norm1(conv1(x)));
    x = torch::relu(batch_norm2(conv2(x)));
    x = torch::relu(batch_norm3(conv3(x)));
    x = torch::tanh(conv4(x));
    return x;
  }

  nn::ConvTranspose2d conv1, conv2, conv3, conv4;
  nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(DCGANGenerator);
```

### Implementing the Discriminator

The discriminator uses regular convolutions to classify images:

```cpp
nn::Sequential discriminator(
  // Layer 1
  nn::Conv2d(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
  nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
  // Layer 2
  nn::Conv2d(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
  nn::BatchNorm2d(128),
  nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
  // Layer 3
  nn::Conv2d(nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
  nn::BatchNorm2d(256),
  nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
  // Layer 4
  nn::Conv2d(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
  nn::Sigmoid());
```

## Loading the MNIST Dataset

### Setting Up Data Loading

The C++ frontend provides a multi-threaded data loader. Here's how to set it up for MNIST:

```cpp
// Define constants
const int64_t kBatchSize = 64;
const int64_t kNoiseSize = 100;

// Load and transform MNIST dataset
auto dataset = torch::data::datasets::MNIST("./mnist")
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());

// Create data loader with multiple workers
auto data_loader = torch::data::make_data_loader(
    std::move(dataset),
    torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));
```

**Note**: You'll need to download the MNIST dataset to the `./mnist` directory. Use [this script](https://gist.github.com/jbschlosser/94347505df6188f8764793ee29fd1bdd) to download it.

### Verifying Data Loading

Test your data loader with a simple loop:

```cpp
for (torch::data::Example<>& batch : *data_loader) {
  std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
  for (int64_t i = 0; i < batch.data.size(0); ++i) {
    std::cout << batch.target[i].item<int64_t>() << " ";
  }
  std::cout << std::endl;
}
```

## Implementing the Training Loop

### Step 1: Initialize Models and Optimizers

```cpp
// Create generator and discriminator
DCGANGenerator generator(kNoiseSize);
nn::Sequential discriminator(/* ... as defined above ... */);

// Create optimizers
torch::optim::Adam generator_optimizer(
    generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
torch::optim::Adam discriminator_optimizer(
    discriminator->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.5)));
```

### Step 2: Implement the Training Loop

```cpp
const int64_t kNumberOfEpochs = 30;
const int64_t batches_per_epoch = 938; // MNIST has 60000 samples, batch size 64

for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
  int64_t batch_index = 0;
  for (torch::data::Example<>& batch : *data_loader) {
    // Train discriminator with real images
    discriminator->zero_grad();
    torch::Tensor real_images = batch.data;
    torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0);
    torch::Tensor real_output = discriminator->forward(real_images).reshape(real_labels.sizes());
    torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
    d_loss_real.backward();

    // Train discriminator with fake images
    torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1});
    torch::Tensor fake_images = generator->forward(noise);
    torch::Tensor fake_labels = torch::zeros(batch.data.size(0));
    torch::Tensor fake_output = discriminator->forward(fake_images.detach()).reshape(fake_labels.sizes());
    torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
    d_loss_fake.backward();

    torch::Tensor d_loss = d_loss_real + d_loss_fake;
    discriminator_optimizer.step();

    // Train generator
    generator->zero_grad();
    fake_labels.fill_(1);
    fake_output = discriminator->forward(fake_images).reshape(fake_labels.sizes());
    torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
    g_loss.backward();
    generator_optimizer.step();

    // Print progress
    std::printf(
        "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
        epoch,
        kNumberOfEpochs,
        ++batch_index,
        batches_per_epoch,
        d_loss.item<float>(),
        g_loss.item<float>());
  }
}
```

## Adding GPU Support

### Step 1: Configure Device Selection

Add device configuration at the beginning of your training script:

```cpp
torch::Device device = torch::kCPU;
if (torch::cuda::is_available()) {
  std::cout << "CUDA is available! Training on GPU." << std::endl;
  device = torch::kCUDA;
}
```

### Step 2: Move Models and Data to GPU

Update your code to use the selected device:

```cpp
// Move models to device
generator->to(device);
discriminator->to(device);

// Update tensor creations to include device
torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);

// Move data tensors to device
torch::Tensor real_images = batch.data.to(device);
torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
```

## Implementing Checkpointing

### Step 1: Save Checkpoints Periodically

Add checkpointing to your training loop:

```cpp
const int64_t kCheckpointEvery = 100;
int64_t checkpoint_counter = 0;

// Inside training loop
if (batch_index % kCheckpointEvery == 0) {
  // Save model states
  torch::save(generator, "generator-checkpoint.pt");
  torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
  torch::save(discriminator, "discriminator-checkpoint.pt");
  torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
  
  // Save sample images
  torch::Tensor samples = generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
  torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
  
  std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
}
```

### Step 2: Add Checkpoint Restoration

Add checkpoint loading before the training loop:

```cpp
const bool kRestoreFromCheckpoint = false; // Set to true to restore from checkpoint

if (kRestoreFromCheckpoint) {
  torch::load(generator, "generator-checkpoint.pt");
  torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
  torch::load(discriminator, "discriminator-checkpoint.pt");
  torch::load(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
}
```

## Visualizing Results

### Python Script for Image Display

Create a Python script to visualize generated images:

```python
import argparse
import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--sample-file", required=True)
parser.add_argument("-o", "--out-file", default="out.png")
parser.add_argument("-d", "--dimension", type=int, default=3)
options = parser.parse_args()

module = torch.jit.load(options.sample_file)
images = list(module.parameters())[0]

for index in range(options.dimension * options.dimension):
    image = images[index].detach().cpu().reshape(28, 28).mul(255).to(torch.uint8)
    array = image.numpy()
    axis = plt.subplot(options.dimension, options.dimension, 1 + index)
    plt.imshow(array, cmap="gray")
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

plt.savefig(options.out_file)
print("Saved", options.out_file)
```

### Running the Visualization

After training, visualize generated images:

```bash
python display.py -i dcgan-sample-100.pt
```

## Complete Training Command

Here's the complete command to build and run your DCGAN:

```bash
# Build the application
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release

# Run training
./dcgan
```

## Next Steps and Resources

### Further Learning
- Experiment with different architectures and hyperparameters
- Try training on other datasets
- Implement additional GAN variants (WGAN, StyleGAN, etc.)

### Useful Resources
- [PyTorch C++ Documentation](https://pytorch.org/cppdocs/)
- [C++ Frontend Design Notes](https://pytorch.org/cppdocs/frontend.html)
- [Full DCGAN Example Code](https://github.com/pytorch/examples/tree/master/cpp/dcgan)

### Getting Help
- [PyTorch Forums](https://discuss.pytorch.org/)
- [GitHub Issues](https://github.com/pytorch/pytorch/issues)

## Conclusion

You've successfully built and trained a DCGAN using the PyTorch C++ frontend. This tutorial covered:
- Setting up the C++ development environment
- Defining neural network modules
- Implementing data loading pipelines
- Creating training loops for GANs
- Adding GPU acceleration and checkpointing

The C++ frontend provides a powerful, performant alternative to Python for production deployments and research in constrained environments. With its API closely mirroring Python PyTorch, you can leverage your existing knowledge while benefiting from C++'s performance characteristics.
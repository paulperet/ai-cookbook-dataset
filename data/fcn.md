# Implementing a Fully Convolutional Network for Semantic Segmentation

## Overview
In this tutorial, you will build a Fully Convolutional Network (FCN) for semantic segmentation. Unlike standard CNNs used for image classification, an FCN transforms image pixels directly into pixel-level class predictions. This is achieved by using transposed convolutional layers to upsample feature maps back to the original input dimensions, enabling a one-to-one correspondence between input pixels and output class predictions.

## Prerequisites
Ensure you have the following libraries installed. The tutorial supports both MXNet and PyTorch.

```bash
# Install necessary packages (if using pip)
# For MXNet:
pip install mxnet d2l

# For PyTorch:
pip install torch torchvision d2l
```

## Step 1: Import Libraries
First, import the required libraries. The code is provided for both MXNet and PyTorch—choose the framework you are using.

```python
# For MXNet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
# For PyTorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

## Step 2: Construct the FCN Model
The FCN model uses a pretrained CNN backbone (ResNet-18) to extract features, followed by a 1×1 convolutional layer to adjust the channel count to the number of classes, and finally a transposed convolutional layer to upsample the feature maps to the original image size.

### 2.1 Load the Pretrained Backbone
We'll use a ResNet-18 model pretrained on ImageNet. The final layers (global average pooling and fully connected layer) are not needed for the FCN.

```python
# MXNet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
print(pretrained_net.features[-3:], pretrained_net.output)
```

```python
# PyTorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
print(list(pretrained_net.children())[-3:])
```

### 2.2 Create the FCN by Truncating the Backbone
Build the FCN by taking all layers from the pretrained network except the last two (global pooling and fully connected layers).

```python
# MXNet
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```python
# PyTorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

### 2.3 Verify Feature Map Dimensions
Check the output shape after the backbone to understand the downsampling factor. For a 320×480 input, the spatial dimensions are reduced to 10×15 (a factor of 32).

```python
# MXNet
X = np.random.uniform(size=(1, 3, 320, 480))
print(net(X).shape)
```

```python
# PyTorch
X = torch.rand(size=(1, 3, 320, 480))
print(net(X).shape)
```

### 2.4 Add the Final Layers
Add a 1×1 convolution to produce the correct number of output channels (21 for Pascal VOC2012) and a transposed convolution to upsample by a factor of 32.

```python
# MXNet
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))
```

```python
# PyTorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## Step 3: Initialize the Transposed Convolutional Layer
Transposed convolutional layers are often initialized using bilinear interpolation kernels to perform upsampling. This helps preserve spatial structure.

### 3.1 Define the Bilinear Kernel Function
The following function generates a kernel for bilinear interpolation upsampling.

```python
# MXNet
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```python
# PyTorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

### 3.2 Test Bilinear Upsampling
Apply the bilinear kernel to a transposed convolutional layer and visualize the upsampling effect on a sample image.

```python
# MXNet
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))

img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)

print('input image shape:', img.shape)
print('output image shape:', out_img.shape)
```

```python
# PyTorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()

print('input image shape:', img.permute(1, 2, 0).shape)
print('output image shape:', out_img.shape)
```

### 3.3 Initialize the FCN's Final Layers
Initialize the transposed convolutional layer with the bilinear kernel and the 1×1 convolutional layer with Xavier initialization.

```python
# MXNet
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```python
# PyTorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)
```

## Step 4: Load the Dataset
Load the Pascal VOC2012 semantic segmentation dataset. The images are randomly cropped to 320×480 pixels, ensuring dimensions are divisible by 32.

```python
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## Step 5: Train the Model
Define the loss function and optimizer, then train the FCN. The loss is computed per-pixel across the channel dimension.

```python
# MXNet
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```python
# PyTorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Step 6: Make Predictions
After training, define a prediction function that standardizes the input image and runs it through the network.

```python
# MXNet
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```python
# PyTorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

### 6.1 Map Predictions to Colors
Convert predicted class indices back to the label colors defined in the dataset for visualization.

```python
# MXNet
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```python
# PyTorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

### 6.2 Visualize Predictions on Test Images
Load test images, crop them to 320×480, and display the original, predicted, and ground-truth segmentation masks.

```python
# MXNet
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
```

```python
# PyTorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
```

## Summary
In this tutorial, you implemented a Fully Convolutional Network for semantic segmentation. The key steps were:
1. Using a pretrained CNN backbone for feature extraction.
2. Adding a 1×1 convolutional layer to adjust channel dimensions to the number of classes.
3. Employing a transposed convolutional layer initialized with bilinear interpolation to upsample feature maps to the original image size.
4. Training the model on a semantic segmentation dataset and visualizing the predictions.

## Exercises
1. Experiment with Xavier initialization for the transposed convolutional layer instead of bilinear interpolation. How do the results change?
2. Try tuning hyperparameters (learning rate, weight decay, number of epochs) to improve accuracy.
3. Extend the prediction function to process entire test images (not just cropped regions) by using a sliding window or tiling approach.
4. Implement the original FCN architecture that also incorporates intermediate CNN layer outputs (skip connections) for improved detail.
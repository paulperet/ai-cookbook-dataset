# Neural Style Transfer: A Step-by-Step Guide

This guide walks you through implementing neural style transfer, a technique that applies the artistic style of one image to the content of another. We'll use a pre-trained VGG-19 network to extract features and optimize a synthesized image.

## Prerequisites

First, ensure you have the necessary libraries installed. This guide provides code for both PyTorch and MXNet frameworks.

**For PyTorch:**
```bash
pip install torch torchvision matplotlib
```

**For MXNet:**
```bash
pip install mxnet gluoncv matplotlib
```

We'll also use a utility module `d2l` (Dive into Deep Learning). You can typically import it if following along with the associated book materials, or install it via:
```bash
pip install d2l
```

## 1. Load Content and Style Images

Let's start by loading our two input images: the **content image** (the photo whose structure we want to keep) and the **style image** (the painting whose artistic style we want to apply).

```python
# PyTorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
style_img = d2l.Image.open('../img/autumn-oak.jpg')
```

```python
# MXNet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
d2l.set_figsize()
content_img = image.imread('../img/rainier.jpg')
style_img = image.imread('../img/autumn-oak.jpg')
```

## 2. Define Image Preprocessing and Postprocessing

Neural networks expect normalized input. We'll define functions to preprocess images for the network and postprocess the output back to a viewable format.

**Key Steps:**
- Resize the image to a consistent shape.
- Normalize the RGB channels using the mean and standard deviation from the ImageNet dataset.
- For postprocessing, clip pixel values to the valid [0, 1] range.

```python
# PyTorch
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
```

```python
# MXNet
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)
```

## 3. Load the Pre-Trained Feature Extractor

We'll use a VGG-19 model pre-trained on ImageNet. This network will serve as our fixed feature extractor; we won't update its weights.

```python
# PyTorch
pretrained_net = torchvision.models.vgg19(pretrained=True)
```

```python
# MXNet
pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
```

## 4. Select Content and Style Layers

Different layers of a CNN capture different types of information. Lower layers capture fine details (edges, textures), while higher layers capture more abstract content (objects, scenes).

- **Content Layer:** We choose a higher layer (the last convolutional layer of the 4th block) to capture the global structure of the content image.
- **Style Layers:** We choose the first convolutional layer of each of the five blocks to capture texture and style patterns at multiple scales.

```python
# Common to both frameworks
style_layers = [0, 5, 10, 19, 28]
content_layers = [25]
```

Now, construct a new network that includes only the layers up to the deepest one we need.

```python
# PyTorch
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])
```

```python
# MXNet
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

## 5. Extract Features

We need a function that runs an image through our truncated network and saves the outputs from our specified content and style layers.

```python
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
```

Let's create helper functions to extract the target features from our content and style images. We do this once before training begins.

```python
# PyTorch
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```python
# MXNet
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).copyto(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).copyto(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

## 6. Define the Loss Functions

The total loss for style transfer is a weighted combination of three components.

### 6.1 Content Loss
This ensures the synthesized image matches the content image's high-level structure. We use the Mean Squared Error (MSE) between feature maps from the content layer.

```python
# PyTorch
def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean()
```

```python
# MXNet
def content_loss(Y_hat, Y):
    return np.square(Y_hat - Y).mean()
```

### 6.2 Style Loss
To match style, we compare the *Gram matrices* of the feature maps. The Gram matrix computes the correlations between different filter responses, which effectively captures texture information.

First, define a function to compute the Gram matrix.

```python
def gram(X):
    num_channels, n = X.shape[1], d2l.size(X) // X.shape[1]
    X = d2l.reshape(X, (num_channels, n))
    return d2l.matmul(X, X.T) / (num_channels * n)
```

Now, the style loss is the MSE between the Gram matrices of the synthesized and style images.

```python
# PyTorch
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
```

```python
# MXNet
def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()
```

### 6.3 Total Variation Loss
This acts as a regularizer to reduce high-frequency noise (checkerboard patterns) and encourage spatial smoothness in the synthesized image.

```python
def tv_loss(Y_hat):
    return 0.5 * (d2l.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  d2l.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
```

### 6.4 Combined Loss Function
Finally, we combine all losses with tunable weights.

```python
content_weight, style_weight, tv_weight = 1, 1e4, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

## 7. Initialize the Synthesized Image

The image we want to generate is the only "model parameter" we will train. We create a simple model class where the parameter is the image tensor itself.

```python
# PyTorch
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

```python
# MXNet
class SynthesizedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

We need an initialization function that creates this model instance, sets its initial value (usually the content image), and pre-computes the Gram matrices for the style image's features.

```python
# PyTorch
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```python
# MXNet
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

## 8. Train the Model

The training loop repeatedly:
1. Extracts features from the current synthesized image.
2. Calculates the total loss.
3. Backpropagates the loss to update the pixel values of the synthesized image.

```python
# PyTorch
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```python
# MXNet
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs], ylim=[0, 20],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.8)
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X).asnumpy())
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

## 9. Run the Training

Now, let's put everything together and start the optimization process. We'll resize our images, move everything to the GPU if available, and begin training.

```python
# PyTorch
device, image_shape = d2l.try_gpu(), (300, 450)  # (height, width)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
```

```python
# MXNet
device, image_shape = d2l.try_gpu(), (450, 300)  # (width, height)
net.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.9, 500, 50)
```

After training, `output` contains your final synthesized image. You can visualize it using `postprocess(output)`.

## Summary

In this guide, you implemented neural style transfer by:
1. Using a pre-trained VGG-19 network as a fixed feature extractor.
2. Defining a loss function with three components: **content loss** (preserves structure), **style loss** (transfers texture via Gram matrices), and **total variation loss** (encourages smoothness).
3. Treating the synthesized image as the only trainable parameter and optimizing it via gradient descent.

The final image retains the objects and layout of the content photo while adopting the colors and brushstroke textures of the style painting.

## Exercises

1. **Experiment with Layers:** Try using different layers for `content_layers` and `style_layers`. How does using a lower layer for content affect the output?
2. **Tune the Weights:** Adjust `content_weight`, `style_weight`, and `tv_weight`. Can you achieve a sharper content preservation or a stronger style effect?
3. **Try New Images:** Use your own content and style images. What interesting combinations can you create?
4. **Beyond Images:** Could this Gram matrix-based style loss be adapted for other data types, like text? (See the survey on style transfer for text.)
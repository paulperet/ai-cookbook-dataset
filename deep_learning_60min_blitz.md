# Deep Learning with PyTorch: A 60 Minute Blitz

**Author**: [Soumith Chintala](http://soumith.ch)

```{=html}
<div style="margin-top:10px; margin-bottom:10px;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/u7x8RXwLKcA" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```
## What is PyTorch?

PyTorch is a Python-based scientific computing package serving two broad
purposes:

-   A replacement for NumPy to use the power of GPUs and other
    accelerators.
-   An automatic differentiation library that is useful to implement
    neural networks.

## Goal of this tutorial:

-   Understand PyTorch's Tensor library and neural networks at a high
    level.
-   Train a small neural network to classify images

To run the tutorials below, make sure you have the
[torch](https://github.com/pytorch/pytorch),
[torchvision](https://github.com/pytorch/vision), and
[matplotlib](https://github.com/matplotlib/matplotlib) packages
installed.

::: {.toctree hidden=""}
/beginner/blitz/tensor_tutorial /beginner/blitz/autograd_tutorial
/beginner/blitz/neural_networks_tutorial
/beginner/blitz/cifar10_tutorial
:::

::::::: grid
4

::: {.grid-item-card link="blitz/tensor_tutorial.html"}
`file-code;1em`{.interpreted-text role="octicon"} Tensors

In this tutorial, you will learn the basics of PyTorch tensors. +++
`code;1em`{.interpreted-text role="octicon"} Code
:::

::: {.grid-item-card link="blitz/autograd_tutorial.html"}
`file-code;1em`{.interpreted-text role="octicon"} A Gentle Introduction
to torch.autograd

Learn about autograd. +++ `code;1em`{.interpreted-text role="octicon"}
Code
:::

::: {.grid-item-card link="blitz/neural_networks_tutorial.html"}
`file-code;1em`{.interpreted-text role="octicon"} Neural Networks

This tutorial demonstrates how you can train neural networks in PyTorch.
+++ `code;1em`{.interpreted-text role="octicon"} Code
:::

::: {.grid-item-card link="blitz/cifar10_tutorial.html"}
`file-code;1em`{.interpreted-text role="octicon"} Training a Classifier

Learn how to train an image classifier in PyTorch by using the CIFAR10
dataset. +++ `code;1em`{.interpreted-text role="octicon"} Code
:::
:::::::

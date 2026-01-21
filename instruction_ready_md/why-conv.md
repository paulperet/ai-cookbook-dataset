# From Fully Connected Layers to Convolutions

## Introduction

When working with tabular data, where rows are examples and columns are features, Multi-Layer Perceptrons (MLPs) are often sufficient. However, for high-dimensional perceptual data like images, unstructured MLPs become impractical. This guide walks through the conceptual transition from fully connected layers to convolutional layers, explaining the core principles that make Convolutional Neural Networks (CNNs) effective for computer vision.

## The Problem with MLPs on Images

Consider a task like distinguishing cats from dogs using one-megapixel photographs. Each input has one million dimensions. Even reducing this to a hidden layer of 1000 dimensions would require:

\[
10^6 \times 10^3 = 10^9 \text{ parameters}
\]

Training a network with billions of parameters is computationally infeasible without immense resources. Yet, both humans and computers perform this task well. The key insight is that images have rich, exploitable structure. CNNs are designed to leverage this structure efficiently.

## Core Principles for Vision Architectures

To design a neural network suitable for images, we establish three guiding principles:

1.  **Translation Invariance:** The network should respond similarly to a pattern regardless of its location in the image.
2.  **Locality:** Early layers should focus on local regions, aggregating information later for global predictions.
3.  **Hierarchical Representation:** Deeper layers should capture longer-range, more complex features.

## Constraining the MLP: The Path to Convolutions

We begin with an MLP where both the 2D input image \(\mathbf{X}\) and hidden representation \(\mathbf{H}\) are matrices of the same shape.

Let \([\mathbf{X}]_{i, j}\) and \([\mathbf{H}]_{i, j}\) denote the pixel at location \((i,j)\). A fully connected layer would require a fourth-order weight tensor \(\mathsf{W}\):

\[
[\mathbf{H}]_{i, j} = [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}
\]

This formulation is unworkable. For a \(1000 \times 1000\) image mapped to a same-sized hidden layer, it requires \(10^{12}\) parameters.

### Step 1: Enforcing Translation Invariance

Invoking translation invariance means a shift in the input leads to a corresponding shift in the hidden representation. This forces the weights \(\mathsf{V}\) and biases \(\mathbf{U}\) to be independent of the spatial location \((i, j)\). We can then simplify:

\[
[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}
\]

This is a **convolution**. We weight pixels around \((i, j)\) with a shared kernel \([\mathbf{V}]_{a, b}\). The parameter count drops dramatically from \(10^{12}\) to roughly \(4 \times 10^6\) (for a kernel spanning the full image).

### Step 2: Enforcing Locality

We invoke locality: to compute \([\mathbf{H}]_{i, j}\), we shouldn't need information from pixels far away. We restrict the kernel to a local window of size \(\Delta\):

\[
[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}
\]
:label:`eq_conv-layer`

This is the definition of a **convolutional layer**. The number of parameters is now \(4 \Delta^2\). With \(\Delta < 10\), we reduce parameters by several more orders of magnitude. The kernel \(\mathbf{V}\) is a learnable filter.

## Mathematical Convolution vs. Cross-Correlation

In mathematics, the convolution between two functions \(f\) and \(g\) is defined as:

\[
(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}
\]

For discrete 2D tensors, this becomes:

\[
(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b)
\]
:label:`eq_2d-conv-discrete`

Our layer definition in :eqref:`eq_conv-layer` uses \((i+a, j+b)\) instead of the difference \((i-a, j-b)\). This operation is technically a **cross-correlation**. In deep learning practice, the learned kernel adapts to either formulation, so the distinction is cosmetic. We commonly refer to the layer operation as a convolution.

## Handling Multiple Channels

Real images have three color channels (Red, Green, Blue). They are third-order tensors of shape `height × width × channels`. Our formulation must adapt.

1.  The input \(\mathsf{X}\) is now indexed as \([\mathsf{X}]_{i, j, k}\).
2.  The convolutional filter becomes a fourth-order tensor \([\mathsf{V}]_{a, b, c, d}\).
3.  We want our hidden representation \(\mathsf{H}\) to also be a third-order tensor (multiple feature maps or channels), indexed by \([\mathsf{H}]_{i,j,d}\).

The general convolutional layer for multiple input and output channels is:

\[
[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},
\]
:label:`eq_conv-layer-channels`

where \(d\) indexes the output channels. This allows the network to learn specialized features (e.g., edges, textures) in different channels at various layers.

## Summary

We derived the convolutional layer from first principles by applying the constraints of **translation invariance** and **locality** to a fully connected layer. This led to a dramatic reduction in parameters, turning an intractable model into a feasible one. The addition of **channels** restored necessary complexity, allowing the network to build rich, hierarchical representations of images.

Convolutional Neural Networks systematize these ideas, providing an efficient and powerful architecture for computer vision and beyond.

## Exercises

1.  Assume the convolution kernel size is \(\Delta = 0\). Show that in this case, the convolutional layer implements an MLP independently for each set of channels. This leads to the *Network in Network* architecture.
2.  Audio data is often a 1D sequence.
    *   When might you impose locality and translation invariance for audio?
    *   Derive the 1D convolution operation for audio.
    *   Can you treat audio using the same tools as computer vision? (Hint: consider spectrograms).
3.  Why might translation invariance *not* always be a good idea? Provide an example.
4.  Could convolutional layers be applicable to text data? What challenges might arise?
5.  What happens with convolutions when an object is at the boundary of an image?
6.  Prove that the convolution operation is symmetric: \(f * g = g * f\).
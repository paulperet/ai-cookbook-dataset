# D2L API Reference Guide

This guide provides a comprehensive reference for the `d2l` package, a core component of the *Dive into Deep Learning* (D2L) book. It lists the available classes and functions, helping you locate their implementations and understand their purpose within the book's tutorials.

## Overview

The `d2l` package is a utility library designed to support the hands-on examples in the D2L book. It provides consistent abstractions and helper functions across different deep learning frameworks (PyTorch, TensorFlow, JAX, and MXNet), allowing you to focus on core concepts rather than framework-specific boilerplate.

**Source Code:** You can explore the full implementation on the [D2L GitHub Repository](https://github.com/d2l-ai/d2l-en/tree/master/d2l).

## API Structure

The package is organized into modules corresponding to each supported framework. The classes and functions below are available within these modules.

### Classes

The following classes are implemented in the `d2l` package. Each class link points to its detailed documentation within the book.

| Class | Description |
| :--- | :--- |
| **`AdditiveAttention`** | Implements additive/bahdanau attention mechanism. |
| **`AddNorm`** | A residual connection followed by layer normalization. |
| **`AttentionDecoder`** | The base decoder interface for models with attention. |
| **`Classifier`** | A base class for classification models. |
| **`DataModule`** | A base class for data loading and preprocessing. |
| **`Decoder`** | The base decoder interface for sequence-to-sequence models. |
| **`DotProductAttention`** | Implements scaled dot-product attention. |
| **`Encoder`** | The base encoder interface. |
| **`EncoderDecoder`** | The base class for encoder-decoder architectures. |
| **`FashionMNIST`** | A DataModule for the Fashion-MNIST dataset. |
| **`GRU`** | A concise implementation of a Gated Recurrent Unit. |
| **`HyperParameters`** | A utility class for managing model hyperparameters. |
| **`LeNet`** | An implementation of the LeNet-5 convolutional network. |
| **`LinearRegression`** | A linear regression model (framework-native version). |
| **`LinearRegressionScratch`** | A linear regression model implemented from scratch. |
| **`Module`** | A base class for all neural network modules. |
| **`MTFraEng`** | A DataModule for the English-French machine translation dataset. |
| **`MultiHeadAttention`** | Implements multi-head attention. |
| **`PositionalEncoding`** | Injects positional information into embeddings. |
| **`PositionWiseFFN`** | The position-wise feed-forward network used in Transformers. |
| **`ProgressBoard`** | A utility for plotting training progress in notebooks. |
| **`Residual`** | A residual block implementation. |
| **`ResNeXtBlock`** | Implements a ResNeXt bottleneck block. |
| **`RNN`** | A concise implementation of a Recurrent Neural Network. |
| **`RNNLM`** | An RNN-based language model (framework-native version). |
| **`RNNLMScratch`** | An RNN-based language model implemented from scratch. |
| **`RNNScratch`** | An RNN implemented from scratch. |
| **`Seq2Seq`** | A sequence-to-sequence model for machine translation. |
| **`Seq2SeqEncoder`** | An RNN encoder for sequence-to-sequence models. |
| **`SGD`** | A minimal stochastic gradient descent optimizer. |
| **`SoftmaxRegression`** | A softmax regression model for multiclass classification. |
| **`SyntheticRegressionData`** | A DataModule for generating synthetic linear regression data. |
| **`TimeMachine`** | A DataModule for *The Time Machine* dataset. |
| **`Trainer`** | A training loop utility class. |
| **`TransformerEncoder`** | The full Transformer encoder. |
| **`TransformerEncoderBlock`** | A single block of the Transformer encoder. |
| **`Vocab`** | A utility class for text tokenization and vocabulary building. |

### Functions

The `d2l` package also includes numerous utility functions to simplify common tasks.

| Function | Description |
| :--- | :--- |
| **`add_to_class`** | A utility for dynamically adding methods to a class. |
| **`bleu`** | Calculates the BLEU score for machine translation evaluation. |
| **`check_len`** | Asserts the length of a data structure. |
| **`check_shape`** | Asserts the shape of a tensor. |
| **`corr2d`** | Performs 2D cross-correlation (the core operation of convolution). |
| **`cpu`** | Moves all tensors in a structure to the CPU. |
| **`gpu`** | Moves all tensors in a structure to the GPU. |
| **`init_cnn`** | Initializes weights for a CNN using the Xavier method. |
| **`init_seq2seq`** | Initializes weights for a sequence-to-sequence model. |
| **`masked_softmax`** | Performs a softmax operation with masking for variable-length sequences. |
| **`num_gpus`** | Returns the number of available GPUs. |
| **`plot`** | A simplified interface for creating line plots. |
| **`set_axes`** | Configures the properties of matplotlib axes. |
| **`set_figsize`** | Sets the default figure size for matplotlib. |
| **`show_heatmaps`** | Visualizes attention weights or other matrices as heatmaps. |
| **`show_list_len_pair_hist`** | Plots a histogram of sequence length pairs. |
| **`try_all_gpus`** | Returns a list of all available GPUs. |
| **`try_gpu`** | Returns the first available GPU, or CPU if none are available. |
| **`use_svg_display`** | Configures Jupyter to use SVG for sharper plot rendering. |

## How to Use This Reference

1.  **Identify the Component:** Find the class or function you need from the tables above.
2.  **Locate the Implementation:** Click the link in the original book chapter (or navigate the GitHub repository) to see the full source code, detailed explanations, and usage examples.
3.  **Import in Your Code:** Import the component from the correct framework-specific submodule. For example, in a PyTorch project:
    ```python
    from d2l import torch as d2l
    # Use a class
    encoder = d2l.TransformerEncoder(...)
    # Use a function
    d2l.try_gpu()
    ```

This API reference serves as a centralized index. For a complete understanding of how these components work together, follow the step-by-step tutorials in the corresponding chapters of the *Dive into Deep Learning* book.
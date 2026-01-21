# A Guide to Hybrid Programming for Deep Learning

This guide explores the concepts of imperative and symbolic programming in the context of deep learning frameworks. You will learn how modern frameworks combine these paradigms to offer both ease of development and high-performance execution.

## Prerequisites

Ensure you have the necessary libraries installed. The code examples are provided for three major frameworks: MXNet, PyTorch, and TensorFlow.

```bash
# Install D2L book library for framework-specific utilities
pip install d2l
```

## 1. Understanding Imperative Programming

Imperative programming, the style used in standard Python, executes operations line-by-line, immediately changing the program's state.

Consider this simple example:

```python
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

**Output:**
```
10
```

**How it works:** The Python interpreter executes each statement sequentially. It computes `e = add(1, 2)`, stores the result, then computes `f = add(3, 4)`, and finally `g = add(e, f)`. This approach is intuitive and easy to debug, as you can inspect any intermediate variable (like `e` or `f`).

However, this flexibility comes at a cost. The interpreter overhead can become a bottleneck, especially when running computations on fast hardware like GPUs. The program must also retain intermediate values in memory until they are no longer needed.

## 2. Introducing Symbolic Programming

Symbolic programming takes a different approach. You first define the entire computation graph, then compile it into an efficient, low-level program before execution. This is the traditional method used by frameworks like Theano and TensorFlow 1.x.

The process involves three steps:
1. Define the operations.
2. Compile them into an executable program.
3. Execute the compiled program with the required inputs.

This allows for significant optimizations. A compiler can see the whole computation (e.g., `((1+2)+(3+4))`) and simplify it (e.g., to `10`). It can also optimize memory usage and execution order.

To illustrate the difference, here's a simulation where we build and compile Python code as a string:

```python
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

**Output:**
```
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
print(fancy_func(1, 2, 3, 4))
10
```

**Key Trade-offs:**
*   **Imperative (Interpreted):** Easier to write and debug.
*   **Symbolic (Compiled):** More efficient and portable to non-Python environments.

## 3. The Hybrid Approach

Modern frameworks bridge this gap through **hybrid programming**. You develop and debug using the familiar imperative style, then convert your model to a symbolic representation for high-performance inference and deployment.

*   **MXNet** uses the `hybridize()` method on `HybridBlock` or `HybridSequential` models.
*   **PyTorch** uses TorchScript via the `torch.jit.script()` function.
*   **TensorFlow** uses graph-mode execution via the `tf.function()` decorator or function.

## 4. Implementing a Hybrid Model

Let's create a simple Multi-Layer Perceptron (MLP) and see how to hybridize it in each framework.

### Step 1: Define the Model

First, we define a factory function to create our neural network.

**For MXNet:**
```python
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def get_net():
    net = nn.HybridSequential()  # Note the use of HybridSequential
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net
```

**For PyTorch:**
```python
from d2l import torch as d2l
import torch
from torch import nn

def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net
```

**For TensorFlow:**
```python
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net
```

### Step 2: Run the Model in Imperative Mode

Create some dummy data and run a forward pass to ensure the model works.

```python
# MXNet
x = np.random.normal(size=(1, 512))
net = get_net()
print(net(x))

# PyTorch
x = torch.randn(size=(1, 512))
net = get_net()
print(net(x))

# TensorFlow
x = tf.random.normal([1,512])
net = get_net()
print(net(x))
```

### Step 3: Convert to Symbolic Execution

Now, we convert the model to its hybrid/symbolic form. The computation result should remain identical.

**For MXNet:**
```python
net.hybridize()
net(x)
```

**For PyTorch:**
```python
net = torch.jit.script(net)
net(x)
```

**For TensorFlow:**
```python
net = tf.function(net) # Converts to graph mode
# net = tf.function(net, jit_compile=True) # Optional: Enable XLA for further GPU optimizations
net(x)
```

## 5. Benchmarking Performance

Let's quantify the performance gain from hybridization. We'll use a simple timer class.

```python
class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

Now, compare execution times.

**For MXNet:**
```python
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall() # Wait for all async operations (MXNet-specific)

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

**For PyTorch:**
```python
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

**For TensorFlow:**
```python
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```

**Expected Outcome:** You should observe a significant speedup for the hybrid/graph-mode execution, especially over many iterations, as the framework overhead is minimized.

## 6. Serialization for Deployment

A major advantage of symbolic models is portability. You can serialize (save) the optimized computation graph and parameters to disk, independent of the Python front-end.

**For MXNet:**
```python
net.export('my_mlp') # Saves 'my_mlp-symbol.json' and 'my_mlp-xxxx.params'
```

**For PyTorch:**
```python
net.save('my_mlp.pt')
```

**For TensorFlow:**
```python
tf.saved_model.save(net, 'my_mlp_saved_model')
```

These saved models can be loaded and executed in different environments (e.g., C++ servers, mobile devices), providing both performance and deployment flexibility.

## 7. Important Considerations and Limitations

Hybridization is powerful but has constraints. When a model is compiled, dynamic Python features are frozen or removed.

**Example in MXNet:** If you define a custom `HybridBlock`, you must use `hybrid_forward` instead of `forward`. Debug print statements inside `hybrid_forward` will only execute during the first run (the tracing run) and are omitted thereafter.

```python
# MXNet-specific example
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x): # Uses F and hybrid_forward
        print('Tracing with x:', x) # This will print only once
        x = F.npx.relu(self.hidden(x))
        return self.output(x)

net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x) # Prints trace
net.hybridize()
net(x) # No print output
net(x) # No print output
```

**Key Limitations:**
*   Complex control flow (e.g., dynamic `if` conditions based on input data) may not be convertible.
*   Operations that rely on the Python interpreter (like certain NumPy conversions) are disallowed after hybridization.
*   In-place operations (e.g., `a += b`) often need to be rewritten.

## Summary

*   **Imperative Programming** offers flexibility and ease of debugging.
*   **Symbolic Programming** provides performance optimizations and deployment portability.
*   **Hybrid Programming**, available in MXNet (`hybridize`), PyTorch (`torch.jit`), and TensorFlow (`tf.function`), lets you leverage the best of both worlds: develop imperatively, then deploy symbolically.

## Exercises

1.  **Review Past Models:** Look at models from previous chapters. Identify which could benefit from re-implementation using your framework's hybrid features.
2.  **MXNet-Specific:**
    *   Try adding `x.asnumpy()` inside a `hybrid_forward` function. What error do you get and why?
    *   Experiment with adding Python `if` or `for` statements within `hybrid_forward`. How does hybridization behave?
3.  **Benchmark Your Own Model:** Apply the hybridization technique to a small project of your own and measure the performance difference.

By mastering hybrid programming, you can write clean, debuggable code without sacrificing the performance required for production-scale deep learning.
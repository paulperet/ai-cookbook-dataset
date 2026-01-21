# PyTorch C++ Autograd Tutorial

This guide walks you through using PyTorch's automatic differentiation (autograd) system in C++. If you're familiar with PyTorch's Python autograd API, you'll find the C++ frontend provides similar functionality, making translation straightforward.

## Prerequisites

Ensure you have PyTorch C++ libraries installed. For basic setup, include the necessary headers:

```cpp
#include <torch/torch.h>
```

## Basic Autograd Operations

### 1. Creating Tensors with Gradient Tracking

Start by creating a tensor and enabling gradient computation:

```cpp
auto x = torch::ones({2, 2}, torch::requires_grad());
std::cout << x << std::endl;
```

Output:
```
1 1
1 1
[ CPUFloatType{2,2} ]
```

### 2. Performing Operations

Perform tensor operations. The resulting tensor automatically tracks its computation history:

```cpp
auto y = x + 2;
std::cout << y << std::endl;
```

Output:
```
3  3
3  3
[ CPUFloatType{2,2} ]
```

Check the gradient function associated with `y`:

```cpp
std::cout << y.grad_fn()->name() << std::endl;
```

Output:
```
AddBackward1
```

### 3. Chaining Operations

Continue with more operations:

```cpp
auto z = y * y * 3;
auto out = z.mean();

std::cout << z << std::endl;
std::cout << z.grad_fn()->name() << std::endl;
std::cout << out << std::endl;
std::cout << out.grad_fn()->name() << std::endl;
```

Output:
```
27  27
27  27
[ CPUFloatType{2,2} ]
MulBackward1
27
[ CPUFloatType{} ]
MeanBackward0
```

### 4. Modifying Gradient Requirements

Use `.requires_grad_()` to modify gradient tracking in-place:

```cpp
auto a = torch::randn({2, 2});
a = ((a * 3) / (a - 1));
std::cout << a.requires_grad() << std::endl;

a.requires_grad_(true);
std::cout << a.requires_grad() << std::endl;

auto b = (a * a).sum();
std::cout << b.grad_fn()->name() << std::endl;
```

Output:
```
false
true
SumBackward0
```

### 5. Computing Gradients

Since `out` is a scalar, call `.backward()` to compute gradients:

```cpp
out.backward();
```

Now examine the gradients with respect to `x`:

```cpp
std::cout << x.grad() << std::endl;
```

Output:
```
4.5000  4.5000
4.5000  4.5000
[ CPUFloatType{2,2} ]
```

You should see a matrix of `4.5`. For the mathematical derivation, refer to the [Python autograd tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients).

### 6. Vector-Jacobian Product

For non-scalar outputs, you can compute vector-Jacobian products by passing a gradient vector to `.backward()`:

```cpp
x = torch::randn(3, torch::requires_grad());

y = x * 2;
while (y.norm().item<double>() < 1000) {
  y = y * 2;
}

std::cout << y << std::endl;
std::cout << y.grad_fn()->name() << std::endl;
```

Output:
```
-1021.4020
  314.6695
 -613.4944
[ CPUFloatType{3} ]
MulBackward1
```

Now compute the vector-Jacobian product:

```cpp
auto v = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
y.backward(v);

std::cout << x.grad() << std::endl;
```

Output:
```
102.4000
1024.0000
  0.1024
[ CPUFloatType{3} ]
```

### 7. Disabling Gradient Tracking

Use `torch::NoGradGuard` to temporarily disable gradient computation:

```cpp
std::cout << x.requires_grad() << std::endl;
std::cout << x.pow(2).requires_grad() << std::endl;

{
  torch::NoGradGuard no_grad;
  std::cout << x.pow(2).requires_grad() << std::endl;
}
```

Output:
```
true
true
false
```

Alternatively, use `.detach()` to create a tensor without gradient tracking:

```cpp
std::cout << x.requires_grad() << std::endl;
y = x.detach();
std::cout << y.requires_grad() << std::endl;
std::cout << x.eq(y).all().item<bool>() << std::endl;
```

Output:
```
true
false
true
```

## Computing Higher-Order Gradients

Higher-order gradients are useful for techniques like gradient penalty. Here's an example using `torch::autograd::grad`:

```cpp
#include <torch/torch.h>

auto model = torch::nn::Linear(4, 3);

auto input = torch::randn({3, 4}).requires_grad_(true);
auto output = model(input);

// Calculate loss
auto target = torch::randn({3, 3});
auto loss = torch::nn::MSELoss()(output, target);

// Use norm of gradients as penalty
auto grad_output = torch::ones_like(output);
auto gradient = torch::autograd::grad({output}, {input}, /*grad_outputs=*/{grad_output}, /*create_graph=*/true)[0];
auto gradient_penalty = torch::pow((gradient.norm(2, /*dim=*/1) - 1), 2).mean();

// Add gradient penalty to loss
auto combined_loss = loss + gradient_penalty;
combined_loss.backward();

std::cout << input.grad() << std::endl;
```

Output:
```
-0.1042 -0.0638  0.0103  0.0723
-0.2543 -0.1222  0.0071  0.0814
-0.1683 -0.1052  0.0355  0.1024
[ CPUFloatType{3,4} ]
```

Refer to the documentation for [`torch::autograd::backward`](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1403bf65b1f4f8c8506a9e6e5312d030.html) and [`torch::autograd::grad`](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1ab9fa15dc09a8891c26525fb61d33401a.html) for more details.

## Implementing Custom Autograd Functions

To add custom operations to `torch::autograd`, subclass `torch::autograd::Function`. You must implement `forward` and `backward` methods.

### Example: Linear Function

Here's an implementation of a `Linear` function similar to `torch::nn::Linear`:

```cpp
#include <torch/torch.h>

using namespace torch::autograd;

// Inherit from Function
class LinearFunction : public Function<LinearFunction> {
 public:
  // Note that both forward and backward are static functions

  // bias is an optional argument
  static torch::Tensor forward(
      AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
    ctx->save_for_backward({input, weight, bias});
    auto output = input.mm(weight.t());
    if (bias.defined()) {
      output += bias.unsqueeze(0).expand_as(output);
    }
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output.mm(weight);
    auto grad_weight = grad_output.t().mm(input);
    auto grad_bias = torch::Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    return {grad_input, grad_weight, grad_bias};
  }
};
```

Use the custom function:

```cpp
auto x = torch::randn({2, 3}).requires_grad_();
auto weight = torch::randn({4, 3}).requires_grad_();
auto y = LinearFunction::apply(x, weight);
y.sum().backward();

std::cout << x.grad() << std::endl;
std::cout << weight.grad() << std::endl;
```

Output:
```
0.5314  1.2807  1.4864
0.5314  1.2807  1.4864
[ CPUFloatType{2,3} ]
3.7608  0.9101  0.0073
3.7608  0.9101  0.0073
3.7608  0.9101  0.0073
3.7608  0.9101  0.0073
[ CPUFloatType{4,3} ]
```

### Example: Function with Non-Tensor Arguments

Here's a function that multiplies by a constant scalar:

```cpp
#include <torch/torch.h>

using namespace torch::autograd;

class MulConstant : public Function<MulConstant> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor tensor, double constant) {
    // ctx is a context object that can be used to stash information
    // for backward computation
    ctx->saved_data["constant"] = constant;
    return tensor * constant;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    // We return as many input gradients as there were arguments.
    // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
    return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::Tensor()};
  }
};
```

Use the function:

```cpp
auto x = torch::randn({2}).requires_grad_();
auto y = MulConstant::apply(x, 5.5);
y.sum().backward();

std::cout << x.grad() << std::endl;
```

Output:
```
5.5000
5.5000
[ CPUFloatType{2} ]
```

For more details, see the [`torch::autograd::Function` documentation](https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html).

## Translating Autograd Code from Python to C++

If you have working Python autograd code, you can translate it to C++ using this reference table:

| Python | C++ |
|--------|-----|
| `torch.autograd.backward` | [`torch::autograd::backward`](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1403bf65b1f4f8c8506a9e6e5312d030.html) |
| `torch.autograd.grad` | [`torch::autograd::grad`](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1ab9fa15dc09a8891c26525fb61d33401a.html) |
| `torch.Tensor.detach` | [`torch::Tensor::detach`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor6detachEv) |
| `torch.Tensor.detach_` | [`torch::Tensor::detach_`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7detach_Ev) |
| `torch.Tensor.backward` | [`torch::Tensor::backward`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor8backwardERK6Tensorbb) |
| `torch.Tensor.register_hook` | [`torch::Tensor::register_hook`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4I0ENK2at6Tensor13register_hookE18hook_return_void_tI1TERR1T) |
| `torch.Tensor.requires_grad` | [`torch::Tensor::requires_grad_`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor14requires_grad_Eb) |
| `torch.Tensor.retain_grad` | [`torch::Tensor::retain_grad`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor11retain_gradEv) |
| `torch.Tensor.grad` | [`torch::Tensor::grad`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor4gradEv) |
| `torch.Tensor.grad_fn` | [`torch::Tensor::grad_fn`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7grad_fnEv) |
| `torch.Tensor.set_data` | [`torch::Tensor::set_data`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor8set_dataERK6Tensor) |
| `torch.Tensor.data` | [`torch::Tensor::data`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor4dataEv) |
| `torch.Tensor.output_nr` | [`torch::Tensor::output_nr`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor9output_nrEv) |
| `torch.Tensor.is_leaf` | [`torch::Tensor::is_leaf`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7is_leafEv) |

## Conclusion

You now have a solid foundation for using PyTorch's C++ autograd API. The examples shown here are available in the [PyTorch examples repository](https://github.com/pytorch/examples/tree/master/cpp/autograd). For further assistance, visit the [PyTorch forum](https://discuss.pytorch.org/) or [GitHub issues](https://github.com/pytorch/pytorch/issues).
# Dequantizing Embeddings: A Practical Guide

This guide demonstrates how to dequantize various quantized representations (int8, uint8, binary, ubinary) of embeddings back to float values using PyTorch and NumPy.

## Prerequisites

First, ensure you have the required packages installed:

```bash
pip install torch numpy
```

## Implementation

### 1. Import Required Libraries

```python
import torch
import numpy as np
```

### 2. Define the Core Unpacking Function

This helper function unpacks bit-packed tensors into a float32 tensor with values of ±1.

```python
def _unpack_sign_bits(x_packed: torch.Tensor, signed: bool, C: int) -> torch.Tensor:
    """
    Unpacks a bit-packed tensor into a ±1 float32 tensor.

    Parameters:
    ----------
    x_packed : torch.Tensor
        A tensor containing packed bit values (usually uint8) representing the sign bits.
    signed : bool
        Indicates if the original data was signed ('binary') or unsigned ('ubinary').
    C : int
        The number of original channels or dimensions to unpack (used to trim padding bits).

    Returns:
    -------
    torch.Tensor
        A float32 tensor on the same device as x_packed, with values ±1 and shape (..., C).
    """
    arr = (x_packed.cpu().numpy() + (128 if signed else 0)).astype(np.uint8)
    bits = np.unpackbits(arr, axis=-1)[..., :C]  # remove pad bits
    return torch.from_numpy(bits * 2 - 1).float().to(x_packed.device)
```

### 3. Define the Main Dequantization Function

This is the primary function that handles all supported quantization types.

```python
def dequantize(
    q: torch.Tensor | list, quant: str, orig_dim: int | None = None
) -> torch.Tensor:
    """
    Dequantizes a quantized tensor or list back to float32 values in [-1, 1].

    Parameters:
    ----------
    q : torch.Tensor or list
        The quantized data, either as a tensor or list.
    quant : str
        The quantization type. Supported values:
        - 'fp8': already float, just converted.
        - 'int8': scaled by 127.
        - 'uint8': scaled and shifted to [-1,1].
        - 'binary' or 'ubinary': unpacked sign bits (requires orig_dim).
    orig_dim : int or None, optional
        The original number of dimensions/channels before packing,
        required when quant is 'binary' or 'ubinary' to correctly unpack bits.

    Returns:
    -------
    torch.Tensor
        A float32 tensor of shape (B, T, C) with values in [-1, 1].

    Raises:
    ------
    ValueError
        If an unsupported quantization type is provided or if `orig_dim` is missing
        for 'binary'/'ubinary' unpacking.
    """
    if isinstance(q, list):
        q = torch.tensor(q)
    if quant == "fp8":
        return q.float()
    if quant == "int8":
        return q.float() / 127.0
    if quant == "uint8":
        return q.float() / 127.5 - 1.0
    if quant in {"binary", "ubinary"}:
        if orig_dim is None:
            raise ValueError("orig_dim needed for (u)binary unpack")
        return _unpack_sign_bits(q, quant == "binary", orig_dim)
    raise ValueError(f"Invalid quantization {quant}")
```

## Practical Examples

Now, let's test the dequantization function with sample data in different formats.

### 1. Prepare Sample Data

We'll create sample embeddings in five different quantized formats.

```python
# Original float values (for reference)
embed_float = [-0.11944580078125,-0.2734375,0.040771484375,0.3056640625,-0.1470947265625,-0.11749267578125,0.0799560546875,0.08282470703125,-0.04205322265625,0.220947265625,0.0015048980712890625,-0.00397491455078125,-0.01099395751953125,-0.052642822265625,0.0504150390625,0.01605224609375,0.029693603515625,-0.024078369140625]

# Quantized representations
embed_int = [-15,-35,5,39,-19,-15,10,11,-5,28,0,0,-2,-7,6,2,4,-3]
embed_uint = [112,93,133,166,109,112,138,138,122,156,128,127,126,121,134,130,131,124]
embed_bin = [-77,-29,0]
embed_ubin = [51,99,128]
```

### 2. Dequantize Binary Representations

Binary and ubinary formats require the `orig_dim` parameter to correctly unpack the bits.

```python
# Dequantize binary representation
result_bin = dequantize(embed_bin, quant="binary", orig_dim=18)
print("Binary dequantization result:")
print(result_bin)
```

```python
# Dequantize ubinary representation
result_ubin = dequantize(embed_ubin, quant="ubinary", orig_dim=18)
print("\nUbinary dequantization result:")
print(result_ubin)
```

### 3. Dequantize Integer Representations

The int8 and uint8 formats don't require the `orig_dim` parameter.

```python
# Dequantize int8 representation
result_int = dequantize(embed_int, quant="int8")
print("\nInt8 dequantization result:")
print(result_int)
```

```python
# Dequantize uint8 representation
result_uint = dequantize(embed_uint, quant="uint8")
print("\nUint8 dequantization result:")
print(result_uint)
```

## Key Takeaways

1. **Flexible Input**: The `dequantize` function accepts both PyTorch tensors and Python lists.
2. **Multiple Formats**: Supports fp8, int8, uint8, binary, and ubinary quantization types.
3. **Binary Handling**: Binary and ubinary formats require the `orig_dim` parameter to correctly unpack the bit-packed data.
4. **Output Range**: All dequantized values are normalized to the range [-1, 1].

The dequantized results will closely approximate the original float values, with minor differences due to quantization precision loss.
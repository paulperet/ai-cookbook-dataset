# Dequantization embeddings

This notebook demonstrates how to dequantize various quantized representations (int8, uint8, binary, ubinary) of embeddings back to float values using PyTorch and NumPy.

## Installation

Install the required packages:


```python
!pip install torch numpy
```

[First Entry, ..., Last Entry]

## define the dequantization functions


```python
import torch
import numpy as np

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

## Examples


```python
embed_float = [-0.11944580078125,-0.2734375,0.040771484375,0.3056640625,-0.1470947265625,-0.11749267578125,0.0799560546875,0.08282470703125,-0.04205322265625,0.220947265625,0.0015048980712890625,-0.00397491455078125,-0.01099395751953125,-0.052642822265625,0.0504150390625,0.01605224609375,0.029693603515625,-0.024078369140625]
embed_int = [-15,-35,5,39,-19,-15,10,11,-5,28,0,0,-2,-7,6,2,4,-3]
embed_uint = [112,93,133,166,109,112,138,138,122,156,128,127,126,121,134,130,131,124]
embed_bin = [-77,-29,0]
embed_ubin = [51,99,128]
```


```python
dequantize(embed_bin, quant="binary", orig_dim=18)
```




    tensor([255., 255.,   1.,   1., 255., 255.,   1.,   1., 255.,   1.,   1., 255.,
            255., 255.,   1.,   1.,   1., 255.])




```python
dequantize(embed_ubin, quant="ubinary", orig_dim=18)
```




    tensor([255., 255.,   1.,   1., 255., 255.,   1.,   1., 255.,   1.,   1., 255.,
            255., 255.,   1.,   1.,   1., 255.])




```python
dequantize(embed_int, quant="int8")
```




    tensor([-0.1181, -0.2756,  0.0394,  0.3071, -0.1496, -0.1181,  0.0787,  0.0866,
            -0.0394,  0.2205,  0.0000,  0.0000, -0.0157, -0.0551,  0.0472,  0.0157,
             0.0315, -0.0236])




```python
dequantize(embed_uint, quant="uint8")
```




    tensor([-0.1216, -0.2706,  0.0431,  0.3020, -0.1451, -0.1216,  0.0824,  0.0824,
            -0.0431,  0.2235,  0.0039, -0.0039, -0.0118, -0.0510,  0.0510,  0.0196,
             0.0275, -0.0275])




```python

```
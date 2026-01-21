# PyTorch torch.export Cookbook: Solving Common Export Challenges

## Overview

This tutorial builds upon the [Introduction to torch.export](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html) guide, demonstrating how to export popular AI models while addressing common challenges. You'll learn practical solutions for exporting video classifiers, speech recognition models, image captioning systems, and segmentation models.

## Prerequisites

- PyTorch 2.4 or later
- Basic understanding of `torch.export` and PyTorch eager inference
- Familiarity with the models covered (MViT, Whisper, BLIP, SAM2)

## Key Concept: No Graph Breaks

Unlike `torch.compile`, which allows graph breaks for unsupported operations, `torch.export` requires your entire model to be a single, unbroken computation graph. This is because `torch.export` aims to create a fully traceable, deployable model without Python interpreter fallbacks.

To check for graph breaks in your code:
```bash
TORCH_LOGS="graph_breaks" python <file_name>.py
```

The models in this tutorial have no graph breaks but present other export challenges that we'll solve.

---

## 1. Video Classification with MViT

MViT (MultiScale Vision Transformers) is trained for video classification and action recognition. Let's export it with dynamic batch size support.

### Initial Export Attempt

```python
import numpy as np
import torch
from torchvision.models.video import MViT_V1_B_Weights, mvit_v1_b
import traceback as tb

# Load model
model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

# Create input batch (2 videos, 16 frames each)
input_frames = torch.randn(2, 16, 224, 224, 3)
input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))  # [batch, channels, frames, height, width]

# Export with static batch size
exported_program = torch.export.export(
    model,
    (input_frames,),
)

# Try inference with different batch size
input_frames = torch.randn(4, 16, 224, 224, 3)
input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
try:
    exported_program.module()(input_frames)
except Exception:
    tb.print_exc()
```

**Error:** `RuntimeError: Expected input at *args[0].shape[0] to be equal to 2, but got 4`

By default, `torch.export` assumes all input shapes are static. Changing the batch size after export causes this error.

### Solution: Dynamic Batch Sizes

Specify dynamic dimensions using `torch.export.Dim`:

```python
import numpy as np
import torch
from torchvision.models.video import MViT_V1_B_Weights, mvit_v1_b

# Load model
model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

# Create input batch
input_frames = torch.randn(2, 16, 224, 224, 3)
input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))

# Define dynamic batch dimension (range 1-16)
batch_dim = torch.export.Dim("batch", min=2, max=16)

# Export with dynamic batch size
exported_program = torch.export.export(
    model,
    (input_frames,),
    dynamic_shapes={"x": {0: batch_dim}},  # First dimension is dynamic
)

# Now works with batch size 4
input_frames = torch.randn(4, 16, 224, 224, 3)
input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
result = exported_program.module()(input_frames)
print(f"Success! Output shape: {result.shape}")
```

**Note:** The `min=2` parameter addresses the [0/1 Specialization Problem](https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit).

---

## 2. Automatic Speech Recognition with Whisper

Whisper is a Transformer-based encoder-decoder model for speech recognition. Let's export the tiny version.

### Initial Export Attempt

```python
import torch
from transformers import WhisperForConditionalGeneration

# Load model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Create dummy inputs
input_features = torch.randn(1, 80, 3000)
attention_mask = torch.ones(1, 3000)
decoder_input_ids = torch.tensor([[1, 1, 1, 1]]) * model.config.decoder_start_token_id

model.eval()

# Export with strict mode (default)
exported_program = torch.export.export(
    model, 
    args=(input_features, attention_mask, decoder_input_ids,)
)
```

**Error:** `torch._dynamo.exc.InternalTorchDynamoError: AttributeError: 'DynamicCache' object has no attribute 'key_cache'`

The strict tracing mode in TorchDynamo encounters unsupported Python features in the Whisper implementation.

### Solution: Non-Strict Mode

Use `strict=False` to trace using Python interpreter semantics:

```python
import torch
from transformers import WhisperForConditionalGeneration

# Load model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Create dummy inputs
input_features = torch.randn(1, 80, 3000)
attention_mask = torch.ones(1, 3000)
decoder_input_ids = torch.tensor([[1, 1, 1, 1]]) * model.config.decoder_start_token_id

model.eval()

# Export with non-strict mode
exported_program = torch.export.export(
    model,
    args=(input_features, attention_mask, decoder_input_ids,),
    strict=False  # Use Python interpreter for tracing
)

print(f"Successfully exported Whisper model")
```

This mode replaces all `Tensor` objects with `ProxyTensors` that record operations into a graph while maintaining Python execution semantics.

---

## 3. Image Captioning with BLIP

BLIP generates textual descriptions for images. Let's export it for deployment.

### Initial Export Attempt

```python
import torch
from models.blip import blip_decoder

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 384
image = torch.randn(1, 3, 384, 384).to(device)
caption_input = ""

# Load model
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

# Export
exported_program = torch.export.export(
    model,
    args=(image, caption_input,),
    strict=False
)
```

**Error:** `RuntimeError: cannot mutate tensors with frozen storage`

The error occurs at line 112 in the BLIP implementation where the code attempts to modify a tensor's storage directly.

### Solution: Clone the Tensor

Modify the BLIP source code at the problematic location:

```python
# Original problematic code in blip.py line 112:
# text.input_ids[:,0] = self.tokenizer.bos_token_id

# Fixed version:
text.input_ids = text.input_ids.clone()  # Clone the tensor first
text.input_ids[:,0] = self.tokenizer.bos_token_id
```

After making this change, the export will succeed:

```python
# Now the export works
exported_program = torch.export.export(
    model,
    args=(image, caption_input,),
    strict=False
)

print(f"Successfully exported BLIP model")
```

**Note:** This constraint has been relaxed in PyTorch 2.7 nightlies and should work out-of-the-box in PyTorch 2.7.

---

## 4. Promptable Image Segmentation with SAM2

SAM2 provides unified segmentation across images and videos. The challenge is exporting a class method rather than a `torch.nn.Module`.

### Initial Export Attempt

```python
# Inside SAM2ImagePredictor.predict() method
ep = torch.export.export(
    self._predict,  # This is a class method, not a torch.nn.Module
    args=(unnorm_coords, labels, unnorm_box, mask_input, multimask_output),
    kwargs={"return_logits": return_logits},
    strict=False,
)
```

**Error:** `ValueError: Expected 'mod' to be an instance of 'torch.nn.Module', got <class 'method'>`

`torch.export` requires the object being exported to inherit from `torch.nn.Module`.

### Solution: Create a Wrapper Module

Create a helper class that wraps the method:

```python
class ExportHelper(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
    
    def forward(self, *args, **kwargs):
        # Call the original _predict method
        return self.predictor._predict(*args, **kwargs)

# Usage in SAM2ImagePredictor.predict():
model_to_export = ExportHelper(self)
ep = torch.export.export(
    model_to_export,
    args=(unnorm_coords, labels, unnorm_box, mask_input, multimask_output),
    kwargs={"return_logits": return_logits},
    strict=False,
)

print(f"Successfully exported SAM2 predictor")
```

This wrapper inherits from `torch.nn.Module` and delegates to the original `_predict` method in its `forward` method.

---

## Next Steps

Once you've successfully exported your models, you can deploy them using:

1. **AOTInductor** for server deployment: [AOTI Tutorial](https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html)
2. **ExecuTorch** for edge devices: [ExecuTorch Tutorial](https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html)

## Summary

This tutorial demonstrated solutions to common `torch.export` challenges:

1. **Dynamic shapes** for variable batch sizes using `torch.export.Dim`
2. **Non-strict mode** for models with unsupported Python features
3. **Tensor cloning** to fix storage mutation errors
4. **Wrapper modules** for exporting class methods

By applying these patterns, you can export a wide variety of PyTorch models for production deployment.
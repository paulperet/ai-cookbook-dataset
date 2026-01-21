# Guide: Image Interpolation with Stable Diffusion

_Authored by: [Rustam Akimov](https://github.com/AkiRusProd)_

This guide demonstrates how to use Stable Diffusion to interpolate between images—creating smooth visual transitions from one image to another by navigating the model's learned latent space.

**Use Cases:**
- **Data Augmentation:** Generate synthetic training data between existing samples to improve model generalization.
- **Product Design:** Visualize subtle design variations and prototypes.
- **Media Production:** Create intermediate frames for smoother animations and transitions.

## Prerequisites

Install the required Python packages:

```bash
pip install -q diffusers transformers xformers accelerate
pip install -q numpy scipy ftfy Pillow
```

## Setup

Import the necessary modules and configure the environment.

```python
import torch
import numpy as np
import os
import time

from PIL import Image
from IPython import display as IPdisplay
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler
from transformers import logging

logging.set_verbosity_error()
```

Verify CUDA availability and set the device.

```python
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

Optimize PyTorch for CUDA performance:

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

## Initialize the Model

We'll use the `runwayml/stable-diffusion-v1-5` model with the `LMSDiscreteScheduler`. This combination offers a good balance of speed, memory efficiency, and quality.

```python
model_name_or_path = "runwayml/stable-diffusion-v1-5"

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000
)

pipe = StableDiffusionPipeline.from_pretrained(
    model_name_or_path,
    scheduler=scheduler,
    torch_dtype=torch.float32,
).to(device)

pipe.set_progress_bar_config(disable=True)
```

### Memory Optimization (Optional)

If you have limited VRAM, enable these optimizations:

```python
pipe.enable_model_cpu_offload()
pipe.unet.to(memory_format=torch.channels_last)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.enable_xformers_memory_efficient_attention()
```

## Helper Function: `display_images`

This function converts a list of image arrays into a GIF, saves it, and returns the GIF for display.

```python
def display_images(images, save_path):
    try:
        images = [
            Image.fromarray(np.array(image[0], dtype=np.uint8)) for image in images
        ]
        filename = time.strftime("%H:%M:%S", time.localtime()).replace(":", "-")
        images[0].save(
            f"{save_path}/{filename}.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
    except Exception as e:
        print(e)

    return IPdisplay.Image(f"{save_path}/{filename}.gif")
```

## Configuration Parameters

Define the key parameters for generation and interpolation.

```python
seed = None  # Set to an integer for reproducibility
if seed is not None:
    generator = torch.manual_seed(seed)
else:
    generator = None

guidance_scale = 8
num_inference_steps = 15
num_interpolation_steps = 30
height = 512
width = 512

save_path = "/output"
if not os.path.exists(save_path):
    os.makedirs(save_path)
```

---

## Example 1: Prompt Interpolation

Interpolate between positive and negative prompt embeddings to explore the conceptual space between them.

### Step 1: Tokenize and Encode Prompts

First, we convert the text prompts into embeddings.

```python
prompt = "Epic shot of Sweden, ultra detailed lake with an ren dear, nostalgic vintage, ultra cozy and inviting, wonderful light atmosphere, fairy, little photorealistic, digital painting, sharp focus, ultra cozy and inviting, wish to be there. very detailed, arty, should rank high on youtube for a dream trip."
negative_prompt = "poorly drawn,cartoon, 2d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry"

step_size = 0.001

prompt_tokens = pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
prompt_embeds = pipe.text_encoder(prompt_tokens.input_ids.to(device))[0]

if negative_prompt is None:
    negative_prompt = [""]

negative_prompt_tokens = pipe.tokenizer(
    negative_prompt,
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
negative_prompt_embeds = pipe.text_encoder(negative_prompt_tokens.input_ids.to(device))[0]
```

### Step 2: Generate Initial Latents and Interpolate Embeddings

Create a random latent vector and build a series of interpolated embeddings.

```python
latents = torch.randn(
    (1, pipe.unet.config.in_channels, height // 8, width // 8),
    generator=generator,
)

walked_embeddings = []
for i in range(num_interpolation_steps):
    walked_embeddings.append(
        [prompt_embeds + step_size * i, negative_prompt_embeds + step_size * i]
    )
```

### Step 3: Generate and Display Images

Generate an image for each interpolated embedding pair and compile them into a GIF.

```python
images = []
for latent in tqdm(walked_embeddings):
    images.append(
        pipe(
            height=height,
            width=width,
            num_images_per_prompt=1,
            prompt_embeds=latent[0],
            negative_prompt_embeds=latent[1],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
        ).images
    )

display_images(images, save_path)
```

---

## Example 2: Diffusion Latents Interpolation (Single Prompt)

Here, we interpolate between two random points in the diffusion model's latent space using spherical linear interpolation (SLERP).

### Step 1: Define the SLERP Function

```python
def slerp(v0, v1, num, t0=0, t1=1):
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    def interpolation(t, v0, v1, DOT_THRESHOLD=0.9995):
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1
        return v2

    t = np.linspace(t0, t1, num)
    v3 = torch.tensor(np.array([interpolation(t[i], v0, v1) for i in range(num)]))
    return v3
```

### Step 2: Prepare Prompts and Latents

```python
prompt = "Sci-fi digital painting of an alien landscape with otherworldly plants, strange creatures, and distant planets."
negative_prompt = "poorly drawn,cartoon, 3d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry"

latents = torch.randn(
    (2, pipe.unet.config.in_channels, height // 8, width // 8),
    generator=generator,
)
```

### Step 3: Interpolate and Generate

```python
interpolated_latents = slerp(latents[0], latents[1], num_interpolation_steps)

images = []
for latent_vector in tqdm(interpolated_latents):
    images.append(
        pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latent_vector[None, ...],
        ).images
    )

display_images(images, save_path)
```

---

## Example 3: Interpolation Between Multiple Prompts

Create smooth transitions across a sequence of prompts by interpolating between consecutive pairs.

### Step 1: Tokenize Multiple Prompts

```python
prompts = [
    "A cute dog in a beautiful field of lavander colorful flowers everywhere, perfect lighting, leica summicron 35mm f2.0, kodak portra 400, film grain",
    "A cute cat in a beautiful field of lavander colorful flowers everywhere, perfect lighting"
]

prompt_embeddings = []
for p in prompts:
    tokens = pipe.tokenizer(
        p,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    emb = pipe.text_encoder(tokens.input_ids.to(device))[0]
    prompt_embeddings.append(emb)
```

### Step 2: Interpolate Between Consecutive Embeddings

Use the `slerp` function defined earlier to interpolate between each pair of embeddings.

```python
all_interpolated = []
for i in range(len(prompt_embeddings) - 1):
    interpolated = slerp(prompt_embeddings[i], prompt_embeddings[i+1], num_interpolation_steps)
    all_interpolated.extend(interpolated)
```

### Step 3: Generate the Image Sequence

```python
images = []
for emb in tqdm(all_interpolated):
    images.append(
        pipe(
            height=height,
            width=width,
            num_images_per_prompt=1,
            prompt_embeds=emb,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images
    )

display_images(images, save_path)
```

## Summary

You've learned three techniques for image interpolation with Stable Diffusion:

1. **Prompt Interpolation:** Linearly interpolate between positive and negative prompt embeddings.
2. **Latent Space Interpolation:** Use SLERP to smoothly transition between two random points in the diffusion model's latent space.
3. **Multi-Prompt Interpolation:** Create transitions across a sequence of prompts by interpolating between consecutive embeddings.

Experiment with different prompts, step counts, and interpolation methods to achieve your desired visual effects.
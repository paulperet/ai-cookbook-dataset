## Images Interpolation with Stable Diffusion

_Authored by: [Rustam Akimov](https://github.com/AkiRusProd)_

This notebook shows how to use Stable Diffusion to interpolate between images.  Image interpolation using Stable Diffusion is the process of creating intermediate images that smoothly transition from one given image to another, using a generative model based on diffusion.         

Here are some various use cases for image interpolation with Stable Diffusion:
- Data Augmentation: Stable Diffusion can augment training data for machine learning models by generating synthetic images that lie between existing data points. This can improve the generalization and robustness of machine learning models, especially in tasks like image generation, classification or object detection.   
- Product Design and Prototyping: Stable Diffusion can aid in product design by generating variations of product designs or prototypes with subtle differences. This can be useful for exploring design alternatives, conducting user studies, or visualizing design iterations before committing to physical prototypes.       
- Content Generation for Media Production: In media production, such as film and video editing, Stable Diffusion can be used to generate intermediate frames between key frames, enabling smoother transitions and enhancing visual storytelling. This can save time and resources compared to manual frame-by-frame editing.

In the context of image interpolation, Stable Diffusion models are often used to navigate through a high-dimensional latent space. Each dimension represents a specific feature that has been learned by the model. By walking through this latent space and interpolating between different latent representations of images, the model is able to generate a sequence of intermediate images which show a smooth transition between the original images. There are two types of latents in stable diffusion: prompt latents and image latents.      

Latent space walking involves moving through a latent space along a path defined by two or more points (representing images). By carefully selecting these points and the path between them, it is possible to control the features of the generated images, such as style, content, and other visual aspects.      

In this Notebook, we will explore examples of image interpolation using Stable Diffusion and demonstrate how latent space walking can be implemented and utilized to create smooth transitions between images. We'll provide code snippets and visualizations that illustrate this process in action, allowing for a deeper understanding of how generative models can manipulate and morph image representations in meaningful ways.


First, let's install all the required modules.


```python
!pip install -q diffusers transformers xformers accelerate
!pip install -q numpy scipy ftfy Pillow
```

Import modules


```python
import torch
import numpy as np
import os

import time

from PIL import Image
from IPython import display as IPdisplay
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from transformers import logging

logging.set_verbosity_error()
```

Let's check if CUDA is available.





```python
print(torch.cuda.is_available())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

These settings are used to optimize the performance of PyTorch models on CUDA-enabled GPUs, especially when using mixed precision training or inference, which can be beneficial in terms of speed and memory usage.       
Source: https://huggingface.co/docs/diffusers/optimization/fp16#memory-efficient-attention


```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

### Model

The [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) model and the [`LMSDiscreteScheduler`](https://huggingface.co/docs/diffusers/en/api/schedulers/lms_discrete) scheduler were chosen to generate images. Despite being an older technology, it continues to enjoy popularity due to its fast performance, minimal memory requirements, and the availability of numerous community fine-tuned models built on top of SD1.5. However, you are free to experiment with other models and schedulers to compare the results.


```python
model_name_or_path = "runwayml/stable-diffusion-v1-5"

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)


pipe = StableDiffusionPipeline.from_pretrained(
    model_name_or_path,
    scheduler=scheduler,
    torch_dtype=torch.float32,
).to(device)

# Disable image generation progress bar, we'll display our own
pipe.set_progress_bar_config(disable=True)
```

These methods are designed to reduce the memory consumed by the GPU. If you have enough VRAM, you can skip this cell.   

More detailed information can be found here: https://huggingface.co/docs/diffusers/en/optimization/opt_overview     
In particular, information about the following methods can be found here: https://huggingface.co/docs/diffusers/optimization/memory



```python
# Offloading the weights to the CPU and only loading them on the GPU can reduce memory consumption to less than 3GB.
pipe.enable_model_cpu_offload()

# Tighter ordering of memory tensors.
pipe.unet.to(memory_format=torch.channels_last)

# Decoding large batches of images with limited VRAM or batches with 32 images or more by decoding the batches of latents one image at a time.
pipe.enable_vae_slicing()

# Splitting the image into overlapping tiles, decoding the tiles, and then blending the outputs together to compose the final image. 
pipe.enable_vae_tiling()

# Using Flash Attention; If you have PyTorch >= 2.0 installed, you should not expect a speed-up for inference when enabling xformers.
pipe.enable_xformers_memory_efficient_attention()

```

The `display_images` function converts a list of image arrays into a GIF, saves it to a specified path and returns the GIF object for display. It names the GIF file using the current time and handles any errors by printing them out.


```python
def display_images(images, save_path):
    try:
        # Convert each image in the 'images' list from an array to an Image object.
        images = [
            Image.fromarray(np.array(image[0], dtype=np.uint8)) for image in images
        ]

        # Generate a file name based on the current time, replacing colons with hyphens
        # to ensure the filename is valid for file systems that don't allow colons.
        filename = (
            time.strftime("%H:%M:%S", time.localtime())
            .replace(":", "-")
        )
        # Save the first image in the list as a GIF file at the 'save_path' location.
        # The rest of the images in the list are added as subsequent frames to the GIF.
        # The GIF will play each frame for 100 milliseconds and will loop indefinitely.
        images[0].save(
            f"{save_path}/{filename}.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
    except Exception as e:
        # If there is an error during the process, print the exception message.
        print(e)

    # Return the saved GIF as an IPython display object so it can be displayed in a notebook.
    return IPdisplay.Image(f"{save_path}/{filename}.gif")
```

### Generation parameters


* `seed`: This variable is used to set a specific random seed for reproducibility.      
* `generator`: This is set to a PyTorch random number generator object if a seed is provided, otherwise it is None. It ensures that the operations using it have reproducible outcomes.     
* `guidance_scale`: This parameter controls the extent to which the model should follow the prompt in text-to-image generation tasks, with higher values leading to stronger adherence to the prompt.       
* `num_inference_steps`: This specifies the number of steps the model takes to generate an image. More steps can lead to a higher quality image but take longer to generate.        
* `num_interpolation_steps`: This determines the number of steps used when interpolating between two points in the latent space, affecting the smoothness of transitions in generated       animations.        
* `height`: The height of the generated images in pixels.       
* `width`: The width of the generated images in pixels.     
* `save_path`: The file system path where the generated gifs will be saved.       


```python
# The seed is set to "None", because we want different results each time we run the generation.
seed = None

if seed is not None:
    generator = torch.manual_seed(seed)
else:
    generator = None

# The guidance scale is set to its normal range (7 - 10).
guidance_scale = 8

# The number of inference steps was chosen empirically to generate an acceptable picture within an acceptable time.
num_inference_steps = 15

# The higher you set this value, the smoother the interpolations will be. However, the generation time will increase. This value was chosen empirically.
num_interpolation_steps = 30

# I would not recommend less than 512 on either dimension. This is because this model was trained on 512x512 image resolution.
height = 512 
width = 512

# The path where the generated GIFs will be saved
save_path = "/output"

if not os.path.exists(save_path):
    os.makedirs(save_path)

```

### Example 1: Prompt interpolation

In this example, interpolation between positive and negative prompt embeddings allows exploration of space between two conceptual points defined by prompts, potentially leading to variety of images blending characteristics dictated by prompts gradually. In this case, interpolation involves adding scaled deltas to original embeddings, creating a series of new embeddings that will be used later to generate images with smooth transitions between different states based on the original prompt.

First of all, we need to tokenize and obtain embeddings for both positive and negative text prompts. The positive prompt guides the image generation towards the desired characteristics, while the negative prompt steers it away from unwanted features.


```python
# The text prompt that describes the desired output image.
prompt = "Epic shot of Sweden, ultra detailed lake with an ren dear, nostalgic vintage, ultra cozy and inviting, wonderful light atmosphere, fairy, little photorealistic, digital painting, sharp focus, ultra cozy and inviting, wish to be there. very detailed, arty, should rank high on youtube for a dream trip."
# A negative prompt that can be used to steer the generation away from certain features; here, it is empty.
negative_prompt = "poorly drawn,cartoon, 2d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry"

# The step size for the interpolation in the latent space.
step_size = 0.001

# Tokenizing and encoding the prompt into embeddings.
prompt_tokens = pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
prompt_embeds = pipe.text_encoder(prompt_tokens.input_ids.to(device))[0]


# Tokenizing and encoding the negative prompt into embeddings.
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

Now let's look at the code part that generates a random initial vector using a normal distribution that is structured to match the dimensions expected by the diffusion model (UNet). This allows for the reproducibility of the results by optionally using a random number generator. After creating the initial vector, the code performs a series of interpolations between the two embeddings (positive and negative prompts), by incrementally adding a small step size for each iteration. The results are stored in a list named "walked_embeddings".


```python
# Generating initial latent vectors from a random normal distribution, with the option to use a generator for reproducibility.
latents = torch.randn(
    (1, pipe.unet.config.in_channels, height // 8, width // 8),
    generator=generator,
)

walked_embeddings = []

# Interpolating between embeddings for the given number of interpolation steps.
for i in range(num_interpolation_steps):
    walked_embeddings.append(
        [prompt_embeds + step_size * i, negative_prompt_embeds + step_size * i]
    )
```

Finally, let's generate a series of images based on interpolated embeddings and then displaying these images. We'll iterate over an array of embeddings, using each to generate an image with specified characteristics like height, width, and other parameters relevant to image generation. Then we'll collect these images into a list. Once generation is complete we'll call the `display_image` function to save and display these images as GIF at a given save path.


```python
# Generating images using the interpolated embeddings.
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

# Display of saved generated images.
display_images(images, save_path)
```

### Example 2: Diffusion latents interpolation for a single prompt
Unlike the first example, in this one, we are performing interpolation between the two embeddings of the diffusion model itself, not the prompts. Please note that in this case, we use the slerp function for interpolation. However, there is nothing stopping us from adding a constant value to one embedding instead.

The function presented below stands for Spherical Linear Interpolation. It is a method of interpolation on the surface of a sphere. This function is commonly used in computer graphics to animate rotations in a smooth manner and can also be used to interpolate between high-dimensional data points in machine learning, such as latent vectors used in generative models.       

The source is from Andrej Karpathy's gist: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355.        
A more detailed explanation of this method can be found at: https://en.wikipedia.org/wiki/Slerp.


```python
def slerp(v0, v1, num, t0=0, t1=1):
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    def interpolation(t, v0, v1, DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""
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


```python
# The text prompt that describes the desired output image.
prompt = "Sci-fi digital painting of an alien landscape with otherworldly plants, strange creatures, and distant planets."
# A negative prompt that can be used to steer the generation away from certain features.
negative_prompt = "poorly drawn,cartoon, 3d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry"

# Generating initial latent vectors from a random normal distribution. In this example two latent vectors are generated, which will serve as start and end points for the interpolation.
# These vectors are shaped to fit the input requirements of the diffusion model's U-Net architecture.
latents = torch.randn(
    (2, pipe.unet.config.in_channels, height // 8, width // 8),
    generator=generator,
)

# Getting our latent embeddings
interpolated_latents = slerp(latents[0], latents[1], num_interpolation_steps)

# Generating images using the interpolated embeddings.
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

# Display of saved generated images.
display_images(images, save_path)
```

### Example 3: Interpolation between multiple prompts

In contrast to the first example, where we moved away from a single prompt, in this example, we will be interpolating between any number of prompts. To do so, we will take consecutive pairs of prompts and create smooth transitions between them. Then, we will combine the interpolations of these consecutive pairs, and instruct the model to generate images based on them. For interpolation we will use the slerp function, as in the second example.

Once again, let's tokenize and obtain embeddings but this time for multiple positive and negative text prompts.


```python
# Text prompts that describes the desired output image.
prompts = [
    "A cute dog in a beautiful field of lavander colorful flowers everywhere, perfect lighting, leica summicron 35mm f2.0, kodak portra 400, film grain",
    "A cute cat in a beautiful field of lavander colorful flowers everywhere, perfect lighting
# Video Generation with Google's Veo: A Developer Guide

## Introduction

Google's Veo is a powerful video generation model that enables creators to produce high-quality videos with incredible detail, minimal artifacts, and extended durations in resolutions up to 1080p. Veo supports both text-to-video and image-to-video generation.

With Veo 3 and 3.1, you can create videos with:
- **Advanced language understanding**: Captures nuance and tone of complex prompts
- **Unprecedented creative control**: Understands cinematic terms and effects
- **Videos with audio**: Automatically generates synchronized audio
- **More accurate controls**: Better lighting, physics, and camera controls

**Important**: Veo is a paid feature and requires billing to be enabled on your Google Cloud account.

## Prerequisites

### 1. Install the SDK

First, install the Google Generative AI SDK:

```bash
pip install -U "google-genai>=1.44.0"
```

### 2. Set Up Authentication

You'll need a Google API key with access to the Gemini API. Store your API key securely:

```python
from google.colab import userdata
from google import genai
from google.genai import types

GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)
```

### 3. Select Your Veo Model

Choose from available Veo models:

```python
VEO_MODEL_ID = "veo-3.1-fast-generate-preview"  # Fast version for rapid generation
# Other options:
# - "veo-3.0-fast-generate-001"
# - "veo-3.0-generate-001"
# - "veo-3.1-generate-preview"
```

## Text-to-Video Generation

### Prompting Tips for Veo

To get the best results from Veo, incorporate specific video terminology into your prompts:

- **Shot composition**: "single shot", "two shot", "over-the-shoulder shot"
- **Camera positioning**: "eye level", "high angle", "dolly shot", "zoom shot", "pan shot"
- **Focus effects**: "shallow focus", "deep focus", "macro lens", "wide-angle lens"
- **Style and subject**: "sci-fi", "romantic comedy", "cityscape", "nature", "animals"

### Optional Parameters

While the prompt is mandatory, you can customize your video with these optional parameters:

- `negative_prompt`: What you don't want to see in the video
- `person_generation`: Control whether adults can appear in videos (children are always blocked)
- `duration_seconds`: 4, 6, or 8 seconds with Veo 3.1 (always 8s for Veo 3)
- `aspect_ratio`: Either `"16:9"` (landscape) or `"9:16"` (portrait)
- `resolution`: Either `"720p"` or `"1080p"`

### Basic Text-to-Video Generation

Let's start with a simple example: generating a video from a text prompt.

```python
import time

# Set this to True to acknowledge you understand Veo is a paid feature
I_am_aware_that_veo_is_a_paid_feature = True

if I_am_aware_that_veo_is_a_paid_feature:
    prompt = "a close-up shot of a golden retriever playing in a field of sunflowers"
    
    # Optional parameters
    negative_prompt = "barking, woofing"
    aspect_ratio = "16:9"
    resolution = "1080p"
    
    # Generate the video
    operation = client.models.generate_videos(
        model=VEO_MODEL_ID,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            negative_prompt=negative_prompt,
        ),
    )
    
    # Wait for generation to complete (typically ~1 minute)
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)
        print(f"Operation status: {operation}")
    
    # Process and save the generated videos
    print(f"Generated videos: {operation.result.generated_videos}")
    
    for n, generated_video in enumerate(operation.result.generated_videos):
        # Download the video file
        client.files.download(file=generated_video.video)
        # Save to local file
        generated_video.video.save(f'video{n}.mp4')
        print(f"Video saved as 'video{n}.mp4'")
else:
    print("Veo is a paid feature. Please set 'I_am_aware_that_veo_is_a_paid_feature' to True if you are okay with paying to run it.")
```

### Portrait Video Generation

You can create vertical videos by changing the aspect ratio:

```python
if I_am_aware_that_veo_is_a_paid_feature:
    prompt = "a unicorn takes off from the top of the arc de triomphe and fly to the Eiffel tower."
    
    # Set portrait aspect ratio
    aspect_ratio = "9:16"
    resolution = "720p"
    negative_prompt = "airplanes"
    
    operation = client.models.generate_videos(
        model=VEO_MODEL_ID,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            negative_prompt=negative_prompt,
        ),
    )
    
    # Wait for completion
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)
    
    # Save the video
    for n, generated_video in enumerate(operation.result.generated_videos):
        client.files.download(file=generated_video.video)
        generated_video.video.save(f'portrait_video{n}.mp4')
```

### Advanced Lighting Control

Veo 3 provides excellent control over lighting in your generated videos:

```python
if I_am_aware_that_veo_is_a_paid_feature:
    prompt = """a solitary, ancient oak tree silhouetted against a dramatic sunset. 
    Emphasize the exquisite control over lighting: capture the deep, warm hues of the 
    setting sun backlighting the tree, with subtle rays of light piercing through the 
    branches, highlighting the texture of the bark and leaves with a golden glow. 
    The sky should transition from fiery orange at the horizon to soft purples and 
    blues overhead, with a single, faint star appearing as dusk deepens. 
    Include the gentle sound of a breeze rustling through the leaves, and the 
    distant call of an owl."""
    
    operation = client.models.generate_videos(
        model=VEO_MODEL_ID,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            resolution="1080p",
        ),
    )
    
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)
    
    for n, generated_video in enumerate(operation.result.generated_videos):
        client.files.download(file=generated_video.video)
        generated_video.video.save(f'lighting_demo{n}.mp4')
```

### Camera Control and Movement

Control camera shots and behavior with detailed prompts:

```python
if I_am_aware_that_veo_is_a_paid_feature:
    prompt = """a realistic video of a futuristic red sportscar speeding down a 
    winding coastal highway at dusk. Begin with a high-angle drone shot that slowly 
    descends, transitioning into a close-up, low-angle tracking shot that perfectly 
    follows the car as it rounds a curve, emphasizing its speed and the gleam of its 
    paint under the fading light. Then, execute a smooth, rapid dolly zoom, making 
    the background compress as the car remains the same size, conveying a sense of 
    intense focus and speed. Finally, end with a perfectly stable, slow-motion shot 
    from a fixed roadside perspective as the car blurs past, its taillights streaking 
    across the frame. Include the immersive sound of the engine roaring, the tires 
    gripping the asphalt, and the distant crash of waves."""
    
    operation = client.models.generate_videos(
        model=VEO_MODEL_ID,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            resolution="1080p",
        ),
    )
    
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)
    
    for n, generated_video in enumerate(operation.result.generated_videos):
        client.files.download(file=generated_video.video)
        generated_video.video.save(f'camera_control{n}.mp4')
```

### Audio Control

Veo 3 automatically generates synchronized audio, but you can guide it with your prompt:

```python
if I_am_aware_that_veo_is_a_paid_feature:
    prompt = """fireworks at a beautiful city skyline scene with many different 
    fireworks colors and sounds. sounds from excited people enjoying the show 
    surrounding the camera POV can be heard too."""
    
    operation = client.models.generate_videos(
        model=VEO_MODEL_ID,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            resolution="1080p",
        ),
    )
    
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)
    
    for n, generated_video in enumerate(operation.result.generated_videos):
        client.files.download(file=generated_video.video)
        generated_video.video.save(f'audio_demo{n}.mp4')
```

## Best Practices and Tips

1. **Be Specific**: The more detailed your prompt, the better the results
2. **Use Video Terminology**: Incorporate camera shots, angles, and movement terms
3. **Consider Audio**: Describe the sounds you want to accompany your video
4. **Use Negative Prompts**: Exclude unwanted elements from your generated videos
5. **Experiment with Aspect Ratios**: Choose between landscape (16:9) and portrait (9:16) based on your use case

## Next Steps

Now that you've mastered basic text-to-video generation with Veo, you can explore:

- **Image-to-video generation**: Use reference images to guide video creation
- **Video extension**: Add 7 more seconds to existing videos (up to 141 seconds total)
- **Starting and ending frames**: Control how your video begins and concludes
- **Batch generation**: Create multiple videos with different parameters

Remember to check the [official Veo documentation](https://ai.google.dev/gemini-api/docs/video) for the latest features and best practices.
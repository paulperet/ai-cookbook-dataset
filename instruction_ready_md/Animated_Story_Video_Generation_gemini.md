# Animated Story Video Generation with Gemini API

## Overview
This guide demonstrates how to create an animated story video by combining multiple Google AI services:
1. **Story Generation**: Create structured story sequences using Gemini API with JSON output
2. **Image Generation**: Generate scene visuals using Imagen API
3. **Audio Generation**: Create narration using Gemini Live API
4. **Video Assembly**: Combine images and audio into a final video using MoviePy

## Prerequisites

### Install Required Packages
```bash
!apt-get update -qq && apt-get install -qq locales
!locale-gen en_US.UTF-8
!update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
!apt-get -qq -y install espeak-ng > /dev/null 2>&1
%pip install -q google-generativeai moviepy Pillow nest_asyncio
```

### Import Libraries
```python
import os
import json
import numpy as np
import time
import asyncio
import contextlib
import wave
from io import BytesIO
from base64 import b64encode

# Image handling
from PIL import Image
from IPython.display import display, HTML

# Video processing
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips

# Google AI
from google import genai
import typing_extensions as typing

# Async support
import nest_asyncio
nest_asyncio.apply()
```

### Configure API Access
```python
from google.colab import userdata
import os

# Set your API key (stored as a Colab Secret named GOOGLE_API_KEY)
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')

# Initialize clients
client = genai.Client(http_options={'api_version': 'v1alpha'})  # v1alpha for Live API
MODEL = "models/gemini-2.5-flash-lite"
IMAGE_MODEL_ID = "imagen-3.0-generate-002"
```

## Step 1: Generate Structured Story Sequence

### Define Story Structure
We'll use Gemini's structured output capability to ensure consistent formatting across all scenes.

```python
# Define the structure for each story segment
class StorySegment(typing.TypedDict):
    image_prompt: str
    audio_text: str
    character_description: str

# Define the overall story response structure
class StoryResponse(typing.TypedDict):
    complete_story: list[StorySegment]
    pages: int

def generate_story_sequence(complete_story: str, pages: int) -> list[StorySegment]:
    """
    Generate a structured story sequence using Gemini API.
    
    Args:
        complete_story: The theme or topic of the story
        pages: Number of scenes to generate
        
    Returns:
        List of story segments with image prompts, audio text, and character descriptions
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=f'''You are an animation video producer. Generate a story sequence about {complete_story} in {pages} scenes (with interactions and characters), 1 sec each scene. Write:

image_prompt: (define art style for kids animation, consistent for all characters, no violence) a full description of the scene, the characters in it, and the background in 20 words or less. Progressively shift the scene as the story advances.
audio_text: a one-sentence dialogue/narration for the scene.
character_description: no people ever, only animals and objects. Describe all characters (consistent names, features, clothing, etc.) with an art style reference (e.g., "Pixar style," "photorealistic," "Ghibli") in 30 words or less.
''',
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[StoryResponse]
        }
    )

    try:
        story_data_text = response.text
        story_data_list = json.loads(story_data_text)
        if isinstance(story_data_list, list) and len(story_data_list) > 0:
            story_data = story_data_list[0]
            return story_data.get('complete_story', [])
        else:
            return []
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing JSON: {e}")
        return []
```

### Generate Your Story
```python
# Customize your story theme and number of scenes
theme = "a cat and a dog playing"
num_scenes = 3

# Generate story segments
story_segments = generate_story_sequence(theme, num_scenes)
print("\nGenerated Story Segments:")
print(json.dumps(story_segments, indent=2))
```

**Example Output:**
```json
[
  {
    "audio_text": "Whiskers the cat pounces playfully, startling Buster the dog in the sunny garden.",
    "character_description": "Whiskers: A playful ginger cat with big green eyes. Art style: Simple, rounded shapes, bright colors, kid-friendly animation style. Buster: A golden retriever puppy, floppy ears, big paws. Art style: Same as Whiskers.",
    "image_prompt": "Kids animation style: Whiskers jumps at Buster in a flower filled garden with a blue fence."
  },
  {
    "audio_text": "Buster wags his tail and chases Whiskers around a colorful mushroom house.",
    "character_description": "Whiskers: A playful ginger cat with big green eyes. Art style: Simple, rounded shapes, bright colors, kid-friendly animation style. Buster: A golden retriever puppy, floppy ears, big paws. Art style: Same as Whiskers.",
    "image_prompt": "Kids animation style: Buster chases Whiskers around a red mushroom house with white spots."
  },
  {
    "audio_text": "Together, Whiskers and Buster slide down a rainbow into a pool filled with toys.",
    "character_description": "Whiskers: A playful ginger cat with big green eyes. Art style: Simple, rounded shapes, bright colors, kid-friendly animation style. Buster: A golden retriever puppy, floppy ears, big paws. Art style: Same as Whiskers.",
    "image_prompt": "Kids animation style: Whiskers and Buster slide down a rainbow into a toy-filled pool."
  }
]
```

## Step 2: Helper Functions for Audio Generation

### WAV File Helper
```python
@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    """
    Context manager for creating WAV files with proper configuration.
    
    Args:
        filename: Output WAV file path
        channels: Number of audio channels (1 for mono)
        rate: Sample rate in Hz
        sample_width: Bytes per sample (2 for 16-bit)
    """
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
```

### Audio Generation Function
```python
def generate_audio_live(api_text, output_filename):
    """
    Generate audio narration using Gemini Live API.
    
    Args:
        api_text: Text to convert to speech
        output_filename: Path to save the WAV file
        
    Returns:
        Path to the generated audio file
    """
    collected_audio = bytearray()
    
    async def _generate():
        config = {"response_modalities": ["AUDIO"]}
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            await session.send(input=api_text, end_of_turn=True)
            async for response in session.receive():
                if response.data:
                    collected_audio.extend(response.data)
        return bytes(collected_audio)
    
    # Run async function and save audio
    audio_bytes = asyncio.run(_generate())
    with wave_file(output_filename) as wf:
        wf.writeframes(audio_bytes)
    return output_filename
```

## Step 3: Process Each Scene

Now we'll iterate through each story segment to generate images and audio, then create video clips.

```python
# Initialize tracking lists
temp_audio_files = []
temp_image_files = []
video_clips = []

# System instruction to ensure natural narration
audio_negative_prompt = "Don't say OK, I will do this or that. Just read this story using voice expressions without introductions or endings. More segments are coming. Don't say OK, I will do this or that:\n"

# Process each scene
for i, segment in enumerate(story_segments):
    print(f"\nProcessing scene {i + 1}/{len(story_segments)}:")
    
    # Extract scene details
    image_prompt = segment['image_prompt']
    audio_text = audio_negative_prompt + segment['audio_text']
    char_desc = segment['character_description']
    
    print(f"Image Prompt: {image_prompt}")
    print(f"Audio Text: {segment['audio_text']}")
    print(f"Character Description: {char_desc}")
    print("-" * 40)
    
    # --- Image Generation ---
    combined_prompt = f"detailed children book animation style {image_prompt} {char_desc}"
    
    try:
        result = client.models.generate_images(
            model=IMAGE_MODEL_ID,
            prompt=combined_prompt,
            config={
                "number_of_images": 1,
                "output_mime_type": "image/jpeg",
                "person_generation": "DONT_ALLOW",
                "aspect_ratio": "1:1"
            }
        )
        
        if not result.generated_images:
            raise ValueError("No images generated. The prompt might have been flagged as harmful.")
        
        # Save the generated image
        generated_image = result.generated_images[0]
        image = Image.open(BytesIO(generated_image.image.image_bytes))
        image_path = f"image_{i}.png"
        image.save(image_path)
        temp_image_files.append(image_path)
        
        # Display the image
        display(image)
        
    except Exception as e:
        print(f"Image generation failed: {e}")
        continue
    
    # --- Audio Generation ---
    audio_path = f"audio_{i}.wav"
    try:
        audio_path = generate_audio_live(audio_text, audio_path)
        temp_audio_files.append(audio_path)
    except Exception as e:
        print(f"Audio generation failed: {e}")
        continue
    
    # --- Create Video Clip ---
    try:
        # Load audio
        audio_clip = AudioFileClip(audio_path)
        
        # Convert image to numpy array for MoviePy
        np_image = np.array(image)
        
        # Create image clip with same duration as audio
        image_clip = ImageClip(np_image).set_duration(audio_clip.duration)
        
        # Combine image and audio
        composite_clip = CompositeVideoClip([image_clip]).set_audio(audio_clip)
        video_clips.append(composite_clip)
        
        print(f"✓ Scene {i + 1} processed successfully")
        
    except Exception as e:
        print(f"Video clip creation failed for scene {i + 1}: {e}")
```

## Step 4: Assemble Final Video

Now we'll combine all individual scene clips into a final video.

```python
# Combine all video clips
if video_clips:
    final_video = concatenate_videoclips(video_clips)
    
    # Generate output filename with timestamp
    output_filename = f"{int(time.time())}_output_video.mp4"
    print(f"\nWriting final video to: {output_filename}")
    
    # Export video
    final_video.write_videofile(output_filename, fps=24)
    
    # Display video in notebook
    def show_video(video_path):
        """Display video in notebook"""
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
        video_b64 = b64encode(video_bytes).decode()
        video_tag = f'<video width="640" height="480" controls><source src="data:video/mp4;base64,{video_b64}" type="video/mp4"></video>'
        return HTML(video_tag)
    
    display(show_video(output_filename))
    print("✓ Video generation complete!")
else:
    print("No video clips were created. Check the processing steps above.")
```

## Step 5: Cleanup Temporary Files

After generating your video, clean up the temporary files to free up space.

```python
# Close video clips to release resources
if 'final_video' in locals():
    final_video.close()

for clip in video_clips:
    clip.close()

# Remove temporary files
for file in temp_audio_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed: {file}")

for file in temp_image_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed: {file}")

print("Cleanup complete!")
```

## Best Practices and Tips

### 1. **Story Generation**
- Use specific themes for better consistency
- Limit scenes to 5-10 for reasonable processing time
- Experiment with different art style instructions in the character description

### 2. **Image Generation**
- Combine image prompt with character description for better results
- Use aspect ratio "1:1" for square videos
- Add "detailed children book animation style" prefix for consistent visual style

### 3. **Audio Generation**
- Use the negative prompt to prevent AI narration artifacts
- Ensure audio text is complete sentences for natural flow
- Test with short segments first to verify quality

### 4. **Performance Optimization**
- Process scenes sequentially to avoid API rate limits
- Use temporary files for intermediate storage
- Clean up files after video generation

## Troubleshooting

### Common Issues:

1. **No images generated**: Check if your prompt contains restricted content. Try simplifying the description.
2. **Audio generation fails**: Ensure you're using the v1alpha API version for Live API.
3. **Video assembly errors**: Verify all image and audio files exist and are readable.
4. **Memory issues**: Reduce the number of scenes or image resolution.

### API References:
- [Structured Outputs Documentation](https://ai.google.dev/gemini-api/docs/structured-outputs)
- [Imagen Pricing](https://ai.google.dev/pricing#2_0flash)
- [Imagen Prompt Guide](https://ai.google.dev/gemini-api/docs/imagen-prompt-guide)

## Next Steps

1. **Experiment with different themes**: Try fantasy, sci-fi, or educational stories
2. **Customize art styles**: Modify the character description to change visual style
3. **Add background music**: Use MoviePy to add background tracks to your video
4. **Create longer videos**: Chain multiple story sequences together
5. **Export in different formats**: Modify MoviePy parameters for different video formats

Your animated story video is now complete! The final MP4 file contains all your generated scenes with synchronized narration.
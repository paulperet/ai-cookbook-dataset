# Guide: Video Generation with Google's Veo API

## Overview
This guide demonstrates how to generate high-quality videos using Google's Veo models via the REST API. You will learn to create videos from text prompts and from existing images.

## Prerequisites

### 1. API Access
Veo is a paid feature. Ensure you have:
- A Google AI Studio account with billing enabled
- A valid Gemini API key

### 2. Environment Setup
Set up your API key and base URL:

```python
import os

# Set your Gemini API key
os.environ['GEMINI_API_KEY'] = 'YOUR_API_KEY_HERE'
os.environ['BASE_URL'] = "https://generativelanguage.googleapis.com/v1beta"

# Choose your Veo model
os.environ['VEO_MODEL_ID'] = "veo-3.0-fast-generate-001"
```

Available models:
- `veo-2.0-generate-001`
- `veo-3.0-fast-generate-001` (faster, ideal for rapid content generation)
- `veo-3.0-generate-001` (highest quality)

### 3. Install Required Tools
Install `jq` for JSON processing:

```bash
sudo apt update && sudo apt install -y jq
```

## Understanding Veo Capabilities

Veo enables high-quality video generation with:
- **Advanced language understanding**: Captures nuance and complex prompts
- **Creative control**: Understands cinematic terms like "timelapse" or "aerial shots"
- **Audio generation**: Automatically includes audio in videos
- **Video controls**: Accurate lighting, physics, and camera controls

### Prompting Tips
Incorporate specific video terminology for best results:

- **Shot composition**: "single shot", "two shot", "over-the-shoulder shot"
- **Camera movement**: "dolly shot", "zoom shot", "pan shot", "tracking shot"
- **Focus effects**: "shallow focus", "deep focus", "macro lens"
- **Style**: "sci-fi", "romantic comedy", "animation"

## Part 1: Text-to-Video Generation

### Step 1: Start Video Generation
Video generation uses the `predictLongRunning` method. Create a JSON request with your prompt and parameters:

```bash
# Create the generation request
curl "${BASE_URL}/models/${VEO_MODEL_ID}:predictLongRunning?key=${GEMINI_API_KEY}" \
  -H "Content-Type: application/json" \
  -X "POST" \
  -d '{
    "instances": [{
        "prompt": "A captivating aerial drone shot of the Notre Dame during fireworks showcasing the majestic cathedral from a unique perspective. The camera slowly pans around the structure, revealing its intricate details and grandeur."
      }
    ],
    "parameters": {
      "aspectRatio": "16:9",
      "resolution": "1080p",
      "negativePrompt": "cars",
      "sampleCount": 1
    }
  }' | tee result.json | jq .name | sed 's/"//g' > op_name
```

**Parameters Explained:**
- `prompt`: Your video description (required)
- `negativePrompt`: What to exclude from the video
- `aspectRatio`: "16:9" (landscape) or "9:16" (portrait)
- `resolution`: "720p" or "1080p" (1080p only available in 16:9)
- `sampleCount`: Always 1 for Veo 3

### Step 2: Check Generation Status
The response contains an operation name. Use it to check status:

```bash
# Read the operation name
op_name=$(cat op_name)

# Check the status
curl "${BASE_URL}/${op_name}?key=${GEMINI_API_KEY}" \
  -H "Content-Type: application/json" \
  -X "GET"
```

### Step 3: Wait for Completion
Video generation takes time. Poll the API until completion:

```bash
STATUS="null"

while [[ $STATUS == "null" ]]
do
  sleep 5
  
  # Check status
  curl "${BASE_URL}/${op_name}?key=${GEMINI_API_KEY}" \
    -H "Content-Type: application/json" \
    -X "GET" > temp_response.json
  
  # Extract response
  jq '.response' temp_response.json > response.json
  STATUS=$(jq -r '.' response.json)
  
  echo "Current status: $STATUS"
done
```

### Step 4: Extract and Download Video
Once complete, extract the video URI and download:

```bash
# Extract video URI
jq '.generateVideoResponse.generatedSamples[].video.uri' -r response.json | tee uris.txt

# Download the video
idx=0
while read uri; do
  curl "${uri}&key=${GEMINI_API_KEY}" \
    --location \
    -H "Content-Type: video/mp4" \
    -0 > video_${idx}.mp4
  idx=$((idx+1))
done < uris.txt
```

### Step 5: View the Video (Optional)
In a Python environment, you can display the video:

```python
import pathlib
from IPython.display import display, Video

# Show downloaded videos
for fpath in pathlib.Path('.').glob("video_*.mp4"):
  display(Video(str(fpath), embed=True))
```

## Part 2: Image-to-Video Generation

### Optional: Generate Base Image with Imagen
You can create a starting image using Imagen 4:

```bash
# Create Imagen request
cat <<EOF > imagen_request.json
{
  "instances": [
    {
      "prompt": "a kitten and a puppy playing in a puddle"
    }
  ],
  "parameters": {
    "sampleCount": 1,
    "safetySetting": "block_low_and_above",
    "personGeneration": "allow_adult",
    "aspectRatio": "16:9"
  }
}
EOF

# Generate image
curl "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict?key=${GEMINI_API_KEY}" \
  -H "Content-Type: application/json" \
  -X "POST" \
  -d @imagen_request.json \
  | tee imagen_result.json | jq -r '.predictions[0].bytesBase64Encoded' > base64_image.jpeg

# Clean up
rm imagen_request.json
```

**Note:** Image-to-video has limitations, including no person generation.

### Convert Image to Video
Use the base64-encoded image as input for Veo:

```bash
# Read the base64 image
IMAGE_BASE64=$(cat base64_image.jpeg)

# Generate video from image
curl "${BASE_URL}/models/${VEO_MODEL_ID}:predictLongRunning?key=${GEMINI_API_KEY}" \
  -H "Content-Type: application/json" \
  -X "POST" \
  -d '{
    "instances": [{
        "image": {
          "bytesBase64Encoded": "'"${IMAGE_BASE64}"'"
        },
        "prompt": "The animals continue playing as raindrops fall"
      }
    ],
    "parameters": {
      "aspectRatio": "16:9",
      "sampleCount": 1
    }
  }' | tee image_to_video_result.json | jq .name | sed 's/"//g' > image_op_name
```

Follow the same status checking and download process as in Part 1.

## Best Practices

1. **Prompt Specificity**: The more detailed your prompt, the better the results
2. **Use Negative Prompts**: Exclude unwanted elements for cleaner output
3. **Consider Aspect Ratio**: Choose based on your display platform
4. **Monitor Costs**: Video generation consumes more resources than image generation
5. **Error Handling**: Implement proper error checking in production code

## Troubleshooting

- **API Key Issues**: Verify your key has Veo permissions enabled
- **Long Wait Times**: Video generation can take several minutes
- **Format Errors**: Ensure JSON is properly formatted
- **Download Failures**: Check network connectivity and API quotas

## Next Steps
- Experiment with different cinematic terms in prompts
- Chain multiple video generations for longer sequences
- Integrate with your application backend for automated content creation
- Explore the [Veo prompt guide](https://ai.google.dev/gemini-api/docs/video/veo-prompt-guide) for advanced techniques

This guide provides the foundation for integrating Veo's powerful video generation capabilities into your projects. Remember to review Google's pricing and usage policies before scaling your implementation.
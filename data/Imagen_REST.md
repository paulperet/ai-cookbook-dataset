# Image Generation with the Gemini API REST Endpoint

This guide walks you through using Google's `imagen-3.0-generate-002` model via its REST API to generate high-quality images from text prompts. The model excels at producing detailed images with natural language understanding, effective text rendering, and minimal artifacts.

> **Note:** Image generation is a paid feature. Ensure your account is not on the free tier. Review the [pricing details](https://ai.google.dev/pricing#imagen3) before proceeding.

## Prerequisites

You will need a Google AI Studio API key. This guide assumes you are running in a Google Colab environment.

## 1. Environment Setup

First, set your API key as an environment variable. In Colab, you can store it in a Secret named `GOOGLE_API_KEY`.

```python
import os
from google.colab import userdata

os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

You will also need the `jq` tool to parse JSON responses from the API. Install it with the following command:

```bash
!apt install -q jq
```

## 2. Configure Your Image Generation Request

Define the parameters for your image generation. The model performs best with detailed, descriptive prompts.

Create a new file to store these variables so they can be accessed in subsequent steps.

```bash
%%bash

# Define your parameters here
SAMPLE_COUNT=3
ASPECT_RATIO="1:1"
PERSON_GENERATION="allow_adult"
PROMPT="a photorealistic single calico cat lounging in a soft cat bed by a window with the sun shining on them. Include shadows so only the cat is touched by the sun."

# Save variables to a file for later use
cat >./vars.sh <<-EOF
  export SAMPLE_COUNT=${SAMPLE_COUNT}
  export ASPECT_RATIO="${ASPECT_RATIO}"
  export PERSON_GENERATION="${PERSON_GENERATION}"
  export PROMPT="${PROMPT}"
EOF

echo "Parameters saved."
```

**Parameter Details:**
*   `sampleCount`: Number of images to generate (1-4, default is 4).
*   `aspectRatio`: Image dimensions. Supported: `1:1`, `3:4`, `4:3`, `16:9`, `9:16`.
*   `personGeneration`: Controls generation of adult figures. Options: `"dont_allow"` or `"allow_adult"`. Generation of children is always blocked.
*   `prompt`: Your descriptive text prompt.

> **Tip:** For guidance on crafting effective prompts, consult the [official prompt guide](https://ai.google.dev/gemini-api/docs/imagen-prompt-guide).

## 3. Send the Generation Request

Now, send a `POST` request to the Imagen REST endpoint using `curl`. The response will contain your generated images encoded in base64 format.

```bash
%%bash
# Load the variables from the file you created
. vars.sh

curl "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-fast:predict?key=${GOOGLE_API_KEY}" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
    "instances": [
        {
            "prompt": "'"${PROMPT}"'"
        }
    ],
    "parameters": {
        "sampleCount": '${SAMPLE_COUNT}',
        "personGeneration": "'${PERSON_GENERATION}'",
        "aspectRatio": "'${ASPECT_RATIO}'",
    }
}' 2>/dev/null >response.json

echo "Request sent. Response saved to 'response.json'."
```

## 4. Inspect the Response

Verify the response contains the expected number of images and check their MIME types.

```bash
print("Number of predictions: ")
!jq '.[] | length' response.json
print("--")
print("Mime types: ")
!jq -C .predictions[].mimeType response.json
```

You should see output confirming the count (e.g., `3`) and the image format (e.g., `"image/png"`).

## 5. Decode and Save the Images

The images are returned as base64 strings. The following script decodes each one and saves it as a `.png` file.

```bash
%%bash
. vars.sh

# Iterate through predictions array
for i in $(seq 0 $(expr ${SAMPLE_COUNT} - 1)); do
  # Extract base64 encoded image data
  base64_data=$(jq -r ".predictions[$i].bytesBase64Encoded" response.json)

  # Decode base64 data and save as image
  echo "$base64_data" | base64 -d > "image${i}.png"
  echo "Saved image${i}.png"
done
```

## 6. Display the Generated Images

Finally, use Python to display the generated images directly in your notebook. This code scales them down for easier viewing.

```python
import os
from IPython.display import Image, display
from PIL import Image as PILImage

for i in range(4): # Adjust range if you generated fewer than 4 images
    filename = f"image{i}.png"
    if os.path.exists(filename):
        try:
            with PILImage.open(filename) as img:
                # Display image at half its original size
                width, height = img.size
                new_width = width // 2
                new_height = height // 2
                display(Image(filename, width=new_width, height=new_height))
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        # Stop the loop if the file doesn't exist (e.g., SAMPLE_COUNT was less than 4)
        break
```

## Next Steps

*   **Improve Your Prompts:** For advanced techniques, read the [Imagen Prompt Guide](https://ai.google.dev/gemini-api/docs/imagen-prompt-guide).
*   **Explore Creative Examples:** See Imagen in action by checking out the [Book Illustration example](../examples/Book_illustration.ipynb), which uses Gemini and Imagen together.
*   **Discover Gemini's Multimodal Capabilities:** Gemini can also understand images and videos. Explore the [Spatial Understanding](./Spatial_understanding.ipynb) and [Video Understanding](./Video_understanding.ipynb) guides.

---
**All generated images include a non-visible digital [SynthID](https://deepmind.google/technologies/synthid/) watermark.**
# Get started with Video generation using Veo and REST API

_If you're reading this notebook on Github, open it in Colab by clicking the above button to see the generated videos._

Veo is a paid only feature. It won't run with the Free Tier.  
(cf. pricing for more details).


```bash
%%bash
# Uncomment the next line if you want to run the colab
# touch I_am_aware_that_veo_is_a_paid_feature
```

## Why Veo

Veo enables developers to create high quality videos with incredible detail, minimal artifacts, and extended durations in resolutions up to 720p. Veo supports both text-to-video and image-to-video.

With [Veo 3](https://ai.google.dev/gemini-api/docs/video), developers can create videos with:
* **Advanced language understanding**: Veo deeply understands natural language and visual semantics, capturing the nuance and tone of complex prompts to render intricate details in extended scenes, including cinematic terms like "timelapse" or "aerial shots."
* **Unprecedented creative control**: Veo provides an unprecedented level of creative control, understanding prompts for all kinds of cinematic effects, like timelapses or aerial shots of a landscape.
* **Videos with audio**: Veo 3 generates videos with audio automatically, with no additional effort from the developer.
* **More accurate video controls**: Veo 3 is more accurate on lighting, accurate physics, and camera controls.

The Veo 3 family of models includes both [**Veo 3**](https://ai.google.dev/gemini-api/docs/video?example=dialogue#veo-3) as well as [**Veo 3 Fast**](https://ai.google.dev/gemini-api/docs/video?example=dialogue#veo-3-fast), which is a faster and more accessible version of Veo 3. Veo 3 Fast is ideal for backend services that programmatically generate ads, tools for rapid A/B testing of creative concepts, or apps that need to quickly produce social media content.

## Setup


```
from google.colab import userdata
import os

os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
os.environ['BASE_URL']="https://generativelanguage.googleapis.com/v1beta"
os.environ['VEO_MODEL_ID']="veo-3.0-fast-generate-001" # @param ['veo-2.0-generate-001', 'veo-3.0-fast-generate-001', 'veo-3.0-generate-001'] {"allow-input":true, isTemplate: true}
```

Install `jq` so you can process the JSON results.


```bash
%%bash

# Install jq for JSON processing
sudo apt install -q jq
```

    [Reading package lists..., Building dependency tree..., Reading state information..., jq is already the newest version (1.6-2.1ubuntu3.1)., 0 upgraded, 0 newly installed, 0 to remove and 35 not upgraded., WARNING: apt does not have a stable CLI interface. Use with caution in scripts.]


## Text-to-video

Veo is able to generate videos from text prompts (see in the code comments for ideas). In these examples you are using the latest Google Veo 3 to generate your videos, which includes audio in the videos.

### Prompting Tips for Veo
To get the most out of Veo, consider incorporating specific video terminology into your prompts  Veo understands a wide range of terms related to:

* **Shot composition**: Specify the framing and number of subjects in the shot (e.g., "*single shot*", "*two shot*", "*over-the-shoulder shot*").
* **Camera positioning and movement**:  Control the camera's location and movement using terms like "*eye level*", "*high angle*", "*worms eye*", "*dolly shot*", "*zoom shot*", "*pan shot," and "*tracking shot*".
* **Focus and lens effects**:  Use terms like "*shallow focus*", "*deep focus*", "*soft focus*", "*macro lens*", and "*wide-angle lens*" to achieve specific visual effects.
* **Overall style and subject**: Guide Veo's creative direction by specifying styles like "*sci-fi*", "*romantic comedy*", "*action movie*" or "*animation*". You can also describe the subjects and backgrounds you want, such as "*cityscape*", "*nature*", "*vehicles*", or "animals."

Check the [Veo prompt guide](https://ai.google.dev/gemini-api/docs/video/veo-prompt-guide) for more details and tips.

### Optional parameters for Veo 3
The prompt is the only mandatory parameters, the others are all optional.

* **negative_prompt**: What you don't want to see in the video,
* **person_generation**: Tell you model if it's allowed to generate adults in the videos or not. Children are always blocked,
* **number_of_videos**: With Veo 3, it is always a single shot generation (one video generated),
* **duration_seconds**: With Veo 3, it is always 8 seconds long videos generated,
* **aspect ratio**: Either `16:9` (landscape) or `9:16` (portrait),
* **resolution**: Either `720p` or `1080p` (only in 16:9)

### Start the generation

Video generation uses the `predictLongRunning` method:



```bash
%%bash

if [[ -e "I_am_aware_that_veo_is_a_paid_feature" ]]; then

  # Use curl to send a POST request to the predictLongRunning endpoint
  # The request body includes the prompt for video generation
  curl "${BASE_URL}/models/${VEO_MODEL_ID}:predictLongRunning?key=${GEMINI_API_KEY}" \
    -H "Content-Type: application/json" \
    -X "POST" \
    -d '{
      "instances": [{
          "prompt": "A captivating aerial drone shot of the Notre dame, during fireworks showcasing the majestic cathedral from a unique perspective. The camera slowly pans around the structure, revealing its intricate details and grandeur."
        }
      ],
      "parameters": {
        "aspectRatio": "16:9",
        "resolution": "1080p",
        "negativePrompt": "cars",
        "sampleCount": 1,
      }
    }' | tee result.json | jq .name | sed 's/"//g' > op_name

else
  echo "Veo is a paid feature. Please change the variable 'I_am_aware_that_veo_is_a_paid_feature' to True if you are okay with paying to run it."
fi
```

    [% Total % Received % Xferd Average Speed Time Time Time Current, Dload Upload Total Spent Left Speed, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 100 514 0 73 100 441 132 798 --:--:-- --:--:-- --:--:-- 931]


Have a look at the output:


```bash
%%bash
cat result.json
```

    {
      "name": "models/veo-3.0-fast-generate-001/operations/321kvdcepmcg"
    }



```bash
%%bash
cat op_name
```

    models/veo-3.0-fast-generate-001/operations/321kvdcepmcg


### Check the status

The op-name tells you where to check the status of the video generation. It initially contains nothing, or just the op name.


```bash
%%bash

# Check the status of the video generation using the operation name
curl "${BASE_URL}/${op_name}?key=${GEMINI_API_KEY}" \
  -H "Content-Type: application/json" \
  -X "GET"
```

    [% Total % Received % Xferd Average Speed Time Time Time Current, Dload Upload Total Spent Left Speed, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0]


### Wait for generation to finish

When generation finished it will contain a response field.


```bash
%%bash

# Wait for the video generation to finish
# Poll the API every 5 seconds until the response contains a 'response' field

# Retrieve the operation name from the file
op_name=$(cat op_name)

STATUS="null"

while [[ $STATUS == "null" ]]
do
  sleep 5

  # Store the entire response in a temporary file
  curl "${BASE_URL}/${op_name}?key=${GEMINI_API_KEY}" \
    -H "Content-Type: application/json" \
    -X "GET" > temp_response.json

  # Extract the response field to response.json
  jq '.response' temp_response.json > response.json

  # Read the status correctly using jq with the -r flag
  STATUS=$(jq -r '.' response.json)

  # Optional: print status for debugging
  echo "Current status: $STATUS"
done
```

    [Current status: null, Current status: null, Current status: null, Current status: null, Current status: {, "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.PredictLongRunningResponse",, "generateVideoResponse": {, "generatedSamples": [, {, "video": {, "uri": "https://generativelanguage.googleapis.com/v1beta/files/qu4d28uipuqh:download?alt=media", }, }, ], }, }, % Total % Received % Xferd Average Speed Time Time Time Current, Dload Upload Total Spent Left Speed, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 100 73 0 73 0 0 178 0 --:--:-- --:--:-- --:--:-- 178, % Total % Received % Xferd Average Speed Time Time Time Current, Dload Upload Total Spent Left Speed, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 100 73 0 73 0 0 210 0 --:--:-- --:--:-- --:--:-- 210, % Total % Received % Xferd Average Speed Time Time Time Current, Dload Upload Total Spent Left Speed, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 100 73 0 73 0 0 193 0 --:--:-- --:--:-- --:--:-- 194, % Total % Received % Xferd Average Speed Time Time Time Current, Dload Upload Total Spent Left Speed, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 100 73 0 73 0 0 211 0 --:--:-- --:--:-- --:--:-- 211, % Total % Received % Xferd Average Speed Time Time Time Current, Dload Upload Total Spent Left Speed, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 100 443 0 443 0 0 1122 0 --:--:-- --:--:-- --:--:-- 1124]


Check the reponse then extract the uri.


```bash
%%bash

# Pretty print the response json for readability (Optional)
jq '' -r response.json
```

    {
      "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.PredictLongRunningResponse",
      "generateVideoResponse": {
        "generatedSamples": [
          {
            "video": {
              "uri": "https://generativelanguage.googleapis.com/v1beta/files/qu4d28uipuqh:download?alt=media"
            }
          }
        ]
      }
    }



```bash
%%bash

# Extract the video URIs from the response and save them to uris.txt
jq '.generateVideoResponse.generatedSamples[].video.uri' -r response.json | tee uris.txt
```

    https://generativelanguage.googleapis.com/v1beta/files/qu4d28uipuqh:download?alt=media


And finally download the video(s).


```bash
%%bash

# Download the generated videos

idx=0

while read uri; do
  # Use curl to download each video
  # --location -> follow redirects
  curl "${uri}&key=${GEMINI_API_KEY}" \
    --location \
    -H "Content-Type: video/mp4" \
    -0 > video_${idx}.mp4 # Save each video to a separate file

  idx=$((idx+1))
done < uris.txt # Read URIs from the uris.txt file
```

    [% Total % Received % Xferd Average Speed Time Time Time Current, Dload Upload Total Spent Left Speed, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 0 5657k 0 772 0 0 733 0 2:11:42 0:00:01 2:11:41 733, 100 5657k 100 5657k 0 0 4144k 0 0:00:01 0:00:01 --:--:-- 17.7M]


### Show the videos (optional - python)


```bash
%%bash
ls video_*
```

    video_0.mp4



```
import pathlib
from IPython.display import display, Video

# Show the downloaded videos
for fpath in pathlib.Path('.').glob("video_*.mp4"):
  display(Video(str(fpath), embed=True))
```

## Generate a video from an image

### Optional - Generate an image using Imagen 4


You can use one of your images, but you can also use Imagen to generate your base image. It gives you more control over what the video will look like but comes with some limitation, image-to-video doesn't allowing for person generation being the main one.

Check the [Imagen](./Get_started_imagen_REST.ipynb) dedicated guide for more details on the image generation model.


```bash
%%bash
# Create a JSON file named 'imagen_request.json' that contains the
# parameters for the Imagen request.
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

if [[ -e "I_am_aware_that_veo_is_a_paid_feature" ]]; then

  # Send a request to the Imagen API to generate an image based on
  # the provided JSON request and saves the base64 encoded image to a file.
  curl "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict?key=${GEMINI_API_KEY}" \
    -H "Content-Type: application/json" \
    -X "POST" \
    -d @imagen_request.json \
    | tee imagen_result.json | jq -r '.predictions[0].bytesBase64Encoded' > base64_image.jpeg

  # Remove the request json file as it is no longer needed
  rm imagen_request.json

  echo "Image generated."

else
  echo "Imagen is a paid feature. Please change the variable 'I_am_aware_that_veo_is_a_paid_feature' to True if you are okay with paying to run it."
fi
```

    [Image generated., % Total % Received % Xferd Average Speed Time Time Time Current, Dload Upload Total Spent Left Speed, 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0, 100 237 0 0 100 237 0 191 0:00:01 0:00:01 --:--:-- 191, 100 237 0 0 100 237 0 105 0:00:02 0:00:02 --:--:-- 105, 100 237 0 0 100 237 0 73 0:00:03 0:00:03 --:--:-- 73, 100 237 0 0 100 237 0 55 0:00:04 0:00:04 --:--:-- 55, 100 237 0 0 100 237 0 45 
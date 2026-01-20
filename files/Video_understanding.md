# Video understanding with Gemini

Gemini has from the begining been a multimodal model, capable of analyzing all sorts of medias using its [long context window](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/).

[Gemini models](https://ai.google.dev/gemini-api/docs/models/) bring video analysis to a whole new level as illustrated in [this video](https://www.youtube.com/watch?v=Mot-JEU26GQ):

This notebook will show you how to easily use Gemini to perform the same kind of video analysis. Each of them has different prompts that you can select using the dropdown, also feel free to experiment with your own.

You can also check the [live demo](https://aistudio.google.com/starter-apps/video) and try it on your own videos on [AI Studio](https://aistudio.google.com/starter-apps/video).

## Setup

This section install the SDK, set it up using your [API key](../quickstarts/Authentication.ipynb), imports the relevant libs, downloads the sample videos and upload them to Gemini.

Expand the section if you are curious, but you can also just run it (it should take a couple of minutes since there are large files) and go straight to the examples.

### Install SDK


```
%pip install -U -q "google-genai>=1.16.0"
```

[First Entry, ..., Last Entry]

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
```

### Initialize SDK client

With the new SDK you now only need to initialize a client with you API key (or OAuth if using [Vertex AI](https://cloud.google.com/vertex-ai)). The model is now set in each call.


```
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Select the Gemini model

Video understanding works best with Gemini 2.5 models. You can also select former models to compare their behavior but it is recommended to use at least the 2.0 ones.

For more information about all Gemini models, check the [documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for extended information on each of them.



```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input": true, isTemplate: true}
```

### Get sample videos

You will start with uploaded videos, as it's a more common use-case, but you will also see later that you can also use Youtube videos.


```
# Load sample images
!wget https://storage.googleapis.com/generativeai-downloads/videos/Pottery.mp4 -O Pottery.mp4 -q
!wget https://storage.googleapis.com/generativeai-downloads/videos/Jukin_Trailcam_Videounderstanding.mp4 -O Trailcam.mp4 -q
!wget https://storage.googleapis.com/generativeai-downloads/videos/post_its.mp4 -O Post_its.mp4 -q
!wget https://storage.googleapis.com/generativeai-downloads/videos/user_study.mp4 -O User_study.mp4 -q
```

### Upload the videos

Upload all the videos using the File API. You can find modre details about how to use it in the [Get Started](../quickstarts/Get_started.ipynb#scrollTo=KdUjkIQP-G_i) notebook.

This can take a couple of minutes as the videos will need to be processed and tokenized.


```
import time

def upload_video(video_file_name):
  video_file = client.files.upload(file=video_file_name)

  while video_file.state == "PROCESSING":
      print('Waiting for video to be processed.')
      time.sleep(10)
      video_file = client.files.get(name=video_file.name)

  if video_file.state == "FAILED":
    raise ValueError(video_file.state)
  print(f'Video processing complete: ' + video_file.uri)

  return video_file

pottery_video = upload_video('Pottery.mp4')
trailcam_video = upload_video('Trailcam.mp4')
post_its_video = upload_video('Post_its.mp4')
user_study_video = upload_video('User_study.mp4')
```

[First Entry, ..., Last Entry]

### Imports


```
import json
from PIL import Image
from IPython.display import display, Markdown, HTML
```

# Search within videos

First, try using the model to search within your videos and describe all the animal sightings in the trailcam video.


```
prompt = "For each scene in this video, generate captions that describe the scene along with any spoken text placed in quotation marks. Place each caption into an object with the timecode of the caption in the video."  # @param ["For each scene in this video, generate captions that describe the scene along with any spoken text placed in quotation marks. Place each caption into an object with the timecode of the caption in the video.", "Organize all scenes from this video in a table, along with timecode, a short description, a list of objects visible in the scene (with representative emojis) and an estimation of the level of excitement on a scale of 1 to 10"] {"allow-input":true}

video = trailcam_video # @param ["trailcam_video", "pottery_video", "post_its_video", "user_study_video"] {"type":"raw","allow-input":true}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        video,
        prompt,
    ]
)

Markdown(response.text)
```




```json
[
  {
    "time": "00:00 - 00:17",
    "caption": "Two gray foxes in the wild, foraging. One comes into view from the right, followed by another. They are sniffing the ground, and one climbs onto a rock."
  },
  {
    "time": "00:17 - 00:34",
    "caption": "A mountain lion is seen sniffing the ground in a forest, then briefly looking up and walking off. (Night vision)"
  },
  {
    "time": "00:34 - 00:50",
    "caption": "Two foxes are captured at night. One digs in the ground, and then they engage in a brief, aggressive interaction before running out of frame. (Night vision with IR flash)"
  },
  {
    "time": "00:50 - 01:04",
    "caption": "A bright flash occurs, followed by two foxes in a rocky area at night. They move around, one looks at the camera, and then another bright flash illuminates the scene. (Night vision with IR flash)"
  },
  {
    "time": "01:04 - 01:17",
    "caption": "A mountain lion walks from right to left across the frame in the dark. (Night vision)"
  },
  {
    "time": "01:17 - 01:29",
    "caption": "Two mountain lions are seen at night. The larger one walks past the camera in the foreground, while a smaller one (possibly a cub) walks on top of a rock in the background. (Night vision)"
  },
  {
    "time": "01:29 - 01:51",
    "caption": "A bobcat is seen at night, foraging on the ground, then digging a hole, and looking directly at the camera with glowing eyes. (Night vision)"
  },
  {
    "time": "01:51 - 01:56",
    "caption": "A brown bear walks away from the camera through a sun-dappled forest. (Daylight)"
  },
  {
    "time": "01:56 - 02:04",
    "caption": "A mountain lion walks into the frame from the left, looks at the camera, and then walks out of frame to the right. (Night vision)"
  },
  {
    "time": "02:04 - 02:22",
    "caption": "Two bears, possibly a mother and cub, are walking through the forest. One briefly obstructs the camera's view before they both move off into the distance. (Daylight)"
  },
  {
    "time": "02:22 - 02:34",
    "caption": "A fox is seen at night on a hill overlooking a city with twinkling lights. It sniffs the ground and then sits up to look out over the city. (Night vision)"
  },
  {
    "time": "02:34 - 02:41",
    "caption": "A bear walks past the camera at night, with a city lights landscape visible in the background. (Night vision)"
  },
  {
    "time": "02:41 - 02:51",
    "caption": "A mountain lion walks past the camera at night, with the illuminated city in the distance. (Night vision)"
  },
  {
    "time": "02:51 - 03:04",
    "caption": "A mountain lion walks towards a tree and then sniffs around on the ground. (Night vision)"
  },
  {
    "time": "03:04 - 03:22",
    "caption": "A brown bear stands in the forest, looks around, then directly at the camera, before walking off. (Daylight)"
  },
  {
    "time": "03:22 - 03:40",
    "caption": "Two brown bears are seen foraging on the ground in the forest. One bear briefly obstructs the camera's view as it moves closer. (Daylight)"
  },
  {
    "time": "03:40 - 04:03",
    "caption": "Two brown bears walk away from the camera. One sits down and scratches itself, then they both continue walking into the distance. (Daylight)"
  },
  {
    "time": "04:03 - 04:22",
    "caption": "Two brown bears walk towards the camera. One walks past, while the other remains in view, sniffing the ground. (Daylight)"
  },
  {
    "time": "04:22 - 04:30",
    "caption": "A bobcat with bright, glowing eyes looks at the camera, then walks past and out of frame. (Night vision)"
  },
  {
    "time": "04:30 - 04:49",
    "caption": "A fox appears in the distance with glowing eyes, walks closer to the camera, and then suddenly dashes out of frame. (Night vision)"
  },
  {
    "time": "04:49 - 04:57",
    "caption": "A fox is seen walking away from the camera into the dark forest. (Night vision)"
  },
  {
    "time": "04:57 - 05:10",
    "caption": "A mountain lion walks towards a tree, sniffs the ground, and then walks past the camera. (Night vision)"
  }
]
```



The prompt used is quite a generic one, but you can get even better results if you cutomize it to your needs (like asking specifically for foxes).

The [live demo on AI Studio](https://aistudio.google.com/starter-apps/video) shows how you can postprocess this output to jump directly to the the specific part of the video by clicking on the timecodes. If you are interested, you can check the [code of that demo on Github](https://github.com/google-gemini/starter-applets/tree/main/video).

# Extract and organize text

Gemini models can also read what's in the video and extract it in an organized way. You can even use Gemini reasoning capabilities to generate new ideas for you.


```
prompt = "Transcribe the sticky notes, organize them and put it in a table. Can you come up with a few more ideas?" # @param ["Transcribe the sticky notes, organize them and put it in a table. Can you come up with a few more ideas?", "Which of those names who fit an AI product that can resolve complex questions using its thinking abilities?"] {"allow-input":true}

video = post_its_video # @param ["trailcam_video", "pottery_video", "post_its_video", "user_study_video"] {"type":"raw","allow-input":true}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        video,
        prompt,
    ]
)

Markdown(response.text)
```




Here are the transcribed project names from the sticky notes, organized alphabetically in a table, along with a few more ideas:

## Brainstorm: Project Names

| Project Name         | Project Name         |
| :------------------- | :------------------- |
| Aether               | Leo Minor            |
| Andromeda's Reach    | Lunar Eclipse        |
| Astral Forge         | Lyra                 |
| Athena               | Lynx                 |
| Athena's Eye         | Medusa               |
| Bayes Theorem        | Odin                 |
| Canis Major          | Orion's Belt         |
| Celestial Drift      | Orion's Sword        |
| Centaurus            | Pandora's Box        |
| Cerberus             | Persius Shield       |
| Chaos Field          | Phoenix              |
| Chaos Theory         | Prometheus Rising    |
| Chimera Dream        | Riemann's Hypothesis |
| Comets Tail          | Sagitta              |
| Convergence          | Serpens              |
| Delphinus            | Stellar Nexus        |
| Draco                | Stokes Theorem       |
| Echo                 | Supernova Echo       |
| Equilibrium          | Symmetry             |
| Euler's Path         | Taylor Series        |
| Fractal              | Titan                |
| Galactic Core        | Vector               |
| Golden Ratio         | Zephyr               |
| Hera                 |                      |
| Infinity Loop        |                      |

---

## A Few More Project Name Ideas:

1.  **Pulsar:** (Astronomical, suggests powerful and rhythmic energy)
2.  **Axiom:** (Mathematical/logical, implies a fundamental truth or starting point)
3.  **Artemis:** (Mythological, associated with precision, exploration, and the moon)
4.  **Quantum Leap:** (Scientific, indicates a significant and sudden advancement)
5.  **Vortex:** (Implies a central point of activity, energy, or convergence)



# Structure information

Gemini is not only able to read text but also to reason and structure about real world objects. Like in this video about a display of ceramics with handwritten prices and notes.


```
prompt = "Give me a table of my items and notes" # @param ["Give me a table of my items and notes", "Help me come up with a selling pitch for my potteries"] {"allow-input":true}

video = pottery_video # @param ["trailcam_video", "pottery_video", "post_its_video", "user_study_video"] {"type":"raw","allow-input":true}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        video,
        prompt,
    ],
    config = types.GenerateContentConfig(
        system_instruction="Don't forget to escape the dollar signs",
    )
)

Markdown(response.text)
```




Here's a table summarizing the items and notes from the image:

| Category          | Item                | Description                                                                                                                                                                                                                             | Dimensions                     | Price   | Additional Notes                  |
| :---------------- | :------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------- | :------ | :-------------------------------- |
| Drinkware         | Tumblers            | Stacked and individual tumblers with an earthy brown/beige base and a light blue/white wavy glaze towards the top. Two small, round ceramic samples displaying the base and blue/grey glaze are shown next to them. | 4"h x 3"d (approx.)            | \$20    | \#5 Artichoke double dip          |
| Bowls             | Small Bowls         | Two bowls with a speckled, rustic brown/orange exterior and a darker, possibly iridescent, interior with hints of blue/green.                                                                                                        | 3.5"h x 6.5"d                  | \$35    |                                   |
| Bowls             | Medium Bowls        | Two larger bowls, similar in appearance to the small bowls with a speckled, rustic brown/orange exterior and a darker, iridescent interior with hints of blue/green.                                                               | 4"h x 7"d                      | \$40    |                                   |
| Glaze Sample/Test | Gemini Double Dip   | A rectangular ceramic tile with "6b6" inscribed, displaying a brown/rust speckled glaze on one side and a blue/grey glaze on the other.                                                                                              | N/A (sample tile)              | N/A     | \#6 Gemini double dip, Slow Cool |



As you can see, Gemini is able to grasp to with item corresponds each note, including the last one.

# Analyze screen recordings for key moments

You can also use the model to analyze screen recordings. Let's say you're doing user studies on how people use your product, so you end up with lots of screen recordings, like this one, that you have to manually comb through.
With just one prompt, the model can describe all the actions in your video.


```
prompt = "Generate a paragraph that summarizes this video. Keep it to 3 to 5 sentences with corresponding timecodes." # @param ["Generate a paragraph that summarizes this video. Keep it to 3 to 5 sentences with corresponding timecodes.", "Choose 5 key shots from this video and put them in a table with the timecode, text description of 10 words or less, and a list of objects visible in the scene (with representative emojis).", "Generate bullet points for the video. Place
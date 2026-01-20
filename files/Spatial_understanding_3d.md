# Pointing and 3D Spatial Understanding with Gemini (Experimental)

This colab highlights some of the exciting use cases for Gemini in spatial understanding. It focuses on how [Gemini](https://ai.google.dev/gemini-api/docs/models/gemini-v2)'s image and real world understanding capabilities including pointing and 3D spatial understanding as briefly teased in the [Building with Gemini 2.0: Spatial understanding](https://www.youtube.com/watch?v=-XmoDzDMqj4) video.

Points and 3D bounding boxes are experimental. Use [2D bounding boxes](../quickstarts/Spatial_understanding.ipynb) for higher accuracy.

Pointing is an important capability for vision language models, because that allows the model to refer to an entity precisely. Gemini Flash has improved accuracy on spatial understanding, with 2D point prediction as an experimental feature. Below you'll see that pointing can be combined with reasoning.

Traditionally, a Vision Language Model (VLM) sees the world in 2D, however, [Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-v2) can perform 3D detection. The model has a general sense of the space and knows where the objects are in 3D space.

The model will respond to spatial understanding-related requests in json format to facilitate parsing, and the coordinates always have the same conventions. For this example to be more readable, it overlays the spatial signals on the image, and the readers can hover their cursor on the image to get the complete response. The coordinates are in the image frame, and are normalized into an integer between 0-1000. The top left is `(0,0)` and the bottom right is `(1000,1000)`. The point is in `[y, x]` order, and 2d bounding boxes are in `y_min, x_min, y_max, x_max` order.

 Additionally, 3D bounding boxes are represented with 9 numbers, the first 3 numbers represent the center of the object in camera frame, they are in metric units; the next 3 numbers represent the size of the object in meters, and the last 3 numbers are Euler angles representing row, pitch and yaw, they are in degree.

To learn more about 2D spatial understanding, please take a look at [2d examples](../quickstarts/Spatial_understanding.ipynb) and the [Spatial understanding example](https://aistudio.google.com/starter-apps/spatial) from [Google AI Studio](https://aistudio.google.com).

## Setup

### Install SDK


```
%pip install -U -q google-genai
```

[Installation logs, ..., Installation logs]

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

from PIL import Image

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Select a model

3d spatial understanding and pointing are two new capabilities introduced in the Gemini 2.0 Flash model. Later generation models are also capable of using those capabilities.

For more information about all Gemini models, check the [documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for extended information on each of them.



```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

### Load sample images


```
# Load sample images
!wget https://storage.googleapis.com/generativeai-downloads/images/kitchen.jpg -O kitchen.jpg -q
!wget https://storage.googleapis.com/generativeai-downloads/images/room-clock.jpg -O room.jpg -q
!wget https://storage.googleapis.com/generativeai-downloads/images/spill.jpg -O spill.jpg -q
!wget https://storage.googleapis.com/generativeai-downloads/images/tool.png -O tool.png -q
!wget https://storage.googleapis.com/generativeai-downloads/images/music_0.jpg -O music_0.jpg -q
!wget https://storage.googleapis.com/generativeai-downloads/images/music_1.jpg -O music_1.jpg -q
!wget https://storage.googleapis.com/generativeai-downloads/images/traj_00.jpg -O traj_00.jpg -q
!wget https://storage.googleapis.com/generativeai-downloads/images/traj_01.jpg -O traj_01.jpg -q
!wget https://storage.googleapis.com/generativeai-downloads/images/shoe_bench_0.jpg -O shoe_bench_0.jpg -q
!wget https://storage.googleapis.com/generativeai-downloads/images/shoe_bench_1.jpg -O shoe_bench_1.jpg -q
```

## Pointing to items using Gemini

Instead of asking for [bounding boxes](../quickstarts/Spatial_understanding.ipynb), you can ask Gemini to points are things on the image. Depending on your use-case it might be sufficent and will less clutter the images.

Just be careful that the format Gemini knows the best is (y, x), so it's better to stick to it.

To prevent the model from repeating itself, it is recommended to use a temperature over 0, in this case 0.5. Limiting the number of items (10 in this case) is also a way to prevent the model from looping and to speed up the decoding of the corrdinates. You can experiment with these parameters and find what works best for your use-case.

### Analyze the image using Gemini


```
# Load and resize image
img = Image.open("tool.png")
img = img.resize((800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS) # Resizing to speed-up rendering

# Analyze the image using Gemini
image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        img,
        """
          Point to no more than 10 items in the image, include spill.
          The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000.
        """
    ],
    config = types.GenerateContentConfig(
        temperature=0.5
    )
)

# Check response
print(image_response.text)
```

    ```json
    [
      {"point": [130, 760], "label": "handle"},
      {"point": [427, 517], "label": "screw"},
      {"point": [472, 201], "label": "clamp arm"},
      {"point": [466, 345], "label": "clamp arm"},
      {"point": [685, 312], "label": "3 inch"},
      {"point": [493, 659], "label": "screw"},
      {"point": [402, 474], "label": "screw"},
      {"point": [437, 664], "label": "screw"},
      {"point": [427, 784], "label": "handle"},
      {"point": [452, 852], "label": "handle"}
    ]
    ```



```
# @title Point visualization code

import IPython

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def generate_point_html(pil_image, points_json):
    # Convert PIL image to base64 string
    import base64
    from io import BytesIO
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    points_json = parse_json(points_json)

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Point Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #fff;
            color: #000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}

        .point-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}

        .point {{
            position: absolute;
            width: 12px;
            height: 12px;
            background-color: #2962FF;
            border: 2px solid #fff;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 0 40px rgba(41, 98, 255, 0.6);
            opacity: 0;
            transition: all 0.3s ease-in;
            pointer-events: auto;
        }}

        .point.visible {{
            opacity: 1;
        }}

        .point.fade-out {{
            animation: pointFadeOut 0.3s forwards;
        }}

        .point.highlight {{
            transform: translate(-50%, -50%) scale(1.1);
            background-color: #FF4081;
            box-shadow: 0 0 40px rgba(255, 64, 129, 0.6);
            z-index: 100;
        }}

        @keyframes pointFadeOut {{
            from {{
                opacity: 1;
            }}
            to {{
                opacity: 0.7;
            }}
        }}

        .point-label {{
            position: absolute;
            background-color: #2962FF;
            color: #fff;
            font-size: 14px;
            padding: 4px 12px;
            border-radius: 4px;
            transform: translate(20px, -10px);
            white-space: nowrap;
            opacity: 0;
            transition: all 0.3s ease-in;
            box-shadow: 0 0 30px rgba(41, 98, 255, 0.4);
            pointer-events: auto;
            cursor: pointer;
        }}

        .point-label.visible {{
            opacity: 1;
        }}

        .point-label.fade-out {{
            opacity: 0.45;
        }}

        .point-label.highlight {{
            background-color: #FF4081;
            box-shadow: 0 0 30px rgba(255, 64, 129, 0.4);
            transform: translate(20px, -10px) scale(1.1);
            z-index: 100;
        }}
    </style>
</head>
<body>
    <div id="container" style="position: relative;">
        <canvas id="canvas" style="background: #000;"></canvas>
        <div id="pointOverlay" class="point-overlay"></div>
    </div>

    <script>
        function annotatePoints(frame) {{
            // Add points with fade effect
            const pointsData = {points_json};

            const pointOverlay = document.getElementById('pointOverlay');
            pointOverlay.innerHTML = '';

            const points = [];
            const labels = [];

            pointsData.forEach(pointData => {{
                // Skip entries without coodinates.
                if (!(pointData.hasOwnProperty("point")))
                  return;

                const point = document.createElement('div');
                point.className = 'point';
                const [y, x] = pointData.point;
                point.style.left = `${{x/1000.0 * 100.0}}%`;
                point.style.top = `${{y/1000.0 * 100.0}}%`;

                const pointLabel = document.createElement('div');
                pointLabel.className = 'point-label';
                pointLabel.textContent = pointData.label;
                point.appendChild(pointLabel);

                pointOverlay.appendChild(point);
                points.push(point);
                labels.push(pointLabel);

                setTimeout(() => {{
                    point.classList.add('visible');
                    pointLabel.classList.add('visible');
                }}, 0);

                // Add hover effects
                const handleMouseEnter = () => {{
                    // Highlight current point and label
                    point.classList.add('highlight');
                    pointLabel.classList.add('highlight');

                    // Fade out other points and labels
                    points.forEach((p, idx) => {{
                        if (p !== point) {{
                            p.classList.add('fade-out');
                            labels[idx].classList.add('fade-out');
                        }}
                    }});
                }};

                const handleMouseLeave = () => {{
                    // Remove highlight from current point and label
                    point.classList.remove('highlight');
                    pointLabel.classList.remove('highlight');

                    // Restore other points and labels
                    points.forEach((p, idx) => {{
                        p.classList.remove('fade-out');
                        labels[idx].classList.remove('fade-out');
                    }});
                }};

                point.addEventListener('mouseenter', handleMouseEnter);
                point.addEventListener('mouseleave', handleMouseLeave);
                pointLabel.addEventListener('mouseenter', handleMouseEnter);
                pointLabel.addEventListener('mouseleave', handleMouseLeave);
            }});
        }}

        // Initialize canvas
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const container = document.getElementById('container');

        // Load and draw the image
        const img = new Image();
        img.onload = () => {{
            const aspectRatio = img.height / img.width;
            canvas.width = 800;
            canvas.height = Math.round(800 * aspectRatio);
            container.style.width = canvas.width + 'px';
            container.style.height = canvas.height + 'px';

            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            frame.width = canvas.width;
            frame.height = canvas.height;
            annotatePoints(frame);
        }};
        img.src = '';

        const frame = {{
            width: canvas.width,
            height: canvas.height
        }};

        annotatePoints(frame);
    </script>
</body>
</html>
"""
```

The script create an HTML rendering of the image and the points. It is similar to the one used in the [Spatial understanding example](https://aistudio.google.com/starter-apps/spatial) from [Google AI Studio](https://aistudio.google.com).

Of course this is just an example and you are free to just write your own.


```
# Display the dots on the image
IPython.display.HTML(generate_point_html(img, image_response.text))
```

### Pointing and reasoning

You can use Gemini's reasoning capabilities on top of its pointing ones as in the [2d bounding box](../quickstarts/Spatial_understanding.ipynb#scrollTo=GZbhjYkUA86w) example and ask for more detailled labels.

In this case you can do it by adding this sentence to the prompt: "Explain how to use each part, put them in the label field, remove duplicated parts and instructions".


```
# Load and resize image
img = Image.open("tool.png")
img = img.resize((800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)

# Analyze the image using Gemini
image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        img,
        """
          Pinpoint no more than 10 items in the image.
          The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000. One element a line.
          Explain how to use each part, put them in the label field, remove duplicated parts and instructions.
        """
    ],
    config = types.GenerateContentConfig(
        temperature=0.5
    )
)

# Display the dots on the image
IPython.display.HTML(generate_point_html(img, image_response.text))
```

### More pointing and reasoning examples

Expend this section to see more examples of images and prompts you can use. Experiment with them and find what works bets for your use-case.

#### Kitchen safety



```
# Load and resize image
img = Image.open("kitchen.jpg")
img = img.resize((800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)

# Analyze the image using Gemini
image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        img,
        """
          Point to no more than 10 items in the image.
          The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000. One element a line.
          Explain how to prevent kids from getting hurt, put them in the label field, remove duplicated parts and instructions.
        """
    ],
    config = types.GenerateContentConfig(
        temperature=0.5
    )
)

# Display the dots on the image
IPython.display.HTML(generate_point
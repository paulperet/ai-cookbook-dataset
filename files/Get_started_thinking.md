##### Copyright 2025 Google LLC.


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Use Gemini thinking

---
> **Gemini 3 models**: If you are only interested in the new [Gemini 3](https://ai.google.dev/gemini-api/docs/models#gemini-3-pro) new thinking levels, jump directly to the [dedicated section](#gemini3) at the end of this notebook that also includes a [migration guide](#gemini3migration).

---


All Gemini models from the 2.5 generation and the new [Gemini 3 generation](https://ai.google.dev/gemini-api/docs/models#gemini-3-pro) are trained to do a [thinking process](https://ai.google.dev/gemini-api/docs/thinking-mode) (or reasoning) before getting to a final answer. As a result, those models are capable of stronger reasoning capabilities in its responses than previous models.

You'll see examples of those reasoning capabilities with [code understanding](#code_execution), [geometry](#geometry) and [math](#math) problems.

As you will see, the model is exposing its thoughts so you can have a look at its reasoning and how it did reach its conclusions.

## Understanding the thinking models

Thinking models are optimized for complex tasks that need multiple rounds of strategyzing and iteratively solving.

[Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-preview-04-17) in particular, brings the flexibility of using `thinking_budget` - a parameter
that offers fine-grained control over the maximum number of tokens a model can generate while thinking. Alternatively, you can designate a precise token allowance for the
"thinking" stage through the adjusment of the `thinking_budget` parameter. This allowance can vary between 0 and 24576 tokens for 2.5 Flash.

For more information about all Gemini models, check the [documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for extended information on each of them.

On this notebook most examples are using the `thinking_budget` parameter since it's compatible with both the 2.5 and the 3 generations of models. For more information about using the `thinking_budget` with the Gemini thinking model, check the [documentation](https://ai.google.dev/gemini-api/docs/thinking).

**NEW: thinking levels:** [Gemini 3 models](https://ai.google.dev/gemini-api/docs/gemini-3) introduced a new, easier way to manage the thinking buget by setting a `thinking_level` that is documented in the [section of this guide dedicated to Gemini 3](#gemini3).

## Setup

This section install the SDK, set it up using your [API key](../quickstarts/Authentication.ipynb), imports the relevant libs, downloads the sample videos and upload them to Gemini.

Just collapse (click on the little arrow on the left of the title) and run this section if you want to jump straight to the examples (just don't forget to run it otherwise nothing will work).

### Install SDK

The **[Google Gen AI SDK](https://ai.google.dev/gemini-api/docs/sdks)** provides programmatic access to Gemini models using both the [Google AI for Developers](https://ai.google.dev/gemini-api/docs) and [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview) APIs. With a few exceptions, code that runs on one platform will run on both. This means that you can prototype an application using the Developer API and then migrate the application to Vertex AI without rewriting your code.

More details about this new SDK on the [documentation](https://ai.google.dev/gemini-api/docs/sdks) or in the [Getting started ![image](https://storage.googleapis.com/generativeai-downloads/images/colab_icon16.png)](../quickstarts/Get_started.ipynb) notebook.


```
%pip install -U -q 'google-genai>=1.51.0' # 1.51 is needed for Gemini 3 pro thinking levels support
```

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication ![image](https://storage.googleapis.com/generativeai-downloads/images/colab_icon16.png)](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
```

### Initialize SDK client

With the new SDK you now only need to initialize a client with you API key (or OAuth if using [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview)). The model is now set in each call.


```
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

### Imports


```
import json
from PIL import Image
from IPython.display import display, Markdown
```

## Using the thinking models

Here are some quite complex examples of what Gemini thinking models can solve.

In each of them you can select different models to see how this new model compares to its predecesors.

In some cases, you'll still get the good answer from the other models, in that case, re-run it a couple of times and you'll see that Gemini thinking models are more consistent thanks to their thinking step.

### Using adaptive thinking

You can start by asking the model to explain a concept and see how it does reasoning before answering.

Starting with the adaptive `thinking_budget` - which is the default when you don't specify a budget - the model will dynamically adjust the budget based on the complexity of the request.

The animal it should find is a [**Platypus**](https://en.wikipedia.org/wiki/Platypus), but as you'll see it is not the first answer it thinks of depending on how much thinking it does.


```
prompt = """
    You are playing the 20 question game. You know that what you are looking for
    is a aquatic mammal that doesn't live in the sea, is venomous and that's
    smaller than a cat. What could that be and how could you make sure?
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

Markdown(response.text)
```




This is a fantastic set of clues! It points to a very specific and unusual animal.

Based on your clues, what you are looking for is almost certainly a **Platypus**.

Here's why:

1.  **Aquatic Mammal:** Yes, they are semi-aquatic mammals.
2.  **Doesn't live in the sea:** They are found in freshwater rivers and streams in eastern Australia and Tasmania.
3.  **Venomous:** This is the most unique clue for a mammal. Male platypuses possess a venomous spur on their hind legs, which can inflict considerable pain on humans and be lethal to smaller animals.
4.  **Smaller than a cat:** An adult platypus typically ranges from 30 to 45 cm (12-18 inches) in body length, plus a tail, and weighs between 0.7 to 2.4 kg (1.5-5.3 lbs), which is generally smaller than an average domestic cat.

---

**How you could make sure (in the 20 questions game):**

To confirm it's a platypus, you'd want to ask questions that narrow down to its unique characteristics:

1.  **"Does it lay eggs?"** (This would confirm it's a monotreme, an egg-laying mammal, which is incredibly rare and applies to platypuses.)
2.  **"Is it native to Australia?"** (Platypuses are endemic to Australia.)
3.  **"Does it have a bill like a duck?"** (This is its most distinctive physical feature.)
4.  **"Do the males have a special defense mechanism on their hind legs?"** (This hints at the venomous spur without giving it away.)
5.  **"Is its primary diet invertebrates found in the water?"** (They are carnivores that forage for worms, insect larvae, and freshwater shrimp.)



Looking to the response metadata, you can see not only the amount of tokens on your input and the amount of tokens used for the response, but also the amount of tokens used for the thinking step - As you can see here, the model used around 1400 tokens in the thinking steps:


```
print("Prompt tokens:",response.usage_metadata.prompt_token_count)
print("Thoughts tokens:",response.usage_metadata.thoughts_token_count)
print("Output tokens:",response.usage_metadata.candidates_token_count)
print("Total tokens:",response.usage_metadata.total_token_count)
```

    Prompt tokens: 59
    Thoughts tokens: 1451
    Output tokens: 815
    Total tokens: 2325


### Disabling the thinking steps

You can also disable the thinking steps by setting the `thinking_budget` to 0 (but not with the pro models). You'll see that in this case, the model doesn't think of the platypus as a possible answer.


```
if "-pro" not in MODEL_ID:
  prompt = """
      You are playing the 20 question game. You know that what you are looking for
      is a aquatic mammal that doesn't live in the sea, is venomous and that's
      smaller than a cat. What could that be and how could you make sure?
  """

  response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
      thinking_config=types.ThinkingConfig(
        thinking_budget=0
      )
    )
  )

  Markdown(response.text)

else:
  print("You can't disable thinking for pro models.")
```




This is a fun and tricky one, because *aquatic mammal that doesn't live in the sea, is venomous, and smaller than a cat* sounds like it doesn't exist! Let's break it down and see if we can find a plausible (even if highly improbable) answer, and then how to confirm.

**The Challenges with Your Clues:**

*   **Aquatic Mammal that doesn't live in the sea:** This limits us to freshwater or semi-aquatic mammals.
*   **Venomous:** This is the *major* sticking point. Very few mammals are venomous. The most well-known are the platypus (males have a venomous spur) and several species of shrews and solenodons (which have venomous saliva).
*   **Smaller than a cat:** This eliminates the platypus (which can be cat-sized or larger), and most of the larger freshwater mammals.

**Possible Candidates (and why they're a stretch):**

1.  **A Freshwater-dwelling Shrew:**
    *   **Aquatic/Freshwater:** Some shrews are highly aquatic or semi-aquatic, like the American water shrew (Sorex palustris) or the Eurasian water shrew (Neomys fodiens). They live near streams, rivers, and ponds.
    *   **Venomous:** Yes! Many species of shrews (and solenodons) possess a neurotoxic venom in their saliva, which they use to immobilize prey.
    *   **Smaller than a cat:** Absolutely. Shrews are tiny, mouse-sized creatures.
    *   **The Catch:** While they are "aquatic" in the sense they forage and swim in water, they are not typically thought of as "aquatic mammals" in the same vein as beavers or otters, which are much more adapted to a watery life. They are terrestrial mammals that have adapted to an aquatic foraging niche. However, given the constraints, this is the most likely candidate.

    **So, my best guess is a Freshwater Water Shrew (e.g., American Water Shrew or Eurasian Water Shrew).**

**How to make sure it's a Freshwater Water Shrew:**

To confirm this, you'd need to ask very specific clarifying questions about its characteristics:

1.  **"Does it spend a significant portion of its life in water, specifically freshwater?"**
    *   *Expected Answer:* Yes, it swims and dives regularly to hunt.

2.  **"Is its venom delivered via a bite or a spur?"**
    *   *Expected Answer:* Through a bite, from its saliva.

3.  **"Is it smaller than a house mouse?"** (This would confirm its tiny size and distinguish it from a rat or larger rodent)
    *   *Expected Answer:* Yes.

4.  **"Is it primarily an insectivore or carnivore?"** (To distinguish it from rodents)
    *   *Expected Answer:* It primarily eats insects and small invertebrates, often aquatic ones.

5.  **"Is it known for having an extremely high metabolism, requiring it to eat almost constantly?"** (A very common shrew characteristic)
    *   *Expected Answer:* Yes.

If you get "yes" to these questions, especially about the venom delivery, size, and aquatic habits, you've almost certainly identified a freshwater water shrew.



Now you can see that the response is faster as the model didn't perform any thinking step. Also you can see that no tokens were used for the thinking step:


```
print("Prompt tokens:",response.usage_metadata.prompt_token_count)
print("Thoughts tokens:",response.usage_metadata.thoughts_token_count)
print("Output tokens:",response.usage_metadata.candidates_token_count)
print("Total tokens:",response.usage_metadata.total_token_count)
```

    Prompt tokens: 59
    Thoughts tokens: None
    Output tokens: 688
    Total tokens: 747


<a name="physics"></a>
### Solving a physics problem

Now, try with a simple physics comprehension example. First you can disable the `thinking_budget` to see how the model performs:


```
if "-pro" not in MODEL_ID:
  prompt = """
      A cantilever beam of length L=3m has a rectangular cross-section (width b=0.1m, height h=0.2m) and is made of steel (E=200 GPa).
      It is subjected to a uniformly distributed load w=5 kN/m along its entire length and a point load P=10 kN at its free end.
      Calculate the maximum bending stress (σ_max).
  """

  response = client.models.generate_content(
      model=MODEL_ID,
      contents=prompt,
      config=types.GenerateContentConfig(
          thinking_config=types.ThinkingConfig(
              thinking_budget=0
          )
      )
  )

  Markdown(response.text)

else:
  print("You can't disable thinking for pro models.")
```




Here's how to calculate the maximum bending stress for the given cantilever beam:

**1. Understand the Concepts**

*   **Cantilever Beam:** A beam fixed at one end and free at the other.
*   **Bending Moment (M):** The internal resistance of a beam to bending. It's maximum where the most "pull" or "push" is happening.
*   **Moment of Inertia (I):** A geometric property of a cross-section that indicates its resistance to bending. A larger 'I' means less bending for a given load.
*   **Bending Stress (σ):** The normal stress developed in a beam due to bending. It's maximum at the top and bottom surfaces of the beam.
*   **Flexure Formula:** $\sigma = \frac{M \cdot y}{I}$
    *   M: Bending moment
    *   y: Distance from the neutral axis to the point where stress is calculated (for maximum stress, y = c, where c is the distance to the extreme fiber).
    *   I: Moment of inertia

**2. Calculate the Moment of Inertia (I)**

For a rectangular cross-section:
$I = \frac{b \cdot h^3}{12}$

Given:
*   b = 0.1 m
*   h = 0.2 m

$I = \frac{0.1 \cdot (0.2)^3}{12} = \frac{0.1 \cdot 0.008}{12} = \frac{0.0008}{12} = 6.6667 \times 10^{-5} \ m^4$

**3. Determine the Maximum Bending Moment (M_max)**

For a cantilever beam, the maximum bending moment always occurs at the fixed end. We need to consider both the distributed load and the point load.

*   **Moment due to uniformly distributed load (w):**
    $M_w = w \cdot L \cdot \frac{L}{2} = \frac{w \cdot L^2}{2}$
    $M_w = \frac{5 \text{ kN/m} \cdot (3 \text{ m})^2}{2} = \frac{5 \cdot 9}{2} = \frac{45}{2} = 22.5 \text{ kN} \cdot \text{m}$

*   **Moment due to point load (P):**
    $M_P = P \cdot L$
    $M_P = 10 \text{ kN} \cdot 3 \text{ m} = 30 \text{ kN} \cdot \text{m}$

*   **Total Maximum Bending Moment:**
    $M_{max} = M_w + M_P = 22.5 \text{ kN} \cdot \text{m} + 30 \text{ kN} \cdot \text{m} = 52.5 \text{ kN} \cdot \text{m}$
    $M_{max} = 52.5 \times 10^3 \text{ N} \cdot \text{m}$ (converting kN to N)

**4. Determine 'y' (distance to the extreme fiber)**

For a rectangular cross-section, the neutral axis is at the centroid. The maximum stress occurs at the top and bottom surfaces.
$y = c = \frac{h}{
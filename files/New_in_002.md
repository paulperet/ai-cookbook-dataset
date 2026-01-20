

# What's new in Gemini-1.5-pro-002 and Gemini-1.5-flash-002

This notebook explores the new options added with the 002 versions of the 1.5 series models:

* Candidate count
* Presence and frequency penalties
* Response logprobs

## Setup

Install a `002` compatible version of the SDK:


```
%pip install -q "google-generativeai>=0.8.2"
```

import the package and give it your API-key


```
import google.generativeai as genai
```


```
from google.colab import userdata
genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))
```

Import other packages.


```
from IPython.display import display, Markdown, HTML
```

Check available 002 models


```
for model in genai.list_models():
  if '002' in model.name:
    print(model.name)
```

    models/bard-lmsys-002
    models/gemini-1.5-pro-002
    models/gemini-1.5-flash-002
    models/gemini-1.5-pro-002-test
    models/gemini-1.5-flash-002-vertex-global-test
    models/imagen-3.0-generate-002
    models/imagen-3.0-generate-002-exp



```
model_name = "models/gemini-1.5-flash-002"
test_prompt="Why don't people have tails"
```

## Quick refresher on `generation_config` [Optional]


```
model = genai.GenerativeModel(model_name, generation_config={'temperature':1.0})
response = model.generate_content('hello', generation_config = genai.GenerationConfig(max_output_tokens=5))
```

Note:

* Each `generate_content` request is sent with a `generation_config` (`chat.send_message` uses `generate_content`).
* You can set the `generation_config` by either passing it to the model's initializer, or passing it in the arguments to `generate_content` (or `chat.send_message`).
* Any `generation_config` attributes set in `generate_content` override the attributes set on the model.
* You can pass the `generation_config` as either a Python `dict`, or a `genai.GenerationConfig`.
* If you're ever unsure about the parameters of `generation_config` check `genai.GenerationConfig`.

## Candidate count

With 002 models you can now use `candidate_count > 1`.


```
model = genai.GenerativeModel(model_name)
```


```
generation_config = dict(candidate_count=2)
```


```
response = model.generate_content(test_prompt, generation_config=generation_config)
```

But note that the `.text` quick-accessor only works for the simple 1-candidate case.


```
try:
  response.text # Fails with multiple candidates, sorry!
except ValueError as e:
  print(e)
```

    Invalid operation: The `response.parts` quick accessor retrieves the parts for a single candidate. This response contains multiple candidates, please use `result.candidates[index].text`.


With multiple candidates you have to handle the list of candidates yourself:


```
for candidate in response.candidates:
  display(Markdown(candidate.content.parts[0].text))
  display(Markdown("-------------"))

```


Humans don't have tails because of evolutionary changes over millions of years.  Our primate ancestors had tails, but as humans evolved, the tail gradually disappeared.  The reasons aren't fully understood, but several theories exist:

* **Loss of function:**  As our ancestors transitioned to bipedalism (walking upright), the function of a tail for balance and climbing diminished.  Natural selection favored individuals with smaller, less developed tails, as the energy needed to maintain a tail was no longer offset by its usefulness.  Essentially, it became an energetically expensive feature with little benefit.

* **Genetic changes:**  Mutations affecting genes controlling tail development likely occurred and were favored by natural selection.  These mutations could have caused the tail to become smaller in successive generations until it was completely lost.  The coccyx (tailbone) is a vestigial structure – a remnant of our tailed ancestors.

* **Developmental changes:**  Changes in the timing and regulation of genes involved in embryonic development may have led to the shortening and eventual disappearance of the tail.  The genes that once directed tail growth might have been altered to cease development of a tail at an early stage of embryonic growth.

It's important to note that these are interconnected factors.  The loss of tail function made it less crucial for survival, and genetic mutations that led to its reduced size and eventual disappearance were then naturally selected for.  The process happened gradually over a long period of evolutionary time.




-------------



Humans don't have tails because of evolution.  Our ancestors did have tails, but over millions of years of evolution, the tail gradually became smaller and less functional until it was essentially absorbed into the body.  There's no single reason, but rather a combination of factors likely contributed:

* **Loss of Functionality:** As our ancestors became bipedal (walking upright), the tail's primary function for balance and locomotion became less crucial.  Other adaptations, like changes in our skeletal structure and leg musculature, compensated for the loss of the tail's balancing role.

* **Genetic Changes:**  Mutations that affected the genes controlling tail development accumulated over time.  These mutations might have been initially neutral or even slightly advantageous in other ways, and natural selection didn't actively remove them because the tail's importance diminished.

* **Energy Conservation:**  Maintaining a tail requires energy.  As our ancestors transitioned to different environments and lifestyles, the energy cost of maintaining a tail may have become a disadvantage, especially in resource-scarce environments.  Those with less pronounced tails, or even the complete loss of tails, might have had a slight survival and reproductive advantage.

* **Sexual Selection:**  It's possible that at some point, a tailless or nearly tailless phenotype became a desirable trait from a sexual selection perspective.  This is difficult to prove, but it's a factor considered in the evolution of various traits.

In short, the absence of a tail in humans is a result of a gradual evolutionary process where the tail's usefulness decreased, genetic changes accumulated, and natural selection favored individuals with less prominent tails.  The coccyx, the small bone at the base of our spine, is the remnant of our evolutionary tail.




-------------


The response contains multiple full `Candidate` objects.


```
response
```




    response:
    GenerateContentResponse(
        done=True,
        iterator=None,
        result=protos.GenerateContentResponse({
          "candidates": [
            {
              "content": {
                "parts": [
                  {
                    "text": "Humans don't have tails because of evolutionary changes over millions of years.  Our primate ancestors had tails, but as humans evolved, the tail gradually disappeared.  The reasons aren't fully understood, but several theories exist:\n\n* **Loss of function:**  As our ancestors transitioned to bipedalism (walking upright), the function of a tail for balance and climbing diminished.  Natural selection favored individuals with smaller, less developed tails, as the energy needed to maintain a tail was no longer offset by its usefulness.  Essentially, it became an energetically expensive feature with little benefit.\n\n* **Genetic changes:**  Mutations affecting genes controlling tail development likely occurred and were favored by natural selection.  These mutations could have caused the tail to become smaller in successive generations until it was completely lost.  The coccyx (tailbone) is a vestigial structure \u2013 a remnant of our tailed ancestors.\n\n* **Developmental changes:**  Changes in the timing and regulation of genes involved in embryonic development may have led to the shortening and eventual disappearance of the tail.  The genes that once directed tail growth might have been altered to cease development of a tail at an early stage of embryonic growth.\n\nIt's important to note that these are interconnected factors.  The loss of tail function made it less crucial for survival, and genetic mutations that led to its reduced size and eventual disappearance were then naturally selected for.  The process happened gradually over a long period of evolutionary time.\n"
                  }
                ],
                "role": "model"
              },
              "finish_reason": "STOP",
              "avg_logprobs": -0.42928099152225774
            },
            {
              "content": {
                "parts": [
                  {
                    "text": "Humans don't have tails because of evolution.  Our ancestors did have tails, but over millions of years of evolution, the tail gradually became smaller and less functional until it was essentially absorbed into the body.  There's no single reason, but rather a combination of factors likely contributed:\n\n* **Loss of Functionality:** As our ancestors became bipedal (walking upright), the tail's primary function for balance and locomotion became less crucial.  Other adaptations, like changes in our skeletal structure and leg musculature, compensated for the loss of the tail's balancing role.\n\n* **Genetic Changes:**  Mutations that affected the genes controlling tail development accumulated over time.  These mutations might have been initially neutral or even slightly advantageous in other ways, and natural selection didn't actively remove them because the tail's importance diminished.\n\n* **Energy Conservation:**  Maintaining a tail requires energy.  As our ancestors transitioned to different environments and lifestyles, the energy cost of maintaining a tail may have become a disadvantage, especially in resource-scarce environments.  Those with less pronounced tails, or even the complete loss of tails, might have had a slight survival and reproductive advantage.\n\n* **Sexual Selection:**  It's possible that at some point, a tailless or nearly tailless phenotype became a desirable trait from a sexual selection perspective.  This is difficult to prove, but it's a factor considered in the evolution of various traits.\n\nIn short, the absence of a tail in humans is a result of a gradual evolutionary process where the tail's usefulness decreased, genetic changes accumulated, and natural selection favored individuals with less prominent tails.  The coccyx, the small bone at the base of our spine, is the remnant of our evolutionary tail.\n"
                  }
                ],
                "role": "model"
              },
              "finish_reason": "STOP",
              "index": 1,
              "avg_logprobs": -0.4348024821413156
            }
          ],
          "usage_metadata": {
            "prompt_token_count": 7,
            "candidates_token_count": 660,
            "total_token_count": 667
          },
          "model_version": "gemini-1.5-flash-002"
        }),
    )



## Penalties

The `002` models expose `penalty` arguments that let you affect the statistics of output tokens.

### Presence penalty

The `presence_penalty` penalizes tokens that have already been used in the output, so it induces variety in the model's output. This is detectible if you count the unique words in the output.

Here's a function to run a prompt a few times and report the fraction of unique words (words don't map perfectly to tokens but it's a simple way to see the effect).


```
from statistics import mean
```


```
def unique_words(prompt, generation_config, N=10):
  responses = []
  vocab_fractions = []
  for n in range(N):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(contents=prompt, generation_config=generation_config)
    responses.append(response)

    words = response.text.lower().split()
    score = len(set(words))/len(words)
    print(score)
    vocab_fractions.append(score)

  return vocab_fractions
```


```
prompt='Tell me a story'
```


```
# baseline
v = unique_words(prompt, generation_config={})
```

    [0.5698689956331878, 0.5426008968609866, 0.6184834123222749, 0.55741127348643, 0.6084070796460177, 0.545054945054945, 0.5891304347826087, 0.5920398009950248, 0.5663716814159292, 0.5831485587583148]



```
mean(v)
```




    0.577251707895572




```
# the penalty encourages diversity in the oputput tokens.
v = unique_words(prompt, generation_config=dict(presence_penalty=1.999))
```

    [0.6214833759590793, 0.5617529880478087, 0.5894495412844036, 0.5789473684210527, 0.5781990521327014, 0.6389684813753582, 0.6061320754716981, 0.5727482678983834, 0.5864485981308412, 0.565410199556541]



```
mean(v)
```




    0.5899539948277868




```
# a negative penalty discourages diversity in the output tokens.
v = unique_words(prompt, generation_config=dict(presence_penalty=-1.999))
```

    [0.5555555555555556, 0.6472148541114059, 0.5839598997493735, 0.6132075471698113, 0.5858369098712446, 0.5823389021479713, 0.5895691609977324, 0.5978021978021978, 0.5604166666666667, 0.5741626794258373]



```
mean(v)
```




    0.5890064373497796



The `presence_penalty` has a small effect on the vocabulary statistics.

### Frequency Penalty

Frequency penalty is similar to the `presence_penalty` but  the penalty is multiplied by the number of times a token is used. This effect is much stronger than the `presence_penalty`.

The easiest way to see that it works is to ask the model to do something repetitive. The model has to get creative while trying to complete the task.


```
model = genai.GenerativeModel(model_name)
response = model.generate_content(contents='please repeat "Cat" 50 times, 10 per line',
                                  generation_config=dict(frequency_penalty=1.999))
```


```
print(response.text)
```

    Cat Cat Cat Cat Cat Cat Cat Cat Cat Cat
    Cat Cat Cat Cat Cat Cat Cat Cat CaT CaT
    Cat cat cat cat cat cat cat cat cat cat
    Cat Cat Cat Cat Cat Cat Cat Cat Cat Cat
    Cat Cat Cat Cat Cat Cat Cat CaT CaT CaT
    Cat cat cat cat cat cat cat cat cat cat
    Cat Cat Cat Cat Cat Cat Cat CaT CaT CaT
    Cat CAT CAT CAT CAT CAT cAT cAT cAT CA
    t Cat Cat Cat Cat cat cat cat cat CAT
    


Since the frequency penalty accumulates with usage, it can have a much stronger effect on the output compared to the presence penalty.

> Caution: Be careful with negative frequency penalties: A negative penalty makes a token more likely the more it's used. This positive feedback quickly leads the model to just repeat a common token until it hits the `max_output_tokens` limit (once it starts the model can't produce the `<STOP>` token).


```
response = model.generate_content(
    prompt,
    generation_config=genai.GenerationConfig(
        max_output_tokens=400,
        frequency_penalty=-2.0))
```


```
Markdown(response.text)  # the, the, the, ...
```




Elara, a wisp of a girl with eyes the colour of a stormy sea, lived in a lighthouse perched precariously on the edge of the Whispering Cliffs.  Her only companions were the relentless rhythm of the waves and the lonely cries of the gulls.  Her father, the lighthouse keeper, was a man of the sea, his face etched with the map of the ocean's moods.  He’d taught her the language of the waves, the the the the way the wind whispered secrets to the rocks, and the constellations that guided lost ships home.

One day, a storm unlike any Elara had ever seen descended.  The lighthouse shuddered, the wind howled like a banshee, and the waves crashed against the cliffs with the fury of a thousand angry giants.  During the tempest, a ship, its masts splintered and its sails ripped, was tossed onto the rocks below.  Elara’s father, his face grim, prepared his small, sturdy boat, defying the monstrous waves to reach the stricken vessel.

He never returned.

Days bled into weeks.  Elara, her heart a frozen wasteland, kept the light burning, a tiny, defiant flame against the overwhelming darkness.  She scanned the horizon every day, hoping, praying, for a sign, a glimpse of a familiar sail, a flicker of a known light.

One evening, a faint, almost imperceptible glow appeared on the horizon.  It was weak, flickering, but undeniably there.  It was a signal, a desperate plea for help.  Elara, her heart pounding, launched her father’s boat, her small form a mere speck against the immensity of the ocean.

The storm, though, had subsided.  The sea was calm. The glow was guiding.

She reached the ship, a small fishing trawler, battered, but afloat.  A lone figure, an old woman with silver hair, lay clinging to the




```
response.candidates[0].finish_reason
```




    <FinishReason.MAX_TOKENS: 2>
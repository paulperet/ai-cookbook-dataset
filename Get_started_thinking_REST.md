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

When making REST API calls, you control the thinking behavior by including a thinkingConfig object within the generationConfig in your JSON request payload.

[Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-preview-04-17) in particular, brings the flexibility of using `thinkingBudget` - a parameter
that offers fine-grained control over the maximum number of tokens a model can generate while thinking. Alternatively, you can designate a precise token allowance for the
"thinking" stage through the adjusment of the `thinkingBudget` parameter. This allowance can vary between 0 and 24576 tokens for 2.5 Flash.

For more information about all Gemini models, check the [documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for extended information on each of them.

On this notebook most examples are using the `thinkingBudget` parameter since it's compatible with both the 2.5 and the 3 generations of models. For more information about using the `thinkingBudget` with the Gemini thinking model, check the [documentation](https://ai.google.dev/gemini-api/docs/thinking).

**NEW: thinking levels:** [Gemini 3 Pro](https://ai.google.dev/gemini-api/docs/models#gemini-3-pro) introduced a new, easier way to manage the thinking buget by setting a `thinkingLevel` that is documented in the [section of this guide dedicated to Gemini 3](#gemini3).

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
```


```
import json
import requests

MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={GOOGLE_API_KEY}"
```

# Examples

Here are some quite complex examples of what Gemini **thinking** model can solve.

In each of them you can select different models to see how this new model compares to its predecesors.

In some cases, you'll still get the good answer from the other models, in that case, re-run it a couple of times and you'll see that Gemini 2.5 thinking is more consistent thanks to its thinking step.

## Example #1: Brain Teaser with Thinking

You can start by asking the model to explain a concept and see how it does reasoning before answering.

Starting with the adaptive `thinkingBudget` - which is the default when you don't specify a budget - the model will dynamically adjust the budget based on the complexity of the request.

`includeThoughts` tells the model to include its thoughts in the output.


```
prompt = """
    You are playing the 20 question game. You know that what you are looking for
    is a aquatic mammal that doesn't live in the sea, and that's smaller than a
    cat. What could that be and how could you make sure?
"""
# The animal I'm thinking of is a platipus

data = {
    "contents": [
        {
            "parts": [
                {"text": prompt}
            ]
        }
    ],
    "generationConfig": {
          "thinkingConfig": {
              "includeThoughts": True,
          }
    }
}

response = requests.post(
   url,
   headers={'Content-Type': 'application/json'},
   data=json.dumps(data)
).json()

print("THOUGHTS:")
print(response['candidates'][0]['content']['parts'][0]['text'])
print()
print("OUTPUT:")
print(response['candidates'][0]['content']['parts'][1]['text'])
```

    THOUGHTS:
    **Here's how I'm thinking about this:**
    
    Okay, so we've got an aquatic mammal, freshwater, smaller than a cat. My mind immediately starts running through the usual suspects. I'm going through the marine mammals first, ruling them out: whales, dolphins, sea lions... nope, definitely not. Manatees and dugongs, too big. Then, I start on the freshwater side. Otters are a possibility, though some are much bigger. Beavers? Too big. Muskrat and Mink are looking promising, and I'm adding platypus because it's aquatic and I need to be inclusive. Water voles, various shrews, and a quick check on the hippo (nope, obviously).
    
    Now, refining based on size, the really strong candidates emerge: the water shrew and vole are ideal. The platypus, being unique, is a great fit. Mink also look good. Muskrat... borderline but possible. I'll include it to show I thought it through.
    
    Based on the information, I'll go with the Water Shrew, Water Vole, Platypus, and Mink, as distinct options. They cover the space and show I've done a proper job.
    
    Now, how would I confirm this? I start formulating those 20-questions style questions. The Platypus gets a special question: "Does it lay eggs?" because it's the most unusual. I'd ask a yes/no question to differentiate the different candidates and confirm the key features.  I want to narrow down the options with each question, so I'm thinking about webbed feet, tail type, habitat, and overall size. I need to design the best strategy with the fewest questions.
    
    
    OUTPUT:
    This is a fun one! Let's break down the clues:
    
    1.  **Aquatic mammal:** Lives in or around water, has fur, breathes air, nurses young.
    2.  **Doesn't live in the sea:** Exclusively freshwater (rivers, lakes, ponds, swamps).
    3.  **Smaller than a cat:** A domestic cat is generally 8-10 lbs (3.6-4.5 kg) and about 18-24 inches (45-60 cm) long including the tail.
    
    Given these clues, here are a few strong possibilities:
    
    ---
    
    ### What could that be?
    
    1.  **Water Shrew:**
        *   **Fits:** These are very small (much smaller than a cat, often only a few inches long), highly aquatic (excellent swimmers and divers), feed on aquatic insects and small fish, and live in freshwater habitats across many continents.
    
    2.  **Water Vole:**
        *   **Fits:** Roughly the size of a large rat (smaller than most cats), strong swimmers, live in freshwater rivers, streams, and ponds, primarily herbivorous but will eat insects. Common in Europe and parts of Asia.
    
    3.  **Platypus:**
        *   **Fits:** Unique semi-aquatic mammal (monotreme, meaning it lays eggs), found exclusively in eastern Australia (freshwater rivers and lakes), and is smaller than a cat (typically 12-15 inches/30-38 cm long, not including the tail).
    
    4.  **Mink:**
        *   **Fits:** A semi-aquatic weasel, smaller than a cat (body length usually 12-18 inches/30-45 cm), excellent swimmers and hunters of fish, frogs, and other small prey in freshwater environments.
    
    ---
    
    ### How could you make sure? (20 Questions style - Yes/No answers)
    
    To narrow it down and confirm, you'd ask questions that target the specific characteristics of these animals:
    
    1.  **Is it native to Australia?** (Yes = Platypus)
        *   *If yes, you've likely found it. If no, proceed...*
    
    2.  **Does it have webbed feet?** (Yes = Platypus, Water Shrew, Mink (partially), some Water Voles have fringed feet)
        *   *This helps rule out things like land mammals that just drink from water.*
    
    3.  **Does it belong to the rodent family?** (Yes = Water Vole, Muskrat - if considering muskrat)
        *   *This would rule out shrews, minks, and platypus.*
    
    4.  **Is its diet primarily insects and small invertebrates?** (Yes = Water Shrew, Platypus to some extent)
        *   *This would lean away from the more herbivorous water vole or the more generalist mink.*
    
    5.  **Does it have a long, pointed snout with whiskers?** (Yes = Water Shrew)
        *   *Distinctive feature of shrews.*
    
    6.  **Does it have a short, blunt snout and round ears almost hidden in its fur?** (Yes = Water Vole)
        *   *Distinguishes it from shrews and minks.*
    
    7.  **Is it known for being a skilled hunter of fish and frogs?** (Yes = Mink, Water Shrew)
        *   *Mink are formidable predators.*
    
    8.  **Is its body typically less than 6 inches (15 cm) long (excluding the tail)?** (Yes = Water Shrew)
        *   *This is a size check that firmly places it as very small.*
    
    9.  **Does it belong to the weasel family?** (Yes = Mink)
        *   *A direct family identification.*
    
    10. **Is it active year-round, even in freezing conditions?** (Many of these are, but can be a differentiator for specific species behavior).
    
    11. **Does it spend a significant amount of time burrowing in riverbanks?** (Yes = Water Vole, Platypus, Mink (for dens))
    
    12. **Is it generally considered solitary?** (Many small aquatic mammals are).
    
    By asking these types of questions, you'd quickly narrow down whether it's a water shrew, water vole, platypus, or mink, or another similar small freshwater aquatic mammal.



```
print(response)
```

    {'error': {'code': 400, 'message': 'Invalid JSON payload received. Unknown name "includeThoughts" at \'generation_config\': Cannot find field.', 'status': 'INVALID_ARGUMENT', 'details': [{'@type': 'type.googleapis.com/google.rpc.BadRequest', 'fieldViolations': [{'field': 'generation_config', 'description': 'Invalid JSON payload received. Unknown name "includeThoughts" at \'generation_config\': Cannot find field.'}]}]}}


Inspecting the Response Metadata: After making the REST call, the response JSON contains usageMetadata. This object provides information about the token counts for the request. Look for the `thoughtsTokenCount` field within usageMetadata to see how many tokens were consumed by the thinking process for this request. You'll also see `promptTokenCount`, `candidatesTokenCount` (for the final output), and `totalTokenCount`. As you can see here, the model used a significant number of tokens in the thinking steps.


```
print("Prompt tokens:",response["usageMetadata"]["promptTokenCount"])
print("Thoughts tokens:",response["usageMetadata"]["thoughtsTokenCount"])
print("Output tokens:", response["usageMetadata"]["candidatesTokenCount"])
print("Total tokens:", response["usageMetadata"]["totalTokenCount"])
```

    Prompt tokens: 58
    Thoughts tokens: 1477
    Output tokens: 820
    Total tokens: 2355


### Disabling the thinking steps

You can explicitly disable the thinking steps by including the `thinkingConfig` object in the `generationConfig` and setting the `thinkingBudget` parameter to `0` in the JSON payload. This tells the model not to perform any internal reasoning steps before generating the final output. You'll likely see that in this case, the model doesn't think of the platypus as a possible answer.

Note that you can't disable thinking on pro models.


```
if "-pro" not in MODEL_ID:
  prompt = """
      You are playing the 20 question game. You know that what you are looking for
      is a aquatic mammal that doesn't live in the sea, and that's smaller than a
      cat. What could that be and how could you make sure?
  """

  data = {
      "contents": [
          {
              "parts": [
                  {"text": prompt}
              ]
          }
      ],
      "generationConfig": {
            "thinkingConfig": {
                "thinkingBudget": 0,
            }
      }
  }

  response = requests.post(
    url,
    headers={'Content-Type': 'application/json'},
    data=json.dumps(data)
  ).json()

  print(response['candidates'][0]['content']['parts'][0]['text'])

else
  print("You can't disable thinking for pro models.")
```

    This is a fun challenge! Let's break it down.
    
    **The Object:** A aquatic mammal that doesn't live in the sea, and that's smaller than a cat.
    
    **Possible Candidate:** The most likely candidate that fits all those criteria is a **river otter pup**.
    
    **Why a River Otter Pup?**
    
    *   **Aquatic Mammal:** Yes, otters are well-known aquatic mammals.
    *   **Doesn't Live in the Sea:** While some otters live in coastal areas, river otters specifically inhabit freshwater environments like rivers, lakes, and marshes.
    *   **Smaller than a Cat:** An adult river otter is definitely larger than a cat. However, a newborn or very young **pup** (baby otter) would be significantly smaller than a domestic cat. They are born blind and helpless, and quite tiny.
    
    **How to Make Sure (Your 20 Questions Strategy):**
    
    To confirm this, you'd use your 20 questions to narrow down the possibilities. Here's a possible line of questioning, aiming to eliminate other options and pinpoint the river otter pup:
    
    1.  **Is it an animal?** (Yes - establishes the category)
    2.  **Is it a mammal?** (Yes - narrows it significantly)
    3.  **Is it aquatic?** (Yes - crucial qualifier)
    4.  **Does it primarily live in freshwater?** (Yes - this eliminates sea mammals like dolphins, seals, whales, sea otters, etc., and focuses on rivers, lakes, ponds)
    5.  **Is it native to North America?** (Yes - helps narrow down specific species of freshwater aquatic mammals)
    6.  **Does it have fur?** (Yes - separates it from some amphibians or fish if there was confusion)
    7.  **Is it known for being playful?** (Yes - strong hint towards otters)
    8.  **Does it belong to the weasel family (Mustelidae)?** (Yes - This is a powerful question as it directly points to otters, badgers, minks, ferrets, etc. among aquatic mammals, otters are the obvious choice here)
    9.  **Does it eat fish, crustaceans, and amphibians?** (Yes - typical otter diet)
    10. **Does it build dens called "holts"?** (Yes - confirms it's an otter, not a beaver or muskrat)
    11. **Is an adult of this species typically larger than a domestic cat?** (Yes - this is key to setting up the next question)
    12. **Are we referring to a *baby* or *juvenile* of this species?** (Yes - This is the crucial question that explains why it's "smaller than a cat" despite the adult being larger.)
    13. **Is it born helpless and blind?** (Yes - characteristic of otter pups)
    14. **Does it learn to swim from its mother?** (Yes - typical otter pup behavior)
    15. **Is it called a "pup"?** (Yes - common term for baby otters)
    
    At this point, you've almost certainly confirmed it's a river otter pup. You could ask a few more specific questions if you wanted to be absolutely sure, but the combination of freshwater aquatic mammal, Mustelidae family, playful, and specifically being a *baby* when the adult is larger, makes the river otter pup an incredibly strong fit.
    
    **Other Less Likely but Possible Considerations (and why they're not as good):**
    
    *   **Muskrat pup:** Muskrats are aquatic, freshwater, and pups are smaller than a cat. However, they are rodents, not typically considered "playful," and less commonly thought of as a primary "aquatic mammal" in the same way an otter is (
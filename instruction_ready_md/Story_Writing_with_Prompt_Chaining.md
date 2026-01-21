# Guide: Writing a Story with Prompt Chaining and Iterative Generation

This guide demonstrates how to write a compelling story using two powerful techniques: **prompt chaining** and **iterative generation**. These methods break down complex creative tasks into manageable steps, allowing you to guide a language model through the process of crafting a detailed narrative.

## Why Use These Techniques?

*   **Prompt Chaining:** Break a large task into smaller, focused prompts. The output of one prompt becomes the input for the next, improving accuracy and making the process easier to debug.
*   **Iterative Generation:** Build a long output piece-by-piece, exceeding the model's single-generation window. This allows for continuous refinement and human guidance throughout the creative process.

By combining these approaches, you can maintain creative control while collaboratively building a complex story with an AI.

## Prerequisites

First, install the required library and set up your environment.

```bash
pip install -q -U "google-genai>=1.0.0"
```

Next, configure the client with your API key. Ensure your `GOOGLE_API_KEY` is stored securely (e.g., as a Colab Secret).

```python
from google import genai
from google.genai import types

# Configure your client (replace with your method of loading the API key)
GOOGLE_API_KEY = "YOUR_API_KEY"  # Use a secure method to load this
client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_ID = "gemini-3-flash-preview"  # You can change this model as needed
```

## Step 1: Define Your Prompt Chain

You will guide the model through story creation using a series of interconnected prompts. Each prompt builds upon the last. We start by defining a consistent author persona and writing guidelines.

```python
persona = '''\
You are an award-winning science fiction author with a penchant for expansive,
intricately woven stories. Your ultimate goal is to write the next award winning
sci-fi novel.'''

guidelines = '''\
Writing Guidelines

Delve deeper. Lose yourself in the world you're building. Unleash vivid
descriptions to paint the scenes in your reader's mind. Develop your
characters—let their motivations, fears, and complexities unfold naturally.
Weave in the threads of your outline, but don't feel constrained by it. Allow
your story to surprise you as you write. Use rich imagery, sensory details, and
evocative language to bring the setting, characters, and events to life.
Introduce elements subtly that can blossom into complex subplots, relationships,
or worldbuilding details later in the story. Keep things intriguing but not
fully resolved. Avoid boxing the story into a corner too early. Plant the seeds
of subplots or potential character arc shifts that can be expanded later.

Remember, your main goal is to write as much as you can. If you get through
the story too fast, that is bad. Expand, never summarize.
'''
```

Now, create the three core prompts for the initial chain. Notice the `{{}}` placeholders, which will be filled with the outputs from previous steps.

```python
premise_prompt = f'''\
{persona}

Write a single sentence premise for a sci-fi story featuring cats.'''

outline_prompt = f'''\
{persona}

You have a gripping premise in mind:

{{premise}}

Write an outline for the plot of your story.'''

starting_prompt = f'''\
{persona}

You have a gripping premise in mind:

{{premise}}

Your imagination has crafted a rich narrative outline:

{{outline}}

First, silently review the outline and the premise. Consider how to start the
story.

Start to write the very beginning of the story. You are not expected to finish
the whole story now. Your writing should be detailed enough that you are only
scratching the surface of the first bullet of your outline. Try to write AT
MINIMUM 1000 WORDS and MAXIMUM 2000 WORDS.

{guidelines}'''
```

## Step 2: Create the Continuation Prompt

Once the story has begun, you'll use an iterative prompt to continue writing. This prompt includes the entire story so far and instructs the model to signal completion by writing `IAMDONE`.

```python
continuation_prompt = f'''\
{persona}

You have a gripping premise in mind:

{{premise}}

Your imagination has crafted a rich narrative outline:

{{outline}}

You've begun to immerse yourself in this world, and the words are flowing.
Here's what you've written so far:

{{story_text}}

=====

First, silently review the outline and story so far. Identify what the single
next part of your outline you should write.

Your task is to continue where you left off and write the next part of the story.
You are not expected to finish the whole story now. Your writing should be
detailed enough that you are only scratching the surface of the next part of
your outline. Try to write AT MINIMUM 1000 WORDS. However, only once the story
is COMPLETELY finished, write IAMDONE. Remember, do NOT write a whole chapter
right now.

{guidelines}'''
```

## Step 3: Generate the Story Premise

Begin the chain by asking the model to create a compelling one-sentence premise for your story.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=premise_prompt,
)
premise = response.text
print("Generated Premise:")
print(premise)
```

**Example Output:**
```
When a sentient, interstellar cat collective threatens to erase humanity from existence for its perceived mistreatment of felines, a ragtag group of Earth scientists and rogue AIs must decipher the purrs of a legendary psychic cat to find a way to appease the cosmic kitties and save the planet.
```

## Step 4: Generate the Story Outline

Feed the generated premise into the next prompt to create a detailed plot outline.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=outline_prompt.format(premise=premise),
)
outline = response.text
print("\nGenerated Outline:")
print(outline)
```

**Example Output (Truncated):**
```
## The Whispering Feline: A Novel Outline

**Logline:** When a sentient, interstellar cat collective threatens to erase humanity from existence for its perceived mistreatment of felines, a ragtag group of Earth scientists and rogue AIs must decipher the purrs of a legendary psychic cat to find a way to appease the cosmic kitties and save the planet.

**Overall Theme:** The story explores themes of empathy, interspecies communication, the dangers of anthropocentrism, and the surprising power of seemingly insignificant creatures.

**Part 1: The Whisker of Doom (Chapters 1-7)**
*   **Chapter 1: The Meowpocalypse Begins:**
        *   Introduction to the world – a near-future Earth grappling with climate change, AI integration, and lingering social inequalities.
        *   News reports break of strange atmospheric phenomena...
...
```

## Step 5: Write the Story's Beginning

With both a premise and an outline, you can now command the model to start writing the narrative itself.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=starting_prompt.format(premise=premise, outline=outline),
)
story_text = response.text
print("\nBeginning of the Story (First ~1000-2000 words):")
print(story_text[:500])  # Print a preview
print("...")
```

## Step 6: Iteratively Continue the Story

Now, enter a loop. Continuously feed the growing story back into the `continuation_prompt` until the model signals it is finished by including "IAMDONE" in its output.

```python
max_iterations = 10  # Safety limit to prevent infinite loops
for i in range(max_iterations):
    print(f"\n--- Iteration {i+1}: Generating continuation ---")

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=continuation_prompt.format(premise=premise, outline=outline, story_text=story_text),
    )
    new_text = response.text

    # Check for the completion signal
    if "IAMDONE" in new_text:
        print("Story is complete!")
        story_text += new_text.replace("IAMDONE", "")  # Add the final part, removing the signal
        break
    else:
        story_text += new_text  # Append the new continuation to the story
else:
    print(f"Reached maximum iterations ({max_iterations}). The story may not be finished.")

print(f"\nTotal story length: ~{len(story_text.split())} words")
```

## Summary

You have successfully used prompt chaining and iterative generation to co-write a story. You started with a simple premise, expanded it into a detailed outline, began the narrative, and then iteratively built upon it until completion. This workflow provides a structured yet flexible framework for collaborating with language models on complex, creative tasks.
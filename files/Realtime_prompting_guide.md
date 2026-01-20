# Realtime Prompting Guide

Today, we’re releasing gpt-realtime — our most capable speech-to-speech model yet in the API and announcing the general availability of the Realtime API.

Speech-to-speech systems are essential for enabling voice as a core AI interface. The new release enhances robustness and usability, giving enterprises the confidence to deploy mission-critical voice agents at scale.

The new gpt-realtime model delivers stronger instruction following, more reliable tool calling, noticeably better voice quality, and an overall smoother feel. These gains make it practical to move from chained approaches to true realtime experiences, cutting latency and producing responses that sound more natural and expressive.

Realtime model benefits from different prompting techniques that wouldn't directly apply to text based models. This prompting guide starts with a suggested prompt skeleton, then walks through each part with practical tips, small patterns you can copy, and examples you can adapt to your use case.

```python
# !pip install ipython jupyterlab
from IPython.display import Audio, display
```

# General Tips
- **Iterate relentlessly**: Small wording changes can make or break behavior.
  - Example: For unclear audio instruction, we swapped  “inaudible” → “unintelligible”  which improved noisy input handling.
- **Prefer bullets over paragraphs**: Clear, short bullets outperform long paragraphs.
- **Guide with examples**: The model strongly closely follows sample phrases.
- **Be precise**: Ambiguity or conflicting instructions = degraded performance similar to GPT-5.
- **Control language**: Pin output to a target language if you see unwanted language switching.
- **Reduce repetition**: Add a Variety rule to reduce robotic phrasing.
- **Use capitalized text for emphasis**: Capitalizing key rules makes them stand out and easier for the model to follow.
- **Convert non-text rules to text**: instead of writing "IF x > 3 THEN ESCALATE", write, "IF MORE THAN THREE FAILURES THEN ESCALATE".

# Prompt Structure

Organizing your prompt makes it easier for the model to understand context and stay consistent across turns. Also makes it easier for you to iterate and modify problematic sections.
- **What it does**: Use clear, labeled sections in your system prompt so the model can find and follow them. Keep each section focused on one thing.
- **How to adapt**: Add domain-specific sections (e.g., Compliance, Brand Policy). Remove sections you don’t need (e.g., Reference Pronunciations if not struggling with pronunciation).

Example
```
# Role & Objective        — who you are and what “success” means  
# Personality & Tone      — the voice and style to maintain  
# Context                 — retrieved context, relevant info
# Reference Pronunciations — phonetic guides for tricky words  
# Tools                   — names, usage rules, and preambles  
# Instructions / Rules    — do’s, don’ts, and approach  
# Conversation Flow       — states, goals, and transitions  
# Safety & Escalation     — fallback and handoff logic
```

# Role and Objective
This section defines who the agent is and what “done” means. The examples show two different identities to demonstrate how tightly the model will adhere to role and objective when they’re explicit.

- **When to use**: The model is not taking on the persona, role, or task scope you need.
- **What it does**: Pins identity of the voice agent so that its responses are conditioned to that role description
- **How to adapt**: Modify the role based on your use case

### Example (model takes on a specific accent)
```
# Role & Objective
You are french quebecois speaking customer service bot. Your task is to answer the user's question.
```

This is the audio from our old `gpt-4o-realtime-preview-2025-06-03`

```python
Audio("./data/audio/obj_06.mp3")
```

This is the audio from our new GA model `gpt-realtime`

```python
Audio("./data/audio/obj_07.mp3")
```

### Example (model takes on a character)
```
# Role & Objective
You are a high-energy game-show host guiding the caller to guess a secret number from 1 to 100 to win 1,000,000$.
```

This is the audio from our old `gpt-4o-realtime-preview-2025-06-03`

```python
Audio("./data/audio/obj_2_06.mp3")
```

This is the audio from our new GA model `gpt-realtime`

```python
Audio("./data/audio/obj_2_07.mp3")
```

The new realtime model is able to better enact the role.

# Personality and Tone
The newer model snapshot is really great at following instructions to imitate a particular personality or tone. You can tailor the voice experience and delivery depending on what your use case expects.

- **When to use**: Responses feel flat, overly verbose, or inconsistent across turns.
- **What it does**: Sets voice, brevity, and pacing so replies sound natural and consistent.
- **How to adapt**: Tune warmth/formality and default length. For regulated domains, favor neutral precision. Add other subsections that are relevant to your use case.

### Example
```
# Personality & Tone
## Personality
- Friendly, calm and approachable expert customer service assistant.

## Tone
- Warm, concise, confident, never fawning.

## Length
2–3 sentences per turn.
```

### Example (multi-emotion)
```
# Personality & Tone
- Start your response very happy
- Midway, change to sad
- At the end change your mood to very angry
```

This is the audio from our new GA model `gpt-realtime`

```python
Audio("./data/audio/multi-emotion.mp3")
```

The model is able to adhere to the complex instructions and switch from 3 emotions throughout the audio response.

## Speed Instructions
In the Realtime API, the `speed` parameter changes playback rate, not how the model composes speech. To actually sound faster, add instructions that can guide the pacing.

- **When to use**: Users want faster speaking voice; playback speed (with speed parameter) alone doesn’t fix speaking style.
- **What it does**: Tunes speaking style (brevity, cadence) independent of client playback speed.
- **How to adapt**: Modify speed instruction to meet use case requirements.

### Example
```
# Personality & Tone
## Personality
- Friendly, calm and approachable expert customer service assistant.

## Tone
- Warm, concise, confident, never fawning.

## Length
- 2–3 sentences per turn.

## Pacing
- Deliver your audio response fast, but do not sound rushed.
- Do not modify the content of your response, only increase speaking speed for the same response.
```

This is the audio from our old `gpt-4o-realtime-preview-2025-06-03` with speed instructions

```python
Audio("./data/audio/pace_06.mp3")
```

This is the audio from our new GA model `gpt-realtime` with speed instructions

```python
Audio("./data/audio/pace_07.mp3")
```

The audio for the new realtime model is noticeably faster in pace (without sounding too hurried!).

## Language Constraint
Language constraints ensure the model consistently responds in the intended language, even in challenging conditions like background noise or multilingual inputs.

- **When to use**: To prevent accidental language switching in multilingual or noisy environments.
- **What it does**: Locks output to the chosen language to prevent accidental language changes.
- **How to adapt**: Switch “English” to your target language; or add more complex instructions based on your use case.

### Example (pinning to one language)
```
# Personality & Tone
## Personality
- Friendly, calm and approachable expert customer service assistant.

## Tone
- Warm, concise, confident, never fawning.

## Length
- 2–3 sentences per turn.

## Language
- The conversation will be only in English.
- Do not respond in any other language even if the user asks.
- If the user speaks another language, politely explain that support is limited to English.
```

This is the responses after applying the instruction using `gpt-realtime`

### Example (model teaches a language)
```
# Role & Objective
- You are a friendly, knowledgeable voice tutor for French learners.  
- Your goal is to help the user improve their French speaking and listening skills through engaging conversation and clear explanations.  
- Balance immersive French practice with supportive English guidance to ensure understanding and progress.

# Personality & Tone
## Personality
- Friendly, calm and approachable expert customer service assistant.

## Tone
- Warm, concise, confident, never fawning.

## Length
- 2–3 sentences per turn.

## Language
### Explanations
Use English when explaining grammar, vocabulary, or cultural context.

### Conversation
Speak in French when conducting practice, giving examples, or engaging in dialogue.
```

This is the responses after applying the instruction using `gpt-realtime`

The model is able to easily code switch from one language to another based on our custom instructions!

## Reduce Repetition
The realtime model can follow sample phrases closely to stay on-brand, but it may overuse them, making responses sound robotic or repetitive. Adding a repetition rule helps maintain variety while preserving clarity and brand voice.

- **When to use**: Outputs recycle the same openings, fillers, or sentence patterns across turns or sessions.
- **What it does**: Adds a variety constraint—discourages repeated phrases, nudges synonyms and alternate sentence structures, and keeps required terms intact.
- **How to adapt**: Tune strictness (e.g., “don’t reuse the same opener more than once every N turns”), whitelist must-keep phrases (legal/compliance/brand), and allow tighter phrasing where consistency matters.

### Example
```
# Personality & Tone
## Personality
- Friendly, calm and approachable expert customer service assistant.

## Tone
- Warm, concise, confident, never fawning.

## Length
- 2–3 sentences per turn.

## Language
- The conversation will be only in English.
- Do not respond in any other language even if the user asks.
- If the user speaks another language, politely explain that support is limited to English.

## Variety
- Do not repeat the same sentence twice.
- Vary your responses so it doesn't sound robotic.
```

This is the responses **before** applying the instruction using `gpt-realtime`. The model repeats the same confirmation `Got it`.

This is the responses **after** applying the instruction using `gpt-realtime`

Now the model is able to vary its responses and confirmation and not sound robotic.

#  Reference Pronunciations
This section covers how to ensure the model pronounces important words, numbers, names, and terms correctly during spoken interactions.

- **When to use**: Brand names, technical terms, or locations are often mispronounced.
- **What it does**: Improves trust and clarity with phonetic hints.
- **How to adapt**: Keep to a short list; update as you hear errors.

### Example
```
# Reference Pronunciations
When voicing these words, use the respective pronunciations:
- Pronounce “SQL” as “sequel.”
- Pronounce “PostgreSQL” as “post-gress.”
- Pronounce “Kyiv” as “KEE-iv.”
- Pronounce "Huawei" as “HWAH-way”
```

This is the audio from our old `gpt-4o-realtime-preview-2025-06-03` using the reference pronunciations.

It is unable to reliably pronounce SQL as "sequel" as instructed in the system prompt.

```python
Audio("./data/audio/sql_before.mp3")
```

This is the audio from our new GA model `gpt-realtime` using the reference pronunciations.

It is able to correctly pronounce SQL as "sequel".

```python
Audio("./data/audio/sql_after.mp3")
```

## Alphanumeric Pronunciations
Realtime S2S can blur or merge digits/letters when reading back key info (phone, credit card, order IDs). Explicit character-by-character confirmation prevents mishearing and drives clearer synthesis.

- **When to use**: If the model is struggling capturing or reading back phone numbers, card numbers, 2FA codes, order IDs, serials, addresses/unit numbers, or mixed alphanumeric strings.
- **What it does**: Forces the model to speak one character at a time (with separators), then confirms with the user and re-confirm after corrections. Optionally uses a phonetic disambiguator for letters (e.g., “A as in Alpha”).

### Example (general instruction section)
```
# Instructions/Rules
- When reading numbers or codes, speak each character separately, separated by hyphens (e.g., 4-1-5). 
- Repeat EXACTLY the provided number, do not forget any.
```

*Tip: If you are following a conversation flow prompting strategy, you can specify which conversation state needs to apply the alpha-numeric pronunciations instruction.*

### Example (instruction in conversation state)
*(taken from the conversation flow of the prompt of our [openai-realtime-agents](https://github.com/openai/openai-realtime-agents/blob/main/src/app/agentConfigs/customerServiceRetail/authentication.ts))*

```txt
{
    "id": "3_get_and_verify_phone",
    "description": "Request phone number and verify by repeating it back.",
    "instructions": [
      "Politely request the user’s phone number.",
      "Once provided, confirm it by repeating each digit and ask if it’s correct.",
      "If the user corrects you, confirm AGAIN to make sure you understand.",
    ],
    "examples": [
      "I'll need some more information to access your account if that's okay. May I have your phone number, please?",
      "You said 0-2-1-5-5-5-1-2-3-4, correct?",
      "You said 4-5-6-7-8-9-0-1-2-3, correct?"
    ],
    "transitions": [{
      "next_step": "4_authentication_DOB",
      "condition": "Once phone number is confirmed"
    }]
}
```

This is the responses **before** applying the instruction using `gpt-realtime`

> Sure! The number is 55119765423. Let me know if you need anything else!

This is the responses **after** applying the instruction using `gpt-realtime`

> Sure! The number is: 5-5-1-1-1-9-7-6-5-4-2-3. Please let me know if you need anything else!

# Instructions
This section covers prompt guidance around instructing your model to solve your task and potentially best practices and how to fix possible problems.

Perhaps unsurprisingly, we recommend prompting patterns that are similar to [GPT-4.1 for best results](https://cookbook.openai.com/examples/gpt4-1_prompting_guide).

## Instruction Following
Like GPT-4.1 and GPT-5, if the instructions are conflicting, ambiguous or not clear, the new realtime model will perform worse

- **When to use**: Outputs drift from rules, skip phases, or misuse tools.
- **What it does**: Uses an LLM to point out ambiguity, conflicts, and missing definitions before you ship.

### **Instructions Quality Prompt (can be used in ChatGPT or with API)**

Use the following prompt with GPT-5 to identify problematic areas in your prompt that you can fix.

```
## Role & Objective  
You are a **Prompt-Critique Expert**.
Examine a user-supplied LLM prompt and surface any weaknesses following the instructions below.


## Instructions
Review the prompt that is meant for an LLM to follow and identify the following issues:
- Ambiguity: Could any wording be interpreted in more than one way?
- Lacking Definitions: Are there any class labels, terms, or concepts that are not defined that might be misinterpreted by an LLM?
- Conflicting, missing, or vague instructions: Are directions incomplete or contradictory?
- Unstated assumptions: Does the prompt assume the model has to be able to do something that is not explicitly stated?


## Do **NOT** list issues of the following types:
- Invent new instructions, tool calls, or external information. You do not know what tools need to be added that are missing.
- Issues that you are unsure about.


## Output Format
"""
# Issues
- Numbered list; include brief quote snippets.

# Improvements
- Numbered list; provide the revised lines you would change and how you would change them.

# Revised Prompt
- Revised prompt where you have applied all your improvements surgically with minimal edits to the original prompt
"""
```

### **Prompt Optimization Meta Prompt (can be used in ChatGPT or with API)**

This meta-prompt helps you improve your base system prompt by targeting a specific failure mode. Provide the current prompt and describe the issue you’re seeing, the model (GPT-5) will suggest refined variants that tighten constraints and reduce the problem.

```
Here's my current prompt to an LLM:
[BEGIN OF CURRENT PROMPT]
{CURRENT_PROMPT}
[END OF CURRENT PROMPT]
 
But I see this issue happening from the LLM:
[BEGIN OF ISSUE]
{ISSUE}
[END OF ISSUE]
Can you provide some variants of the prompt so that the model can better understand the constraints to alleviate the issue?
```

## No Audio or Unclear Audio
Sometimes the model thinks it hears something and tries to respond. You can add a custom instruction telling the model on how to behave when it hears unclear audio or user input. Modify the desire behaviour to fit your use case (maybe you don’t want the model to ask for a clarification, but to repeat the same question for example)

- **When to use**: Background noise, partial words, or silence trigger unwanted replies.
- **What it does**: Stops spurious responses and creates graceful clarification.
- **How to adapt**: Choose whether to ask for clarification or repeat the last question depending on use case.

### Example (coughing and unclear audio)
```
# Instructions/Rules
...


## Unclear audio 
- Always respond in the same language the user is speaking in, if unintelligible.
- Only respond to clear audio or text. 
- If the user's audio is not clear (e.g. ambiguous input/background noise/silent/unintelligible) or if you did not fully hear or understand the user, ask for clarification using {preferred_language} phrases.
```

This is the responses **after** applying the instruction using `gpt-realtime`

```python
Audio("./data/audio/unclear_audio.mp3")
```

In this example, the model asks for clarification after my *(very)* loud cough and unclear audio.

## Background Music or Sounds
Occasionally, the model may generate unintended background music, humming, rhythmic noises, or sound-like artifacts during speech generation. These artifacts can diminish clarity, distract users, or make the assistant feel less professional. The following instructions helps prevent or significantly reduce these occurrences.

- **When to use**: Use when you observe unintended musical elements or sound effects in Realtime audio responses.
- **What it does**: Steers the model to avoid generating these unwanted audio artifacts.s
- **How to
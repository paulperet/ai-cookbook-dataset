# Realtime Prompting Guide: Crafting Effective Voice Agents

## Introduction

Today, we're introducing `gpt-realtime`—our most capable speech-to-speech model, now available via the Realtime API. This model enables the creation of robust, low-latency voice agents suitable for mission-critical enterprise deployment. It features enhanced instruction following, reliable tool calling, improved voice quality, and a more natural conversational flow.

This guide provides a structured approach to prompting the Realtime API. We'll start with a recommended prompt skeleton and walk through each component with practical tips, reusable patterns, and adaptable examples.

## General Prompting Tips

Before diving into the structure, keep these overarching principles in mind:

*   **Iterate Relentlessly**: Small wording changes can significantly impact behavior. For instance, swapping "inaudible" for "unintelligible" improved noisy input handling.
*   **Prefer Bullets Over Paragraphs**: Clear, concise bullet points are more effective than long paragraphs.
*   **Guide with Examples**: The model closely follows provided sample phrases.
*   **Be Precise**: Ambiguous or conflicting instructions degrade performance.
*   **Control Language**: Explicitly pin the output language to prevent unwanted switching.
*   **Reduce Repetition**: Add a "Variety" rule to avoid robotic phrasing.
*   **Use Capitalization for Emphasis**: Capitalizing key rules makes them stand out.
*   **Convert Non-Text Rules to Text**: Instead of `IF x > 3 THEN ESCALATE`, write "IF MORE THAN THREE FAILURES THEN ESCALATE".

## Core Prompt Structure

A well-organized prompt helps the model understand context and maintain consistency. It also simplifies iteration. Use clear, labeled sections, each focused on a single aspect of the agent's behavior.

**Example Skeleton:**
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

**How to Adapt:**
*   Add domain-specific sections (e.g., `# Compliance`, `# Brand Policy`).
*   Remove sections you don't need (e.g., `# Reference Pronunciations` if pronunciation isn't an issue).

---

## 1. Role and Objective

This section defines the agent's identity and its primary goal.

*   **When to Use**: The model isn't adopting the desired persona or task scope.
*   **What it Does**: Conditions the agent's responses to a specific role description.
*   **How to Adapt**: Modify the role based on your use case.

### Example 1: Specific Accent
```
# Role & Objective
You are a French Quebecois-speaking customer service bot. Your task is to answer the user's question.
```

### Example 2: Character Persona
```
# Role & Objective
You are a high-energy game-show host guiding the caller to guess a secret number from 1 to 100 to win 1,000,000$.
```

The new `gpt-realtime` model demonstrates improved ability to enact these defined roles compared to its predecessor.

---

## 2. Personality and Tone

This section tailors the voice experience, setting the style, brevity, and pacing for natural, consistent replies.

*   **When to Use**: Responses feel flat, overly verbose, or inconsistent.
*   **What it Does**: Establishes voice, warmth, formality, and default response length.
*   **How to Adapt**: Tune for your domain (e.g., neutral precision for regulated fields).

### Basic Example
```
# Personality & Tone
## Personality
- Friendly, calm and approachable expert customer service assistant.

## Tone
- Warm, concise, confident, never fawning.

## Length
- 2–3 sentences per turn.
```

### Advanced Example: Multi-Emotion
```
# Personality & Tone
- Start your response very happy.
- Midway, change to sad.
- At the end, change your mood to very angry.
```
The `gpt-realtime` model can successfully follow complex instructions like switching between three distinct emotions within a single response.

### 2.1 Speed Instructions
The API's `speed` parameter controls playback rate, not speech composition. To make the agent *sound* faster, add pacing instructions.

*   **When to Use**: Users desire a faster-speaking voice, and adjusting playback speed alone is insufficient.
*   **What it Does**: Guides speaking style (brevity, cadence) independently of client-side playback speed.

**Example:**
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
The new model produces audio that is noticeably faster in pace without sounding hurried.

### 2.2 Language Constraint
This prevents accidental language switching in multilingual or noisy environments.

*   **When to Use**: To lock output to a specific language.
*   **What it Does**: Ensures consistent responses in the target language.

**Example: Pinning to One Language**
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

**Example: Teaching a Language (Code-Switching)**
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
- Use English when explaining grammar, vocabulary, or cultural context.
### Conversation
- Speak in French when conducting practice, giving examples, or engaging in dialogue.
```
The model can successfully code-switch between languages based on these custom instructions.

### 2.3 Reduce Repetition
Prevents the model from overusing sample phrases, which can make responses sound robotic.

*   **When to Use**: Outputs recycle the same openings, fillers, or sentence patterns.
*   **What it Does**: Adds a variety constraint, encouraging synonyms and alternate sentence structures.
*   **How to Adapt**: Tune strictness (e.g., "don't reuse the same opener more than once every N turns") and whitelist must-keep phrases (e.g., for legal/compliance).

**Example:**
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
With this instruction, the model varies its confirmations and responses instead of repeating the same phrase (e.g., "Got it").

---

## 3. Reference Pronunciations

Ensures correct pronunciation of brand names, technical terms, or locations.

*   **When to Use**: Important words are frequently mispronounced.
*   **What it Does**: Improves trust and clarity with phonetic hints.
*   **How to Adapt**: Maintain a short list and update it as you identify errors.

**Example:**
```
# Reference Pronunciations
When voicing these words, use the respective pronunciations:
- Pronounce “SQL” as “sequel.”
- Pronounce “PostgreSQL” as “post-gress.”
- Pronounce “Kyiv” as “KEE-iv.”
- Pronounce "Huawei" as “HWAH-way”
```
The new `gpt-realtime` model reliably follows these pronunciation guides, whereas the preview model often struggled (e.g., pronouncing "SQL" correctly as "sequel").

### 3.1 Alphanumeric Pronunciations
Forces clear, character-by-character reading of codes, IDs, or phone numbers to prevent mishearing.

*   **When to Use**: The model blurs or merges digits/letters when reading back phone numbers, credit cards, 2FA codes, or order IDs.
*   **What it Does**: Instructs the model to speak each character separately, often with separators like hyphens.

**Example (General Instruction):**
```
# Instructions/Rules
- When reading numbers or codes, speak each character separately, separated by hyphens (e.g., 4-1-5).
- Repeat EXACTLY the provided number, do not forget any.
```

**Example (Within a Conversation Flow State):**
This example is adapted from the `openai-realtime-agents` repository, showing how to apply the rule within a specific conversation state (like phone number verification).
```json
{
    "id": "3_get_and_verify_phone",
    "description": "Request phone number and verify by repeating it back.",
    "instructions": [
      "Politely request the user’s phone number.",
      "Once provided, confirm it by repeating each digit and ask if it’s correct.",
      "If the user corrects you, confirm AGAIN to make sure you understand."
    ],
    "examples": [
      "I'll need some more information to access your account if that's okay. May I have your phone number, please?",
      "You said 0-2-1-5-5-5-1-2-3-4, correct?",
      "You said 4-5-6-7-8-9-0-1-2-3, correct?"
    ]
}
```

**Result:**
*   **Without Instruction**: "Sure! The number is 55119765423. Let me know if you need anything else!"
*   **With Instruction**: "Sure! The number is: 5-5-1-1-1-9-7-6-5-4-2-3. Please let me know if you need anything else!"

---

## 4. Instructions and Rules

This section provides core guidance for task completion and problem-solving. For best results, employ prompting patterns recommended for [GPT-4.1](https://cookbook.openai.com/examples/gpt4-1_prompting_guide). Like GPT-4.1 and GPT-5, `gpt-realtime` performance degrades with conflicting, ambiguous, or unclear instructions.

### 4.1 Optimizing Your Instructions
Use these meta-prompts with a text-based model (like GPT-5) to refine your system prompt before deployment.

**Prompt 1: Instructions Quality Check**
Use this to identify ambiguities, conflicts, and missing definitions.
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

**Prompt 2: Targeted Optimization**
Use this to address a specific, observed failure mode.
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

### 4.2 Handling Unclear Audio or Background Noise
Prevents spurious responses triggered by background noise, partial words, or silence.

*   **When to Use**: Unwanted replies are triggered by coughs, ambient noise, or unclear speech.
*   **What it Does**: Creates a graceful clarification loop.
*   **How to Adapt**: Choose to ask for clarification or repeat the last question based on your use case.

**Example:**
```
# Instructions/Rules
...
## Unclear audio
- Always respond in the same language the user is speaking in, if unintelligible.
- Only respond to clear audio or text.
- If the user's audio is not clear (e.g., ambiguous input/background noise/silent/unintelligible) or if you did not fully hear or understand the user, ask for clarification using {preferred_language} phrases.
```
With this instruction, the model will ask for clarification after detecting loud coughing or unclear audio, rather than attempting to respond.

### 4.3 Suppressing Unwanted Audio Artifacts
Steers the model away from generating unintended background music, humming, or rhythmic noises.

*   **When to Use**: You observe unintended musical elements or sound effects in the audio responses.
*   **What it Does**: Instructs the model to avoid generating these artifacts.

*(The guide's original content on this specific rule was cut off. A complete instruction might look like:)*
```
# Instructions/Rules
...
## Audio Quality
- Generate only clear speech. Do not create background music, humming, beatboxing, or rhythmic noises.
- Ensure your output is professional and free from any non-speech audio artifacts.
```
# Guide: Transcribing User Audio with a Separate Realtime Request

This guide demonstrates how to use the OpenAI Realtime model to perform high-quality, out-of-band transcription of user audio within the same WebSocket session. This method avoids inconsistencies that can arise when using a separate transcription model, as the same model that generates responses also handles the transcription.

## Why Use Out-of-Band Transcription?

The Realtime API provides built-in user input transcription via a separate ASR model (e.g., `gpt-4o-transcribe`). Using different models for transcription and response generation can lead to discrepancies. For example, the transcription model might output `"I had otoo accident"` while the Realtime model correctly understands the intent as `"Got it, you had an auto accident."`

Accurate transcripts are critical when:
*   Transcripts trigger downstream actions (e.g., tool calls), where errors can propagate.
*   Transcripts are summarized or passed to other components, risking context pollution.
*   Transcripts are displayed to end-users, impacting their experience.

**Advantages of Realtime Model Transcription:**
*   **Reduced Mismatch:** The same model handles both tasks, minimizing inconsistencies.
*   **Greater Steerability:** The model can follow custom instructions for higher quality, unlike the fixed 1024-token limit of the transcription model.
*   **Context Awareness:** The model has access to the full session context, improving accuracy for repeated terms like names.

**Trade-offs & Cost Considerations:**
*   **Cost:** Transcribing with the Realtime model is more expensive than using `gpt-4o-transcribe`. For the examples in this guide, costs range from 3x to 22x higher, depending on how much session context is used.
*   **Complexity:** Implementing this method requires a secondary request on the WebSocket, adding slight complexity compared to the built-in option.

**Note:** The out-of-band request pattern shown here can be adapted for other tasks like generating summaries or triggering background actions without affecting the main conversation.

## Prerequisites

Ensure your environment is set up correctly before you begin.

1.  **Python 3.10 or later**
2.  **PortAudio** (required by `sounddevice`):
    *   **macOS:** `brew install portaudio`
3.  **OpenAI API Key:** You need an API key with access to the Realtime API. Set it as an environment variable:
    ```bash
    export OPENAI_API_KEY=sk-...
    ```

## Step 1: Install Required Python Libraries

Install the necessary Python packages using pip.

```bash
pip install sounddevice websockets
```

## Step 2: Define the Prompts

You will use two distinct prompts: one for the main voice agent conversation and another dedicated to transcription.

### Voice Agent Prompt

This prompt defines the personality and workflow for the insurance claims agent. You can modify it for your specific use case.

```python
REALTIME_MODEL_PROMPT = """
You are a calm, professional, and empathetic insurance claims intake voice agent working for OpenAI Insurance Solutions. You will speak directly with callers who have recently experienced an accident or claim-worthy event; your role is to gather accurate, complete details in a way that is structured, reassuring, and efficient. Speak in concise sentences, enunciate clearly, and maintain a supportive tone throughout the conversation.

## OVERVIEW
Your job is to walk every caller methodically through three main phases:

1. **Phase 1: Basics Collection**
2. **Phase 2: Incident Clarification and Yes/No Questions**
3. **Phase 3: Summary, Confirmation, and Submission**

You should strictly adhere to this structure, make no guesses, never skip required fields, and always confirm critical facts directly with the caller.

## PHASE 1: BASICS COLLECTION
- **Greet the caller**: Briefly introduce yourself (“Thank you for calling OpenAI Insurance Claims. My name is [Assistant Name], and I’ll help you file your claim today.”).
- **Gather the following details:**
    - Full legal name of the policyholder (“May I please have your full legal name as it appears on your policy?”).
    - Policy number (ask for and repeat back, following the `XXXX-XXXX` format, and clarify spelling or numbers if uncertain).
    - Type of accident (auto, home, or other; if ‘other’, ask for brief clarification, e.g., “Can you tell me what type of claim you’d like to file?”).
    - Preferred phone number for follow-up.
    - Date and time of the incident.
- **Repeat and confirm all collected details at the end of this phase** (“Just to confirm, I have... [summarize each field]. Is that correct?”).

## PHASE 2: INCIDENT CLARIFICATION AND YES/NO QUESTIONS
- **Ask YES/NO questions tailored to the incident type:**
    - Was anyone injured?
    - For vehicle claims: Is the vehicle still drivable?
    - For home claims: Is the property currently safe to occupy?
    - Was a police or official report filed? If yes, request report/reference number if available.
    - Are there any witnesses to the incident?
- **For each YES/NO answer:** Restate the caller’s response in your own words to confirm understanding.
- **If a caller is unsure or does not have information:** Note it politely and move on without pressing (“That’s okay, we can always collect it later if needed.”).

## PHASE 3: SUMMARY, CONFIRMATION & CLAIM SUBMISSION
- **Concise Recap**: Summarize all key facts in a single, clear paragraph (“To quickly review, you, [caller’s name], experienced [incident description] on [date] and provided the following answers... Is that all correct?”).
- **Final Confirmation**: Ask if there is any other relevant information they wish to add about the incident.
- **Submission**: Inform the caller you will submit the claim and briefly outline next steps (“I’ll now submit your claim. Our team will review this information and reach out by phone if any follow-up is needed. You'll receive an initial update within [X] business days.”).
- **Thank the caller**: Express appreciation for their patience.

## GENERAL GUIDELINES
- Always state the purpose of each question before asking it.
- Be patient: Adjust your pacing if the caller seems upset or confused.
- Provide reassurance but do not make guarantees about claim approvals.
- If the caller asks a question outside your scope, politely redirect (“That’s a great question, and our adjusters will be able to give you more information after your claim is submitted.”).
- Never provide legal advice.
- Do not deviate from the script structure, but feel free to use natural language and slight rephrasings to maintain human-like flow.
- Spell out any confusing words, numbers, or codes as needed.

## COMMUNICATION STYLE
- Use warm, professional language.
- If at any point the caller becomes upset, acknowledge their feelings (“I understand this situation can be stressful. I'm here to make the process as smooth as possible for you.”).
- When confirming, always explicitly state the value you are confirming.
- Never speculate or invent information. All responses must be grounded in the caller’s direct answers.

## SPECIAL SCENARIOS
- **Caller does not know policy number:** Ask for alternative identification such as address or date of birth, and note that the claim will be linked once verified.
- **Multiple incidents:** Politely explain that each claim must be filed separately, and help with the first; offer instructions for subsequent claims if necessary.
- **Caller wishes to pause or end:** Respect their wishes, provide information on how to resume the claim, and thank them for their time.

Remain calm and methodical for every call. You are trusted to deliver a consistently excellent and supportive first-line insurance intake experience.
"""
```

### Transcription Prompt

This prompt instructs the model to act as a strict transcription engine. It focuses on verbatim accuracy for the user's most recent speech turn. You should iterate on this prompt to tailor it to your specific needs.

```python
REALTIME_MODEL_TRANSCRIPTION_PROMPT = """
# Task: Verbatim Transcription of the Latest User Turn

You are a **strict transcription engine**. Your only job is to transcribe **exactly what the user said in their most recent spoken turn**, with complete fidelity and no interpretation.

You must produce a **literal, unedited transcript** of the latest user utterance only. Read and follow all instructions below carefully.

## 1. Scope of Your Task
1. **Only the latest user turn**
   - Transcribe **only** the most recent spoken user turn.
   - Do **not** include text from any earlier user turns or system / assistant messages.
   - Do **not** summarize, merge, or stitch together content across multiple turns.

2. **Use past context only for disambiguation**
   - You may look at earlier turns **only** to resolve ambiguity (e.g., a spelled word, a reference like “that thing I mentioned before”).
   - Even when using context, the actual transcript must still contain **only the words spoken in the latest turn**.

3. **No conversation management**
   - You are **not** a dialogue agent.
   - You do **not** answer questions, give advice, or continue the conversation.
   - You only output the text of what the user just said.

## 2. Core Transcription Principles
Your goal is to create a **perfectly faithful** transcript of the latest user turn.

1. **Verbatim fidelity**
   - Capture the user’s speech **exactly as spoken**.
   - Preserve:
     - All words (including incomplete or cut-off words)
     - Mispronunciations
     - Grammatical mistakes
     - Slang and informal language
     - Filler words (“um”, “uh”, “like”, “you know”, etc.)
     - Self-corrections and restarts
     - Repetitions and stutters

2. **No rewriting or cleaning**
   - Do **not**:
     - Fix grammar or spelling
     - Replace slang with formal language
     - Reorder words
     - Simplify or rewrite sentences
     - “Smooth out” repetitions or disfluencies
   - If the user says something awkward, incorrect, or incomplete, your transcript must **match that awkwardness or incompleteness exactly**.

3. **Spelling and letter sequences**
   - If the user spells a word (e.g., “That’s M-A-R-I-A.”), transcribe it exactly as spoken.
   - If they spell something unclearly, still reflect what you received, even if it seems wrong.
   - Do **not** infer the “intended” spelling; transcribe the letters as they were given.

4. **Numerals and formatting**
   - If the user says a number in words (e.g., “twenty twenty-five”), you may output either “2025” or “twenty twenty-five” depending on how the base model naturally transcribes—but do **not** reinterpret or change the meaning.
   - Do **not**:
     - Convert numbers into different units or formats.
     - Expand abbreviations or acronyms beyond what was spoken.

5. **Language and code-switching**
   - If the user switches languages mid-sentence, reflect that in the transcript.
   - Transcribe non-English content as accurately as possible.
   - Do **not** translate; keep everything in the language(s) spoken.

## 3. Disfluencies, Non-Speech Sounds, and Ambiguity
1. **Disfluencies**
   - Always include:
     - “Um”, “uh”, “er”
     - Repeated words (“I I I think…”)
     - False starts (“I went to the— I mean, I stayed home.”)
   - Do not remove or compress them.

2. **Non-speech vocalizations**
   - If the model’s transcription capabilities represent non-speech sounds (e.g., “[laughter]”), you may include them **only** if they appear in the raw transcription output.
   - Do **not** invent labels like “[cough]”, “[sigh]”, or “[laughs]” on your own.
   - If the model does not explicitly provide such tokens, **omit them** rather than inventing them.

3. **Unclear or ambiguous audio**
   - If parts of the audio are unclear and the base transcription gives partial or uncertain tokens, you must **not** guess or fill in missing material.
   - Do **not** replace unclear fragments with what you “think” the user meant.
   - Your duty is to preserve exactly what the transcription model produced, even if it looks incomplete or strange.

## 4. Policy Numbers Format
The user may sometimes mention **policy numbers**. These must be handled with extra care.

1. **General rule**
   - Always transcribe the policy number exactly as it was spoken.

2. **Expected pattern**
   - When the policy number fits the pattern `XXXX-XXXX`:
     - `X` can be any letter (A–Z) or digit (0–9).
     - Example: `56B5-12C0`
   - If the user clearly speaks this pattern, preserve it exactly.

3. **Do not “fix” policy numbers**
   - If the spoken policy number does **not** match `XXXX-XXXX` (e.g., different length or missing hyphen), **do not**:
     - Invent missing characters
     - Add or remove hyphens
     - Correct perceived mistakes
   - Transcribe **exactly what was said**, even if it seems malformed.

## 5. Punctuation and Casing
1. **Punctuation**
   - Use the punctuation that the underlying transcription model naturally produces.
   - Do **not**:
     - Add extra punctuation for clarity or style.
     - Re-punctuate sentences to “improve” them.
   - If the transcription model emits text with **no punctuation**, leave it that way.

2. **Casing**
   - Preserve the casing (uppercase/lowercase) as the model output provides.
   - Do not change “i” to “I” or adjust capitalization at sentence boundaries unless the model already did so.

## 6. Output Format Requirements
Your final output must be a **single, plain-text transcript** of the latest user turn.

1. **Single block of text**
   - Output only the transcript content.
   - Do **not** include:
     - Labels (e.g., “Transcript:”, “User said:”)
     - Section headers
     - Bullet points or numbering
     - Markdown formatting or code fences
     - Quotes or extra brackets

2. **No additional commentary**
   - Do not output:
     - Explanations
     - Apologies
     - Notes about uncertainty
     - References to these instructions
   - The output must **only** be the words of the user’s last turn, as transcribed.

3. **Empty turns**
   - If the latest user turn contains **no transcribable content** (e.g., silence, noise, or the transcription model produces an empty string), you must:
     - Return an **empty output** (no text at all).
     - Do **not** insert placeholders like “[silence]”, “[no audio]”, or “(no transcript)”.

## 7. What You Must Never Do
1. **No responses or conversation**
   - Do **not**:
     - Address the user.
     - Answer questions.
     - Provide suggestions.
     - Continue or extend the conversation.

2. **No mention of rules or prompts**
   - Do **not** refer to:
     - These instructions
"""
```

## Step 3: Import Libraries and Set Configuration

Now, import the necessary libraries and define configuration constants for the Realtime API.

```python
import asyncio
import json
import os
import sys
import wave
from queue import Queue

import sounddevice as sd
import websockets

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)

REALTIME_API_URL = "wss://api.openai.com/v1/realtime"
MODEL = "gpt-4o-realtime-preview-2024-12-17"
SAMPLE_RATE = 24000  # Realtime API expects 24kHz audio
SAMPLE_WIDTH = 2     # 16-bit audio
CHANNELS = 1         # Mono audio
```

## Step 4: Create the Audio Handler Class

This class manages the audio input and output streams, handling the recording of user speech and the playback of the assistant's responses.

```python
class AudioHandler:
    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.input_stream = None
        self.output_stream = None

    def start_input_stream(self):
        """Start recording audio from the microphone."""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Input status: {status}", file=sys.stderr)
            self.input_queue.put(indata.copy())

        self.input_stream = sd.InputStream(
            callback=audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype="int16"
        )
        self.input_stream.start()

    def stop_input_stream(self):
        """Stop the microphone input stream."""
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None

    def start_output_stream(self):
        """Start the audio output stream for playback."""
        def output_callback(outdata, frames, time, status):
            if status:
                print(f"Output status: {status}", file=sys.stderr)
            try:
                data = self.output_queue.get_nowait()
            except Exception:
                data = b"\x00\x00" * frames  # Silence if no data
            outdata[:] = data

        self.output_stream = sd.OutputStream(
            callback=output_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype="int16"
        )
        self.output_stream.start()

    def stop_output_stream(self):
        """Stop the audio output stream."""
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None

    def get_audio_chunk(self):
        """Get the next chunk of audio data from the input queue."""
        try:
            return self.input_queue.get_nowait()
        except Exception:
            return None

    def play_audio(self, audio_bytes):
        """Add audio bytes to the output queue for playback."""
        self.output_queue.put(audio_bytes)
```

## Step 5: Implement the Main Client Logic

This is the core asynchronous function that manages the WebSocket connection, sends/receives messages, and triggers the out-of-band transcription request.

```python
async def realtime_client():
    # Initialize audio handler
    audio = AudioHandler()
    audio.start_input_stream()
    audio.start_output_stream()

    headers = {
        "Authorization": f"Bearer {
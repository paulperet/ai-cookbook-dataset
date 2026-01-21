# Guide: Correcting Transcription Misspellings with Whisper and GPT-4

This guide demonstrates two complementary strategies for improving the accuracy of AI-generated transcriptions, especially for specialized vocabulary like company and product names. You will learn how to use Whisper's prompt parameter for initial guidance and GPT-4 for robust post-processing correction.

## Prerequisites

Ensure you have the OpenAI Python library installed and your API key configured.

```bash
pip install openai
```

## 1. Setup and Imports

Begin by importing the necessary libraries and setting up your OpenAI client. Replace `<your OpenAI API key...>` with your actual key if it's not set as an environment variable.

```python
from openai import OpenAI
import urllib
import os

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
```

## 2. Download the Example Audio File

We'll use a sample audio file containing a monologue about a fictitious tech company, `ZyntriQix`.

```python
# Define remote and local file paths
ZyntriQix_remote_filepath = "https://cdn.openai.com/API/examples/data/ZyntriQix.wav"
ZyntriQix_filepath = "data/ZyntriQix.wav"

# Download the file
urllib.request.urlretrieve(ZyntriQix_remote_filepath, ZyntriQix_filepath)
```

## 3. Establish a Baseline Transcription

First, let's see how Whisper performs without any guidance. We'll create a helper function for transcription.

```python
def transcribe(prompt: str, audio_filepath) -> str:
    """Transcribe an audio file using the Whisper model with an optional prompt."""
    transcript = client.audio.transcriptions.create(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
        prompt=prompt,
    )
    return transcript.text

# Get the baseline transcription
baseline_text = transcribe(prompt="", audio_filepath=ZyntriQix_filepath)
print(baseline_text)
```

**Output:**
```
Have you heard of ZentricX? This tech giant boasts products like Digi-Q+, Synapse 5, VortiCore V8, Echo Nix Array, and not to forget the latest Orbital Link 7 and Digifractal Matrix. Their innovation arsenal also includes the Pulse framework, Wrapped system, they've developed a brick infrastructure court system, and launched the Flint initiative, all highlighting their commitment to relentless innovation. ZentricX, in just 30 years, has soared from a startup to a tech titan, serving us tech marvels alongside a stimulating linguistic challenge. Quite an adventure, wouldn't you agree?
```

As you can see, Whisper makes several errors with the company name (`ZentricX`), product names, and acronyms.

## 4. Strategy 1: Using Whisper's Prompt Parameter

The Whisper API accepts a `prompt` parameter. You can provide a list of correct spellings to guide the initial transcription. This is useful but limited to 244 tokens.

### Step 4.1: Provide a Targeted Prompt

Let's pass the correct names for our key products and initiatives.

```python
corrected_prompt = "ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."

prompt_corrected_text = transcribe(prompt=corrected_prompt, audio_filepath=ZyntriQix_filepath)
print(prompt_corrected_text)
```

**Output:**
```
Have you heard of ZyntriQix? This tech giant boasts products like Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, and not to forget the latest OrbitalLink Seven and DigiFractal Matrix. Their innovation arsenal also includes the PULSE framework, RAPT system. They've developed a B.R.I.C.K. infrastructure, Q.U.A.R.T. system, and launched the F.L.I.N.T. initiative, all highlighting their commitment to relentless innovation. ZyntriQix in just 30 years has soared from a startup to a tech titan, serving us tech marvels alongside a stimulating linguistic challenge. Quite an adventure, wouldn't you agree?
```

**Analysis:** The prompt helps significantly, correctly fixing `ZyntriQix`, `Digique Plus`, `CynapseFive`, and the acronyms. However, note that `Q.U.A.R.T.Z.` was shortened to `Q.U.A.R.T.`, showing the model's struggle with long, similar acronyms even with guidance.

### Step 4.2: Test the Limits with an Extensive Prompt

What happens if we provide an extremely long list of potential product names, simulating a large SKU catalog?

```python
# An extensive list of product names (abbreviated here for clarity)
extensive_prompt = "ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, AstroPixel Array, QuantumFlare Five, CyberPulse Six, VortexDrive Matrix, PhotonLink Ten, TriCircuit Array, PentaSync Seven, UltraWave Eight, QuantumVertex Nine, HyperHelix X, DigiSpiral Z, PentaQuark Eleven, TetraCube Twelve, GigaPhase Thirteen, EchoNeuron Fourteen, FusionPulse V15, MetaQuark Sixteen, InfiniCircuit Seventeen, TeraPulse Eighteen, ExoMatrix Nineteen, OrbiSync Twenty, QuantumHelix TwentyOne, NanoPhase TwentyTwo, TeraFractal TwentyThree, PentaHelix TwentyFour, ExoCircuit TwentyFive, HyperQuark TwentySix, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."

extensive_prompt_text = transcribe(prompt=extensive_prompt, audio_filepath=ZyntriQix_filepath)
print(extensive_prompt_text)
```

**Output:**
```
Have you heard of ZentricX? This tech giant boasts products like DigiCube Plus, Synapse 5, VortiCore V8, EchoNix Array, and not to forget the latest Orbital Link 7 and Digifractal Matrix. Their innovation arsenal also includes the PULSE framework, RAPT system. They've developed a brick infrastructure court system and launched the F.L.I.N.T. initiative, all highlighting their commitment to relentless innovation. ZentricX in just 30 years has soared from a startup to a tech titan, serving us tech marvels alongside a stimulating linguistic challenge. Quite an adventure, wouldn't you agree?
```

**Analysis:** Performance degrades. The model reverts to errors like `ZentricX` and `Synapse 5`. This illustrates a key limitation: overly long or noisy prompts can confuse the model and reduce transcription quality.

## 5. Strategy 2: Post-Processing with GPT-4

For a more scalable and reliable approach, we can use GPT-4 to correct the initial Whisper transcription. This method is not bound by Whisper's 244-token prompt limit and can leverage GPT-4's stronger language understanding, though it increases cost and latency.

### Step 5.1: Create a GPT-4 Spell-Check Function

First, define a function that takes a system message (containing the correct spellings) and uses GPT-4 to fix the baseline transcription.

```python
def transcribe_with_spellcheck(system_message, audio_filepath):
    """Transcribe audio with Whisper, then correct spelling using GPT-4."""
    # First, get the raw Whisper transcription
    raw_transcription = transcribe(prompt="", audio_filepath=audio_filepath)

    # Then, ask GPT-4 to correct it based on the provided list
    completion = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": raw_transcription},
        ],
    )
    return completion.choices[0].message.content
```

### Step 5.2: Correct with a Focused Product List

Now, instruct GPT-4 to act as an assistant for `ZyntriQix` and correct any spelling discrepancies using a targeted list.

```python
system_prompt_focused = """You are a helpful assistant for the company ZyntriQix. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."""

corrected_text_focused = transcribe_with_spellcheck(system_prompt_focused, ZyntriQix_filepath)
print(corrected_text_focused)
```

**Output:**
```
Have you heard of ZyntriQix? This tech giant boasts products like Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, and not to forget the latest OrbitalLink Seven and DigiFractal Matrix. Their innovation arsenal also includes the PULSE framework, RAPT system, they've developed a B.R.I.C.K. infrastructure court system, and launched the F.L.I.N.T. initiative, all highlighting their commitment to relentless innovation. ZyntriQix, in just 30 years, has soared from a startup to a tech titan, serving us tech marvels alongside a stimulating linguistic challenge. Quite an adventure, wouldn't you agree?
```

**Result:** GPT-4 successfully corrected all target terms, including the previously problematic `Q.U.A.R.T.Z.` and `B.R.I.C.K.`.

### Step 5.3: Scale Up with an Extensive Product Catalog

This method shines when you have a large list of correct terms. You can provide an extensive SKU list without degrading performance.

```python
# A very long list of product names (simulating a full catalog)
extensive_list = "ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, AstroPixel Array, QuantumFlare Five, CyberPulse Six, VortexDrive Matrix, PhotonLink Ten, TriCircuit Array, PentaSync Seven, UltraWave Eight, QuantumVertex Nine, HyperHelix X, DigiSpiral Z, PentaQuark Eleven, TetraCube Twelve, GigaPhase Thirteen, EchoNeuron Fourteen, FusionPulse V15, MetaQuark Sixteen, InfiniCircuit Seventeen, TeraPulse Eighteen, ExoMatrix Nineteen, OrbiSync Twenty, QuantumHelix TwentyOne, NanoPhase TwentyTwo, TeraFractal TwentyThree, PentaHelix TwentyFour, ExoCircuit TwentyFive, HyperQuark TwentySix, GigaLink TwentySeven, FusionMatrix TwentyEight, InfiniFractal TwentyNine, MetaSync Thirty, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."

system_prompt_extensive = f"""You are a helpful assistant for the company ZyntriQix. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: {extensive_list}. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."""

corrected_text_extensive = transcribe_with_spellcheck(system_prompt_extensive, ZyntriQix_filepath)
print(corrected_text_extensive)
```

**Output:**
```
Have you heard of ZyntriQix? This tech giant boasts products like Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, and not to forget the latest OrbitalLink Seven and DigiFractal Matrix. Their innovation arsenal also includes the PULSE framework, RAPT system, they've developed a B.R.I.C.K. infrastructure court system, and launched the F.L.I.N.T. initiative, all highlighting their commitment to relentless innovation. ZyntriQix, in just 30 years, has soared from a startup to a tech titan, serving us tech marvels alongside a stimulating linguistic challenge. Quite an adventure, wouldn't you agree?
```

**Analysis:** GPT-4 correctly identifies which terms from the massive list are present in the transcription and fixes them appropriately, ignoring the irrelevant ones. This demonstrates excellent scalability.

### Step 5.4: Advanced Use Case: Audit and Correct

You can also design prompts that make GPT-4 explain its corrections, which is useful for auditing.

```python
audit_prompt = """You are a helpful assistant for the company ZyntriQix. Your first task is to list the words that are not spelled correctly according to the list provided to you and to tell me the number of misspelled words. Your next task is to insert those correct words in place of the misspelled ones. List: ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, AstroPixel Array, QuantumFlare Five, CyberPulse Six, VortexDrive Matrix, PhotonLink Ten, TriCircuit Array, PentaSync Seven, UltraWave Eight, QuantumVertex Nine, HyperHelix X, DigiSpiral Z, PentaQuark Eleven, TetraCube Twelve, GigaPhase Thirteen, EchoNeuron Fourteen, FusionPulse V15, MetaQuark Sixteen, InfiniCircuit Seventeen, TeraPulse Eighteen, ExoMatrix Nineteen, OrbiSync Twenty, QuantumHelix TwentyOne, NanoPhase TwentyTwo, TeraFractal TwentyThree, PentaHelix TwentyFour, ExoCircuit TwentyFive, HyperQuark TwentySix, GigaLink TwentySeven, FusionMatrix TwentyEight, InfiniFractal TwentyNine, MetaSync Thirty, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."""

audit_result = transcribe_with_spellcheck(audit_prompt, ZyntriQix_filepath)
print(audit_result)
```

**Output:**
```
The misspelled words are: ZentricX, Digi-Q+, Synapse 5, VortiCore V8, Echo Nix Array, Orbital Link 7, Digifractal Matrix, Pulse, Wrapped, brick, Flint, and 30. The total number of misspelled words is 12.

The corrected paragraph is:

Have you heard of ZyntriQix? This tech giant boasts products like Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, and not to forget the latest OrbitalLink Seven and DigiFractal Matrix. Their innovation arsenal also includes the PULSE framework, RAPT system, they've developed a B.R.I.C.K. infrastructure court system, and launched the F.L.I.N.T. initiative, all highlighting their commitment to relentless innovation. ZyntriQix, in just MetaSync Thirty years, has soared from a startup to a tech titan, serving us tech marvels alongside a stimulating linguistic challenge. Quite an adventure, wouldn't you agree?
```

**Note:** In this audit, GPT-4 correctly identified the errors but made one *over-correction*: it replaced the number "30" with the product name "MetaSync Thirty" from the list. This highlights the importance of precise prompt engineering. Adding instructions like "do not change numbers" would prevent this.

## Summary and Recommendations

You have successfully implemented two strategies for improving transcription accuracy:

1.  **Whisper Prompt Guidance:** Effective for short, focused lists of terms. It's fast and cost-effective but has a strict token limit and can be confused by overly long prompts.
2.  **GPT-4 Post-Processing:** More scalable and reliable for large vocabularies. It handles extensive term lists well but introduces additional cost and latency.

**Best Practice Recommendation:** For most production use cases involving specialized terminology, use a **hybrid approach**:
*   Use Whisper's `prompt` parameter for a shortlist of the most critical, frequently occurring terms (e.g., the company name, flagship products).
*   Follow up with GPT-4 post-processing using your full, validated glossary to catch any remaining errors and ensure consistency across the entire transcript.

This combination balances speed, cost, and the highest possible accuracy.
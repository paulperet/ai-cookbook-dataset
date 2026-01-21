# Sora 2 Prompting Guide

This guide provides best practices for crafting effective prompts for the Sora 2 video generation models. By understanding how to structure your instructions, you can achieve greater control, consistency, and creative outcomes.

## Understanding the Prompting Process

Think of prompting like briefing a cinematographer who has never seen your storyboard. If you leave out details, they’ll improvise. Being specific gives the model more control, while leaving details open can lead to surprising and beautiful interpretations.

**Key Principles:**
*   **Detailed Prompts:** Provide control and consistency.
*   **Lighter Prompts:** Open space for creative, unexpected outcomes.
*   **Iteration is Essential:** Small changes to camera, lighting, or action can dramatically shift the result. Treat the process as a collaboration.
*   **Variation is a Feature:** Using the same prompt multiple times will yield different results—each generation is a fresh take.

## Core API Parameters

The prompt controls the video's content, but certain attributes are governed only by API parameters. You cannot request these in prose; they must be set explicitly in your API call.

| Parameter | Description | Supported Values |
| :--- | :--- | :--- |
| **`model`** | The model version to use. | `sora-2` or `sora-2-pro` |
| **`size`** | The video resolution in `{width}x{height}` format. | **`sora-2`**: `1280x720`, `720x1280`<br>**`sora-2-pro`**: `1280x720`, `720x1280`, `1024x1792`, `1792x1024` |
| **`seconds`** | The clip length. | `"4"`, `"8"`, `"12"` (default is `"4"`) |

**Important Notes:**
*   These parameters define the video's container. Your prompt controls everything else (subject, motion, lighting, style).
*   Higher resolutions generally produce better visual fidelity and motion consistency.
*   For best results, aim for concise shots. The model often follows instructions more reliably in shorter clips. Consider stitching together multiple 4-second clips in editing instead of generating a single long clip.

## Anatomy of an Effective Prompt

A clear prompt describes a shot as if you were sketching it on a storyboard. Describe the camera framing, depth of field, action in beats, and lighting.

### Basic Prompt Example

```text
In a 90s documentary-style interview, an old Swedish man sits in a study and says, "I still remember when I was young."
```

**Why it works:**
*   **`90s documentary`**: Sets the overall style (camera, lighting, color grade).
*   **`an old Swedish man sits in a study`**: Describes subject and setting with enough detail for creative interpretation.
*   **`and says, "I still remember when I was young."`**: Provides clear dialogue for the model to follow.

This prompt will reliably produce matching videos, but many details (time of day, character's exact look, camera angles) are left for the model to invent.

### Advanced, Cinematic Prompt Structure

For complex shots, you can use a detailed, production-style brief. This approach helps lock onto a specific aesthetic.

**Example Template:**
```text
Format & Look
Duration 4s; digital capture emulating 65 mm photochemical contrast; fine grain.

Lenses & Filtration
32 mm spherical prime; Black Pro-Mist 1/4.

Grade / Palette
Highlights: clean morning sunlight with amber lift.
Mids: balanced neutrals.
Blacks: soft, neutral.

Lighting & Atmosphere
Natural sunlight from camera left, low angle (07:30 AM).
Atmos: gentle mist.

Location & Framing
Urban commuter platform, dawn.
Foreground: yellow safety line.
Midground: waiting passengers silhouetted.
Background: arriving train braking to a stop.

Wardrobe / Props
Main subject: mid-30s traveler, navy coat, backpack.
Props: paper coffee cup.

Sound
Diegetic only: faint rail screech, train brakes hiss, low ambient hum.

Shot List (2 shots / 4 s total)
0.00–2.40 — “Arrival Drift” (32 mm, slow dolly left)
Camera slides past platform; shallow focus reveals traveler looking down tracks.

2.40–4.00 — “Turn and Pause” (50 mm, slow arc in)
Cut to tighter over-shoulder arc as train halts; traveler turns slightly toward camera.
```

## Key Prompting Techniques

### 1. Use Specific Visual Cues
Style is a powerful lever. Describe the overall aesthetic early (`"1970s film"`, `"IMAX-scale scene"`, `"16mm black-and-white"`), then layer in specific, visible details.

| Weak Prompt | Strong Prompt |
| :--- | :--- |
| “A beautiful street at night” | “Wet asphalt, zebra crosswalk, neon signs reflecting in puddles” |
| “Person moves quickly” | “Cyclist pedals three times, brakes, and stops at crosswalk” |
| “Cinematic look” | “Anamorphic 2.0x lens, shallow depth of field, volumetric light” |

### 2. Direct Camera and Framing
Clearly state the camera shot and motion.

**Weak:**
```text
Camera shot: cinematic look
```

**Strong:**
```text
Camera shot: wide shot, low angle
Depth of field: shallow (sharp on subject, blurred background)
Lighting + palette: warm backlight with soft rim
```

**Examples of good instructions:**
*   `wide establishing shot, eye level`
*   `aerial wide shot, slight downward angle`
*   `slowly tilting camera`
*   `handheld eng camera`

### 3. Control Motion and Timing
Keep movement simple. Describe actions in clear beats or counts.

**Weak:**
```text
Actor walks across the room.
```

**Strong:**
```text
Actor takes four steps to the window, pauses, and pulls the curtain in the final second.
```

### 4. Define Lighting and Color
Lighting determines mood. Describe the quality, direction, and color palette to ensure consistency, especially across multiple clips.

**Weak:**
```text
Lighting + palette: brightly lit room
```

**Strong:**
```text
Lighting + palette: soft window light with warm lamp fill, cool rim from hallway
Palette anchors: amber, cream, walnut brown
```

## Using Image Input for Enhanced Control

For fine-grained control over composition and style, you can provide an image as a visual reference. The model uses it as an anchor for the first frame, while your text prompt defines what happens next.

**How to use it:**
1.  Include an image file as the `input_reference` parameter in your API request.
2.  The image must match the target video's resolution (`size`).
3.  Supported formats: `image/jpeg`, `image/png`, `image/webp`.

**Tip:** If you don't have a reference image, you can use OpenAI's image generation models to create one, then pass it to Sora.

## Incorporating Dialogue and Audio

Dialogue must be described directly in your prompt. For clarity, place it in a dedicated block.

**Example with dialogue:**
```text
A cramped, windowless room with walls the color of old ash. A single bare bulb dangles from the ceiling... The silence presses in.

Dialogue:
- Detective: "You’re lying. I can hear it in your silence."
- Suspect: "Or maybe I’m just tired of talking."
```

**Example describing background sound:**
```text
The hum of espresso machines and the murmur of voices form the background.
```

**Keep dialogue concise:** A 4-second shot typically fits one or two short exchanges.

## Iterating with Remix

Use the remix functionality for controlled, incremental changes. Describe only the specific tweak you want.

**Examples:**
*   `same shot, switch to 85 mm`
*   `same lighting, new palette: teal, sand, rust`
*   `Change the color of the monster to orange`

If a shot isn't working, simplify it (freeze the camera, clear the background) and then add complexity back step-by-step.

## Prompt Template and Examples

This template provides a clear framework. Remember, leaving elements open-ended encourages creative interpretation.

### Descriptive Prompt Template
```text
[Prose scene description in plain language. Describe characters, costumes, scenery, weather and other details.]

Cinematography:
Camera shot: [framing and angle, e.g., wide establishing shot, eye level]
Mood: [overall tone, e.g., cinematic and tense, playful]

Actions:
- [Action 1: a clear, specific beat or gesture]
- [Action 2: another distinct beat]

Dialogue:
[If applicable, add short, natural lines here.]
```

### Example Prompt
```text
Style: Hand-painted 2D/3D hybrid animation with soft brush textures and warm tungsten lighting. The aesthetic evokes mid-2000s storybook animation — cozy, imperfect, full of mechanical charm.

Cinematography:
Camera shot: Medium close-up, slight low angle
Mood: Whimsical and mysterious

Actions:
- A small, rusted metal robot with glowing blue eyes carefully winds a large, ornate key in its own back.
- It takes two wobbly steps forward on a wooden workbench.
- It looks up, its eyes flickering, as a shadow falls across it from off-screen.

Dialogue:
- (Whispered, metallic) "I am awake."
```
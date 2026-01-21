# Prompt Migration Guide: Optimizing Prompts for GPT‑4.1

Newer models like GPT‑4.1 are best-in-class for performance and instruction following. As models get smarter, prompts originally written for earlier models must be adapted to remain effective. GPT‑4.1 excels at following instructions precisely, which means unclear or ambiguous phrasing can lead to unexpected results. To leverage its full potential, refine your prompts to be explicit, unambiguous, and aligned with your intended outcomes.

**Objective:** This guide helps you improve an existing prompt into one that is clear, unambiguous, and optimized for GPT‑4.1.

**Workflow Overview:**
1.  **Input your original prompt**
2.  **Identify all instructions in your prompt**
3.  **Ask GPT‑4.1 to critique the prompt**
4.  **Auto-generate a revised system prompt**
5.  **Evaluate and iterate**
6.  **(Optional) Automatically apply GPT‑4.1 best practices**

## Prerequisites

You will need the `openai` Python package and an `OPENAI_API_KEY`.

```bash
pip install openai pydantic tiktoken
```

## Setup: Imports and API Connection

First, import the necessary libraries and set up your OpenAI client.

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Any, Dict, Iterable, List, Optional
import tiktoken
import html
from html import escape
import difflib
import sys

from IPython.display import display, HTML

try:
    from IPython.display import HTML, display
    _IN_IPYTHON = True
except ImportError:
    _IN_IPYTHON = False

client = OpenAI()
MODEL = "gpt-4.1"
```

## Helper Functions for Display

The following helper functions will help us visualize the analysis and modifications to our prompt.

```python
_COLORS = {
    '+': ("#d2f5d6", "#22863a"),   # additions  (green)
    '-': ("#f8d7da", "#b31d28"),   # deletions  (red)
    '@': (None,      "#6f42c1"),   # hunk header (purple)
}

def _css(**rules: str) -> str:
    """Convert kwargs to a CSS string (snake_case → kebab-case)."""
    return ";".join(f"{k.replace('_', '-')}: {v}" for k, v in rules.items())

def _render(html_str: str) -> None:
    """Render inside Jupyter if available, else print to stdout."""
    try:
        display  # type: ignore[name-defined]
        from IPython.display import HTML  # noqa: WPS433
        display(HTML(html_str))
    except NameError:
        print(html_str, flush=True)

# ---------- diff helpers ------------------------------------------------------

def _style(line: str) -> str:
    """Wrap a diff line in a <span> with optional colors."""
    bg, fg = _COLORS.get(line[:1], (None, None))
    css = ";".join(s for s in (f"background:{bg}" if bg else "",
                               f"color:{fg}" if fg else "") if s)
    return f'<span style="{css}">{html.escape(line)}</span>'

def _wrap(lines: Iterable[str]) -> str:
    body = "<br>".join(lines)
    return (
        "<details>"
        "<summary>🕵️‍♂️ Critique & Diff (click to expand)</summary>"
        f'<div style="font-family:monospace;white-space:pre;">{body}</div>'
        "</details>"
    )

def show_critique_and_diff(old: str, new: str) -> str:
    """Display & return a GitHub-style HTML diff between *old* and *new*."""
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(),
                                fromfile="old", tofile="new", lineterm="")
    html_block = _wrap(map(_style, diff))
    _render(html_block)
    return html_block

# ---------- “card” helpers ----------------------------------------------------

CARD    = _css(background="#f8f9fa", border_radius="8px", padding="18px 22px",
               margin_bottom="18px", border="1px solid #e0e0e0",
               box_shadow="0 1px 4px #0001")
TITLE   = _css(font_weight="600", font_size="1.1em", color="#2d3748",
               margin_bottom="6px")
LABEL   = _css(color="#718096", font_size="0.95em", font_weight="500",
               margin_right="6px")
EXTRACT = _css(font_family="monospace", background="#f1f5f9", padding="7px 10px",
               border_radius="5px", display="block", margin_top="3px",
               white_space="pre-wrap", color="#1a202c")

def display_cards(
    items: Iterable[Any],
    *,
    title_attr: str,
    field_labels: Optional[Dict[str, str]] = None,
    card_title_prefix: str = "Item",
) -> None:
    """Render objects as HTML “cards” (or plaintext when not in IPython)."""
    items = list(items)
    if not items:
        _render("<em>No data to display.</em>")
        return

    # auto-derive field labels if none supplied
    if field_labels is None:
        sample = items[0]
        field_labels = {
            a: a.replace("_", " ").title()
            for a in dir(sample)
            if not a.startswith("_")
            and not callable(getattr(sample, a))
            and a != title_attr
        }

    cards = []
    for idx, obj in enumerate(items, 1):
        title_html = html.escape(str(getattr(obj, title_attr, "<missing title>")))
        rows = [f'<div style="{TITLE}">{card_title_prefix} {idx}: {title_html}</div>']

        for attr, label in field_labels.items():
            value = getattr(obj, attr, None)
            if value is None:
                continue
            rows.append(
                f'<div><span style="{LABEL}">{html.escape(label)}:</span>'
                f'<span style="{EXTRACT}">{html.escape(str(value))}</span></div>'
            )

        cards.append(f'<div style="{CARD}">{"".join(rows)}</div>')

    _render("\n".join(cards))
```

## Step 1: Input Your Original Prompt

Begin by providing your existing prompt clearly. This prompt will serve as the baseline for improvement.

For this example, we will use the system prompt for LLM-as-a-Judge from the paper ["Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/pdf/2306.05685).

```python
original_prompt = """
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.

[User Question]
{question}

[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]

[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
"""

encoding = tiktoken.encoding_for_model("gpt-4")
num_tokens = len(encoding.encode(original_prompt))
print("Original prompt length:", num_tokens, "tokens")
```

```
Original prompt length: 243 tokens
```

## Step 2: Identify All Instructions in Your Prompt

In this step, we will extract every **instruction** that the LLM identifies within the system prompt. This allows you to review the list, spot any statements that should not be instructions, and clarify any that are ambiguous.

Carefully review and confirm that each listed instruction is both accurate and essential to retain.

First, define Pydantic models to structure the extracted instructions.

```python
class Instruction(BaseModel):
    instruction_title: str = Field(description="A 2-8 word title of the instruction that the LLM has to follow.")
    extracted_instruction: str = Field(description="The exact text that was extracted from the system prompt that the instruction is derived from.")

class InstructionList(BaseModel):
    instructions: list[Instruction] = Field(description="A list of instructions and their corresponding extracted text that the LLM has to follow.")
```

Now, use GPT‑4.1 to extract the instructions from your original prompt.

```python
EXTRACT_INSTRUCTIONS_SYSTEM_PROMPT = """
## Role & Objective
You are an **Instruction-Extraction Assistant**.  
Your job is to read a System Prompt provided by the user and distill the **mandatory instructions** the target LLM must obey.

## Instructions
1. **Identify Mandatory Instructions**  
   • Locate every instruction in the System Prompt that the LLM is explicitly required to follow.  
   • Ignore suggestions, best-practice tips, or optional guidance.

2. **Generate Rules**  
   • Re-express each mandatory instruction as a clear, concise rule.
   • Provide the extracted text that the instruction is derived from.
   • Each rule must be standalone and imperative.

## Output Format
Return a json object with a list of instructions which contains an instruction_title and their corresponding extracted text that the LLM has to follow. Do not include any other text or comments.

## Constraints
- Include **only** rules that the System Prompt explicitly enforces.  
- Omit any guidance that is merely encouraged, implied, or optional.  
"""

response = client.responses.parse(
    model=MODEL,
    input="SYSTEM_PROMPT TO ANALYZE: " + original_prompt,
    instructions=EXTRACT_INSTRUCTIONS_SYSTEM_PROMPT,
    temperature=0.0,
    text_format=InstructionList,
)

instructions_list = response.output_parsed
```

Finally, display the extracted instructions in a clean, scannable format.

```python
display_cards(
    instructions_list.instructions,
    title_attr="instruction_title",
    field_labels={"extracted_instruction": "Extracted Text"},
    card_title_prefix="Instruction"
)
```

This will render a series of cards, each showing one instruction title and the exact text from the prompt it was derived from. This visual breakdown is your first checkpoint for understanding what your prompt is explicitly asking the model to do.
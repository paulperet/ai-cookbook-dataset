# Context Engineering for Personalization - State Management with Long-Term Memory Notes using OpenAI Agents SDK

Modern AI agents are no longer just reactive assistants—they’re becoming adaptive collaborators. The leap from “responding” to “remembering” defines the new frontier of **context engineering**. At its core, context engineering is about shaping what the model knows at any given moment. By managing what’s stored, recalled, and injected into the model’s working memory, we can make an agent that feels personal, consistent, and context-aware.

The `RunContextWrapper` in the **OpenAI Agents SDK** provides the foundation for this. It allows developers to define structured state objects that persist across runs, enabling memory, notes, or even preferences to evolve over time. When paired with hooks and context-injection logic, this becomes a powerful system for **context personalization**—building agents that learn who you are, remember past actions, and tailor their reasoning accordingly.

This cookbook shows a **state-based long-term memory** pattern:

* **State object** = your local-first memory store (structured profile + notes)
* **Distill** memories during a run (tool call → session notes)
* **Consolidate** session notes into global notes at the end (dedupe + conflict resolution)
* **Inject** a well-crafted state at the start of each run (with precedence rules)

 ## Why Context Personalization Matters

 Context personalization is the **“magic moment”** when an AI agent stops feeling generic and starts feeling like *your* agent.

 It’s when the system remembers your coffee order, your company’s tone of voice, your past support tickets, or your preferred aisle seat—and uses that knowledge naturally, without being prompted.

 From a user perspective, this builds trust and delight: the agent appears to genuinely understand them. From a company perspective, it creates a **strategic moat**—a way to continuously capture, refine, and apply high-quality behavioral data. If implemented carefully, you can capture denser, higher-signal information about your users than typical clicks, impressions, or history data. Each interaction becomes a signal for better service, higher retention, and deeper insight into user needs.

 This value extends beyond the agent itself. When managed rigorously and safely, personalized context can also empower **human-facing roles**—support agents, account managers, travel advisors—by giving them a richer, longitudinal understanding of the customer. Over time, analyzing accumulated memories reveals how user preferences, behaviors, and goals evolve, enabling smarter product decisions and more adaptive systems.

 In practice, effective personalization means maintaining structured state—preferences, constraints, prior outcomes—and injecting only the *relevant* slices into the agent’s context at the right moment. Different agents demand different memory lifecycles: a life-coaching agent may require fast-evolving, nuanced memories, while an IT troubleshooting agent benefits from slower, more predictable state. Done well, personalization transforms a stateless chatbot into a persistent digital collaborator.

## Real-World Scenario: Travel Concierge Agent

We’ll ground this tutorial in a **travel concierge** agent that helps users book flights, hotels, and car rentals with a high degree of personalization.

In this tutorial, you’ll build an agent that:

* starts each session with a structured user profile and curated memory notes
* captures new durable preferences (for example, “I’m vegetarian”) via a dedicated tool
* consolidates those preferences into long-term memory at the end of each run
* resolves conflicts using a clear precedence order: **latest user input → session overrides → global defaults**


**Architecture at a Glance**

This section summarizes how state and memory flow across sessions.

1. Before the Session Starts

* A **state object** (user profile + global memory notes) is stored locally in your system.
* This state represents the agent’s long-term understanding of the user.

2. At the Start of a New Session

* The state object is injected into the **system prompt**:
  * Structured fields are included as **YAML frontmatter**
  * Unstructured memories are included as a **Markdown memory list**

3. During the Session

* As the agent interacts with the user, it captures candidate memories using
  `save_memory_note(...)`.
* These notes are written to **session memory** within the state object.

4. When the Context Is Trimmed

* If context trimming occurs (e.g., to avoid hitting the context limit):
  * Session-scoped memory notes are reinjected into the system prompt
  * This preserves important short-term context across long-running sessions

5. At the End of the Session

* A **consolidation job** runs asynchronously:
  * Session notes are merged into global memory
  * Conflicts are resolved and duplicates are removed

6. Next Run

* The updated state object is reused.
* The lifecycle repeats from the beginning.


## AI Memory Architecture Decisions

AI memory is still a new concept, and there is no one-size-fits-all solution. In this cookbook, we make design decisions based on a well-defined use case: a Travel Concierge agent.

## 1. Retrieval-Based vs State-Based Memory

Considering the many challenges in retrieval-based memory mechanisms including the need to train the model, state-based memory is better suited than retrieval-based memory for a travel concierge AI agent because travel decisions depend on continuity, priorities, and evolving preferences—not ad-hoc search. A travel agent must reason over a *current, coherent user state* (loyalty programs, seat preferences, budgets, visa constraints, trip intent, and temporary overrides like “this time I want to sleep”) and consistently apply it across flights, hotels, insurance, and follow-ups. 

Retrieval-based memory treats past interactions as loosely related documents, making it brittle to phrasing, prone to missing overrides, and unable to reconcile conflicts or updates over time. In contrast, state-based memory encodes user knowledge as structured, authoritative fields with clear precedence (global vs session), supports belief updates instead of fact accumulation, and enables deterministic decision-making without relying on fragile semantic search. This allows the agent to behave less like a search engine and more like a persistent concierge—maintaining continuity across sessions, adapting to context, and reliably using memory whenever it is relevant, not just when it is successfully retrieved.

## 2. Shape of a Memory

The shape of an agent’s memory is entirely driven by the use case. A reliable way to design it is to start with a simple question:

> *If this were a human agent performing the same task, what would they actively hold in working memory to get the job done? What details would they track, reference, or infer in real time?*

This framing grounds memory design in *task-relevance*, not arbitrary persistence.

**Metaprompting for Memory Extraction**

Use this pattern to elicit the memory schema for any workflow:

**Template**

> *You are a **[USE CASE]** agent whose goal is **[GOAL]**.
> What information would be important to keep in working memory during a single session?
> List both **fixed attributes** (always needed) and **inferred attributes** (derived from user behavior or context).*

Combining **predefined structured keys** with **unstructured memory notes** provides the right balance for a travel concierge agent—enabling reliable personalization while still capturing rich, free-form user preferences. In this design, the quality of your internal data systems becomes critical: structured fields should be consistently hydrated and kept up to date from trusted internal sources, while unstructured memories fill in the gaps where flexibility is required.

For this cookbook, we keep things simple by sourcing memory notes only from explicit user messages. In more advanced agents, this definition naturally expands to include signals from tool calls, system actions, and full execution traces, enabling deeper and more autonomous memory formation.

### Structured Memory (Schema-driven, machine-enforceable, predictable)

These should follow strict formats, be validated, and used directly in logic, filtering, or booking APIs.

**Identity & Core Profile**
* Global customer ID
* Full name
* Date of birth
* Gender
* Passport expiry date

**Loyalty & Programs**
* Airline loyalty status
* Hotel loyalty status
* Loyalty IDs

**Preferences & Coverage**
* Seat preference
* Insurance coverage profile:
  * Car rental coverage type
  * Travel medical coverage status
  * Coverage level (e.g., primary, secondary)

**Constraints**
* Visa requirements (array of country / region codes)

### Unstructured Memory (Narrative, contextual, semantic)

These are freeform and optimized for reasoning, personalization, and human-like decision-making.

**Global Memory Notes**
* “User usually prefers aisle seats.”
* “For trips shorter than a week, user generally prefers not to check bags.”
* “User prefers coverage that includes collision damage waiver and zero deductible when available.”


**Tip:** Do not dump all the fields from internal systems into the profile section. Make sure that every single token you add here helps agent to make better decisions. Some these fields might even be an input parameter to a tool call that you can pass from the state object without making it visible to the model.

Using the `RunContextWrapper`, the agent maintains a persistent `state` object containing structured data such as:


## 3. Memory Scope

Separate memory by **scope** to reduce noise and make evolution safer over time.

### User-Level Memory (Global Notes)

Durable preferences that should persist across sessions and influence future interactions.

**Examples:**

* “Prefers aisle seats”
* “Vegetarian”
* “United Gold status”

These are injected at the start of each session and updated cautiously during consolidation.

### Session-Level Memory (Session Notes)

Short-lived or contextual information relevant only to the current interaction.

**Examples:**

* “This trip is a family vacation”
* “Budget under $2,000 for this trip”
* “I prefer window seat this time for the red eye flight.”

Session notes act as a staging area and are promoted to global memory only if they prove durable.

**Rule of thumb:** if it should affect future trips by default, store it globally; if it only matters now, keep it session-scoped.


 ```json
 {
   "profile": {
     "global_customer_id": "crm_12345",
     "name": "John Doe",
     "age": 31,
     "home_city": "San Francisco",
     "currency": "USD",
     "passport_expiry_date": "2029-06-12",
     "loyalty_status": {"airline": "United Gold", "hotel": "Marriott Titanium"},
     "loyalty_ids": {"marriott": "MR998877", "hilton": "HH445566", "hyatt": "HY112233"},
     "seat_preference": "aisle",
     "tone": "concise and friendly",
     "active_visas": ["Schengen", "US"],
     "tight_connection_ok": false,
     "insurance_coverage_profile": {
       "car_rental": "primary_cdw_included",
       "travel_medical": "covered"
     }
   },
   "global_memory": {
     "notes": [
       {
         "text": "For trips shorter than a week, user generally prefers not to check bags.",
         "last_update_date": "2025-04-05",
         "keywords": ["baggage"]
       },
       {
         "text": "User usually prefers aisle seats.",
         "last_update_date": "2024-06-25",
         "keywords": ["seat_preference"]
       },
       {
         "text": "User generally likes staying in central, walkable city-center neighborhoods.",
         "last_update_date": "2024-02-11",
         "keywords": ["neighborhood"]
       },
       {
         "text": "User generally likes to compare options side-by-side.",
         "last_update_date": "2023-02-17",
         "keywords": ["pricing"]
       },
       {
         "text": "User prefers high floors.",
         "last_update_date": "2023-02-11",
         "keywords": ["room"]
       }
     ]
   }
 }
 ```

## 4. Memory Lifecycle

Memory is not static. Over time, you can analyze user behavior to identify different patterns, such as:

* **Stability** — preferences that rarely change (e.g., “seat preference is almost always aisle”)
* **Drift** — gradual changes over time (e.g., “average trip budget has increased month over month”)
* **Contextual variance** — preferences that depend on context (e.g., “business trips vs. family trips behave differently”)

These signals should directly influence your memory architecture:

* Stable, repeatedly confirmed preferences can be **promoted** from free-form notes into structured profile fields.
* Volatile or context-dependent preferences should remain as notes, often with **recency weighting**, confidence scores, or a TTL.

In other words, **memory design should evolve** as the system learns what is durable versus situational.

### 4.1 Memory Distillation

Memory distillation extracts high-quality, durable signals from the conversation and records them as memory notes.

In this cookbook, distillation is performed **during live turns** via a dedicated tool, enabling the agent to capture preferences and constraints as they are explicitly expressed.

An alternative approach is **post-session memory distillation**, where memories are extracted at the end of the session using the full execution trace. This can be especially useful for incorporating signals from tool usage patterns and internal reasoning that may not surface directly in user-facing turns.

### 4.2 Memory Consolidation

Memory consolidation runs asynchronously at the end of each session, graduating eligible session notes into global memory when appropriate.

This is the **most sensitive and error-prone stage** of the lifecycle. Poor consolidation can lead to context poisoning, memory loss, or long-term hallucinations. Common failure modes include:

* Losing meaningful information through over-aggressive pruning
* Promoting noisy, speculative, or unreliable signals
* Introducing contradictions or duplicate memories over time

To maintain a healthy memory system, consolidation must explicitly handle:

* **Deduplication** — merging semantically equivalent memories
* **Conflict resolution** — choosing between competing or outdated facts
* **Forgetting** — pruning stale, low-confidence, or superseded memories

Forgetting is not a bug—it is essential. Without careful pruning, memory stores will accumulate redundant and outdated information, degrading agent quality over time. Well-curated prompts and strict consolidation instructions are critical to controlling the aggressiveness and safety of this step.


### 4.3 Memory Injection

Inject curated memory back into the model context at the start of each session.
In this cookbook, injection is implemented via hooks that run after context trimming and before the agent begins execution, under the global memory section. High-signal memory in the system prompt is extremely effective for latency.



## Techniques Covered

To address these challenges, this cookbook applies a set of design decisions tailored to this specific agent, implemented using the **[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)**. The techniques below work together to enable reliable, controllable memory and context personalization:

* **State Management** – Maintain and evolve the agent’s [persistent state](https://openai.github.io/openai-agents-python/context/) using the `RunContextWrapper` class.

  * Pre-populate and curate key fields from internal systems before each session begins.

* **Memory Injection** – Inject only the relevant portions of state into the agent’s context at the start of each session.

  * Use **YAML frontmatter** for structured, machine-readable metadata.
  * Use **Markdown notes** for flexible, human-readable memory.

* **Memory Distillation** – Capture dynamic insights during active turns by writing session notes via a dedicated tool.

* **Memory Consolidation** – Merge session-level notes into a dense, conflict-free set of global memories.

  * **Forgetting**: Prune stale, overwritten, or low-signal memories during consolidation, and deduplicate aggressively over time.

Two-phase memory processing (note taking → consolidation) is more reliable than one-shot build the whole memory system at once.

All techniques in this cookbook are implemented in a **local-first** manner. Session and global memories live in your own state object and can be kept **ZDR (Zero Data Retention)** by design, as long as you avoid remote persistence.

These approaches are intentionally **zero-shot**—relying on prompting, orchestration, and lightweight scaffolding rather than training. Once the end-to-end design and evaluations are validated, a natural next step is **fine-tuning** to achieve stronger and more consistent memory behaviors such as extraction, consolidation, and conflict resolution.



Over time, the concierge becomes more efficient and human-like:

* It auto-suggests flights that match the user’s seat preference.
* It filters hotels by loyalty tier benefits.
* It pre-fills rental forms with known IDs and preferences.

This pattern exemplifies how **context engineering + state management** turn personalization into a sustainable differentiator. Rather than retraining models or embedding static rules, you evolve the *state layer*—a dynamic, inspectable memory the model can reason over.


## Step 0 — Prerequisites

Before running this cookbook, you must set up the following accounts and complete a few setup actions. These prerequisites are essential to interact with the APIs used in this project.

#### Step 0.1: OpenAI Account and `OPENAI_API_KEY`

- **Purpose:**  
  You need an OpenAI account to access language models and use the Agents SDK featured in this cookbook.

- **Action:**  
  [Sign up for an OpenAI account](https://openai.com) if you don’t already have one. Once you have an account, create an API key by visiting the [OpenAI API Keys page](https://platform.openai.com/api-keys).

**Before running the workflow, set your environment variables:**

```
# Your openai key
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
```

Alternatively, you can set your OpenAI API key for use by the agents via the `set_default_openai_key` function by importing agents library .

```
from agents import set_default_openai_key
set_default_openai_key("YOUR_API_KEY")
```

#### Step 0.2: Install the Required Libraries

Below we install the `openai-agents` library ([OpenAI Agents SDK](https://github.com/openai/openai-agents-python))


```python
%pip install openai-agents nest_asyncio
```

```python
from openai import OpenAI

client = OpenAI()
```

Let's test the installed libraries by defining and running an agent.


```python
import asyncio
from agents import Agent, Runner, set_tracing_disabled

set_tracing_disabled(True)

agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)
# Quick Test
result = await Runner.run(agent, "Tell me why it is important to evaluate AI agents.")
print(result.final_output)
```

    Evaluating AI agents ensures they are accurate, safe, reliable, ethical, and effective for their intended tasks.


## Step 1 — Define the State Object (Local-First Memory Store)

We start by defining a **local-first state object** that serves as the single source of truth for personalization and memory. This state is initialized at the beginning of each run and evolves over time.

The state includes:

* **`profile`**
  Structured, predefined fields (often hydrated from internal systems or CR
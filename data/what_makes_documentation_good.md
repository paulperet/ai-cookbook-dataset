# A Guide to Writing Effective Technical Documentation

This guide provides practical advice for creating clear, helpful, and user-friendly technical documentation. The goal is to transfer useful information efficiently from the page to the reader's mind.

## 1. Make Documentation Easy to Skim

Readers rarely read documentation from start to finish. They scan to find answers. Structure your content to support this behavior.

### 1.1. Use Clear Section Titles
*   **Action:** Break content into logical sections with descriptive titles.
*   **Why:** Titles act as signposts, helping readers decide where to focus.
*   **Tip:** Prefer informative, sentence-like titles over abstract nouns.
    *   **Less Effective:** "Results"
    *   **More Effective:** "Streaming Reduced Time to First Token by 50%"

### 1.2. Provide a Table of Contents
*   **Action:** Include a table of contents (TOC) for longer documents.
*   **Why:** A TOC enables fast navigation and gives readers an immediate overview of the document's scope and value.

### 1.3. Structure Text for Scannability
*   **Keep paragraphs short.** Essential points can stand alone in a one-sentence paragraph.
*   **Start with topic sentences.** Begin paragraphs and sections with a short, standalone sentence that previews the content. Avoid sentences that depend on prior text.
    *   **Less Effective:** "Building on top of this, let’s now talk about a faster way."
    *   **More Effective:** "Vector databases can speed up embeddings search."
*   **Front-load keywords.** Place the topic word at the beginning of your topic sentence.
    *   **Less Scannable:** "Embeddings search can be sped up by vector databases."
    *   **More Scannable:** "Vector databases speed up embeddings search."

### 1.4. Prioritize and Highlight Information
*   **Put takeaways first.** Place the most important information at the top of documents and sections. Avoid long introductions before presenting key results.
*   **Use visual aids.** Employ bulleted lists and tables frequently to present information clearly.
*   **Bold important text.** Use bold formatting to help critical information stand out during a scan.

## 2. Write Clearly and Concisely

Poor writing creates cognitive load. Minimize this burden on your readers.

### 2.1. Simplify Sentence Structure
*   **Keep sentences short.** Split long sentences. Remove unnecessary words and adverbs. Use the imperative mood for instructions.
*   **Ensure clear parsing.** Write sentences that are unambiguous from the start.
    *   **Ambiguous:** "Title sections with sentences." (Is "Title" a noun or verb?)
    *   **Clear:** "Write section titles as sentences."
*   **Avoid left-branching sentences.** These force readers to hold information in memory until they reach the main point. Prefer right-branching structures.
    *   **Left-branching:** "You need flour, eggs, milk, butter and a dash of salt to make pancakes."
    *   **Right-branching:** "To make pancakes, you need flour, eggs, milk, butter, and a dash of salt."

### 2.2. Maintain Clarity and Consistency
*   **Minimize demonstrative pronouns.** Avoid "this" or "that" across sentences, as they require the reader to recall the antecedent. Be specific.
    *   **Less Clear:** "Building on our discussion of the previous topic, now let’s discuss function calling."
    *   **More Clear:** "Building on message formatting, now let’s discuss function calling."
*   **Be rigorously consistent.** Apply the same formatting, naming conventions (e.g., Title Case, terminal commas), and stylistic choices throughout your documentation. Inconsistencies distract readers from the content.

### 2.3. Adopt a Helpful Tone
*   **Avoid presuming reader intent.** Don't tell readers what they are probably thinking or what they must do next.
    *   **Presumptive:** "Now you probably want to understand how to call a function."
    *   **Neutral:** "To call a function, ..."

## 3. Aim for Broad Accessibility

Readers have diverse backgrounds, language skills, and patience levels. Write to help as many as possible.

### 3.1. Use Simple Language
*   **Explain concepts simply.** Assume less knowledge than you think is necessary. This benefits non-native English speakers and those new to the technical domain.
*   **Avoid abbreviations.** Spell terms out on first use. The minor cost to experts is outweighed by the significant benefit to beginners.
    *   **Instead of:** "We'll use RAG for this."
    *   **Write:** "We'll use retrieval-augmented generation (RAG) for this."

### 3.2. Proactively Solve Problems
*   **Err on the side of over-explaining.** Include basic setup steps (e.g., installing a Python package, setting environment variables). Experts can skim past them, but omitting them can block beginners.
*   **Use specific, accurate terminology.** Prefer self-evident terms over field-specific jargon when possible.
    *   **Jargon:** "prompt", "context limit"
    *   **Clearer:** "input", "max token limit"

### 3.3. Create Effective Examples
*   **Keep code examples general and self-contained.** Minimize dependencies and external references. Each example should be simple and runnable on its own.
*   **Teach best practices.** Never demonstrate bad habits, such as hard-coding API keys in source files.
*   **Prioritize high-value topics.** Focus documentation effort on common tasks and frequent problems.

### 3.4. Provide Context
*   **Start with a broad opening.** When introducing a technical topic, briefly connect it to familiar, real-world applications. This grounds the reader and builds confidence before diving into details.

## 4. Apply Judgment

These guidelines are not absolute laws. Documentation is an exercise in empathy.

**Break these rules when you have a good reason.** Always put yourself in the reader's position. The ultimate goal is to create the most helpful resource possible for them. Use these principles as tools to achieve that.
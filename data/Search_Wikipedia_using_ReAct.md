# Build a ReAct Agent with Gemini to Search Wikipedia

This guide walks you through implementing a **ReAct (Reasoning + Acting)** agent using the Google `gemini-2.0-flash` model. You'll configure the model to answer questions by searching Wikipedia, observing the results, and reasoning step-by-step.

By the end, you will know how to:
1.  Set up your environment and authenticate with the Gemini API.
2.  Construct a ReAct prompt with in-context examples.
3.  Connect the model to the Wikipedia API as external tools.
4.  Run a multi-turn conversation where the model reasons and acts to find answers.

## Prerequisites & Setup

First, install the required packages and configure your API key.

```bash
pip install -q "google-generativeai>=0.7.2"
pip install -q wikipedia
```

Now, import the necessary libraries.

```python
import re
import os
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
import google.generativeai as genai
```

Authenticate with the Gemini API. Store your API key in an environment variable named `GOOGLE_API_KEY`.

```python
# Set your API key. In a notebook environment like Colab, you might use:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
```

## Step 1: Construct the ReAct Prompt

ReAct prompts the model to interleave **Thought**, **Action**, and **Observation** steps. We'll use a prompt adapted from the original ReAct paper, which includes detailed instructions and few-shot examples.

Define the core instructions for the model.

```python
model_instructions = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, Observation is understanding relevant information from an Action's output and Action can be of three types:
(1) <search>entity</search>, which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search and you can try to search the information from those topics.
(2) <lookup>keyword</lookup>, which returns the next sentence containing keyword in the current context. This only does exact matches, so keep your searches short.
(3) <finish>answer</finish>, which returns the answer and finishes the task.
"""
```

Provide the model with concrete examples to learn from (few-shot prompting). This significantly improves its ability to follow the ReAct format.

```python
examples = """
Here are some examples.

Question
What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

Thought 1
I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.

Action 1
<search>Colorado orogeny</search>

Observation 1
The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.

Thought 2
It does not mention the eastern sector. So I need to look up eastern sector.

Action 2
<lookup>eastern sector</lookup>

Observation 2
The eastern sector extends into the High Plains and is called the Central Plains orogeny.

Thought 3
The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.

Action 3
<search>High Plains</search>

Observation 3
High Plains refers to one of two distinct land regions

Thought 4
I need to instead search High Plains (United States).

Action 4
<search>High Plains (United States)</search>

Observation 4
The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130m).

Thought 5
High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.

Action 5
<finish>1,800 to 7,000 ft</finish>

Question
Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?

Thought 1
The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.

Action 1
<search>Milhouse</search>

Observation 1
Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Thought 2
The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".

Action 2
<lookup>named after</lookup>

Observation 2
Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.

Thought 3
Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.

Action 3
<finish>Richard Nixon</finish>

Question
Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?

Thought 1
I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.

Action 1
<search>Adam Clayton Powell</search>

Observation 1
Could not find [Adam Clayton Powell]. Similar: [’Adam Clayton Powell III’, ’Seventh Avenue (Manhattan)’, ’Adam Clayton Powell Jr. State Office Building’, ’Isabel Washington Powell’, ’Adam Powell’, ’Adam Clayton Powell (film)’, ’Giancarlo Esposito’].

Thought 2
To find the documentary, I can search Adam Clayton Powell (film).

Action 2
<search>Adam Clayton Powell (film)</search>

Observation 2
Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg. The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.

Thought 3
Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.

Action 3
<finish>The Saimaa Gesture</finish>

Question
What profession does Nicholas Ray and Elia Kazan have in common?

Thought 1
I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.

Action 1
<search>Nicholas Ray</search>

Observation 1
Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.

Thought 2
Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.

Action 2
<search>Elia Kazan</search>

Observation 2
Elia Kazan was an American film and theatre director, producer, screenwriter and actor.

Thought 3
Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.

Action 3
<finish>director, screenwriter, actor</finish>

Question
Which magazine was started first Arthur’s Magazine or First for Women?

Thought 1
I need to search Arthur’s Magazine and First for Women, and find which was started first.

Action 1
<search>Arthur’s Magazine</search>

Observation 1
Arthur’s Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.

Thought 2
Arthur’s Magazine was started in 1844. I need to search First for Women next.

Action 2
<search>First for Women</search>

Observation 2
First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.

Thought 3
First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First for Women), so Arthur’s Magazine was started first.

Action 3
<finish>Arthur’s Magazine</finish>

Question
Were Pavel Urysohn and Leonid Levin known for the same type of work?

Thought 1
I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.

Action 1
<search>Pavel Urysohn</search>

Observation 1
Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.

Thought 2
Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.

Action 2
<search>Leonid Levin</search>

Observation 2
Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.

Thought 3
Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.

Action 3
<finish>yes</finish>

Question
{question}"""
```

Combine the instructions and examples to form the complete prompt. You can save it to a file for reuse.

```python
ReAct_prompt = model_instructions + examples

# Optionally save the prompt
with open('model_instructions.txt', 'w') as f:
    f.write(ReAct_prompt)
```

## Step 2: Build the ReAct Agent Pipeline

Now, you'll create a `ReAct` class that manages the conversation with the Gemini model and handles the tool calls (search, lookup, finish).

### 2.1 Define the Agent Class

The `ReAct` class initializes a Gemini chat session and loads the prompt.

```python
class ReAct:
    def __init__(self, model: str, ReAct_prompt: str | os.PathLike):
        """Prepares Gemini to follow a `Few-shot ReAct prompt` by imitating
        `function calling` technique to generate both reasoning traces and
        task-specific actions in an interleaved manner.

        Args:
            model: name of the model.
            ReAct_prompt: ReAct prompt OR path to the ReAct prompt file.
        """
        self.model = genai.GenerativeModel(model)
        self.chat = self.model.start_chat(history=[])
        self.should_continue_prompting = True
        self._search_history: list[str] = []
        self._search_urls: list[str] = []

        try:
            # Try to read the prompt from a file
            with open(ReAct_prompt, 'r') as f:
                self._prompt = f.read()
        except FileNotFoundError:
            # Assume the parameter is the prompt string itself
            self._prompt = ReAct_prompt

    @property
    def prompt(self):
        return self._prompt

    @classmethod
    def add_method(cls, func):
        """Class method decorator to add functions as methods."""
        setattr(cls, func.__name__, func)
        return func

    @staticmethod
    def clean(text: str):
        """Helper function to clean text responses."""
        text = text.replace("\n", " ")
        return text
```

### 2.2 Implement the Tools (Search, Lookup, Finish)

The agent needs tools to interact with Wikipedia. We'll define them as methods.

**1. The `search` Tool**: Queries Wikipedia and returns a summary.

```python
@ReAct.add_method
def search(self, query: str):
    """Performs search on `query` via Wikipedia API and returns its summary.

    Args:
        query: Search parameter for the Wikipedia API.

    Returns:
        observation: Summary of the Wikipedia page or similar search results.
    """
    observation = None
    query = query.strip()
    try:
        # Try to get the summary for the requested `query`
        observation = wikipedia.summary(query, sentences=4, auto_suggest=False)
        wiki_url = wikipedia.page(query, auto_suggest=False).url
        observation = self.clean(observation)

        # Return the first 2-3 sentences as context
        observation = self.model.generate_content(
            f'Return the first 2 or 3 sentences from the following text: {observation}'
        )
        observation = observation.text

        # Keep track of search history
        self._search_history.append(query)
        self._search_urls.append(wiki_url)
        print(f"Information Source: {wiki_url}")

    # If the page is ambiguous or doesn't exist, return similar search phrases
    except (DisambiguationError, PageError) as e:
        observation = f'Could not find ["{query}"].'
        search_results = wikipedia.search(query)
        observation += f' Similar: {search_results}. You should search for one of those instead.'

    return observation
```

**2. The `lookup` Tool**: Searches for a specific phrase within the last searched page.

```python
@ReAct.add_method
def lookup(self, phrase: str, context_length=200):
    """Searches for the `phrase` in the latest Wikipedia search page.

    Args:
        phrase: Lookup phrase to search for within a page.
        context_length: Number of words to consider around the phrase.

    Returns:
        result: Context related to the `phrase` within the page.
    """
    # Get the last searched Wikipedia page
    page = wikipedia.page(self._search_history[-1], auto_suggest=False)
    page = page.content
    page = self.clean(page)
    start_index = page.find(phrase)

    # Extract context around the phrase
    result = page[max(0, start_index - context_length):start_index + len(phrase) + context_length]
    print(f"Information Source: {self._search_urls[-1]}")
    return result
```

**3. The `finish` Tool**: Stops the agent's execution loop.

```python
@ReAct.add_method
def finish(self, _):
    """Finishes the conversation on encountering <finish> token."""
    self.should_continue_prompting = False
    print(f"Information Sources: {self._search_urls}")
```

### 2.3 Imitate Function Calling with Stop Sequences

The core of the ReAct pipeline is intercepting the model's output when it generates an action token (`<search>`, `<lookup>`, `<finish>`). We use the `stop_sequences` parameter to make the model pause after generating one of these tokens. Then, we parse the token, call the corresponding tool, and feed the result back as an "Observation".

```python
@ReAct.add_method
def __call__(self, user_question, max_calls: int = 8, **generation_kwargs):
    """Starts a multi-turn conversation where the model reasons and acts.

    Args:
        user_question: The question from the user.
        max_calls: Maximum number of reasoning steps (Thought-Action cycles).
        **generation_kwargs: Additional arguments for model generation.

    Returns:
        The final answer from the model.
    """
    # Format the prompt with the user's question
    prompt = self.prompt.format(question=user_question)
    self.chat = self.model.start_chat(history=[])
    self.should_continue_prompting = True
    self._search_history.clear()
    self._search_urls.clear()

    # Define the action tokens that will trigger a tool call
    stop_sequences = ["<search>", "<lookup>", "<finish>"]
    response = self.chat.send_message(
        prompt,
        generation_config=genai.types.GenerationConfig(
            stop_sequences=stop_sequences,
            **generation_kwargs
        )
    )

    # Main loop for ReAct steps
    for i in range(max_calls):
        if not self.should_continue_prompting:
            break

        print(f"\n--- Step {i+1} ---")
        print(f"Thought: {response.text}")

        # Check which action token stopped the generation
        if response.candidates[0].finish_reason == 3:  # STOP
            # Find which stop sequence was matched
            for seq in stop_sequences:
                if seq in response.text:
                    action_token = seq
                    break

            # Extract the content inside the action tags (e.g., 'Colorado orogeny')
            match = re.search(f"{action_token}(.*?){action_token.replace('<', '</')}", response.text + action_token.replace('<', '</'))
            if match:
                action_input = match.group(1).strip()
            else:
                action_input = ""

            print(f"Action: {action_token}{action_input}{action_token.replace('<', '</')}")

            # Map the action token to the corresponding tool and call it
            if action_token == "<search>":
                observation = self.search(action_input)
            elif action_token == "<lookup>":
                observation = self.lookup(action_input)
            elif action_token == "<finish>":
                observation = self.finish(action_input)
                # The finish tool sets self.should_continue_prompting = False
                # The observation is the final answer
                observation = action_input
                print(f"Final Answer: {observation}")
                return observation
            else:
                observation = "Unknown action."

            print(f"Observation: {observation}")

            # Send the observation back to the model to continue the chain
            response = self.chat.send_message(
                f"Observation: {observation}",
                generation_config=genai.types.GenerationConfig(
                    stop_sequences=stop_sequences,
                    **generation_kwargs
                )
            )
        else:
            # If no stop sequence was hit, continue generation
            response = self.chat.send_message(
                "Continue.",
                generation_config=genai.types.GenerationConfig(
                    stop_sequences=stop_sequences,
                    **generation_kwargs
                )
            )

    print("Reached maximum number of steps.")
    return None
```

## Step 3: Run the Agent

Instantiate the agent with the `gemini-2.0-flash` model and your ReAct prompt, then ask it a question.

```python
# Initialize the ReAct agent
agent = ReAct(
    model="gemini-2
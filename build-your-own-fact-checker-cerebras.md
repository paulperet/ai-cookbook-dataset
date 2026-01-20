# **Build your own content fact-checker with OpenAI gpt-oss-120B, Cerebras, and Parallel**

Ever read an article only to discover later that some of the “facts” were fabricated? As information becomes more abundant, verifying its accuracy has become increasingly challenging.

This guide provides a practical, automated way to assess factual accuracy at scale. It extracts claims from any text or URL, retrieves real-world evidence, and evaluates each claim using gpt-oss-120B powered by Cerebras ultra low latency inference.

See demo here: [Content Fact-Checker](https://oss.parallel.ai/agents/cerebras-fact-checker).

For this guide, set up the following accounts: 

- Cerebras API: the fastest inference provider, [get started for free here.](https://cloud.cerebras.ai/?utm_source=DevX&utm_campaign=parallel)

- Parallel API: The search engine for AI, [get started for free here.](https://platform.parallel.ai/)

Learn more about best practices of gpt-oss-120B [here](https://openai.com/index/introducing-gpt-oss/).

### **Step 1: Environment Setup (Colab or local)**

This guide supports both local Jupyter environments and Google Colab. Set the following environment variables:
- CEREBRAS_API_KEY
- PARALLEL_API_KEY


```python
python3 -m pip install -U cerebras_cloud_sdk parallel-web
```


```python
import os
from cerebras.cloud.sdk import Cerebras
from parallel import Parallel

# API keys: Colab userdata (if available) -> env vars fallback
try:
    from google.colab import userdata  # type: ignore
    CEREBRAS_API_KEY = userdata.get("CEREBRAS_API_KEY") or os.getenv("CEREBRAS_API_KEY")
    PARALLEL_API_KEY = userdata.get("PARALLEL_API_KEY") or os.getenv("PARALLEL_API_KEY")
except ImportError:
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
    PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY")

if not CEREBRAS_API_KEY or not PARALLEL_API_KEY:
    raise RuntimeError("Set CEREBRAS_API_KEY and PARALLEL_API_KEY as environment variables.")

cerebras_client = Cerebras(
    api_key=CEREBRAS_API_KEY,
    default_headers={
        "X-Cerebras-3rd-Party-Integration": "parallel-ai-workshop"
    }
)

parallel_client = Parallel(api_key=PARALLEL_API_KEY)

CEREBRAS_MODEL_NAME = "gpt-oss-120B"

print("Clients initialized, model:", CEREBRAS_MODEL_NAME)
```

    Clients initialized, model: gpt-oss-120b


### **Step 2: Set up the LLM**

Now, with the environment ready, create the function that will call the LLM.



```python
def call_cerebras_chat(
    user_content: str,
    system_content: str | None = None,
    model: str = CEREBRAS_MODEL_NAME,
    temperature: float = 1.0,
    top_p= 1.0,
    max_tokens: int = 4096,
    reasoning_effort: str = "medium"
):
    """
    Calls the Cerebras chat completion API.

    Args:
        user_content (str): The user's message.
        system_content (str | None): Optional system message to guide the LLM.
        model (str): The Cerebras model to use.
        temperature (float): Controls the randomness of the output.
        max_tokens (int): The maximum number of tokens in the response.

    Returns:
        str: The content of the LLM's response.
    """
    messages = []
    # Add a system message to guide the model's behavior
    if system_content:
        messages.append({"role": "system", "content": system_content})

    messages.append({"role": "user", "content": user_content})

    # Make the API call to Cerebras chat completions
    resp = cerebras_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    return resp.choices[0].message.content
```

### **Step 3: Connect the LLM to the web**

To fact-check a claim, the model needs to find evidence online, and this step builds the function that connects the LLM to the web.

Notice a few fields:

- `objective` field: Bold text Natural language intent rather than keywords.

- `one-shot` mode: For simplicity and speed, this guide stick to a one-shot setup, which gives high-quality excerpts in a single call.


```python
def search_web(query: str, num: int = 5, mode: str = "one-shot"):
    """
    Search the web using Parallel's Search API.

    Returns a list of dicts with:
      - url
      - title
      - publish_date
      - excerpts (list of strings)
    """

    # Instruct the LLM find quality sources.
    objective = (
        f"Find high-quality, up-to-date sources that answer the question:\n\n{query}\n\n"
        "Prefer authoritative sites (e.g., .gov, .edu, major news, or official org websites)."
    )

    # Initiatiate the LLM for web search
    search = parallel_client.beta.search(
        objective=objective,
        search_queries=[query],
        mode=mode,
        max_results=num,
        excerpts={
            "max_chars_per_result": 8000,
        },
    )

    results = []
    # Process the search results and extract information like URL, title, and excerpts.
    for r in search.results:
        results.append(
            {
                "url": r.url,
                "title": getattr(r, "title", None),
                "publish_date": getattr(r, "publish_date", None),
                "excerpts": list(r.excerpts or []),
            }
        )
    return results
```

### **Step 4 – Organize and summarize web results**

After retrieving information from the web, organize it into a clean, readable format. This step takes the search results and compiles the key excerpts into a simple summary for evaluation.


```python
import textwrap
from typing import List, Dict, Any

def build_evidence_context(results: List[Dict[str, Any]], max_chars: int = 8000) -> str:

    blocks = []
    for idx, r in enumerate(results):
        excerpts_text = "\n\n".join(r["excerpts"][:2])
        block = textwrap.dedent(f"""
        [Source {idx+1}]
        Title: {r['title'] or r['url']}
        URL: {r['url']}
        Publish date: {r['publish_date']}

        Excerpts:
        {excerpts_text}
        """).strip()
        blocks.append(block)

    context = "\n\n".join(blocks)

    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[Context truncated for length]"
    return context
```

### **Step 5 – Find the claims to verify**

Next, identify the specific statements in the text to verify. Rather than analyzing an entire article at once, the LLM should break it into multiple clear, stand-alone claims that can be judged verbatim.

For example, from a short paragraph like: “The unemployment rate fell to 3.5% in March 2024, and Company X announced a $10B merger the same week.”

The LLM should extract individual factual statements such as:

*  “The unemployment rate fell to 3.5% in March 2024.”
*  “Company X announced a $10 billion merger.”

Each one can then be checked independently, which makes the entire fact-checking process precise and reliable.


```python
import json
import re
import time

def extract_claims_from_text(text: str, max_claims: int = 8) -> list[str]:
    """
    Use Cerebras LLM to extract atomic factual claims from text.
    Output format (strict JSON):
    {
      "claims": ["...", "."...]
    }
    """
    # Instruct the LLM to extract factual claims
    system_prompt_content = (
        "You are an information extraction assistant.\n"
        "From the user's text, extract up to {max_claims} atomic factual claims.\n"
        "Each claim should:\n"
        "- Be checkable against external sources (dates, numbers, named entities)\n"
        "- Be concrete and not an opinion.\n\n"
        "Return STRICT JSON:\n"
        "{{\n"
        '  "claims": ["...", "..."]\n'
        "}}\n"
    ).format(max_claims=max_claims)

    # Prompt the LLM for claim extraction
    user_prompt_content = f"Text:\n\n{text}\n\nExtract up to {max_claims} factual claims."

    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_prompt_content}
    ]

    start_time = time.time()
    # Call Cerebras LLM (gpt-oss-120B) for claim extraction
    resp = cerebras_client.chat.completions.create(
        model=CEREBRAS_MODEL_NAME,
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=4096,
        reasoning_effort="medium",
    )
    raw = resp.choices[0].message.content.strip()
    end_time = time.time()
    print(f"Cerebras LLM claim extraction took {end_time - start_time:.2f} seconds")

    # Clean up the raw JSON output
    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)

    try:
        data = json.loads(raw)
        claims = data.get("claims", [])
        claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
        return claims[:max_claims]
    except Exception as e:
        print("Error parsing claims JSON:", e)
        print("Raw model output:\n", raw)
        return []

print("Claim extraction ready")
```

    Claim extraction ready


### **Step 6 – Check claims against evidence (true / false / uncertain)**

After collecting the claims and extracting them into independent factual claims, the LLM can now evaluate each claim for a verdict. The process has two steps:

1) **Retrieve evidence with Parallel:**
First, use Parallel to query authoritative sources related to the claim.

2) **Judge the claim with Cerebras:**
Then, send the evidence and the original claims to Cerebras for evaluation. Here's where Cerebras's ultra-fast inference becomes crucial, where the LLM can analyze multiple pieces of evidence, weigh contradictions, and generate a verdict. 

The model will return one of three structured verdicts:
- **True** — Evidence supports the claim
- **False** — Evidence contradicts the claim
- **Uncertain** — Not enough evidence, or sources conflict

Each verdict comes with an explanation and cited URLs, so the model's reasoning is transparent.


```python
from typing import Dict, Any
import textwrap
import re
import time

def fact_check_single_claim(claim: str) -> Dict[str, Any]:
    """
    Fact-check a single claim using:
      - Parallel Search for evidence
      - Cerebras LLM for verdict

    Args:
        claim (str): The factual claim to be checked.

    Returns:
        Dict[str, Any]: A dictionary containing the claim, verdict, reason, and sources.
        {
          "claim": str,
          "verdict": "true" | "false" | "uncertain",
          "reason": str,
          "sources": [url, ...]
        }
    """
    print(f"\nFact-checking claim: {claim}")

    # Search the web for evidence relevant to the claim
    results = search_web(query=claim, num=6, mode="one-shot")
    print(f"Retrieved {len(results)} evidence sources")

    # Compile the search results into a clean, readable context for the LLM
    evidence_context = build_evidence_context(results)

    # Define the system prompt to instruct the Cerebras LLM (gpt-oss-120B) on how to evaluate each claim
    system_prompt_content = (
        "You are a careful, skeptical fact-checking assistant.\n"
        "You get a factual claim and web search excerpts.\n"
        "Decide if the evidence supports, contradicts, or does not clearly resolve the claim.\n\n"
        "Respond with STRICT JSON:\n"
        "{\n"
        '  "verdict": "true" | "false" | "uncertain",\n'
        '  "reason": "short explanation",\n'
        '  "top_sources": ["url1", "url2", ...]\n'
        "}\n"
        "Use 'true' only when the evidence strongly supports the claim.\n"
        "Use 'false' only when it clearly contradicts the claim.\n"
        "Otherwise use 'uncertain'."
    )

    # Construct the user prompt
    user_prompt_content = textwrap.dedent(f"""
    Claim:
    {claim}

    Evidence (web search excerpts):
    {evidence_context}
    """)

    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_prompt_content}
    ]

    start_time = time.time()
    # Call the Cerebras LLM (gpt-oss-120B) to get a structured verdict
    resp = cerebras_client.chat.completions.create(
        model=CEREBRAS_MODEL_NAME,
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=4096,    
        reasoning_effort="medium"
    )
    raw = resp.choices[0].message.content.strip()
    end_time = time.time()
    print(f"Cerebras LLM judgment for this claim took {end_time - start_time:.2f} seconds")

    # Clean up the raw JSON output from the LLM
    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)

    try:
        data = json.loads(raw)
    except Exception as e:
        print("Error parsing judgment JSON:", e)
        print("Raw model output:\n", raw)
        data = {
            "verdict": "uncertain",
            "reason": "Could not parse model output.",
            "top_sources": [],
        }

    # Extract and normalize the verdict (true, false, or uncertain)
    verdict = str(data.get("verdict", "uncertain")).lower()
    if verdict not in {"true", "false", "uncertain"}:
        verdict = "uncertain"

    # Extract and format the top sources cited by the LLM
    top_sources = data.get("top_sources") or []
    if not isinstance(top_sources, list):
        top_sources = [str(top_sources)]
    top_sources = [str(u) for u in top_sources][:5]

    # Consolidate all the fact-checking results into a single dictionary
    result = {
        "claim": claim,
        "verdict": verdict,
        "reason": data.get("reason", ""),
        "sources": top_sources,
    }

    # Print the detailed fact-checking result for clarity
    print("Verdict:", result["verdict"].upper())
    print("Reason:", result["reason"])
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print("  •", s)

    return result

print("Single-claim fact-checker ready")
```

    Single-claim fact checker ready


### **Step 7 - Fact-check an entire text**

This final step brings everything together. Here, take any piece of text and run each one through the full fact-checking process you built.


```python
def fact_check_text(text: str, max_claims: int = 6):
    # First, extract factual claims from the input text
    claims = extract_claims_from_text(text, max_claims=max_claims)

    print(f"Extracted {len(claims)} claims:")
    for i, c in enumerate(claims, 1):
        print(f"  {i}. {c}")

    all_results = []
    # Iterate through each extracted claim and perform a single fact-check
    for i, claim in enumerate(claims):
        print(f"\n{'='*50}\nFact-checking Claim {i+1} of {len(claims)}: '{claim}'")
        single_claim_result = fact_check_single_claim(claim)
        all_results.append(single_claim_result)
        print(f"{'='*50}")

    # After all claims are checked, print a summary of all results
    print("\n\n--- Summary of All Fact-Checking Results ---\n")
    for result in all_results:
        print(f"Claim: {result['claim']}")
        print(f"Verdict: {result['verdict'].upper()}")
        print(f"Reason: {result['reason']}")
        if result['sources']:
            print("Sources:")
            for s in result['sources']:
                print(f"  • {s}")
        print("\n" + "-"*50 + "\n")

    return all_results

print("Full fact-checking pipeline ready")
```

    Full fact-checking pipeline ready


### **Step 8: Fact check directly from a URL**
Finally, to make the fact-checker even easier, add a function that accepts a URL directly.


```python
import requests
from bs4 import BeautifulSoup

def extract_claims_from_url(url: str, max_claims: int = 8) -> list[str]:
    """
    Extracts atomic factual claims from the main content of a given URL.
    Fetches content using requests/BeautifulSoup and uses Cerebras LLM for claim extraction.
    """
    print(f"Fetching content from URL: {url}")
    try:
        # Fetch the content of the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

       
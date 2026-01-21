# Deep Research API Cookbook: Automate Complex Research Workflows

## Introduction

The Deep Research API enables you to automate complex research workflows that require reasoning, planning, and synthesis across real-world information. It is designed to take a high-level query and return a structured, citation-rich report by leveraging an agentic model capable of decomposing the task, performing web searches, and synthesizing results.

Unlike ChatGPT where this process is abstracted away, the API provides direct programmatic access. When you send a request, the model autonomously plans sub-questions, uses tools like web search and code execution, and produces a final structured response.

You can access Deep Research via the `responses` endpoint using the following models:
*   `o3-deep-research-2025-06-26`: Optimized for in-depth synthesis and higher-quality output.
*   `o4-mini-deep-research-2025-06-26`: Lightweight and faster, ideal for latency-sensitive use cases.

This guide will walk you through setting up, making your first API call, parsing the results, and extending the system with custom tools.

---

## 1. Setup and Authentication

### 1.1 Install the OpenAI Python SDK

First, ensure you have the latest version of the OpenAI Python SDK installed.

```bash
pip install --upgrade openai
```

### 1.2 Initialize the Client

Import the OpenAI client and initialize it with your API key.

```python
from openai import OpenAI

# Replace with your actual API key or set it as an environment variable
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
```

---

## 2. Your First Deep Research Query

Let's walk through an example. Imagine you're at a healthcare financial services firm tasked with producing an in-depth report on the economic implications of semaglutide, a medication for type 2 diabetes and obesity.

Our goal is to synthesize clinical outcomes, cost-effectiveness, and regional pricing data into a structured, citation-backed analysis.

### 2.1 Define the System Prompt and User Query

First, craft a system message to define the role and expectations for the researcher, and formulate the user's query.

```python
system_message = """
You are a professional researcher preparing a structured, data-driven report on behalf of a global health economics team. Your task is to analyze the health question the user poses.

Do:
- Focus on data-rich insights: include specific figures, trends, statistics, and measurable outcomes (e.g., reduction in hospitalization costs, market size, pricing trends, payer adoption).
- When appropriate, summarize data in a way that could be turned into charts or tables, and call this out in the response (e.g., “this would work well as a bar chart comparing per-patient costs across regions”).
- Prioritize reliable, up-to-date sources: peer-reviewed research, health organizations (e.g., WHO, CDC), regulatory agencies, or pharmaceutical earnings reports.
- Include inline citations and return all source metadata.

Be analytical, avoid generalities, and ensure that each section supports data-backed reasoning that could inform healthcare policy or financial modeling.
"""

user_query = "Research the economic impact of semaglutide on global healthcare systems."
```

### 2.2 Make the API Call

Now, send the request to the Deep Research API. We'll configure it to:
*   Use the `o3-deep-research` model.
*   Run in **background mode** (`background=True`) to handle longer execution times asynchronously.
*   Enable the `web_search_preview` and `code_interpreter` tools.
*   Set the reasoning summary to `"auto"` for a concise overview of the agent's steps.

```python
response = client.responses.create(
  model="o3-deep-research",
  input=[
    {
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": system_message,
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": user_query,
        }
      ]
    }
  ],
  reasoning={
    "summary": "auto"
  },
  tools=[
    {
      "type": "web_search_preview"
    },
    {
      "type": "code_interpreter",
      "container": {
        "type": "auto",
        "file_ids": []
      }
    }
  ]
)
```

---

## 3. Parsing the Response

The Deep Research API returns a rich response object containing the final report, inline citations, and a trace of all intermediate steps.

### 3.1 Extract the Final Report

The main text output of the report is located in the last item of the `response.output` list.

```python
# Access the final report from the response object
final_report_text = response.output[-1].content[0].text
print(final_report_text)
```

### 3.2 Access Inline Citations and Source Metadata

Citations in the response text are annotated and linked to source metadata. Each annotation contains the source's title, URL, and its character span in the text.

```python
annotations = response.output[-1].content[0].annotations
for i, citation in enumerate(annotations):
    print(f"Citation {i+1}:")
    print(f"  Title: {citation.title}")
    print(f"  URL: {citation.url}")
    print(f"  Location: chars {citation.start_index}–{citation.end_index}")
```

### 3.3 Inspect Intermediate Steps

You can debug or analyze the agent's process by examining the intermediate steps stored in `response.output`. Each step has a `type` field.

#### Find a Reasoning Step
These represent internal summaries or plans generated by the model.

```python
# Find the first reasoning step
reasoning = next(item for item in response.output if item.type == "reasoning")
for s in reasoning.summary:
    print(s.text)
```

#### Find a Web Search Call
This shows what queries were executed to retrieve information.

```python
# Find the first web search step
search = next(item for item in response.output if item.type == "web_search_call")
print("Query:", search.action["query"])
print("Status:", search.status)
```

#### Find a Code Execution Step
If the model used the code interpreter, you can inspect its input and output.

```python
# Find a code execution step (if any)
code_step = next((item for item in response.output if item.type == "code_interpreter_call"), None)
if code_step:
    print("Code Input:", code_step.input)
    print("Code Output:", code_step.output)
else:
    print("No code execution steps found.")
```

---

## 4. Extending with Custom Tools via MCP

You can integrate your own data sources using the **Model Context Protocol (MCP)**. For example, you can configure a tool that lets the Deep Research agent fetch internal documents.

> **Note:** To learn how to build an MCP server for this purpose, refer to the related cookbook: [How to Build a Deep Research MCP Server](https://cookbook.openai.com/examples/deep_research_api/how_to_build_a_deep_research_mcp_server/readme).

### 4.1 Update the System Prompt

First, update the system message to instruct the agent to use the new internal file lookup tool.

```python
system_message = """
You are a professional researcher... (previous content remains) ...
- Include an internal file lookup tool to retrieve information from our own internal data sources. If you’ve already retrieved a file, do not call fetch again for that same file. Prioritize inclusion of that data.
- Include inline citations and return all source metadata.
...
"""
```

### 4.2 Make the API Call with an MCP Tool

Add an MCP tool configuration to the `tools` list, pointing to your server's URL.

```python
response = client.responses.create(
  model="o3-deep-research-2025-06-26",
  input=[
    {
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": system_message,
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": user_query,
        }
      ]
    }
  ],
  reasoning={
    "summary": "auto"
  },
  tools=[
    {
      "type": "web_search_preview"
    },
    { # ADD MCP TOOL SUPPORT
      "type": "mcp",
      "server_label": "internal_file_lookup",
      "server_url": "https://<your_mcp_server>/sse/", # Update to your MCP server's location
      "require_approval": "never"
    }
  ]
)
```

### 4.3 Review the Response with MCP Citations

After receiving the response, you can filter citations to identify those sourced from your internal MCP server.

```python
# Grab the full report text
report_text = response.output[-1].content[0].text

print("REPORT EXCERPT:")
print(report_text[:100])   # first 100 chars
print("--------")

annotations = response.output[-1].content[0].annotations
target_url = "https://platform.openai.com/storage/files" # Example target domain for internal files

for citation in annotations:
    if citation.url.startswith(target_url):
        start, end = citation.start_index, citation.end_index
        excerpt = report_text[start:end]
        pre_start = max(0, start - 100)
        preceding_txt = report_text[pre_start:start]

        print("MCP CITATION SAMPLE:")
        print(f"  Title:       {citation.title}")
        print(f"  URL:         {citation.url}")
        print(f"  Location:    chars {start}–{end}")
        print(f"  Preceding:   {preceding_txt!r}")
        print(f"  Excerpt:     {excerpt!r}")
        break

print("--------")

# Print the MCP tool calls made during the research
calls = [
    (item.name, item.server_label, item.arguments)
    for item in response.output
    if item.type == "mcp_call" and item.arguments
]
for name, server, args in calls:
    print(f"{name}@{server} → {args}")
print("--------")
```

---

## 5. Optimizing Input: Prompt Rewriting and Clarification

The Deep Research API, unlike ChatGPT, does not ask clarifying questions. To get the best results, you must provide fully-formed, detailed prompts. You can use a two-step approach:

### 5.1 Use a Prompt Rewriter

First, use a lightweight model (like `gpt-4.1`) to expand a vague user query into a detailed set of research instructions.

```python
suggested_rewriting_prompt = """
You will be given a research task by a user. Your job is to produce a set of instructions for a researcher that will complete the task. Do NOT complete the task yourself, just provide instructions on how to complete it.

GUIDELINES:
1. **Maximize Specificity and Detail**
- Include all known user preferences and explicitly list key attributes or dimensions to consider.
- It is of utmost importance that all details from the user are included in the instructions.

2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
- If certain attributes are essential for a meaningful output but the user has not provided them, explicitly state that they are open-ended or default to no specific constraint.

3. **Avoid Unwarranted Assumptions**
- If the user has not provided a particular detail, do not invent one.
- Instead, state the lack of specification and guide the researcher to treat it as flexible or accept all possible options.

4. **Use the First Person**
- Phrase the request from the perspective of the user.

5. **Tables**
- If you determine that including a table will help illustrate, organize, or enhance the information in the research output, you must explicitly request that the researcher provide them.

6. **Headers and Formatting**
- You should include the expected output format in the prompt.
- If the user is asking for content that would be best returned in a structured format (e.g. a report, plan, etc.), ask the researcher to format as a report with the appropriate headers and formatting that ensures clarity and structure.

7. **Language**
- If the user input is in a language other than English, tell the researcher to respond in this language, unless the user query explicitly asks for the response in a different language.

8. **Sources**
- If specific sources should be prioritized, specify them in the prompt.
- For product and travel research, prefer linking directly to official or primary websites.
- For academic or scientific queries, prefer linking directly to the original paper or official journal publication.
- If the query is in a specific language, prioritize sources published in that language.
"""

# Example: Rewrite a vague user query
response = client.responses.create(
    instructions=suggested_rewriting_prompt,
    model="gpt-4.1-2025-04-14",
    input="help me plan a trip to france",
)

new_query = response.output[0].content[0].text
print(new_query)
```

### 5.2 Use a Clarification Step (Alternative)

Alternatively, you can use a model to ask the user clarifying questions *before* starting the research, ensuring the final query is well-scoped.

```python
suggested_clarifying_prompt = """
You will be given a research task by a user. Your job is NOT to complete the task yet, but instead to ask clarifying questions that would help you or another researcher produce a more specific, efficient, and relevant answer.

GUIDELINES:
1. **Maximize Relevance**
- Ask questions that are *directly necessary* to scope the research output.
- Consider what information would change the structure, depth, or direction of the answer.

2. **Surface Missing but Critical Dimensions**
- Identify essential attributes that were not specified in the user’s request (e.g., preferences, time frame, budget, audience).
- Ask about each one *explicitly*, even if it feels obvious or typical.

3. **Do Not Invent Preferences**
- If the user did not mention a preference, *do not assume it*. Ask about it clearly and neutrally.

4. **Use the First Person**
- Phrase your questions from the perspective of the assistant or researcher talking to the user.

5. **Use a Bulleted List if Multiple Questions**
- If there are multiple open questions, list them clearly in bullet format for readability.

6. **Avoid Overasking**
- Prioritize the 3–6 questions that would most reduce ambiguity or scope creep.

7. **Include Examples Where Helpful**
- If a question might be ambiguous, provide a short example of the kind of answer you're looking for.
"""

# You would use this prompt in a separate chat interaction with the user to gather details before submitting to Deep Research.
```

---

## Summary

You've now learned how to:
1.  **Set up** and authenticate with the Deep Research API.
2.  **Make your first query** to generate a structured, cited research report.
3.  **Parse the response** to extract the final report, citations, and agent reasoning steps.
4.  **Extend the system** by integrating custom data sources via MCP tools.
5.  **Optimize your inputs** using prompt rewriting or clarification steps to ensure high-quality, efficient research outputs.

The Deep Research API provides powerful, programmable control over complex research workflows, enabling you to build sophisticated, data-backed analysis tools.
# Building a One-Liner Research Agent

Research tasks consume hours of expert time: market analysts manually gathering competitive intelligence, legal teams tracking regulatory changes, engineers investigating bug reports across documentation. The core challenge isn't finding information but knowing what to search for next based on what you just discovered.

The Claude Agent SDK makes it possible to build agents that autonomously explore external systems without a predefined workflow. Unlike traditional workflow automations that follow fixed steps, research agents adapt their strategy based on what they find--following promising leads, synthesizing conflicting sources, and knowing when they have enough information to answer the question.

## By the end of this cookbook, you'll be able to:

- Build a research agent that autonomously searches and synthesizes information with a few lines of code

This foundation applies to any task where the information needed isn't available upfront: competitive analysis, technical troubleshooting, investment research, or literature reviews.

# Why Research Agents?

Research is an ideal agentic use case for two reasons:

1. **Information isn't self-contained**. The input question alone doesn't contain the answer. The agent must interact with external systems (search engines, databases, APIs) to gather what it needs.
2. **The path emerges during exploration**. You can't predetermine the workflow. Whether an agent should search for company financials or regulatory filings depends on what it discovers about the business model. The optimal strategy reveals itself through investigation.

In its simplest form, a research agent searches the web and synthesizes findings. Below, we'll build exactly that with the Claude Agent SDK's built-in web search tool in just a few lines of code.

Note: You can also view the full list of [Claude Code's built-in tools](https://docs.claude.com/en/docs/claude-code/settings#tools-available-to-claude)

# Prerequisites

Before following this guide, ensure you have:

**Required Knowledge**

* Python fundamentals - comfortable with async/await, functions, and basic data structures
* Basic understanding of agentic patterns - we recommend reading [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) first if you're new to agents

**Required Tools**

* Python 3.11 or higher
* Anthropic API key [(get one here)](https://console.anthropic.com)

**Recommended:**
* Familiarity with the Claude Agent SDK concepts
* Understanding of tool use patterns in LLMs


## Setup

First, install the required dependencies:


```python
%%capture
%pip install -U claude-agent-sdk python-dotenv
```

Note: Ensure your .env file contains:

```bash
ANTHROPIC_API_KEY=your_key_here
```

Load your environment variables and configure the client:


```python
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-5"
```

## Building Your First Research Agent

Let's start with the simplest possible implementation: a research agent that can search the web and synthesize findings. With the Claude Agent SDK, this takes just a few lines of code.

The key is the query() function, which creates a stateless agent interaction. We'll provide Claude with a single tool, WebSearch, and let it autonomously decide when and how to use it based on our research question.


```python
from utils.agent_visualizer import (
    display_agent_response,
    print_activity,
)

from claude_agent_sdk import ClaudeAgentOptions, query

messages = []
async for msg in query(
    prompt="Research the latest trends in AI agents and give me a brief summary and relevant citiations links.",
    options=ClaudeAgentOptions(model=MODEL, allowed_tools=["WebSearch"]),
):
    print_activity(msg)
    messages.append(msg)
```

    [🤖 Using: WebSearch(), ..., 🤖 Thinking...]



```python
display_agent_response(messages)
```



<style>
.pretty-card {
    font-family: ui-sans-serif, system-ui;
    border: 2px solid transparent;
    border-radius: 14px;
    padding: 14px 16px;
    margin: 10px 0;
    background: linear-gradient(#fff, #fff) padding-box,
                linear-gradient(135deg, #3b82f6, #9333ea) border-box;
    color: #111;
    box-shadow: 0 4px 12px rgba(0,0,0,.08);
}
.pretty-title {
    font-weight: 700;
    margin-bottom: 8px;
    font-size: 14px;
    color: #111;
}
.pretty-card pre,
.pretty-card code {
    background: #f3f4f6;
    color: #111;
    padding: 8px;
    border-radius: 8px;
    display: block;
    overflow-x: auto;
    font-size: 13px;
    white-space: pre-wrap;
}
.pretty-card img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
}
/* Tables: both pandas (.pretty-table) and markdown-rendered */
.pretty-card table {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
    color: #111;
    margin: 0.5em 0;
}
.pretty-card th,
.pretty-card td {
    border: 1px solid #e5e7eb;
    padding: 6px 8px;
    text-align: left;
}
.pretty-card th {
    background: #f9fafb;
    font-weight: 600;
}
/* Markdown headings */
.pretty-card h1, .pretty-card h2, .pretty-card h3, .pretty-card h4 {
    margin: 0.5em 0 0.3em 0;
    color: #111;
}
.pretty-card h1 { font-size: 1.4em; }
.pretty-card h2 { font-size: 1.2em; }
.pretty-card h3 { font-size: 1.1em; }
/* Markdown lists and paragraphs */
.pretty-card ul, .pretty-card ol {
    margin: 0.5em 0;
    padding-left: 1.5em;
}
.pretty-card p {
    margin: 0.5em 0;
}
.pretty-card hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1em 0;
}
</style>
<div class="pretty-card"><div class="pretty-title">Agent Response</div><h2>Latest Trends in AI Agents (2025) - Summary</h2>
<h3>🚀 Market Growth &amp; Adoption</h3>
<p>The AI agent market is experiencing explosive growth, nearly doubling from <strong>$3.7 billion (2023) to $7.38 billion (2025)</strong>, with projections reaching <strong>$103.6 billion by 2032</strong>. According to PwC's 2025 survey, <strong>79% of organizations</strong> have adopted AI agents, with <strong>88% of executives</strong> piloting or scaling autonomous agent systems.</p>
<h3>🔑 Key Trends</h3>
<p><strong>1. Rise of Multi-Agent Systems</strong><br />
Instead of single AI systems trying to do everything, 2025 has introduced the "orchestra approach" where multiple specialized agents collaborate—one gathers research, another drafts reports, and a third reviews. Frameworks like <strong>CrewAI, AutoGen, and LangGraph</strong> are enabling this coordination across enterprise departments.</p>
<p><strong>2. From Assistants to Autonomous Decision-Makers</strong><br />
AI agents are evolving from knowledge assistants to <strong>self-directed workers</strong> that can take initiative, make decisions, and complete multi-step tasks without constant human input. By 2029, <strong>80% of customer service issues</strong> are expected to be resolved entirely by autonomous agents.</p>
<p><strong>3. Model Context Protocol (MCP)</strong><br />
Anthropic's open standard provides a "USB-C for AI"—standardizing how language models connect with external systems, enabling structured multi-step workflows and access to real-time information.</p>
<p><strong>4. Two-Speed Enterprise Landscape</strong><br />
A divide is emerging: companies with existing automation are racing ahead with agentic AI, while others watch from the sidelines. Among highly automated enterprises, <strong>50%</strong> have either adopted or are preparing to adopt autonomous agents.</p>
<h3>⚠️ Key Challenges</h3>
<ul>
<li><strong>Integration with legacy systems</strong> (cited by ~60% of AI leaders)</li>
<li><strong>Trust issues</strong> for high-stakes tasks like financial transactions</li>
<li><strong>Enterprise readiness</strong>—organizations need to expose APIs and prepare infrastructure</li>
<li><strong>Reliability concerns</strong>—agents can misinterpret instructions or fail on edge cases</li>
</ul>
<h3>💼 Impact</h3>
<ul>
<li><strong>66%</strong> of adopters report measurable productivity gains</li>
<li>Early movers are cutting operational costs by up to <strong>40%</strong></li>
<li><strong>75% of executives</strong> believe AI agents will reshape the workplace more than the internet did</li>
<li><strong>87%</strong> agree AI agents augment roles rather than replace them</li>
</ul>
<hr />
<h2>Sources:</h2>
<ul>
<li><a href="https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai">The State of AI in 2025 - McKinsey</a></li>
<li><a href="https://www.ibm.com/think/insights/ai-agents-2025-expectations-vs-reality">AI Agents in 2025: Expectations vs. Reality - IBM</a></li>
<li><a href="https://www.pwc.com/us/en/tech-effect/ai-analytics/ai-agent-survey.html">PwC's AI Agent Survey</a></li>
<li><a href="https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage">Seizing the Agentic AI Advantage - McKinsey</a></li>
<li><a href="https://www.gartner.com/en/newsroom/press-releases/2025-08-05-gartner-hype-cycle-identifies-top-ai-innovations-in-2025">Gartner Hype Cycle Identifies Top AI Innovations in 2025</a></li>
<li><a href="https://www.deloitte.com/us/en/services/consulting/blogs/ai-adoption-challenges-ai-trends.html">AI Trends 2025: Adoption Barriers - Deloitte</a></li>
<li><a href="https://aws.amazon.com/blogs/aws-insights/the-rise-of-autonomous-agents-what-enterprise-leaders-need-to-know-about-the-next-wave-of-ai/">The Rise of Autonomous Agents - AWS</a></li>
<li><a href="https://terralogic.com/multi-agent-ai-systems-why-they-matter-2025/">Multi-Agent AI Systems in 2025 - Terralogic</a></li>
<li><a href="https://www.index.dev/blog/ai-agents-statistics">50+ Key AI Agent Statistics - Index.dev</a></li>
<li><a href="https://www.salesforce.com/news/stories/future-of-ai-agents-2025/">Future of AI Agents 2025 - Salesforce</a></li>
<li><a href="https://superagi.com/top-5-agentic-ai-trends-in-2025-from-multi-agent-collaboration-to-self-healing-systems/">Top 5 Agentic AI Trends - SuperAGI</a></li>
</ul></div>


## What's happening here:

- `query()` creates a single-turn agent interaction (no conversation memory)
- `allowed_tools=["WebSearch"]` gives Claude permission to search the web without asking for approval
- The agent autonomously decides when to search, what queries to run, and how to synthesize results

**Visualization utilities from `utils.agent_visualizer`:**
- `print_activity()` - Shows the agent's actions in real-time (tool calls, thinking)
- `display_agent_response()` - Renders the final response as a styled HTML card
- `visualize_conversation()` - Creates a timeline view of the full conversation

That's it! A functional research agent in just a few lines of code. The agent will search for relevant information, follow up on promising leads, and provide a synthesized summary with citations.

The query() function creates a stateless agent interaction. Each call is independent—no conversation memory, no context from previous queries. This makes it perfect for one-off research tasks where you need a quick answer without maintaining state.

**How tool permissions work:**

The `allowed_tools=["WebSearch"]` parameter gives Claude permission to search without asking for approval. This is critical for autonomous operation:

- `Allowed tools` - Claude can use these freely (in this case, WebSearch)
- `Other tools` - Available but require approval before use
- `Read-only tools` - Tools like Read are always allowed by default
- `Disallowed tools` - Add tools to disallowed_tools to remove them entirely from Claude's context

**When to use stateless queries:**

- One-off research questions where context doesn't matter
- Parallel processing of independent research tasks
- Scenarios where you want fresh context for each query

**When not to use stateless queries:**

- Multi-turn investigations that build on previous findings
- Iterative refinement of research based on initial results
- Complex analysis requiring sustained context

Let's inspect what the agent actually did using the visualize_conversation helper:


```python
from utils.agent_visualizer import visualize_conversation

visualize_conversation(messages)
```




<style>
.conversation-timeline {
    font-family: ui-sans-serif, system-ui;
    max-width: 900px;
    margin: 1em 0;
}
.timeline-header {
    background: linear-gradient(135deg, #3b82f6, #9333ea);
    color: white;
    padding: 12px 16px;
    border-radius: 12px 12px 0 0;
    font-weight: 700;
    font-size: 14px;
}
.timeline-body {
    border: 1px solid #e5e7eb;
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 12px;
    background: #fafafa;
}
.msg-block {
    margin: 8px 0;
    padding: 10px 12px;
    border-radius: 8px;
    background: white;
    border-left: 3px solid #e5e7eb;
}
.msg-block.system { border-left-color: #6b7280; }
.msg-block.assistant { border-left-color: #3b82f6; }
.msg-block.tool { border-left-color: #10b981; background: #f0fdf4; }
.msg-block.subagent { border-left-color: #9333ea; background: #faf5ff; }
.msg-block.result { border-left-color: #f59e0b; background: #fffbeb; }
.msg-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 4px;
}
.msg-content {
    font-size: 13px;
    color: #111;
}
.msg-content pre {
    background: #f3f4f6;
    padding: 8px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 12px;
}
.tool-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
}
.tool-badge {
    background: #e0f2fe;
    color: #0369a1;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: monospace;
}
.stats-bar {
    display: flex;
    gap: 16px;
    padding: 10px 12px;
    background: #f9fafb;
    border-radius: 8px;
    font-size: 12px;
    color: #374151;
    margin-top: 8px;
}
.stat-item { display: flex; gap: 4px; }
.stat-label { color: #6b7280; }
</style>

    <div class="conversation-timeline">
        <div class="timeline-header">🤖 Agent Conversation Timeline • claude-opus-4-5</div>
        <div class="timeline-body">
            <div class="msg-block system"><div class="msg-label">⚙️ System</div><div class="msg-content">Initialized (4e8497a9...)</div></div><div class="msg-block tool"><div class="msg-label">🔧 Tools</div><div class="tool-list"><span class="tool-badge">WebSearch: &quot;AI agents trends 2025 latest d...&quot;</span><span class="tool-badge">WebSearch: &quot;autonomous AI agents enterpris...&quot;</span><span class="tool-badge">WebSearch: &quot;multi-agent AI systems trends ...&quot;</span></div></div><div class="msg-block assistant"><div class="msg-label">🤖 Assistant</div><div class="msg-content"><h2>Latest Trends in AI Agents (2025) - Summary</h2>
<h3>🚀 Market Growth &amp; Adoption</h3>
<p>The AI agent market is experiencing explosive growth, nearly doubling from <strong>$3.7 billion (2023) to $7.38 billion (2025)</strong>, with projections reaching <strong>$103.6 billion by 2032</strong>. According to PwC's 2025 survey, <strong>79% of organizations</strong> have adopted AI agents, with <strong>88% of executives</strong> piloting or scaling autonomous agent systems.</p>
<h3>🔑 Key Trends</h3>
<p><strong>1. Rise of Multi-Agent Systems</strong><br />
Instead of single AI systems trying to do everything, 2025 has introduced the "orchestra approach" where multiple specialized agents collaborate—one gathers research, another drafts reports, and a third reviews. Frameworks like <strong
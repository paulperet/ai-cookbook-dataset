# Parallelizing Specialized Agents with the OpenAI Agents SDK

## Introduction

In production AI workflows, you often need to answer multiple independent questions about the same piece of content. Running these analyses sequentially increases latency and can raise costs if a failure forces a retry. By executing multiple specialized agents in parallel and then aggregating their outputs through a final "meta" agent, you can significantly reduce overall latency.

This guide demonstrates two patterns for parallel execution using the OpenAI Agents SDK:
1. **Custom Parallelization with `asyncio`**: For low-latency, deterministic control.
2. **SDK-Managed Parallelization via "Agents as Tools"**: For convenience and dynamic planning.

We'll use a product review analysis as our example, where multiple agents extract different aspects (features, pros/cons, sentiment, recommendation) concurrently.

## Prerequisites

First, install the required packages and set up your environment.

```bash
pip install openai-agents matplotlib nest_asyncio
```

```python
import time
import asyncio
import matplotlib.pyplot as plt
import nest_asyncio

from agents import Agent, Runner, ModelSettings

# Required for running async code in notebooks/scripts
nest_asyncio.apply()
```

## Step 1: Define Your Specialized Agents

Create four focused agents, each with a specific analysis task.

```python
# Agent focusing on product features
features_agent = Agent(
    name="FeaturesAgent",
    instructions="Extract the key product features from the review."
)

# Agent focusing on pros & cons
pros_cons_agent = Agent(
    name="ProsConsAgent",
    instructions="List the pros and cons mentioned in the review."
)

# Agent focusing on sentiment analysis
sentiment_agent = Agent(
    name="SentimentAgent",
    instructions="Summarize the overall user sentiment from the review."
)

# Agent focusing on recommendation summary
recommend_agent = Agent(
    name="RecommendAgent",
    instructions="State whether you would recommend this product and why."
)

# Collect all parallel agents
parallel_agents = [
    features_agent,
    pros_cons_agent,
    sentiment_agent,
    recommend_agent
]

# Meta-agent to combine outputs
meta_agent = Agent(
    name="MetaAgent",
    instructions="You are given multiple summaries labeled with Features, ProsCons, Sentiment, and a Recommendation."
    " Combine them into a concise executive summary of the product review with a 1-5 star rating for each summary area."
)
```

## Step 2: Create Helper Functions for Execution Tracking

Define a function to run a single agent and track its execution time for later visualization.

```python
# Global lists to track execution times
starts, ends = [], []

async def run_agent(agent, review_text: str):
    """Run an agent and record its start/end times."""
    agent_name = agent.name

    start = time.time()
    starts.append((agent_name, start))

    result = await Runner.run(agent, review_text)

    end = time.time()
    ends.append((agent_name, end))

    return result
```

## Step 3: Implement Parallel Execution with `asyncio`

Create a function that runs all specialized agents concurrently using `asyncio.gather()`, then passes their outputs to the meta-agent.

```python
async def run_agents(review_text: str):
    """Execute all parallel agents and combine results via the meta-agent."""
    # Run all specialized agents in parallel
    responses = await asyncio.gather(
        *(run_agent(agent, review_text) for agent in parallel_agents)
    )

    # Format each agent's output with a header
    labeled_summaries = [
        f"### {resp.last_agent.name}\n{resp.final_output}"
        for resp in responses
    ]

    # Combine all summaries into a single string
    collected_summaries = "\n".join(labeled_summaries)
    
    # Pass the combined summaries to the meta-agent
    final_summary = await run_agent(meta_agent, collected_summaries)

    print('Final summary:', final_summary.final_output)
    return final_summary
```

## Step 4: Run the Analysis on Sample Content

Define a sample product review and execute the parallel analysis.

```python
review_text = """
I recently upgraded to the AuroraSound X2 wireless noise-cancelling headphones, and after two weeks of daily use I have quite a bit to share. First off, the design feels premium without being flashy: the matte‐finish ear cups are softly padded and rotate smoothly for storage, while the headband’s memory‐foam cushion barely presses on my temples even after marathon work calls. Connectivity is seamless—pairing with my laptop and phone took under five seconds each time, and the Bluetooth 5.2 link held rock-solid through walls and down the hallway.

The noise-cancelling performance is genuinely impressive. In a busy café with music and chatter swirling around, flipping on ANC immediately quiets low-level ambient hums, and it even attenuates sudden noises—like the barista’s milk frother—without sounding distorted. The “Transparency” mode is equally well‐tuned: voices come through clearly, but the world outside isn’t overwhelmingly loud. Audio quality in standard mode is rich and balanced, with tight bass, clear mids, and a hint of sparkle in the highs. There’s also a dedicated EQ app, where you can toggle between “Podcast,” “Bass Boost,” and “Concert Hall” presets or craft your own curve.

On the control front, intuitive touch panels let you play/pause, skip tracks, and adjust volume with a simple swipe or tap. One neat trick: holding down on the right ear cup invokes your phone’s voice assistant. Battery life lives up to the hype, too—over 30 hours with ANC on, and the quick‐charge feature delivers 2 hours of playtime from just a 10-minute top-up.

That said, it isn’t perfect. For one, the carrying case is a bit bulky, so it doesn’t slip easily into a slim bag. And while the touch interface is mostly reliable, I occasionally trigger a pause when trying to adjust the cup position. The headphones also come in only two colorways—black or white—which feels limiting given the premium price point.
"""

# Execute the parallel workflow
result = asyncio.get_event_loop().run_until_complete(run_agents(review_text))
```

## Step 5: Visualize the Execution Timeline

Create a simple visualization to see the latency benefits of parallel execution.

```python
def plot_timeline(starts, ends):
    """Plot a horizontal bar chart showing agent execution times."""
    # Normalize times to start from zero
    base = min(t for _, t in starts)
    labels = [n for n, _ in starts]
    start_offsets = [t - base for _, t in starts]
    lengths = [ends[i][1] - starts[i][1] for i in range(len(starts))]

    plt.figure(figsize=(8, 3))
    plt.barh(labels, lengths, left=start_offsets)
    plt.xlabel("Seconds since kickoff")
    plt.title("Agent Execution Timeline")
    plt.show()

plot_timeline(starts, ends)
```

**Expected Output:**
```
Final summary: ### Executive Summary

The AuroraSound X2 wireless noise-cancelling headphones offer a blend of premium design and advanced features...
```

The timeline visualization will show all four specialized agents executing concurrently, demonstrating the latency reduction compared to sequential execution.

## Step 6: Alternative Approach - SDK-Managed Parallel Tools

The Agents SDK provides a built-in method for parallelization by treating agents as tools. This approach offers convenience and dynamic planning at the cost of slightly higher latency.

```python
# Reset tracking lists
starts, ends = [], []

# Create a meta-agent with parallel tools enabled
meta_agent_parallel_tools = Agent(
    name="MetaAgent",
    instructions="You are given multiple summaries labeled with Features, ProsCons, Sentiment, and a Recommendation."
    " Combine them into a concise executive summary of the product review with a 1-5 star rating for each summary area.",
    model_settings=ModelSettings(
        parallel_tool_calls=True  # Enable parallel tool execution
    ),
    tools=[
        features_agent.as_tool(
            tool_name="features",
            tool_description="Extract the key product features from the review.",
        ),
        pros_cons_agent.as_tool(
            tool_name="pros_cons",
            tool_description="List the pros and cons mentioned in the review.",
        ),
        sentiment_agent.as_tool(
            tool_name="sentiment",
            tool_description="Summarize the overall user sentiment from the review.",
        ),
        recommend_agent.as_tool(
            tool_name="recommend",
            tool_description="State whether you would recommend this product and why.",
        ),
    ],
)

# Run the analysis using the SDK's parallel tools
result = await run_agent(meta_agent_parallel_tools, review_text)

print('Final summary:', result.final_output)

# Visualize the execution timeline
plot_timeline(starts, ends)
```

**Expected Output:**
```
Final summary: **Executive Summary: AuroraSound X2 Wireless Noise-Cancelling Headphones**

**Features (⭐️⭐️⭐️⭐️⭐️ 5/5):** The headphones boast a premium, matte-finish design...
```

## Summary: Choosing Your Parallelization Strategy

You now have two patterns for parallelizing agents. Choose based on your specific requirements:

| **Consideration** | **`asyncio.gather()` Approach** | **Agents as Tools Approach** |
|-------------------|---------------------------------|------------------------------|
| **Control** | Full control over execution flow | SDK-managed, less customization |
| **Planning** | Deterministic execution order | Dynamic tool selection by planner |
| **Latency** | Lower (no planning overhead) | Higher (additional planning call) |
| **Convenience** | More setup required | Simpler configuration |
| **Use Case** | Custom workflows, latency-sensitive apps | Rapid prototyping, dynamic scenarios |

### Key Decision Factors:

1. **Convenience vs. Customization**
   - Use "agents as tools" for convenience and quick setup.
   - Use `asyncio.gather()` when you need custom control over how agents fan in/out across multiple layers.

2. **Planning vs. Determinism**
   - Use "agents as tools" if you want the meta-agent to dynamically decide which tools to call.
   - Use `asyncio.gather()` for deterministic execution order.

3. **Latency Sensitivity**
   - Use `asyncio` for latency-sensitive applications to avoid planning overhead.
   - Use "agents as tools" when latency is less critical than convenience.

Both approaches effectively parallelize agent execution, reducing overall latency compared to sequential processing. The choice depends on your specific balance of control, convenience, and performance requirements.
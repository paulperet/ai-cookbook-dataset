# Multi-Agent Orchestration with OpenAI Agents SDK: Financial Portfolio Analysis Guide

## Introduction

This guide demonstrates how to build a sophisticated multi-agent collaboration system using the OpenAI Agents SDK. You'll implement a financial analysis workflow where specialized agents work together under a central Portfolio Manager to analyze investment scenarios.

**What You'll Learn**
- Design a hub-and-spoke multi-agent architecture
- Implement the "agent-as-tool" collaboration pattern
- Combine custom tools, managed tools, and MCP servers in a single workflow
- Apply best practices for modularity, parallelism, and observability

**Prerequisites**
- Familiarity with OpenAI models and LLM agents
- Python programming experience
- API keys for OpenAI and FRED (Federal Reserve Economic Data)

## Architecture Overview

Our system follows a **hub-and-spoke design**:
- **Portfolio Manager (PM) Agent**: Central orchestrator that coordinates all analysis
- **Specialist Agents**: Three domain experts:
  - **Macro Agent**: Analyzes economic conditions and interest rate impacts
  - **Fundamental Agent**: Evaluates company financials and business metrics
  - **Quantitative Agent**: Performs statistical and technical analysis

The PM agent treats each specialist as a callable tool, invoking them for specific subtasks and synthesizing their results into a comprehensive investment report.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Set the following environment variables before running the workflow:

```python
import os

# Required API keys
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'
os.environ['FRED_API_KEY'] = 'your-fred-api-key'

# Verify configuration
missing = []
if not os.environ.get('OPENAI_API_KEY'):
    missing.append('OPENAI_API_KEY')
if not os.environ.get('FRED_API_KEY'):
    missing.append('FRED_API_KEY')

if missing:
    print(f"Missing environment variable(s): {', '.join(missing)}")
else:
    print("All required API keys are set.")
```

## Building the Portfolio Manager Agent

### 3. Create Specialist Agent Tools

The PM agent coordinates three specialist agents, each wrapped as a callable tool:

```python
from agents import Agent, ModelSettings, function_tool
from utils import load_prompt, DISCLAIMER

def build_head_pm_agent(fundamental, macro, quant, memo_edit_tool):
    """Create the central Portfolio Manager agent with specialist tools."""
    
    def make_agent_tool(agent, name, description):
        """Wrap an agent as a callable tool."""
        @function_tool(name_override=name, description_override=description)
        async def agent_tool(input):
            return await specialist_analysis_func(agent, input)
        return agent_tool
    
    # Create tool wrappers for each specialist
    fundamental_tool = make_agent_tool(
        fundamental, 
        "fundamental_analysis", 
        "Generate the Fundamental Analysis section."
    )
    
    macro_tool = make_agent_tool(
        macro, 
        "macro_analysis", 
        "Generate the Macro Environment section."
    )
    
    quant_tool = make_agent_tool(
        quant, 
        "quantitative_analysis", 
        "Generate the Quantitative Analysis section."
    )
    
    # Parallel execution tool
    @function_tool(
        name_override="run_all_specialists_parallel",
        description_override="Run all three specialist analyses in parallel."
    )
    async def run_all_specialists_tool(fundamental_input, macro_input, quant_input):
        return await run_all_specialists_parallel(
            fundamental, macro, quant,
            fundamental_input, macro_input, quant_input
        )
    
    # Build the PM agent
    return Agent(
        name="Head Portfolio Manager Agent",
        instructions=(load_prompt("pm_base.md") + DISCLAIMER),
        model="gpt-4.1",
        tools=[fundamental_tool, macro_tool, quant_tool, memo_edit_tool, run_all_specialists_tool],
        model_settings=ModelSettings(
            parallel_tool_calls=True,
            tool_choice="auto",
            temperature=0
        )
    )
```

### 4. Configure the System Prompt

The PM agent's behavior is guided by a detailed system prompt (`prompts/pm_base.md`) that encodes:

- **Firm Philosophy**: Originality, risk awareness, challenging consensus
- **Tool Usage Rules**: When and how to use each specialist tool
- **Workflow Steps**: Structured analysis process
- **Quality Standards**: Review and validation procedures

```markdown
# Portfolio Manager System Prompt

## Firm Philosophy
You are the Head Portfolio Manager at a premier investment firm. Your approach emphasizes:
1. Original, differentiated insights over consensus views
2. Rigorous risk assessment and scenario planning
3. Challenging assumptions with evidence-based analysis

## Workflow Process
1. **Task Analysis**: Determine which specialist agents are needed
2. **Guidance Provision**: Frame questions for each specialist
3. **Parallel Execution**: Run specialists concurrently when possible
4. **Output Review**: Validate and synthesize specialist findings
5. **Memo Assembly**: Structure the final investment report

## Tool Usage Guidelines
- Use `run_all_specialists_parallel` for independent analyses
- Provide clear, focused inputs to each specialist
- Review all outputs for consistency and completeness
- Flag any missing data or contradictory findings
```

## Running the Multi-Agent Workflow

### 5. Execute the Complete Analysis

Now you'll run the entire workflow, from user query to final investment report:

```python
import datetime
import json
import asyncio
from contextlib import AsyncExitStack
from agents import Runner, add_trace_processor, trace
from agents.tracing.processors import BatchTraceProcessor
from utils import FileSpanExporter, output_file
from investment_agents.config import build_investment_agents

# Enable tracing for observability
add_trace_processor(BatchTraceProcessor(FileSpanExporter()))

async def run_workflow():
    """Execute the complete multi-agent analysis."""
    
    # Verify API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY not set")
    
    # Define the analysis question
    today_str = datetime.date.today().strftime("%B %d, %Y")
    question = (
        f"Today is {today_str}. "
        "How would the planned interest rate reduction affect my holdings in GOOGL if they were to happen? "
        "Considering all factors affecting its price (Macro, Technical, Fundamental, etc.), "
        "what is a realistic price target by the end of the year?"
    )
    
    # Build the agent bundle
    bundle = build_investment_agents()
    
    # Connect to MCP servers
    async with AsyncExitStack() as stack:
        for agent in [getattr(bundle, "fundamental", None), getattr(bundle, "quant", None)]:
            if agent is None:
                continue
            for server in getattr(agent, "mcp_servers", []):
                await server.connect()
                await stack.enter_async_context(server)
        
        print("Running multi-agent workflow with tracing enabled...\n")
        
        # Start the trace
        with trace(
            "Investment Research Workflow",
            metadata={"question": question[:512]}
        ) as workflow_trace:
            print(
                f"\n🔗 View the trace in the OpenAI console: "
                f"https://platform.openai.com/traces/trace?trace_id={workflow_trace.trace_id}\n"
            )
            
            # Execute the workflow with timeout protection
            response = None
            try:
                response = await asyncio.wait_for(
                    Runner.run(bundle.head_pm, question, max_turns=40),
                    timeout=1200  # 20-minute timeout
                )
            except asyncio.TimeoutError:
                print("\n❌ Workflow timed out after 20 minutes.")
            
            # Process the response
            report_path = None
            try:
                if hasattr(response, 'final_output'):
                    output = response.final_output
                    if isinstance(output, str):
                        data = json.loads(output)
                        if isinstance(data, dict) and 'file' in data:
                            report_path = output_file(data['file'])
            except Exception as e:
                print(f"Could not parse investment report path: {e}")
            
            print(f"Workflow completed. Investment report created: {report_path if report_path else '[unknown]'}")

# Run the workflow
await run_workflow()
```

## Understanding the Output

### 6. Review the Investment Report

The workflow generates a comprehensive investment memo in the `outputs` directory. Here's an example structure:

```markdown
# Investment Memo: Alphabet Inc. (GOOGL) – Impact of Planned Interest Rate Reduction

## Executive Summary
- Current price: $171.42
- Market cap: $1.88 trillion
- P/E ratio: 16.91
- Primary drivers: AI growth, sector rotation, regulatory factors
- Rate sensitivity: Modest (max correlation ~0.29 with 10Y yield)

## Specialist Analyses

### Macro Analysis
- Interest rate impact assessment
- Economic condition evaluation
- Sector rotation implications

### Fundamental Analysis
- Revenue: $90.2B (Q1 2025)
- Net income: $34.5B
- EPS: $2.81
- Net margin: 38.3%

### Quantitative Analysis
- Technical indicators
- Statistical correlations
- Price target modeling

## Risk Assessment
- Regulatory actions
- Macroeconomic uncertainty
- AI narrative shifts

## Price Targets
- Best case: $200–$210 by year-end
- Worst case: $160–$170 retest
```

## Best Practices

### 7. Design Principles for Multi-Agent Systems

1. **Specialization Over Generalization**
   - Each agent should have a clear, focused domain
   - Avoid overloading agents with multiple responsibilities

2. **Parallel Execution**
   - Use `parallel_tool_calls=True` for independent analyses
   - Structure inputs to enable concurrent processing

3. **Observability**
   - Enable tracing for all agent interactions
   - Use structured logging for debugging

4. **Modular Design**
   - Keep agents independent and swappable
   - Use clear interfaces between components

5. **Prompt Engineering**
   - Encode business logic in system prompts
   - Include validation and review steps
   - Maintain consistency across runs

## Troubleshooting

### Common Issues and Solutions

**Issue**: Workflow times out after 20 minutes
**Solution**: 
- Check network connectivity to OpenAI API
- Reduce `max_turns` parameter
- Simplify the analysis question

**Issue**: Missing specialist outputs
**Solution**:
- Verify MCP server connections
- Check agent tool configurations
- Review specialist agent prompts

**Issue**: Inconsistent analysis quality
**Solution**:
- Refine system prompts with clearer instructions
- Add validation steps in the workflow
- Implement output quality checks

## Next Steps

### Extending the System

1. **Add New Specialist Agents**
   - Technical analysis agent
   - Sentiment analysis agent
   - Competitor analysis agent

2. **Integrate Additional Data Sources**
   - Real-time market data
   - Alternative data providers
   - Proprietary research databases

3. **Enhance Workflow Logic**
   - Dynamic agent selection
   - Iterative refinement loops
   - Confidence scoring for outputs

## Conclusion

You've successfully built a multi-agent orchestration system for financial portfolio analysis. This architecture demonstrates how to:

- Coordinate specialized agents under a central manager
- Leverage multiple tool types in a unified workflow
- Maintain consistency and quality through structured prompts
- Enable observability with comprehensive tracing

The "agent-as-tool" pattern provides a scalable, maintainable approach to complex analysis tasks, with applications extending far beyond financial analysis to any domain requiring expert collaboration.

---

**Disclaimer**: This example is for educational purposes only. Consult a qualified financial professional before making any investment decisions.
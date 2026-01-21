# Building with the Responses API's MCP Tool: A Practical Guide

## Introduction

Building agentic applications often requires connecting to external services. Traditionally, this is done through function calling, where each action requires a round-trip from the model to your backend, then to an external service, and back. This process introduces multiple network hops and significant latency, making it cumbersome to scale.

The hosted Model Context Protocol (MCP) tool in the Responses API simplifies this. Instead of manually wiring each function call, you configure your model once to point to an MCP server. This server acts as a centralized tool host, exposing standard commands like "search product catalog" or "add item to cart." With MCP, the model interacts directly with the MCP server, reducing latency and eliminating backend coordination.

This guide will walk you through the core concepts, best practices, and a practical implementation example for using the MCP tool effectively.

---

## How the MCP Tool Works

At a high level, the MCP tool operates through a three-step process:

1.  **Declare the Server:** When you add an MCP block to the `tools` array, the Responses API runtime detects the server's transport protocol (streamable HTTP or HTTP-over-SSE) and uses it for communication.
2.  **Import the Tool List:** The runtime calls the server's `tools/list` endpoint, passing any headers you provide (like API keys). The resulting list of available tools is written to an `mcp_list_tools` item in the model's context. As long as this item is present, the list won't be fetched again. You can limit the tools the model sees using the `allowed_tools` parameter.
3.  **Call and Approve Tools:** Once the model knows the available actions, it can invoke one. Each invocation produces an `mcp_tool_call` item. By default, the stream pauses for your explicit approval, but you can disable this with `require_approval: "never"` once you trust the server. After execution, the result is streamed back, and the model decides whether to chain another tool or return a final answer.

---

## Prerequisites & Setup

To follow along with the examples, ensure you have:
*   An OpenAI API key.
*   The `curl` command-line tool installed.

All interactions in this guide are demonstrated using the OpenAI Responses API via direct `curl` commands.

---

## Best Practices for Production Use

### 1. Filter Tools to Optimize Performance

Remote servers often expose many tools with verbose definitions, which can add hundreds of tokens to the context and increase latency. Use the `allowed_tools` parameter to explicitly list only the tools your agent needs. This reduces token overhead, improves response time, and focuses the model's decision space.

**Example: Limiting tools for a documentation search server**

```bash
curl https://api.openai.com/v1/responses -i \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1",
    "tools": [
        {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": ["search_tiktoken_documentation", "fetch_tiktoken_documentation"],
            "require_approval": "never"
        }
    ],
    "input": "how does tiktoken work?"
}'
```

### 2. Manage Latency and Tokens with Caching

The `mcp_list_tools` item is cached within a conversation. To maintain this cache across turns and avoid re-fetching the tool list, pass the `previous_response_id` in subsequent API requests. Alternatively, you can manually pass the context items to a new response.

**Token Usage Comparison: Reasoning vs. Non-Reasoning Models**

Using a reasoning model (e.g., `o4-mini`) will significantly increase both input and output tokens due to the internal reasoning process. Choose your model based on task complexity.

*   **Scenario 1: Non-reasoning model (`gpt-4.1`)**
    ```bash
    curl https://api.openai.com/v1/responses \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -d '{
        "model": "gpt-4.1",
        "tools": [
          {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "require_approval": "never"
          }
        ],
        "input": "how does tiktoken work?"
      }'
    ```
    **Sample Usage Output:**
    ```json
    "usage": {
      "input_tokens": 280,
      "output_tokens": 665,
      "total_tokens": 945
    }
    ```

*   **Scenario 2: Reasoning model (`o4-mini`) without cached context**
    ```bash
    curl https://api.openai.com/v1/responses \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -d '{
        "model": "o4-mini",
        "tools": [
          {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "require_approval": "never"
          }
        ],
        "input": "how does tiktoken work?",
        "reasoning": {
          "effort": "medium",
          "summary": "auto"
        }
      }'
    ```
    **Sample Usage Output:**
    ```json
    "usage": {
      "input_tokens": 36436,
      "output_tokens": 1586,
      "output_tokens_details": {
        "reasoning_tokens": 576
      },
      "total_tokens": 38022
    }
    ```

### 3. Prompting Guidelines for Better Tool Calls

To prevent the model from making excessive or vague MCP calls, incorporate specific instructions into your system prompt:

*   **Limit Search Results:** Instruct the model to return only a set number of results (e.g., 4) at a time and ask the user if they want to see more.
*   **Clarify Missing Information:** Guide the model to ask follow-up questions if essential details (size, color, product ID) are missing from the user query, rather than launching a broad search.
*   **Provide Few-Shot Examples:** Include examples in your prompt to show the model how to correctly choose and invoke tools from the available MCP servers.

**Example System Prompt Incorporating Guidelines:**

```text
You are an AI assistant that can call the following MCP servers:
1. allbirds_store
2. gitmcp

Steps
1. Use the MCP servers above to answer the user query. Not every MCP server will be relevant for a given query, so choose which ones to invoke.
2. If the user’s request lacks essential details (size, color, etc.), ask a follow-up question first rather than guessing and using the MCP server tool prematurely.
3. If searching for products in a catalog, only return 4 results at a time before asking the user whether they want to see more.

### Few-shot examples

# EX1 — Direct product search, then fetch variant details
user: Do you have the Allbirds Tree Runner in men’s size 10?
assistant: {"name":"allbirds_store.search_shop_catalog",
            "arguments":{"query":"Tree Runner",
                         "context":"Men’s size 10 shoes"}}
# (assume the search returns product_id "gid://shopify/Product/987")
assistant: {"name":"allbirds_store.get_product_details",
            "arguments":{"product_id":"gid://shopify/Product/987",
                         "options":{"Size":"10"}}}

# EX2 — Clarify missing size, then two-step color variant lookup
user: I want the Tree Runner in blue.
assistant: Sure — what size do you need?
user: Size 10.
assistant: {"name":"allbirds_store.search_shop_catalog",
            "arguments":{"query":"Tree Runner",
                         "context":"Blue variant, size 10"}}
# (assume the search returns product_id "gid://shopify/Product/987")
assistant: {"name":"allbirds_store.get_product_details",
            "arguments":{"product_id":"gid://shopify/Product/987",
                         "options":{"Size":"10","Color":"Blue"}}}
```

---

## Tutorial: Build a Pricing Analyst Agent

Let's build a practical agent that combines the MCP tool with other hosted tools. This agent acts as a pricing analyst for a fictional yoga attire store. It will:
1.  Fetch competitor prices from an Alo Yoga MCP server.
2.  Search for competitor prices from Uniqlo using web search.
3.  Analyze the store's internal sales data using Code Interpreter.
4.  Generate a report flagging significant price discrepancies.

### Step 1: Define the Agent's System Prompt

First, craft a detailed system prompt that outlines the agent's role and the specific steps it must follow.

```python
system_prompt = """You are a pricing analyst for my clothing company. Please use the MCP tool to fetch prices from the Alo Yoga MCP server for the categories of women's shorts, yoga pants, and tank tops. Use only the MCP server for Alo yoga data, don't search the web.

Next, use the web search tool to search for Uniqlo prices for women's shorts, yoga pants, and tank tops.

In each case for Alo Yoga and Uniqlo, extract the price for the top result in each category. Also provide the full URLs.

Using the uploaded CSV file of sales data from my store, and with the code interpreter tool calculate revenue by product item, compute average order-value on a transaction level, and calculate the percentage price gap between the CSV data and Uniqlo/Alo Yoga prices.
Flag products priced 15% or more above or below the market.
Create and output a short report including the findings.

# Steps

1. **Fetch Alo Yoga Prices:**
   - Use the Alo Yoga MCP server to fetch prices for the following products:
     High-Waist Airlift Legging
     Sway Bra Tank
     5" Airlift Energy Short
   - Ensure you find prices for each.
   - Extract the price of the top result for each category.
   - Include URL links.

2. **Query Uniqlo Prices:**
   - Use the Web-Search tool to search non-sale prices for the following Uniqlo products:
     Women's AIRism Soft Biker Shorts
     Women's AIRism Soft Leggings
     Women's AIRism Bra Sleeveless Top
   - Ensure you find non-sale prices for each.
   - Extract the price for the top result in each category.
   - Include URL links.

3. **Sales Data Analysis:**
   - Use the uploaded CSV sales data to calculate revenue across each product item.
   - Determine the average order-value on a transaction level.
   - For each SKU, compute the percentage price gap between the CSV data and Uniqlo/Alo Yoga prices.
   - Flag products priced ≥ 15% above or below the market.

4. **Report:**
   - Compile and output a report including the flagging results.

# Output Format
- A short text report explaining:
  - Any products that are priced ≥ 15% above or below the market, with specific details."""
```

### Step 2: Configure and Execute the Agent

Now, construct the API call. This configuration uses three tools simultaneously: `mcp` for Alo Yoga data, `web_search_preview` for Uniqlo, and `code_interpreter` for analyzing the pre-uploaded sales CSV file.

```bash
curl https://api.openai.com/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1",
    "input": [
      {
        "role": "system",
        "content": [
          {
            "type": "input_text",
            "text": "ABOVE_SYSTEM_PROMPT"
          }
        ]
      }
    ],
    "tools": [
      {
        "type": "web_search_preview",
        "user_location": {
          "type": "approximate",
          "country": "US"
        },
        "search_context_size": "medium"
      },
      {
        "type": "code_interpreter",
        "container": {
          "type": "auto",
          "file_ids": [
            "file-WTiyGcZySaU6n218gj4XxR"
          ]
        }
      },
      {
        "type": "mcp",
        "server_url": "https://www.aloyoga.com/api/mcp",
        "server_label": "aloyoga",
        "allowed_tools": [
          "search_shop_catalog",
          "get_product_details"
        ],
        "require_approval": "never"
      }
    ],
    "temperature": 1,
    "max_output_tokens": 2048,
    "top_p": 1,
    "store": true
  }'
```

**Key Configuration Notes:**
*   Replace `"ABOVE_SYSTEM_PROMPT"` with the actual prompt text from Step 1.
*   Replace `"file-WTiyGcZySaU6n218gj4XxR"` with the actual File ID of your uploaded sales CSV.
*   The `allowed_tools` for the MCP server is filtered to just the two needed for catalog search.
*   `require_approval` is set to `"never"` for autonomous operation.

### Step 3: Interpret the Results

The model will execute the steps sequentially, using the outputs from one tool as context for the next. The final output will be a structured report. Here is an example of the formatted analysis you might receive:

#### **Pricing Comparison and Revenue Analysis Report**

**Your Store's Sales & Price Analysis**
*   **Revenue by Product:**
    *   Shorts: $6,060
    *   Tank tops: $6,150
    *   Yoga pants: $12,210
*   **Average Order Value:** $872.14
*   **Your Store's Average Selling Price by Category:**
    *   Shorts: $60.00
    *   Tank tops: $75.00
    *   Yoga pants: $110.00

#### **Pricing Gaps vs Market**

| Category | Store Avg Price | vs Alo Yoga Gap (%) | Flagged (≥15%) | vs Uniqlo Gap (%) | Flagged (≥15%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Shorts | $60.00 | -31.8% | **YES** | +100.7% | **YES** |
| Tank tops | $75.00 | -14.8% | | +114.9% | **YES** |
| Yoga pants | $110.00 | -14.1% | | +267.9% | **YES** |

#### **Recommendations & Flags**

**Flagged products (≥15% price gap):**
*   **Shorts:** Priced 31.8% below Alo Yoga, but 100.7% above Uniqlo.
*   **Tank tops:** Priced over 114.9% above Uniqlo.
*   **Yoga pants:** Priced 267.9% above Uniqlo.

Shorts are priced significantly below premium competitors (Alo Yoga), but far higher than budget alternatives (Uniqlo). If you want to compete in the premium segment, consider increasing your price. If you want to target budget buyers, a price decrease could be justifiable. Most of your tank tops and yoga pants are similarly positioned—much lower than Alo, but well above Uniqlo.

---

## Conclusion

The MCP tool in the Responses API provides a powerful and efficient way to connect AI agents directly to external services. By following the best practices outlined here—filtering tools, managing context caching, and using effective prompting—you can build robust, multi-step agents that combine MCP with other tools like web search and code interpretation to perform complex, real-world tasks.
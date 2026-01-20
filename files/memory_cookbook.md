# Context Editing & Memory for Long-Running Agents

AI agents that run across multiple sessions or handle long-running tasks face two key challenges: they lose learned patterns between conversations, and context windows fill up during extended interactions.

This cookbook demonstrates how to address these challenges using Claude's memory tool and context editing capabilities.

## Table of Contents

1. [Introduction: Why Memory Matters](#introduction)
2. [Use Cases](#use-cases)
3. [Quick Start Examples](#quick-start)
4. [How It Works](#how-it-works)
5. [Code Review Assistant Demo](#demo)
6. [Real-World Applications](#real-world)
7. [Best Practices](#best-practices)

## Prerequisites

**Required Knowledge:**
- Python fundamentals (functions, classes, async/await basics)
- Basic understanding of REST APIs and JSON

**Required Tools:**
- Python 3.10 or higher
- Anthropic API key

**Recommended:**
- Familiarity with concurrent programming concepts (threads, async)
- Basic understanding of context windows in LLMs

## Setup

### For VSCode Users

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. In VSCode: Select .venv as kernel (top right)
```

### API Key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

Get your API key from: https://console.anthropic.com/

## 1. Introduction: Why Memory Matters {#introduction}

This cookbook demonstrates practical implementations of the context engineering patterns described in Effective context engineering for AI agents. That post covers why context is a finite resource, how attention budgets work, and strategies for building effective agents—the techniques you'll see in action here.

### The Problem

Large language models have finite context windows (200k tokens for the Claude 4 family of models). While this seems large, several challenges emerge:

- **Context limits**: Long conversations or complex tasks can exceed available context
- **Computational cost**: Processing large contexts is expensive - attention mechanisms scale quadratically
- **Repeated patterns**: Similar tasks across conversations require re-explaining context every time
- **Information loss**: When context fills up, earlier important information gets lost

### The Solution

Claude 4 models introduce powerful context management capabilities:

1. **Memory Tool** (`memory_20250818`): Enables cross-conversation learning
   - Claude can write down what it learns for future reference
   - File-based system under `/memories` directory
   - Client-side implementation gives you full control

2. **Context Editing**: Automatically manages context with two strategies:
   - **Tool use clearing** (`clear_tool_uses_20250919`): Clears old tool results when context grows large
   - **Thinking management** (`clear_thinking_20251015`): Manages extended thinking blocks (requires thinking enabled)
   - Configurable triggers and retention policies

### The Benefit

Build AI agents that **get better at your specific tasks over time**:

- **Session 1**: Claude solves a problem, writes down the pattern
- **Session 2**: Claude applies the learned pattern immediately (faster!)
- **Long sessions**: Context editing keeps conversations manageable

Think of it as giving Claude a notebook to take notes and refer back to - just like humans do.

### What You'll Learn

By the end of this cookbook, you will be able to:
- **Implement** the memory tool for cross-conversation learning
- **Configure** context editing to manage long-running sessions
- **Apply** best practices for memory security and organization

## 1. Introduction: Why Memory Matters {#introduction}

This cookbook demonstrates practical implementations of the context engineering patterns described in Effective context engineering for AI agents. That post covers why context is a finite resource, how attention budgets work, and strategies for building effective agents—the techniques you'll see in action here.

### The Problem

Large language models have finite context windows (200k tokens for Claude 4). While this seems large, several challenges emerge:

- **Context limits**: Long conversations or complex tasks can exceed available context
- **Computational cost**: Processing large contexts is expensive - attention mechanisms scale quadratically
- **Repeated patterns**: Similar tasks across conversations require re-explaining context every time
- **Information loss**: When context fills up, earlier important information gets lost

### The Solution

Claude Sonnet 4.5 introduces two powerful capabilities:

1. **Memory Tool** (`memory_20250818`): Enables cross-conversation learning
   - Claude can write down what it learns for future reference
   - File-based system under `/memories` directory
   - Client-side implementation gives you full control

**Supported Models**: Claude Opus 4.1 (`claude-opus-4-1`), Claude Opus 4 (`claude-opus-4`), Claude Sonnet 4.5 (`claude-sonnet-4-5`), Claude Sonnet 4 (`claude-sonnet-4`), and Claude Haiku 4.5 (`claude-haiku-4-5`)

### The Benefit

Build AI agents that **get better at your specific tasks over time**:

- **Session 1**: Claude solves a problem, writes down the pattern
- **Session 2**: Claude applies the learned pattern immediately (faster!)
- **Long sessions**: Context editing keeps conversations manageable

Think of it as giving Claude a notebook to take notes and refer back to - just like humans do.

## 2. Use Cases {#use-cases}

Memory and context management enable powerful new workflows:

### 🔍 Code Review Assistant
- Learns debugging patterns from past reviews
- Recognizes similar bugs instantly in future sessions
- Builds team-specific code quality knowledge
- **Production ready**: Integrate with claude-code-action for GitHub PR reviews

### 📚 Research Assistant
- Accumulates knowledge on topics over multiple sessions
- Connects insights across different research threads
- Maintains bibliography and source tracking

### 💬 Customer Support Bot
- Learns user preferences and communication style
- Remembers common issues and solutions
- Builds product knowledge base from interactions

### 📊 Data Analysis Helper
- Remembers dataset patterns and anomalies
- Stores analysis techniques that work well
- Builds domain-specific insights over time

**Supported Models**: Claude Opus 4.1 (`claude-opus-4-1`) and Claude Sonnet 4.5 (`claude-sonnet-4-5`)

**This cookbook focuses on the Code Review Assistant** as it clearly demonstrates both memory (learning patterns) and context editing (handling long reviews).

## 3. Quick Start Examples {#quick-start}

Let's see memory and context management in action with simple examples.

### Setup

First, install dependencies and configure your environment:


```coconut
%%capture
# Install required packages
# Option 1: From requirements.txt
# %pip install -q -r requirements.txt

# Option 2: Direct install
%pip install -q anthropic python-dotenv ipykernel
```

**⚠️ Important**: Create a `.env` file in this directory:

```bash
# Copy .env.example to .env and add your API key
cp .env.example .env
```

Then edit `.env` to add your Anthropic API key from https://console.anthropic.com/


```coconut
import os
from typing import cast

from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configuration - use alias for automatic updates
MODEL = "claude-sonnet-4-5"  # Can override via ANTHROPIC_MODEL env var
if os.getenv("ANTHROPIC_MODEL"):
    MODEL = os.getenv("ANTHROPIC_MODEL")

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found. Copy .env.example to .env and add your API key.")

if not MODEL:
    raise ValueError("ANTHROPIC_MODEL not found. Copy .env.example to .env and set the model.")

MODEL = cast(str, MODEL)

client = Anthropic(api_key=API_KEY)

print("✓ API key loaded")
print(f"✓ Using model: {MODEL}")
```

    ✓ API key loaded
    ✓ Using model: claude-sonnet-4-5


### Example 1: Basic Memory Usage

Let's see Claude use memory to store information for future reference.

**Helper Functions**

These examples use helper functions from `demo_helpers.py`:

- **`run_conversation_loop()`**: Handles the API conversation loop
  - Calls Claude's API with memory tool enabled
  - Executes tool uses (memory operations)
  - Continues until Claude stops using tools
  - Returns the final response

- **`run_conversation_turn()`**: Single turn (used in Example 3)
  - Same as above but returns after one API call
  - Useful when you need fine-grained control

- **`print_context_management_info()`**: Displays context clearing stats
  - Shows tokens saved, tool uses cleared
  - Helps visualize when context editing triggers

**⚠️ Note on Memory Clearing**

The following cell clears all memory files to provide a clean slate for this demonstration. This is useful for running the notebook multiple times to see consistent results.

**In production applications**, you should carefully consider whether to clear all memory, as it permanently removes learned patterns. Consider using selective deletion or organizing memory into project-specific directories instead.


```coconut
# Import helper functions
from memory_demo.demo_helpers import (
    run_conversation_loop,
    run_conversation_turn,
    print_context_management_info,
)
from memory_tool import MemoryToolHandler

# Initialize
client = Anthropic()
memory = MemoryToolHandler(base_path="./demo_memory")

# Clear any existing memories to start fresh
print("🧹 Clearing previous memories...")
memory.clear_all_memory()
print("✓ Memory cleared\n")

# Load example code with a race condition bug
with open("memory_demo/sample_code/web_scraper_v1.py", "r") as f:
    code_to_review = f.read()

messages = [
    {
        "role": "user",
        "content": f"I'm reviewing a multi-threaded web scraper that sometimes returns fewer results than expected. The count is inconsistent across runs. Can you find the issue?\n\n```python\n{code_to_review}\n```",
    }
]

print("=" * 60)
print("📝 SESSION 1: Learning from a bug")
print("=" * 60)

# Run conversation loop
response = run_conversation_loop(
    client=client,
    model=MODEL,
    messages=messages,
    memory_handler=memory,
    system="You are a code reviewer.",
    max_tokens=2048,
    max_turns=5,
    verbose=True,
)

print("\n" + "=" * 60)
print("✅ Session 1 complete!")
print("=" * 60)
```

    🧹 Clearing previous memories...
    ✓ Memory cleared
    
    ============================================================
    📝 SESSION 1: Learning from a bug
    ============================================================
    
    🔄 Turn 1:
    💬 Claude: I'll review this code for the race condition issue. Let me first check my memory, then analyze the problem.
    
      🔧 Memory tool: view /memories
      ✓ Result: Directory: /memories
    (empty)
    
    🔄 Turn 2:
      🔧 Memory tool: create /memories/review.md
      ✓ Result: File created successfully at /memories/review.md
    
    🔄 Turn 3:
    💬 Claude: ## Code Review: Multi-threaded Web Scraper Race Condition
    
    ### 🔴 **Critical Issue Found: Race Condition in Shared State**
    
    You've correctly identified the problem! The code has **race conditions** caused by multiple threads modifying shared lists without synchronization.
    
    ---
    
    ### **The Problem**
    
    **Lines with race conditions:**
    ```python
    self.results.append(result)  # RACE CONDITION
    ```
    
    **Problem:** 
    - Python's `list.append()` is **NOT thread-safe** for concurrent modifications
    - Multiple threads simultaneously appending can cause:
      - **Lost updates:** One thread's append overwrites another's
      - **Inconsistent state:** Internal list structure can be corrupted
      - **Unpredictable counts:** Results randomly dropped
    
    **Why it happens:**
    `list.append()` involves multiple operations:
    1. Read current list size
    2. Allocate space
    3. Insert item
    4. Update size
    
    # In scrape_urls():
    if "error" in result:
        self.failed_urls.append(result["url"])  # ⚠️ NOT THREAD-SAFE
    else:
        self.results.append(result)  # ⚠️ NOT THREAD-SAFE
    ```
    
    **Why this causes lost results:**
    - Python's `list.append()` is **not atomic** in all scenarios
    - When multiple threads call `append()` simultaneously, the internal list operations can interleave
    - This can cause lost updates where one thread's append overwrites another's
    
    ---
    
    ### **Solutions**
    
    #### **Option 1: Use Thread Locks (Traditional approach)**
    ```python
    import threading
    
    class WebScraper:
        """Web scraper that fetches multiple URLs concurrently."""
    
        def __init__(self, max_workers: int = 10):
            self.max_workers = max_workers
            self.results = []
            self.failed_urls = []
            self.lock = threading.Lock()  # ✅ Add lock
    
        def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.fetch_url, url) for url in urls]
    
                for future in as_completed(futures):
                    result = future.result()
    
                    # ✅ Protect shared state with lock
                    with self.lock:
                        if "error" in result:
                            self.failed_urls.append(result["url"])
                        else:
                            self.results.append(result)
    
            return self.results
    ```
    
    #### **Option 2: Use Thread-Safe Collections**
    ```python
    from queue import Queue
    
    class WebScraper:
        """Web scraper that fetches multiple URLs concurrently."""
    
        def __init__(self, max_workers: int = 10):
            self.max_workers = max_workers
            self.results = Queue()  # ✅ Thread-safe
            self.failed_urls = Queue()  # ✅ Thread-safe
    
        def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.fetch_url, url) for url in urls]
    
                for future in as_completed(futures):
                    result = future.result()
    
                    if "error" in result:
                        self.failed_urls.put(result["url"])
                    else:
                        self.results.put(result)
    
            # Convert Queue to list for return
            return list(self.results.queue)
    ```
    
    #### **Option 3: Collect Results from Futures (Best approach)**
    ```python
    def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        ✅ FIXED: Collect results in main thread, avoiding shared state.
        """
        results = []  # Local to main thread
        failed_urls = []  # Local to main thread
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.fetch_url, url) for url in urls]
    
            for future in as_completed(futures):
                result = future.result()
    
                # ✅ Only main thread accesses these lists
                if "error" in result:
                    failed_urls.append(result["url"])
                else:
                    results.append(result)
    
        # Update instance variables once, in main thread
        self.results = results
        self.failed_urls = failed_urls
        
        return self.results
    ```
    
    ---
    
    ### **Recommended Fix (Complete)**
    
    Here's the fully corrected version using **Option 3** (cleanest approach):
    
    ```python
    """
    Concurrent web scraper - FIXED VERSION
    Results collected in main thread to avoid race conditions.
    """
    
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from typing import List, Dict, Any
    
    import requests
    
    
    class WebScraper:
        """Web scraper that fetches multiple URLs concurrently."""
    
        def __init__(self, max_workers: int = 10):
            self.max_workers = max_workers
            self.results = []
            self.failed_urls = []
    
        def fetch_url(self, url: str) -> Dict[str, Any]:
            """Fetch a single URL and return the result."""
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                return {
                    "url": url,
                    "status": response.status_code,
                    "content_length": len(response.content),
                }
            except requests.exceptions.RequestException as e:
                return {"url": url, "error": str(e)}
    
        def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
            """
            Scrape multiple URLs concurrently.
            
            ✅ FIXED: Results collected in main thread only.
            """
            results = []  # ✅ Local variables - no race condition
            failed_urls = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.fetch_url, url) for url in urls]
    
                for future in as_completed(futures):
                    result = future.result()
    
                    # ✅ Only the main thread modifies these lists
                    if "error" in result:
                        failed_urls.append(result["url"])
                    else:
                        results.append(result)
    
            # Update instance variables after all threads complete
            self.results = results
            self.failed_urls = failed_urls
            
            return self.results
    
        def get_stats(self) -> Dict[str, int]:
            """Get scraping statistics."""
            return {
                "total_results": len(self.results),
                "failed_urls": len(self.failed_urls),
                "success_rate": (

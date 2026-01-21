# Guide: Implementing Context Editing & Memory for Long-Running AI Agents

## Introduction

AI agents that operate across multiple sessions or handle extended tasks face two primary challenges: they lose learned patterns between conversations, and their context windows can fill up during long interactions. This guide demonstrates how to address these challenges using Claude's memory tool and context editing capabilities.

### Prerequisites

**Required Knowledge:**
- Python fundamentals (functions, classes, async/await basics)
- Basic understanding of REST APIs and JSON

**Required Tools:**
- Python 3.10 or higher
- Anthropic API key

**Recommended:**
- Familiarity with concurrent programming concepts
- Basic understanding of context windows in LLMs

## Setup

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install anthropic python-dotenv
```

### 3. Configure API Key

```bash
# Copy the example environment file
cp .env.example .env
```

Edit the `.env` file and add your Anthropic API key from [https://console.anthropic.com/](https://console.anthropic.com/):

```bash
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_MODEL=claude-sonnet-4-5  # Optional: specify your preferred model
```

### 4. Initialize Your Python Environment

```python
import os
from typing import cast
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key and model
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found. Copy .env.example to .env and add your API key.")

# Initialize the Anthropic client
client = Anthropic(api_key=API_KEY)

print("✓ API key loaded")
print(f"✓ Using model: {MODEL}")
```

## Part 1: Implementing Basic Memory Usage

Memory enables Claude to store information for future reference, allowing it to learn patterns across conversations.

### Step 1: Import Helper Functions

First, let's import the necessary helper functions. These will handle the conversation loop and memory operations:

```python
from memory_demo.demo_helpers import (
    run_conversation_loop,
    run_conversation_turn,
    print_context_management_info,
)
from memory_tool import MemoryToolHandler
```

### Step 2: Initialize Memory Handler

Create a memory handler instance that will manage Claude's memory files:

```python
# Initialize the memory handler
client = Anthropic()
memory = MemoryToolHandler(base_path="./demo_memory")

# Clear any existing memories to start fresh
print("🧹 Clearing previous memories...")
memory.clear_all_memory()
print("✓ Memory cleared\n")
```

**Note:** In production applications, carefully consider whether to clear all memory, as it permanently removes learned patterns. Consider using selective deletion or organizing memory into project-specific directories instead.

### Step 3: Prepare Your First Code Review

Let's load a sample code file that contains a race condition bug:

```python
# Load example code with a race condition bug
with open("memory_demo/sample_code/web_scraper_v1.py", "r") as f:
    code_to_review = f.read()

# Prepare the initial message
messages = [
    {
        "role": "user",
        "content": f"I'm reviewing a multi-threaded web scraper that sometimes returns fewer results than expected. The count is inconsistent across runs. Can you find the issue?\n\n```python\n{code_to_review}\n```",
    }
]
```

### Step 4: Run the First Conversation Session

Now, let's run a conversation loop where Claude will analyze the code, identify the bug, and store what it learns in memory:

```python
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

When you run this, Claude will:
1. Check its memory (which will be empty initially)
2. Analyze the code and identify the race condition
3. Create a memory file (`/memories/review.md`) documenting the bug pattern and solution
4. Provide a detailed review with multiple fix options

### Step 5: Examine What Claude Learned

After the first session, Claude has stored information about the race condition pattern. Let's see how this helps in subsequent sessions:

```python
# Load a different but similar code file
with open("memory_demo/sample_code/data_processor_v1.py", "r") as f:
    new_code_to_review = f.read()

# Prepare a new message
messages = [
    {
        "role": "user",
        "content": f"I have another concurrent data processor that's losing data. Can you review it?\n\n```python\n{new_code_to_review}\n```",
    }
]

print("=" * 60)
print("📝 SESSION 2: Applying learned patterns")
print("=" * 60)

# Run another conversation loop
response = run_conversation_loop(
    client=client,
    model=MODEL,
    messages=messages,
    memory_handler=memory,
    system="You are a code reviewer.",
    max_tokens=2048,
    max_turns=3,
    verbose=True,
)
```

In this second session, Claude will:
1. Check its memory and find the previously stored race condition pattern
2. Immediately recognize the similar issue in the new code
3. Provide a faster, more targeted review by applying the learned pattern

## Part 2: Implementing Context Editing

Context editing helps manage long conversations by automatically clearing old tool results and thinking blocks when the context grows too large.

### Step 1: Enable Context Editing

To enable context editing, you need to add specific tools to your API calls. Let's create a function that sets up context editing:

```python
def create_context_editing_tools():
    """Create tools for context editing management."""
    return [
        {
            "type": "clear_tool_uses_20250919",
            "name": "clear_tool_uses",
            "description": "Clear old tool uses when context grows large",
            "config": {
                "trigger_threshold": 0.7,  # Clear when context is 70% full
                "retention_policy": "keep_recent",  # Keep recent tool uses
                "max_tool_uses_to_keep": 10  # Keep last 10 tool uses
            }
        },
        {
            "type": "clear_thinking_20251015",
            "name": "clear_thinking",
            "description": "Manage extended thinking blocks",
            "config": {
                "trigger_threshold": 0.8,  # Clear when context is 80% full
                "max_thinking_blocks_to_keep": 5  # Keep last 5 thinking blocks
            }
        }
    ]
```

### Step 2: Run a Long Conversation with Context Editing

Now let's simulate a long conversation where context editing becomes necessary:

```python
# Prepare a long conversation with multiple turns
long_messages = [
    {
        "role": "user",
        "content": "I need you to review this entire codebase. It's quite large, so we'll go through it piece by piece."
    }
]

# Add context editing tools to the API call
tools = create_context_editing_tools()

print("=" * 60)
print("📝 LONG SESSION: Testing context editing")
print("=" * 60)

# Run multiple conversation turns
for turn in range(1, 11):
    print(f"\n🔄 Turn {turn}:")
    
    # Add more content to simulate a long conversation
    if turn > 1:
        long_messages.append({
            "role": "user",
            "content": f"Here's another file to review (file_{turn}.py):\n\n```python\n# Sample code content for file {turn}\n```"
        })
    
    # Run a single conversation turn
    response = run_conversation_turn(
        client=client,
        model=MODEL,
        messages=long_messages,
        memory_handler=memory,
        tools=tools,
        system="You are a code reviewer.",
        max_tokens=1024,
        verbose=True,
    )
    
    # Add Claude's response to the messages
    long_messages.append({
        "role": "assistant",
        "content": response
    })
    
    # Print context management info if available
    print_context_management_info(response)
```

### Step 3: Monitor Context Usage

As the conversation progresses, you'll see context editing in action. The system will automatically clear old tool results and thinking blocks when the context approaches its limits, keeping the conversation manageable.

## Part 3: Building a Production-Ready Code Review Assistant

Now let's combine memory and context editing to build a robust code review assistant.

### Step 1: Create the Assistant Class

```python
class CodeReviewAssistant:
    """A code review assistant with memory and context management."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.memory = MemoryToolHandler(base_path="./code_review_memories")
        self.conversation_history = []
        
    def create_tools(self):
        """Create tools for the assistant."""
        tools = [
            {
                "type": "memory_20250818",
                "name": "memory",
                "description": "Read and write memories for cross-conversation learning"
            }
        ]
        
        # Add context editing tools for long sessions
        tools.extend(create_context_editing_tools())
        
        return tools
    
    def review_code(self, code: str, context: str = ""):
        """Review a piece of code."""
        # Prepare the message
        user_message = f"Please review this code:\n\n```python\n{code}\n```"
        if context:
            user_message = f"{context}\n\n{user_message}"
        
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Run the conversation
        response = run_conversation_loop(
            client=self.client,
            model=self.model,
            messages=self.conversation_history,
            memory_handler=self.memory,
            tools=self.create_tools(),
            system="You are an expert code reviewer. Look for bugs, security issues, performance problems, and code quality issues.",
            max_tokens=2048,
            max_turns=3,
            verbose=False
        )
        
        # Add the response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("✓ Conversation history cleared")
```

### Step 2: Use the Assistant

```python
# Initialize the assistant
assistant = CodeReviewAssistant(api_key=API_KEY, model=MODEL)

# Review multiple code snippets
code_snippets = [
    ("web_scraper_v1.py", "A multi-threaded web scraper with potential race conditions"),
    ("data_processor_v1.py", "A concurrent data processor that might lose data"),
    ("api_client_v1.py", "An API client with error handling issues")
]

for filename, description in code_snippets:
    print(f"\n{'='*60}")
    print(f"🔍 Reviewing: {filename}")
    print(f"📝 Description: {description}")
    print(f"{'='*60}\n")
    
    # Load and review the code
    with open(f"memory_demo/sample_code/{filename}", "r") as f:
        code = f.read()
    
    review = assistant.review_code(code, f"Review this {description}:")
    print(f"Review:\n{review}\n")
    
    # Clear conversation between unrelated reviews if needed
    if filename != code_snippets[-1][0]:
        assistant.clear_conversation()
```

## Best Practices

### 1. Memory Organization
- Organize memories by project or domain
- Use descriptive memory file names
- Regularly review and prune outdated memories
- Consider implementing memory versioning for important patterns

### 2. Context Editing Configuration
- Adjust trigger thresholds based on your specific use case
- Monitor context usage to find optimal settings
- Balance between keeping enough context and preventing overflow
- Test different retention policies for your workflow

### 3. Security Considerations
- Never store sensitive information in memories
- Implement access controls for memory files
- Regularly audit memory contents
- Consider encrypting sensitive memories

### 4. Performance Optimization
- Use memory for frequently repeated patterns
- Implement caching for common memory lookups
- Monitor API usage and costs
- Batch related operations when possible

## Conclusion

You've now learned how to implement context editing and memory for long-running AI agents. By combining these techniques, you can build agents that:

1. **Learn over time** by storing patterns in memory
2. **Handle long conversations** through automatic context management
3. **Provide consistent, improved performance** across multiple sessions

The key insight is that context is a finite resource that needs active management. With memory and context editing, you can build more efficient, capable, and cost-effective AI agents that improve with use.

## Next Steps

1. **Experiment with different memory organizations** to find what works best for your use case
2. **Monitor context usage patterns** in your applications to optimize trigger thresholds
3. **Implement memory versioning** for critical patterns that evolve over time
4. **Explore integration with existing systems** like version control or project management tools

Remember that these techniques are most powerful when tailored to your specific workflow. Start with the examples provided, then adapt them to solve your unique challenges.
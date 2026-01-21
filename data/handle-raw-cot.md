# Handling Chain of Thought in GPT-OSS Models: A Developer Guide

This guide explains how to properly handle the raw Chain of Thought (CoT) output from GPT-OSS models. The raw CoT is essential for tool calling performance but contains internal reasoning that should not be exposed to end users.

## Understanding the Chain of Thought

GPT-OSS models generate a raw Chain of Thought that:
- Contains the model's internal reasoning process
- Is crucial for proper tool calling functionality
- May include potentially harmful content or implementation details
- Should **not** be shown to end users

## Harmony Chat Template Handling

If you're working with chat templates or handling tokens directly, follow these Harmony response format guidelines:

### Key Principles

1. **Channel Assignment**: CoT is issued to the `analysis` channel
2. **Message Cleanup**: After a message to the `final` channel in a subsequent sampling turn, all `analysis` messages should be dropped
3. **Tool Call Preservation**: If the last assistant message was a tool call, preserve `analysis` messages until the previous `final` message

## Chat Completions API Implementation

Since OpenAI's hosted models don't currently expose raw CoT, follow OpenRouter's convention:

### Request Configuration
```json
{
  "reasoning": {
    "exclude": true  // Set to false to include raw CoT in response
  }
}
```

### Response Structure
- Raw CoT is exposed as a `reasoning` property on message objects
- For streaming responses, deltas include a `reasoning` property
- Subsequent turns should receive previous reasoning as `reasoning`

## Responses API Implementation

The Responses API has been extended to support CoT with these changes:

### Type Definitions
```typescript
type ReasoningItem = {
  id: string;
  type: "reasoning";
  summary: SummaryContent[];
  content: ReasoningTextContent[];  // New: Contains raw CoT
};

type ReasoningTextContent = {
  type: "reasoning_text";
  text: string;
};

type ReasoningTextDeltaEvent = {
  type: "response.reasoning_text.delta";
  sequence_number: number;
  item_id: string;
  output_index: number;
  content_index: number;
  delta: string;
};

type ReasoningTextDoneEvent = {
  type: "response.reasoning_text.done";
  sequence_number: number;
  item_id: string;
  output_index: number;
  content_index: number;
  text: string;
};
```

### Event Flow
```typescript
// CoT streaming begins
{
  type: "response.reasoning_text.delta",
  sequence_number: 14,
  item_id: "rs_67f47a642e788191aec9b5c1a35ab3c3016f2c95937d6e91",
  output_index: 0,
  content_index: 0,
  delta: "The "
}

// CoT streaming completes
{
  type: "response.reasoning_text.done",
  sequence_number: 18,
  item_id: "rs_67f47a642e788191aec9b5c1a35ab3c3016f2c95937d6e91",
  output_index: 0,
  content_index: 0,
  text: "The user asked me to think"
}
```

### Example Response
```typescript
"output": [
  {
    "type": "reasoning",
    "id": "rs_67f47a642e788191aec9b5c1a35ab3c3016f2c95937d6e91",
    "summary": [
      {
        "type": "summary_text",
        "text": "**Calculating volume of gold for Pluto layer**\n\nStarting with the approximation..."
      }
    ],
    "content": [
      {
        "type": "reasoning_text",
        "text": "The user asked me to think..."
      }
    ]
  }
]
```

## Displaying CoT to End Users

### Critical Security Consideration
**Never show raw CoT to end users** because it may contain:
- Potentially harmful content
- Internal implementation details
- Developer instructions
- Unfiltered reasoning that could be misleading

### Recommended Approach
Instead of raw CoT, display a summarized version similar to OpenAI's production implementations:
- Use a summarizer model to review and filter content
- Block harmful or sensitive information
- Present only safe, relevant reasoning to users

## Best Practices Summary

1. **Always separate** raw CoT from user-facing content
2. **Follow Harmony guidelines** for chat template implementations
3. **Use OpenRouter conventions** for Chat Completions API
4. **Leverage the enhanced Responses API** for proper CoT handling
5. **Implement content filtering** before displaying any reasoning to users

By following these guidelines, you can safely leverage the performance benefits of CoT while protecting users from potentially harmful or confusing internal model reasoning.
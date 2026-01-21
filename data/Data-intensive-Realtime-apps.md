# Practical Guide to Data-Intensive Apps with the Realtime API

This guide provides practical strategies for AI Engineers to maximize the effectiveness of OpenAI's Realtime API when dealing with data-intensive function calls. We'll focus on scenarios common in speech-to-speech agents, where large amounts of data must be handled efficiently and reliably.

## Prerequisites

This tutorial assumes you have basic familiarity with the OpenAI Realtime API. We'll focus on optimization techniques rather than initial setup.

## Understanding the Challenge

### What is a Data-Intensive Function Call?

Agents often need to access external data through APIs. When these APIs return large payloads (thousands of tokens), they can overwhelm the Realtime API—leading to slow responses or failures. Our example uses an NBA Scouting Agent that searches draft prospects, demonstrating how to handle realistic, data-heavy scenarios.

## Step 1: Design Focused Function Calls

Start by breaking down monolithic functions into smaller, well-defined ones. Each function should have a clear purpose and return only essential information.

### Initial Problematic Function

Here's an example of a function that returns too much data:

```json
{
  "type": "session.update",
  "session": {
    "tools": [
      {
        "type": "function",
        "name": "searchDraftProspects",
        "description": "Search draft prospects for a given year e.g., Point Guard",
        "parameters": {
          "type": "object",
          "properties": {
            "position": {
              "type": "string",
              "description": "The player position",
              "enum": ["Point Guard", "Shooting Guard", "Small Forward", "Power Forward", "Center", "Any"]
            },
            "year": {
              "type": "number",
              "description": "Draft year e.g., 2025"
            },
            "mockDraftRanking": {
              "type": "number",
              "description": "Predicted Draft Ranking"
            }
          },
          "required": ["position", "year"]
        }
      }
    ],
    "tool_choice": "auto"
  }
}
```

This function returns a massive payload including player details, stats, scouting reports, media links, and agent information—easily thousands of tokens.

### Optimized Function Design

Split the functionality into two focused functions:

```json
{
  "tools": [
    {
      "type": "function",
      "name": "searchDraftProspects",
      "description": "Search NBA draft prospects by position, draft year, and projected ranking, returning only general statistics to optimize response size.",
      "parameters": {
        "type": "object",
        "properties": {
          "position": {
            "type": "string",
            "description": "The player's basketball position.",
            "enum": ["Point Guard", "Shooting Guard", "Small Forward", "Power Forward", "Center", "Any"]
          },
          "year": {
            "type": "number",
            "description": "Draft year, e.g., 2025"
          },
          "maxMockDraftRanking": {
            "type": "number",
            "description": "Maximum predicted draft ranking (e.g., top 10)"
          }
        },
        "required": ["position", "year"]
      }
    },
    {
      "type": "function",
      "name": "getProspectDetails",
      "description": "Fetch detailed information for a specific NBA prospect, including comprehensive stats, agent details, and scouting reports.",
      "parameters": {
        "type": "object",
        "properties": {
          "playerName": {
            "type": "string",
            "description": "Full name of the prospect (e.g., Jalen Storm)"
          },
          "year": {
            "type": "number",
            "description": "Draft year, e.g., 2025"
          },
          "includeAgentInfo": {
            "type": "boolean",
            "description": "Include agent information"
          },
          "includeStats": {
            "type": "boolean",
            "description": "Include detailed player statistics"
          },
          "includeScoutingReport": {
            "type": "boolean",
            "description": "Include scouting report details"
          }
        },
        "required": ["playerName", "year"]
      }
    }
  ],
  "tool_choice": "auto"
}
```

**Why this works:** The first function returns only basic information for filtering. The second function provides details on demand, reducing token usage when full details aren't needed.

## Step 2: Optimize Conversation Context

Realtime conversations have a rolling context window (~16,000 tokens). As conversations progress, context management becomes crucial.

### Strategy 1: Periodic Summarization

Periodically summarize the conversation to reduce context size. This cuts both cost and latency. For implementation details, see the [Context Summarization with Realtime API](https://cookbook.openai.com/examples/context_summarization_with_realtime_api) guide.

### Strategy 2: Role Reminders

If the model loses track of instructions during data-heavy exchanges, periodically remind it of its system prompt and available tools using `session.update`.

## Step 3: Optimize Data Processing

### Apply Response Filtering

Use parameters to filter responses at the data level. Fewer tokens returned mean better quality responses.

```json
{
  "status": {
    "code": 200,
    "message": "SUCCESS"
  },
  "found": 4274,
  "offset": 0,
  "limit": 5,
  "data": [
    {
      "prospectId": 10001,
      "data": {
        "PropertyInfo": {
          "bedroomCount": 2,
          "bathroomCount": 2,
          "livingAreaSize": 1089,
          "yearBuilt": "1985"
        }
      }
    }
  ]
}
```

### Flatten Hierarchical Payloads

Remove unnecessary nesting to reduce token count and improve model comprehension.

```json
{
  "status": {
    "code": 200,
    "message": "SUCCESS"
  },
  "found": 4274,
  "offset": 0,
  "limit": 2,
  "data": [
    {
      "prospectId": 10001,
      "league": "NCAA",
      "firstName": "Jalen",
      "lastName": "Storm",
      "position": "PG",
      "heightFeet": 6,
      "heightInches": 4,
      "weightPounds": 205,
      "hometown": "Springfield",
      "state": "IL",
      "collegeTeam": "Springfield Tigers",
      "gamesPlayed": 32,
      "minutesPerGame": 34.5,
      "FieldGoalPercentage": 47.2,
      "averagePoints": 21.3,
      "averageAssists": 6.8,
      "mockDraftRanking": 5
    }
  ]
}
```

### Experiment with Data Formats

Different formats affect how well the model processes data. JSON and YAML generally work better than tabular formats like Markdown for complex data.

```yaml
status:
  code: 200
  message: "SUCCESS"
found: 4274
offset: 0
limit: 10
data:
  - prospectId: 10001
    firstName: "Jalen"
    lastName: "Storm"
    position: "PG"
    heightFeet: 6
    heightInches: 4
    averagePoints: 21.3
    mockDraftRanking: 5
```

## Step 4: Use Hint Prompts After Data-Heavy Calls

After a function returns complex data, provide a hint prompt to guide the model's interpretation.

```javascript
const prospectSearchPrompt = `
Parse NBA prospect data and provide a concise, engaging response.

General Guidelines
- Act as an NBA scouting expert.
- Highlight key strengths and notable attributes.
- Use conversational language.
- Mention identical attributes once.
- Ignore IDs and URLs.

Player Details
- State height conversationally ("six-foot-eight").
- Round weights to nearest 5 lbs.

Stats & Draft Info
- Round stats to nearest whole number.
- Use general terms for draft ranking ("top-five pick").

Experience
- Refer to players as freshman, sophomore, etc., or mention professional experience.

Location & Team
- Mention hometown city and state/country.
- Describe teams conversationally.

Skip (unless asked explicitly)
- Exact birth dates
- IDs
- Agent/contact details
- URLs

Examples
- "Jalen Storm, a dynamic six-foot-four point guard from Springfield, Illinois, averages 21 points per game."
- "Known for his clutch shooting, he's projected as a top-five pick."

Important: Respond based strictly on provided data, without inventing details.
`;
```

### Implementing Hint Prompts

Add the function call result to the conversation, then emit a response with your hint prompt:

```javascript
// Add function call output to conversation
const conversationItem = {
  type: 'conversation.item.create',
  previous_item_id: output.id,
  item: {
    call_id: output.call_id,
    type: 'function_call_output',
    output: `Draft Prospect Search Results: ${result}`
  }
};

dataChannel.send(JSON.stringify(conversationItem));

// Emit response with hint prompt
const event = {
  type: 'response.create',
  conversation: "none",
  response: {
    instructions: prospectSearchPrompt
  }
};

dataChannel.send(JSON.stringify(event));
```

## Summary

To optimize data-intensive Realtime API applications:

1. **Design focused functions** that return minimal necessary data
2. **Manage conversation context** through summarization and reminders
3. **Optimize data processing** with filtering, flattening, and format experimentation
4. **Use hint prompts** to guide model interpretation after data-heavy calls

These strategies will help you build efficient, reliable real-time conversational agents that handle large data payloads effectively.
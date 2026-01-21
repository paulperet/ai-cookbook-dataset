# Build a Customer Support Voice Agent with GPT Realtime Mini

## Introduction

This guide walks you through building a customer support voice agent using OpenAI's GPT Realtime Mini and the Agents SDK. You'll create a system that handles both general policy questions and authenticated customer-specific queries through a handoff architecture.

## Prerequisites

Before starting, ensure you have:
- Node.js and npm/pnpm installed
- An OpenAI API key with access to GPT Realtime Mini
- Basic familiarity with React and TypeScript

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/openai/openai-agents-js
cd openai-agents-js
```

### 2. Install Dependencies

```bash
npm install @openai/agents zod@3
```

### 3. Run the Example Application

```bash
pnpm examples:realtime-next
```

Navigate to `http://localhost:3000` to see the basic chat interface.

## Understanding the Architecture

Our application uses a **handoff architecture** where a primary agent routes queries to specialized agents:

1. **Main Agent**: Entry point that classifies intent and routes queries
2. **QA Agent**: Handles general policy questions using document lookup
3. **Flight Status Agent**: Manages authenticated customer-specific queries

## Step 1: Create the Main Agent

Open `openai-agents-js/examples/realtime-next/src/app/page.tsx` and add the main agent:

```typescript
const mainAgent = new RealtimeAgent({
  name: 'Main Agent',
  instructions:
    'You are the entry point for all customer queries. Default to the no-auth QA flow. If authentication is needed and validated, escalate to the Auth Layer by handing off to either the Flight Status Checker or Rebooking Agent. Do not answer policy questions from your own knowledge; rely on subordinate agents and tools.',
  tools: [
    checkFlightsTool, // We'll define this later
  ],
  handoffs: [qaAgent], // We'll define this next
});
```

The main agent acts as an intelligent router, analyzing user intent and directing queries to the appropriate specialist.

## Step 2: Build the QA Agent

The QA Agent handles general airline policy questions by searching through documentation. In a production system, this would connect to a vector database, but for this demo we'll use a mock document.

### 2.1 Create the Document Lookup Tool

```typescript
const documentLookupTool = tool({
  name: 'document_lookup_tool',
  description: 'Looks up answers from known airline documentation to handle general questions without authentication.',
  parameters: z.object({
    request: z.string(),
  }),
  execute: async ({ request }) => {
    const mockDocument = `**Airline Customer Support — Quick Reference**

1. Each passenger may bring 1 carry-on (22 x 14 x 9) and 1 personal item.
2. Checked bags must be under 50 lbs; overweight fees apply.
3. Online check-in opens 24 hours before departure.
4. Seat upgrades can be requested up to 1 hour before boarding.
5. Wi‑Fi is complimentary on all flights over 2 hours.
6. Customers can change flights once for free within 24 hours of booking.
7. Exit rows offer extra legroom and require passengers to meet safety criteria.
8. Refunds can be requested for canceled or delayed flights exceeding 3 hours.
9. Pets are allowed in the cabin if under 20 lbs and in an approved carrier.
10. For additional help, contact our support team via chat or call center.`;
    return mockDocument;
  },
});
```

### 2.2 Create the QA Agent

```typescript
const qaAgent = new RealtimeAgent({
  name: 'QA Agent',
  instructions:
    'You handle general customer questions using the document lookup tool. Use only the document lookup for answers. If the request may involve personal data or operations (rebooking, flight status), call the auth check tool. If auth is required and validated, handoff to the appropriate Auth Layer agent.',
  tools: [documentLookupTool],
});
```

## Step 3: Implement Authentication with Flight Status Agent

For customer-specific queries like flight status, we need an authentication layer. The Agents SDK supports this through the `needsApproval` parameter.

### 3.1 Create the Flight Status Tool

```typescript
// Demo-only credential store the tool can read at execution time
const credState: { username?: string; password?: string } = {};

const checkFlightsTool = tool({
  name: 'checkFlightsTool',
  description: 'Call this tool if the user queries about their current flight status',
  parameters: z.object({}),
  // Require approval so the UI can collect creds before executing.
  needsApproval: true,
  execute: async () => {
    if (!credState.username || !credState.password) {
      return 'Authentication missing.';
    }
    return `${credState.username} you are currently booked on the 8am flight from SFO to JFK`;
  },
});
```

### 3.2 Handle Authentication in the UI

Add state management and event listeners to handle authentication requests:

```typescript
export default function Home() {
  const session = useRef<RealtimeSession<any> | null>(null);
  const [credOpen, setCredOpen] = useState(false);
  const [credUsername, setCredUsername] = useState('');
  const [credPassword, setCredPassword] = useState('');
  const [pendingApproval, setPendingApproval] = useState<any | null>(null);

  useEffect(() => {
    session.current = new RealtimeSession(mainAgent, {
      model: 'gpt-realtime-mini',
      outputGuardrailSettings: {
        debounceTextLength: 200,
      },
      config: {
        audio: {
          output: {
            voice: 'cedar',
          },
        },
      },
    });

    // Listen for authentication requests
    session.current.on(
      'tool_approval_requested',
      (_context, _agent, approvalRequest) => {
        setPendingApproval(approvalRequest.approvalItem);
        setCredUsername('');
        setCredPassword('');
        setCredOpen(true);
      },
    );
  }, []);

  // Handle authentication submission
  function handleCredSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!credUsername || !credPassword) return;
    
    // Store credentials for the tool to read
    credState.username = credUsername;
    credState.password = credPassword;
    
    const approval = pendingApproval;
    setCredOpen(false);
    setPendingApproval(null);
    setCredUsername('');
    setCredPassword('');
    
    if (approval) session.current?.approve(approval);
  }

  // Handle authentication cancellation
  function handleCredCancel() {
    const approval = pendingApproval;
    setCredOpen(false);
    setPendingApproval(null);
    if (approval) session.current?.reject(approval);
  }

  return (
    <div className="relative">
      {credOpen && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black/50" />
          <div className="fixed top-0 left-0 right-0 flex justify-center p-4">
            <form
              onSubmit={handleCredSubmit}
              className="w-full max-w-sm rounded-lg bg-white p-4 shadow-xl"
            >
              <div className="mb-2 text-sm font-semibold">Authentication Required</div>
              <div className="mb-3 text-xs text-gray-600">
                Enter username and password to continue.
              </div>
              <input
                className="mb-2 w-full rounded border border-gray-300 p-2 text-sm focus:border-gray-500 focus:outline-none"
                placeholder="Username"
                value={credUsername}
                onChange={(e) => setCredUsername(e.target.value)}
              />
              <input
                type="password"
                className="mb-3 w-full rounded border border-gray-300 p-2 text-sm focus:border-gray-500 focus:outline-none"
                placeholder="Password"
                value={credPassword}
                onChange={(e) => setCredPassword(e.target.value)}
              />
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  className="rounded bg-gray-100 px-3 py-1.5 text-sm hover:bg-gray-200"
                  onClick={handleCredCancel}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded bg-black px-3 py-1.5 text-sm text-white hover:bg-gray-800 disabled:opacity-50"
                  disabled={!credUsername || !credPassword}
                >
                  Submit
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
      {/* Rest of your UI components */}
    </div>
  );
}
```

## Step 4: Complete the Agent Setup

Now let's complete the agent configuration by setting up cross-handoffs:

```typescript
// Set up cross-handoffs so agents can return or escalate
qaAgent.handoffs = [mainAgent];
```

This allows the QA Agent to hand queries back to the Main Agent when authentication is required.

## Step 5: Test Your Application

1. Start the development server:
```bash
pnpm dev
```

2. Navigate to `http://localhost:3000`

3. Test different scenarios:
   - **General question**: "What's your baggage policy?"
   - **Authenticated query**: "What's my flight status?" (will trigger authentication)

## How It Works

### Query Flow
1. User asks a question via voice or text
2. Main Agent analyzes the intent
3. For general questions: Routes to QA Agent → Document Lookup → Response
4. For flight status: Triggers authentication → User provides credentials → Flight Status Tool executes → Personalized response

### Authentication Flow
1. User asks for flight status
2. `checkFlightsTool` with `needsApproval: true` triggers `tool_approval_requested` event
3. UI displays authentication modal
4. User enters credentials
5. Credentials stored in `credState`
6. Tool executes with access to authenticated data
7. Personalized response returned

## Next Steps

To enhance this application:

1. **Replace mock document lookup** with a real vector database (see [Vector Databases Cookbook](https://cookbook.openai.com/examples/vector_databases/pinecone/readme))
2. **Add more specialized agents** for rebooking, cancellations, etc.
3. **Implement real authentication** with your backend system
4. **Add voice customization** options for different agent personas
5. **Implement conversation history** for context-aware responses

## Conclusion

You've successfully built a customer support voice agent using GPT Realtime Mini that:
- Routes queries intelligently between specialized agents
- Handles both general policy questions and authenticated queries
- Implements secure authentication workflows
- Provides a seamless voice interface for customer support

The handoff architecture ensures each query is handled by the most appropriate agent, while the authentication layer protects sensitive customer data. This foundation can be extended to support more complex customer service scenarios and integrated with your existing systems.
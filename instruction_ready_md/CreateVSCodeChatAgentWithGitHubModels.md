# Build a Visual Studio Code Chat Copilot Agent with Phi-3.5 Models

This guide walks you through creating a custom Visual Studio Code Chat Copilot agent powered by GitHub's Phi-3.5 models. You'll build an extension that adds a new chat participant, `@phicoding`, capable of generating code from text and images.

## Prerequisites

Before you begin, ensure you have the following:
*   **Node.js and npm** installed on your system.
*   A **GitHub Models** account and an API key. You can sign up at [https://gh.io/models](https://gh.io/models).
*   **Visual Studio Code** installed.

## Step 1: Set Up Your Development Environment

First, install the tools required for VS Code extension development.

```bash
npm install --global yo generator-code
```

## Step 2: Create the Extension Project

Generate a new VS Code extension project. We'll use TypeScript for development.

1.  Run the Yeoman generator:
    ```bash
    yo code
    ```
2.  Follow the prompts:
    *   Select **New Extension (TypeScript)**.
    *   Name your project (e.g., `phiext`).
    *   Fill in the remaining details as preferred.

3.  Open the newly created project folder in VS Code.

## Step 3: Configure the Extension (`package.json`)

The `package.json` file defines your extension's metadata, commands, and dependencies. Replace the default content with the configuration below.

**Key Sections:**
*   `chatParticipants`: Defines your `@phicoding` agent and its commands (`/help`, `/gen`, `/image`).
*   `configuration`: Creates settings for users to input their GitHub Models endpoint and API key.
*   `dependencies`: Adds the necessary Azure AI Inference SDK and other required packages.

```json
{
  "name": "phiext",
  "displayName": "Phi-3.5 Coding Assistant",
  "description": "A VS Code Chat Copilot agent powered by Phi-3.5 models for code generation.",
  "version": "0.0.1",
  "engines": {
    "vscode": "^1.93.0"
  },
  "categories": [
    "AI",
    "Chat"
  ],
  "activationEvents": [],
  "enabledApiProposals": [
    "chatVariableResolver"
  ],
  "main": "./dist/extension.js",
  "contributes": {
    "chatParticipants": [
      {
        "id": "chat.phicoding",
        "name": "phicoding",
        "description": "Hey! I am Microsoft Phi-3.5. I can help with coding problems, such as generating code from natural language or from charts in images.",
        "isSticky": true,
        "commands": [
          {
            "name": "help",
            "description": "Introduce myself to you"
          },
          {
            "name": "gen",
            "description": "Generate code for you with Microsoft Phi-3.5-mini-instruct"
          },
          {
            "name": "image",
            "description": "Generate code for a chart from an image (PNG or JPG) with Microsoft Phi-3.5-vision-instruct. Provide an image URL."
          }
        ]
      }
    ],
    "commands": [
      {
        "command": "phicoding.namesInEditor",
        "title": "Use Microsoft Phi 3.5 in Editor"
      }
    ],
    "configuration": {
      "type": "object",
      "title": "GitHub Models Configuration",
      "properties": {
        "githubmodels.endpoint": {
          "type": "string",
          "default": "https://models.inference.ai.azure.com",
          "description": "Your GitHub Models Endpoint",
          "order": 0
        },
        "githubmodels.api_key": {
          "type": "string",
          "default": "Your GitHub Models Token",
          "description": "Your GitHub Models API Key",
          "order": 1
        },
        "githubmodels.phi35instruct": {
          "type": "string",
          "default": "Phi-3.5-mini-instruct",
          "description": "Your Phi-3.5-Instruct Model Name",
          "order": 2
        },
        "githubmodels.phi35vision": {
          "type": "string",
          "default": "Phi-3.5-vision-instruct",
          "description": "Your Phi-3.5-Vision Model Name",
          "order": 3
        }
      }
    }
  },
  "scripts": {
    "vscode:prepublish": "npm run package",
    "compile": "webpack",
    "watch": "webpack --watch",
    "package": "webpack --mode production --devtool hidden-source-map",
    "compile-tests": "tsc -p . --outDir out",
    "watch-tests": "tsc -p . -w --outDir out",
    "pretest": "npm run compile-tests && npm run compile && npm run lint",
    "lint": "eslint src",
    "test": "vscode-test"
  },
  "devDependencies": {
    "@types/vscode": "^1.93.0",
    "@types/mocha": "^10.0.7",
    "@types/node": "20.x",
    "@typescript-eslint/eslint-plugin": "^8.3.0",
    "@typescript-eslint/parser": "^8.3.0",
    "eslint": "^9.9.1",
    "typescript": "^5.5.4",
    "ts-loader": "^9.5.1",
    "webpack": "^5.94.0",
    "webpack-cli": "^5.1.4",
    "@vscode/test-cli": "^0.0.10",
    "@vscode/test-electron": "^2.4.1"
  },
  "dependencies": {
    "@types/node-fetch": "^2.6.11",
    "node-fetch": "^3.3.2",
    "@azure-rest/ai-inference": "latest",
    "@azure/core-auth": "latest",
    "@azure/core-sse": "latest"
  }
}
```

After updating the file, run `npm install` in your terminal to install the new dependencies.

## Step 4: Implement the Extension Logic (`src/extension.ts`)

This is the core of your agent. The `activate` function sets up a chat request handler that processes the three commands defined earlier.

Replace the contents of `src/extension.ts` with the following code:

```typescript
import * as vscode from 'vscode';
import ModelClient from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";

// Interface for our custom chat result
interface IPhiChatResult extends vscode.ChatResult {
    metadata: {
        command: string;
    };
}

// Selector for the GPT-4o model (used for the /help command)
const MODEL_SELECTOR: vscode.LanguageModelChatSelector = { vendor: 'copilot', family: 'gpt-4o' };

// Helper function to validate image URLs
function isValidImageUrl(url: string): boolean {
    const regex = /^(https?:\/\/.*\.(?:png|jpg))$/i;
    return regex.test(url);
}

export function activate(context: vscode.ExtensionContext) {

    // The main handler for all @phicoding chat requests
    const codinghandler: vscode.ChatRequestHandler = async (request: vscode.ChatRequest, context: vscode.ChatContext, stream: vscode.ChatResponseStream, token: vscode.CancellationToken): Promise<IPhiChatResult> => {

        // Retrieve user configuration from VS Code settings
        const config: any = vscode.workspace.getConfiguration('githubmodels');
        const endPoint: string = config.get('endpoint');
        const apiKey: string = config.get('api_key');
        const phi35instruct: string = config.get('phi35instruct');
        const phi35vision: string = config.get('phi35vision');

        // Command: /help - Introduces the agent using GPT-4o
        if (request.command === 'help') {
            stream.progress("Welcome to Coding assistant with Microsoft Phi-3.5");

            try {
                const [model] = await vscode.lm.selectChatModels(MODEL_SELECTOR);
                if (model) {
                    const messages = [
                        vscode.LanguageModelChatMessage.User("Please help me express this content in a humorous way: I am a programming assistant who can help you convert natural language into code and generate code based on the charts in the images. Output format like this: 'Hey I am Phi ...'")
                    ];
                    const chatResponse = await model.sendRequest(messages, {}, token);
                    for await (const fragment of chatResponse.text) {
                        stream.markdown(fragment);
                    }
                }
            } catch (err) {
                console.error(err);
            }
            return { metadata: { command: 'help' } };
        }

        // Command: /gen - Generates code using Phi-3.5-mini-instruct
        if (request.command === 'gen') {
            stream.progress("Welcome to use phi-3.5 to generate code");

            const client = new ModelClient(endPoint, new AzureKeyCredential(apiKey));
            const response = await client.path("/chat/completions").post({
                body: {
                    messages: [
                        { role: "system", content: "You are a coding assistant. Help answer all code generation questions." },
                        { role: "user", content: request.prompt }
                    ],
                    model: phi35instruct,
                    temperature: 0.4,
                    max_tokens: 1000,
                    top_p: 1.0
                }
            });

            stream.markdown(response.body.choices[0].message.content);
            return { metadata: { command: 'gen' } };
        }

        // Command: /image - Generates code from an image using Phi-3.5-vision-instruct
        if (request.command === 'image') {
            stream.progress("Welcome to use phi-3.5 to generate code from an image (PNG or JPG).");

            // Validate the input is a proper image URL
            if (!isValidImageUrl(request.prompt)) {
                stream.markdown('Please provide a valid image URL (ending in .png or .jpg)');
                return { metadata: { command: 'image' } };
            } else {
                const client = new ModelClient(endPoint, new AzureKeyCredential(apiKey));
                const response = await client.path("/chat/completions").post({
                    body: {
                        messages: [
                            {
                                role: "system",
                                content: "You are a helpful assistant that describes images in detail."
                            },
                            {
                                role: "user",
                                content: [
                                    {
                                        type: "text",
                                        text: "Please generate code to recreate the chart in this image according to the following requirements:\n1. Keep all information in the chart, including data and text.\n2. Do not generate additional information not included in the chart.\n3. Extract data directly from the image.\n4. Provide code that saves the regenerated chart to ./output/demo.png"
                                    },
                                    {
                                        type: "image_url",
                                        image_url: { url: request.prompt }
                                    }
                                ]
                            }
                        ],
                        model: phi35vision,
                        temperature: 0.4,
                        max_tokens: 2048,
                        top_p: 1.0
                    }
                });

                stream.markdown(response.body.choices[0].message.content);
                return { metadata: { command: 'image' } };
            }
        }

        // Fallback for unrecognized commands
        return { metadata: { command: '' } };
    };

    // Register the chat participant with VS Code
    const phi_ext = vscode.chat.createChatParticipant("chat.phicoding", codinghandler);
    phi_ext.iconPath = new vscode.ThemeIcon('sparkle');

    // Provide a friendly follow-up suggestion
    phi_ext.followupProvider = {
        provideFollowups(result: IPhiChatResult, context: vscode.ChatContext, token: vscode.CancellationToken) {
            return [{
                prompt: 'Let us coding with Phi-3.5 😋',
                label: vscode.l10n.t('Enjoy coding with Phi-3.5'),
                command: 'help'
            } satisfies vscode.ChatFollowup];
        }
    };

    context.subscriptions.push(phi_ext);
}

export function deactivate() { }
```

## Step 5: Build and Run the Extension

1.  **Compile the TypeScript code:**
    ```bash
    npm run compile
    ```
    Or, to watch for changes:
    ```bash
    npm run watch
    ```

2.  **Run the Extension:**
    *   Press `F5` in VS Code. This will open a new Extension Development Host window.
    *   In the new window, open the Chat view (click the chat icon in the activity bar or use the command palette: `View: Open Chat`).

3.  **Configure Your Settings:**
    *   Before using the agent, you must add your GitHub Models credentials.
    *   Open the Settings (`Ctrl+,`).
    *   Search for "GitHub Models Configuration".
    *   Enter your **Endpoint** (default is correct) and your **API Key**.

## Step 6: Test Your Phi-3.5 Agent

In the Chat panel of your development window, you can now interact with your agent.

*   **`@phicoding /help`**: The agent will introduce itself using GPT-4o to generate a humorous greeting.
*   **`@phicoding /gen`**: Follow this with a natural language request for code. For example:
    ```
    @phicoding /gen Write a Python function to calculate the Fibonacci sequence.
    ```
    The agent will respond with code generated by the `Phi-3.5-mini-instruct` model.
*   **`@phicoding /image`**: Provide a publicly accessible URL to a PNG or JPG image containing a chart.
    ```
    @phicoding /image https://example.com/path/to/chart.png
    ```
    The agent will use the `Phi-3.5-vision-instruct` model to analyze the image and generate code to recreate the chart.

## Next Steps & Resources

*   **Package and Share:** Use `npm run package` to create a VSIX file you can share or publish to the marketplace.
*   **Enhance Your Agent:** Explore the [VS Code Chat API Guide](https://code.visualstudio.com/api/extension-guides/chat) to add more features like context awareness or integration with workspace files.
*   **Learn More:**
    *   [GitHub Models Sign-up](https://gh.io/models)
    *   [VS Code Extension API Getting Started](https://code.visualstudio.com/api/get-started/your-first-extension)

You have now successfully built a custom AI-powered assistant directly into Visual Studio Code, leveraging the code-generation strengths of the Phi-3.5 model family.
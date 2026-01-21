# Building a Copilot Application with Phi-3-mini and Semantic Kernel

This guide walks you through integrating the **Phi-3-mini** model into a .NET Copilot application using **Semantic Kernel**. You will learn how to connect to a locally hosted instance of Phi-3-mini, enabling greater autonomy and control, which is especially beneficial for enterprise applications.

## Prerequisites

Before you begin, ensure you have the following:

1. **.NET SDK** (version 6.0 or later) installed.
2. A running local instance of **Phi-3-mini**. You can deploy it using tools like [Ollama](https://ollama.com) or [LM Studio](https://llamaedge.com), or by writing your own server.
3. The **Semantic Kernel NuGet packages** for your project.

## Step 1: Set Up Your .NET Project

Create a new .NET console application and add the necessary Semantic Kernel packages.

```bash
dotnet new console -n Phi3MiniCopilot
cd Phi3MiniCopilot
dotnet add package Microsoft.SemanticKernel
```

## Step 2: Configure the Kernel to Connect to Your Local Server

Semantic Kernel uses a "Kernel" as the core orchestrator. You'll configure it to use a connector that points to your locally hosted Phi-3-mini service.

Create a `Program.cs` file and add the following code:

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

// Initialize the kernel builder
var builder = Kernel.CreateBuilder();

// Add your local Phi-3-mini chat completion service.
// Replace "http://localhost:11434" with your local server's endpoint.
builder.Services.AddOpenAIChatCompletion(
    modelId: "phi3:mini", // This label is for identification within SK
    apiKey: "not-needed-for-local", // API key is typically not required for local servers
    endpoint: new Uri("http://localhost:11434") // Your local Ollama/LM Studio endpoint
);

// Build the kernel
IKernel kernel = builder.Build();
```

**Explanation:**
- The `Kernel.CreateBuilder()` method sets up the foundation for your AI application.
- `AddOpenAIChatCompletion` is used because many local servers (like Ollama) are compatible with the OpenAI API schema. You provide your local endpoint and a placeholder API key.
- The `modelId` parameter is a friendly identifier used within Semantic Kernel; it doesn't need to match the remote model name exactly.

## Step 3: Create and Execute a Chat Interaction

Now, let's use the kernel to send a prompt to your local Phi-3-mini model and receive a response.

Add the following code to your `Program.cs` after the kernel is built:

```csharp
// Get the chat completion service from the kernel
IChatCompletionService chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

// Create a chat history and add a user message
ChatHistory chatHistory = new("You are a helpful AI assistant.");
chatHistory.AddUserMessage("What is the capital of France?");

// Get the response from the model
var reply = await chatCompletionService.GetChatMessageContentAsync(
    chatHistory,
    executionSettings: new OpenAIPromptExecutionSettings { MaxTokens = 500 }
);

// Output the model's response
Console.WriteLine($"Phi-3-mini: {reply.Content}");
```

**Explanation:**
- `IChatCompletionService` is the interface for interacting with chat models.
- `ChatHistory` maintains the conversation context. You initialize it with a system message to define the assistant's behavior.
- `GetChatMessageContentAsync` sends the chat history to the model and returns its response.
- `OpenAIPromptExecutionSettings` allows you to configure parameters like `MaxTokens` to control response length.

## Step 4: Run the Application

Execute your console application to see Phi-3-mini in action.

```bash
dotnet run
```

You should see output similar to the following:

```
Phi-3-mini: The capital of France is Paris.
```

## Step 5: Extend Your Application (Optional)

With the kernel configured, you can now leverage other Semantic Kernel features, such as **plugins** for reusable functions, **planners** for complex task orchestration, or **memories** for contextual recall. This modularity allows you to build sophisticated Copilot applications tailored to your needs.

## Summary

You have successfully connected a .NET Semantic Kernel application to a locally hosted Phi-3-mini model. This approach provides full control over your AI infrastructure, making it ideal for secure, enterprise-grade solutions.

For a complete sample project, including additional configurations and advanced scenarios, refer to the [Phi3MiniSamples repository](https://github.com/kinfey/Phi3MiniSamples/tree/main/semantickernel).
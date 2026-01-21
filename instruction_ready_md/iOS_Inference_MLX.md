# Building an On-Device AI Chat App with Phi Models and Apple MLX

This guide walks you through creating an iOS application that runs the Phi-3 or Phi-4 language model directly on an iPhone or iPad using Apple's MLX framework. You'll build a functional chat interface that communicates with the model entirely on-device, with no external API calls.

## Prerequisites

Before you begin, ensure you have the following:

*   **macOS** with **Xcode 16** or higher installed.
*   A physical **iOS device** (iPhone or iPad) running **iOS 18** or higher. The app requires a device with an Apple Silicon chip (M-series or A17 Pro and later) and at least 8GB of RAM to run the model effectively.
    *   **Important:** The iOS Simulator is **not supported** by the MLX framework.
*   Basic familiarity with **Swift** and **SwiftUI**.

## Step 1: Create a New iOS Project

1.  Launch Xcode and select **"Create a new Xcode project"**.
2.  Choose the **"App"** template and click **Next**.
3.  Enter a product name (e.g., `PhiChat`).
4.  Set the **Interface** to `SwiftUI` and the **Language** to `Swift`.
5.  Choose a location to save your project and click **Create**.

## Step 2: Add the MLX Swift Dependencies

Our app needs the MLX Swift libraries for tensor operations and model handling. We'll add the `mlx-swift-examples` package, which includes convenient utilities for loading language models.

1.  In Xcode, navigate to **File > Add Package Dependencies...**.
2.  In the search bar, enter the package repository URL: `https://github.com/ml-explore/mlx-swift-examples`
3.  Click **Add Package**. Xcode will fetch the package and its dependencies (including the core `mlx-swift` package).
4.  When prompted, add the `MLX`, `MLXLLM`, and `MLXLMCommon` libraries to your application target.

This package provides the tools to download models from Hugging Face, manage tokenization, and run inference.

## Step 3: Configure App Entitlements

To allow the app to download model files and use the necessary system resources, we must add specific entitlements.

1.  In your Xcode project navigator, select your app target and go to the **"Signing & Capabilities"** tab.
2.  Click the **"+ Capability"** button.
3.  Add the following capabilities:
    *   **App Sandbox**
    *   **Network (Client)**
4.  To request increased memory limits for running larger models, you need to add a custom entitlement file.
    1.  Right-click your project folder in the navigator and select **New File...**.
    2.  Choose **"Property List"** and name it `YourAppName.entitlements`.
    3.  Open the new `.entitlements` file and add the following keys as a dictionary:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.files.user-selected.read-only</key>
    <true/>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.developer.kernel.increased-memory-limit</key>
    <true/>
</dict>
</plist>
```

The `com.apple.developer.kernel.increased-memory-limit` entitlement is crucial for allowing the app to allocate the memory required by the language model.

## Step 4: Define the Data Model

First, let's create a simple data structure to represent messages in our chat interface. Create a new Swift file (e.g., `Models.swift`).

```swift
import SwiftUI

enum MessageState {
    case ok
    case waiting
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
    let state: MessageState
}
```

## Step 5: Build the ViewModel

The ViewModel is the core of our application. It manages the state, loads the AI model, and handles the conversation logic. Create a new Swift file named `PhiViewModel.swift`.

```swift
import MLX
import MLXLLM
import MLXLMCommon
import SwiftUI

@MainActor
class PhiViewModel: ObservableObject {
    // Published properties for the UI to observe
    @Published var isLoading: Bool = false
    @Published var isLoadingEngine: Bool = false
    @Published var messages: [ChatMessage] = []
    @Published var prompt: String = ""
    @Published var isReady: Bool = false
    
    private let maxTokens = 1024 // Limit response length
    private var modelContainer: ModelContainer? // Holds the loaded model
    
    func loadModel() async {
        // Update UI to show loading state
        DispatchQueue.main.async {
            self.isLoadingEngine = true
        }
        
        do {
            // Configure MLX to optimize GPU memory usage for mobile
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024) // 20 MB cache
            
            // 1. Choose your model configuration
            // Option A: Use the pre-configured Phi-3.5 Mini (4-bit quantized)
            let modelConfig = ModelRegistry.phi3_5_4bit
            
            // Option B: To use Phi-4 Mini, you would define a custom configuration.
            // See the note on model formats below.
            /*
            let phi4Config = ModelConfiguration(
                id: "mlx-community/Phi-4-mini-instruct-4bit",
                defaultPrompt: "You are a helpful assistant.",
                extraEOSTokens: ["<|end|>"]
            )
            */
            
            print("Loading \(modelConfig.name)...")
            
            // 2. Download and load the model using the LLMModelFactory
            self.modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfig
            ) { progress in
                // Optional: Log download progress
                print("Download progress: \(Int(progress.fractionCompleted * 100))%")
            }
            
            // Log model info upon successful load
            if let container = self.modelContainer {
                let numParams = await container.perform { context in
                    context.model.numParameters()
                }
                print("Model loaded. Parameters: \(numParams / (1024*1024))M")
            }
            
            // Update UI state
            DispatchQueue.main.async {
                self.isLoadingEngine = false
                self.isReady = true
            }
        } catch {
            print("Failed to load model: \(error)")
            DispatchQueue.main.async {
                self.isLoadingEngine = false
            }
        }
    }
    
    func fetchAIResponse() async {
        // Guard against invalid states
        guard !isLoading, let container = self.modelContainer else {
            print("Cannot generate: model not loaded or already processing")
            return
        }
        
        let userQuestion = prompt
        let currentMessages = self.messages
        
        // 1. Update UI: Add user message and a placeholder for the AI response
        DispatchQueue.main.async {
            self.isLoading = true
            self.prompt = "" // Clear the input field
            self.messages.append(ChatMessage(text: userQuestion, isUser: true, state: .ok))
            self.messages.append(ChatMessage(text: "", isUser: false, state: .waiting))
        }
        
        do {
            // 2. Perform inference within the model's context
            let _ = try await container.perform { context in
                // 3. Format the conversation history for the model
                var messageHistory: [[String: String]] = [
                    ["role": "system", "content": "You are a helpful assistant."]
                ]
                
                for message in currentMessages {
                    let role = message.isUser ? "user" : "assistant"
                    messageHistory.append(["role": role, "content": message.text])
                }
                messageHistory.append(["role": "user", "content": userQuestion])
                
                // 4. Prepare the tokenized input
                let input = try await context.processor.prepare(
                    input: .init(messages: messageHistory))
                let startTime = Date()
                
                // 5. Generate the response token-by-token (streaming)
                let result = try MLXLMCommon.generate(
                    input: input,
                    parameters: GenerateParameters(temperature: 0.6), // Controls creativity
                    context: context
                ) { tokens in
                    // This closure is called as each new token is generated
                    let output = context.tokenizer.decode(tokens: tokens)
                    
                    // Stream the partial response back to the UI
                    Task { @MainActor in
                        if let index = self.messages.lastIndex(where: { !$0.isUser }) {
                            self.messages[index] = ChatMessage(
                                text: output,
                                isUser: false,
                                state: .ok
                            )
                        }
                    }
                    
                    // Stop if we reach the token limit
                    if tokens.count >= self.maxTokens {
                        return .stop
                    } else {
                        return .more
                    }
                }
                
                // 6. Finalize and log the result
                let finalOutput = context.tokenizer.decode(tokens: result.tokens)
                Task { @MainActor in
                    if let index = self.messages.lastIndex(where: { !$0.isUser }) {
                        self.messages[index] = ChatMessage(
                            text: finalOutput,
                            isUser: false,
                            state: .ok
                        )
                    }
                    self.isLoading = false
                    
                    // Performance logging
                    print("Inference complete:")
                    print("Tokens: \(result.tokens.count)")
                    print("Tokens/second: \(result.tokensPerSecond)")
                    print("Time: \(Date().timeIntervalSince(startTime))s")
                }
                
                return result
            }
        } catch {
            print("Inference error: \(error)")
            DispatchQueue.main.async {
                // Show error in the chat
                if let index = self.messages.lastIndex(where: { !$0.isUser }) {
                    self.messages[index] = ChatMessage(
                        text: "Sorry, an error occurred: \(error.localizedDescription)",
                        isUser: false,
                        state: .ok
                    )
                }
                self.isLoading = false
            }
        }
    }
}
```

### Key ViewModel Concepts:

*   **Model Loading:** The `loadModel()` function downloads a pre-converted, quantized model from Hugging Face via `LLMModelFactory`. The `ModelRegistry.phi3_5_4bit` points to a community-maintained Phi-3.5 model.
*   **Inference:** The `fetchAIResponse()` function manages the chat history, tokenizes the input, and uses `MLXLMCommon.generate` to produce a response. The `temperature` parameter (set to 0.6) influences the randomness of the output.
*   **Streaming:** The generation happens token-by-token within a closure, allowing the UI to update in real-time as the AI "types" its response.

### Note on Model Formats and Phi-4

**Important:** Standard model formats (like GGUF) are not compatible with MLX. You must use models specifically converted to the MLX format. The `mlx-swift-examples` package references pre-converted models on Hugging Face (e.g., `mlx-community/Phi-3.5-mini-instruct-4bit`).

To use **Phi-4**, you need to reference the `mlx-swift-examples` package directly from its `main` branch, as official releases may not yet include its configuration. In Xcode's package dependencies, use the URL `https://github.com/ml-explore/mlx-swift-examples.git` and select the **"Branch"** option with `main`. You can then define a custom `ModelConfiguration` as shown in the commented code above.

## Step 6: Create the User Interface

Now, let's build the chat interface. Replace the contents of your `ContentView.swift` file with the following code.

```swift
import SwiftUI

struct ContentView: View {
    @StateObject var viewModel = PhiViewModel()

    var body: some View {
        NavigationStack {
            // Show loading button until the model is ready
            if !viewModel.isReady {
                VStack {
                    Spacer()
                    if viewModel.isLoadingEngine {
                        ProgressView("Loading Model...")
                    } else {
                        Button("Load AI Model") {
                            Task {
                                await viewModel.loadModel()
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .padding()
                    }
                    Spacer()
                }
                .navigationTitle("Phi Chat")
            } else {
                // Main chat interface
                VStack(spacing: 0) {
                    // Message List
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 12) {
                                ForEach(viewModel.messages) { message in
                                    MessageView(message: message)
                                }
                            }
                            .padding()
                        }
                        .onChange(of: viewModel.messages.count) { _ in
                            // Auto-scroll to the latest message
                            if let lastMessage = viewModel.messages.last {
                                withAnimation {
                                    proxy.scrollTo(lastMessage.id, anchor: .bottom)
                                }
                            }
                        }
                    }
                    
                    // Input Area
                    HStack {
                        TextField("Ask a question...", text: $viewModel.prompt)
                            .textFieldStyle(.roundedBorder)
                            .padding(.leading)
                            .disabled(viewModel.isLoading) // Disable while generating
                        
                        Button(action: {
                            Task {
                                await viewModel.fetchAIResponse()
                            }
                        }) {
                            Image(systemName: "paperplane.fill")
                                .font(.title2)
                                .foregroundColor(.blue)
                                .padding(.trailing)
                        }
                        .disabled(viewModel.prompt.isEmpty || viewModel.isLoading)
                    }
                    .padding(.vertical)
                    .background(.ultraThinMaterial)
                }
                .navigationTitle("Chat with Phi")
            }
        }
    }
}

// View for a single chat bubble
struct MessageView: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
                Text(message.text)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            } else {
                // AI Message
                if message.state == .waiting {
                    TypingIndicatorView() // Shows "typing" animation
                } else {
                    Text(message.text)
                        .padding()
                        .background(Color.gray.opacity(0.2))
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    Spacer()
                }
            }
        }
        .padding(.horizontal)
    }
}

// Animated "typing" indicator
struct TypingIndicatorView: View {
    @State private var shouldAnimate = false

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { index in
                Circle()
                    .frame(width: 8, height: 8)
                    .foregroundColor(.gray)
                    .offset(y: shouldAnimate ? -4 : 0)
                    .animation(
                        Animation.easeInOut(duration: 0.5)
                            .repeatForever()
                            .delay(Double(index) * 0.2),
                        value: shouldAnimate
                    )
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color.gray.opacity(0.2))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .onAppear { shouldAnimate = true }
        .onDisappear { shouldAnimate = false }
    }
}
```

## Step 7: Build and Run on Device

1.  Connect your compatible iPhone or iPad to your Mac.
2.  In Xcode's top toolbar, select your connected device as the run destination.
3.  Press **Cmd + R** to build and run the application.
4.  On your device, tap **"Load AI Model"**. The first time, this will download the model weights (several GBs) from Hugging Face. You can monitor the progress in Xcode's console.
5.  Once loaded, type a question in the text field and tap the send button. You should see the AI generate a response token-by-token.

## Conclusion

You've successfully built an iOS application that runs a state-of-the-art language model (Phi-3.5 or Phi-4) entirely on-device using Apple's MLX framework. This approach offers privacy, offline functionality, and low-latency interactions.

**Next Steps to Explore:**
*   Experiment with the `temperature` parameter in the `GenerateParameters` to adjust response creativity.
*   Implement a local cache for the downloaded model to avoid re-downloading on each app launch.
*   Explore other MLX-converted models available in the `mlx-community` on Hugging Face.
*   Enhance the UI with features like conversation history, copy/paste, or different AI personas.
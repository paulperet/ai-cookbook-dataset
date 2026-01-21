# Deploying Phi-3-mini on iOS with ONNX Runtime

This guide walks you through the process of deploying Microsoft's Phi-3-mini model on an iOS device using ONNX Runtime. You'll compile the necessary libraries, integrate them into an Xcode project, and run inference using the model's C++ API.

## Prerequisites

Before you begin, ensure your development environment meets these requirements:

*   **Hardware & OS:**
    *   macOS 14 or later.
    *   iPhone 14 (A16 chip) or later running iOS 17.x.
*   **Development Tools:**
    *   Xcode 15 or later.
    *   iOS SDK 17.x.
*   **Software & Libraries:**
    *   Python 3.10+ (Conda is recommended for environment management).
    *   The `flatbuffers` Python library.
    *   CMake.

You can install the required Python library using pip:
```bash
pip install flatbuffers
```

## Step 1: Understand Your Deployment Options

You can access local models like Phi-3-mini through frameworks like **Semantic Kernel**, which simplifies integration. For running quantized models locally, tools like **Ollama** and **LlamaEdge** are popular choices.

*   **Ollama:** Run `ollama run phi3` directly or configure it offline using a `Modelfile` that points to your `.gguf` model file.
*   **LlamaEdge:** A suitable option if you need to use `.gguf` models across both cloud and edge devices.

This tutorial focuses on using the **ONNX Runtime** directly for maximum control and performance on iOS.

## Step 2: Compile ONNX Runtime for iOS

First, you need to compile the core ONNX Runtime library for the iOS platform.

1.  Clone the ONNX Runtime repository:
    ```bash
    git clone https://github.com/microsoft/onnxruntime.git
    cd onnxruntime
    ```

2.  **Configure Xcode:** Before compiling, set Xcode as the active developer directory in your terminal:
    ```bash
    sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer
    ```

3.  Execute the build script. This command compiles a shared library for iOS (`arm64` architecture) targeting iOS 17.5.
    ```bash
    ./build.sh --build_shared_lib --ios --skip_tests --parallel \
               --build_dir ./build_ios \
               --ios --apple_sysroot iphoneos --osx_arch arm64 \
               --apple_deploy_target 17.5 \
               --cmake_generator Xcode \
               --config Release
    ```
    **Note:** Adjust the `--apple_deploy_target` value if you need compatibility with older iOS versions. The build process may take several minutes.

4.  Return to your parent directory:
    ```bash
    cd ../
    ```

## Step 3: Compile ONNX Runtime Generative AI for iOS

The `onnxruntime-genai` extension provides the APIs needed for generative AI tasks like text generation with Phi-3.

1.  Clone the Generative AI extension repository:
    ```bash
    git clone https://github.com/microsoft/onnxruntime-genai
    cd onnxruntime-genai
    ```

2.  Prepare a directory structure for the compiled ONNX Runtime files:
    ```bash
    mkdir ort
    cd ort
    mkdir include
    mkdir lib
    cd ../
    ```

3.  Copy the necessary header file and compiled dynamic library from the ONNX Runtime build you just completed:
    ```bash
    cp ../onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h ort/include/
    cp ../onnxruntime/build_ios/Release/Release-iphoneos/libonnxruntime*.dylib* ort/lib/
    ```

4.  Set an environment variable to avoid a potential build issue with OpenCV in Xcode:
    ```bash
    export OPENCV_SKIP_XCODEBUILD_FORCE_TRYCOMPILE_DEBUG=1
    ```

5.  Build the `onnxruntime-genai` library for iOS:
    ```bash
    python3 build.py --parallel --build_dir ./build_ios \
                     --ios --ios_sysroot iphoneos --ios_arch arm64 \
                     --ios_deployment_target 17.5 \
                     --cmake_generator Xcode \
                     --cmake_extra_defines CMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
    ```

## Step 4: Set Up the Xcode Project

1.  Create a new iOS App project in Xcode.
2.  **Language Choice:** While you can use Swift, this guide uses **Objective-C** for its straightforward compatibility with the C++ APIs from `onnxruntime-genai`. If you use Swift, you will need to create a bridging header.
3.  **Add the Model:** Download the INT4 quantized ONNX format model of Phi-3-mini (e.g., from Hugging Face). In Xcode, add this model file to your project's **Resources** directory. This ensures it's bundled with your application.

## Step 5: Integrate Libraries and Headers into Xcode

For your app to use the compiled libraries, you must configure the Xcode project.

1.  **Add Header Files:** In your project's **Build Settings**, add the path to the `ort/include/` directory (containing `onnxruntime_c_api.h`) and the `onnxruntime-genai/build_ios/Release-iphoneos/include/` directory to **Header Search Paths**.
2.  **Link the Dynamic Libraries:**
    *   Go to your target's **Build Phases**.
    *   In "Link Binary With Libraries", click `+` and choose **Add Other...**.
    *   Navigate to and add the `libonnxruntime.dylib` and `libonnxruntime_genai.dylib` files from your `ort/lib/` and `onnxruntime-genai/build_ios/Release-iphoneos/` directories, respectively.
3.  **Copy Libraries to Bundle:** In the same **Build Phases** tab, add a **New Copy Files Phase**. Set the destination to "Frameworks" and add the two `.dylib` files mentioned above. This ensures they are packaged with your app.
4.  **Enable Objective-C++:** Since you will be using C++ code, rename your main view controller file from `ViewController.m` to `ViewController.mm`. This tells Xcode to compile it as Objective-C++.

## Step 6: Implement Model Inference in Code

Now, you can write the code to load the model and generate a response. The following example should be placed in your `ViewController.mm` file.

This code performs the following steps:
1.  Locates the model file within the app bundle.
2.  Creates a model and tokenizer instance.
3.  Encodes a prompt using the model's specific chat template.
4.  Sets up generator parameters (like maximum output length).
5.  Runs the model to generate a sequence of tokens.
6.  Decodes the tokens back into a readable string.

```objc
// Get the path to the model file included in the app bundle
NSString *modelDirectoryPath = [[NSBundle mainBundle] resourcePath];
const char *modelPath = [modelDirectoryPath cStringUsingEncoding:NSUTF8StringEncoding];

// Initialize the model and tokenizer
auto model = OgaModel::Create(modelPath);
auto tokenizer = OgaTokenizer::Create(*model);

// Define the prompt using the Phi-3 chat template
const char* prompt = "<|system|>You are a helpful AI assistant.<|end|><|user|>Can you introduce yourself?<|end|><|assistant|>";

// Prepare the input sequences
auto sequences = OgaSequences::Create();
tokenizer->Encode(prompt, *sequences);

// Configure the text generation parameters
auto params = OgaGeneratorParams::Create(*model);
params->SetSearchOption("max_length", 100); // Limit the response length
params->SetInputSequences(*sequences);

// Generate the output sequence
auto output_sequences = model->Generate(*params);

// Decode the generated tokens into text
const auto output_sequence_length = output_sequences->SequenceCount(0);
const auto* output_sequence_data = output_sequences->SequenceData(0);
auto out_string = tokenizer->Decode(output_sequence_data, output_sequence_length);

// 'out_string' now contains the model's response
// You can now display this string in your UI (e.g., a UILabel)
```

## Step 7: Build and Run the Application

1.  Connect your iOS device (iPhone 14 or later with iOS 17.x).
2.  Select your device as the run target in Xcode.
3.  Click the **Run** button.
4.  If everything is configured correctly, the app will install on your device, load the Phi-3-mini model, and generate a response to the prompt "Can you introduce yourself?". The output will be the model's introductory response.

## Next Steps and Resources

*   For more advanced features like a chat interface or different model parameters, explore the sample code in the official [Phi-3 Mini Samples repository](https://github.com/Azure-Samples/Phi-3MiniSamples/tree/main/ios).
*   Refer to the [ONNX Runtime Generative AI documentation](https://github.com/microsoft/onnxruntime-genai) for detailed API references and advanced configuration options.
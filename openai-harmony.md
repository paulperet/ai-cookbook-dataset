# OpenAI harmony response format

The [`gpt-oss` models](https://openai.com/open-models) were trained on the harmony response format for defining conversation structures, generating reasoning output and structuring function calls. If you are not using `gpt-oss` directly but through an API or a provider like Ollama, you will not have to be concerned about this as your inference solution will handle the formatting. If you are building your own inference solution, this guide will walk you through the prompt format. The format is designed to mimic the OpenAI Responses API, so if you have used that API before, this format should hopefully feel familiar to you. `gpt-oss` should not be used without using the harmony format, as it will not work correctly.

## Concepts

### Roles

Every message that the model processes has a role associated with it. The model knows about five types of roles:

| Role        | Purpose                                                                                                                                                                                 |
| :---------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `system`    | A system message is used to specify reasoning effort, meta information like knowledge cutoff and built-in tools                                                                         |
| `developer` | The developer message is used to provide information about the instructions for the model (what is normally considered the “system prompt”) and available function tools                |
| `user`      | Typically representing the input to the model                                                                                                                                           |
| `assistant` | Output by the model which can either be a tool call or a message output. The output might also be associated with a particular “channel” identifying what the intent of the message is. |
| `tool`      | Messages representing the output of a tool call. The specific tool name will be used as the role inside a message.                                                                      |

These roles also represent the information hierarchy that the model applies in case there are any instruction conflicts: `system` \> `developer` \> `user` \> `assistant` \> `tool`

#### Channels

Assistant messages can be output in three different “channels”. These are being used to separate between user-facing responses and internal facing messages.

| Channel      | Purpose                                                                                                                                                                                                                                                                                                                                                            |
| :----------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `final`      | Messages tagged in the final channel are messages intended to be shown to the end-user and represent the responses from the model.                                                                                                                                                                                                                                 |
| `analysis`   | These are messages that are being used by the model for its chain of thought (CoT). **Important:** Messages in the analysis channel do not adhere to the same safety standards as final messages do. Avoid showing these to end-users.                                                                                                                             |
| `commentary` | Any function tool call will typically be triggered on the `commentary` channel while built-in tools will normally be triggered on the `analysis` channel. However, occasionally built-in tools will still be output to `commentary`. Occasionally this channel might also be used by the model to generate a [preamble](#preambles) to calling multiple functions. |

## Harmony renderer library

We recommend using our harmony renderer through [PyPI](https://pypi.org/project/openai-harmony/) or [crates.io](https://crates.io/crates/openai-harmony) when possible as it will automatically handle rendering your messages in the right format and turning them into tokens for processing by the model.

Below is an example of using the renderer to construct a system prompt and a short conversation.

```py
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

system_message = (
    SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_conversation_start_date("2025-06-28")
)

developer_message = (
    DeveloperContent.new()
        .with_instructions("Always respond in riddles")
        .with_function_tools(
            [
                ToolDescription.new(
                    "get_current_weather",
                    "Gets the current weather in the provided location.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["location"],
                    },
                ),
            ]
	)
)

convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, "What is the weather in Tokyo?"),
        Message.from_role_and_content(
            Role.ASSISTANT,
            'User asks: "What is the weather in Tokyo?" We need to use get_current_weather tool.',
        ).with_channel("analysis"),
        Message.from_role_and_content(Role.ASSISTANT, '{"location": "Tokyo"}')
        .with_channel("commentary")
        .with_recipient("functions.get_current_weather")
        .with_content_type("<|constrain|> json"),
        Message.from_author_and_content(
            Author.new(Role.TOOL, "functions.get_current_weather"),
            '{ "temperature": 20, "sunny": true }',
        ).with_channel("commentary"),
    ]
)

tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

# After receiving a token response
# Do not pass in the stop token
parsed_response = encoding.parse_messages_from_completion_tokens(new_tokens, Role.ASSISTANT)
```

Additionally the openai_harmony library also includes a StreamableParser for parsing and decoding as the model is generating new tokens. This can be helpful for example to stream output and handle unicode characters during decoding.

```py
from openai_harmony import (
    load_harmony_encoding,
    Role,
    StreamableParser,
    HarmonyEncodingName
)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
stream = StreamableParser(encoding, role=Role.ASSISTANT)

tokens = [
    200005,35644,200008,1844,31064,25,392,4827,382,220,17,659,220,17,16842,12295,81645,
    13,51441,6052,13,200007,200006,173781,200005,17196,200008,17,659,220,17,314,220,19,
    13,200002
]

for token in tokens:
    stream.process(token)
    print("--------------------------------")
    print("current_role", stream.current_role)
    print("current_channel", stream.current_channel)
    print("last_content_delta", stream.last_content_delta)
    print("current_content_type", stream.current_content_type)
    print("current_recipient", stream.current_recipient)
    print("current_content", stream.current_content)
```

## Prompt format

If you choose to build your own renderer, you’ll need to adhere to the following format.

### Special Tokens

The model uses a set of special tokens to identify the structure of your input. If you are using [tiktoken](https://github.com/openai/tiktoken) these tokens are encoded in the `o200k_harmony` encoding. All special tokens follow the format `<|type|>`.

| Special token           | Purpose                                                                                                                                     | Token ID |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :------- |
| <&#124;start&#124;>     | Indicates the beginning of a [message](#message-format). Followed by the “header” information of a message starting with the [role](#roles) | `200006` |
| <&#124;end&#124;>       | Indicates the end of a [message](#message-format)                                                                                           | `200007` |
| <&#124;message&#124;>   | Indicates the transition from the message “header” to the actual content                                                                    | `200008` |
| <&#124;channel&#124;>   | Indicates the transition to the [channel](#channels) information of the header                                                              | `200005` |
| <&#124;constrain&#124;> | Indicates the transition to the data type definition in a [tool call](#receiving-tool-calls)                                                | `200003` |
| <&#124;return&#124;>    | Indicates the model is done with sampling the response message. A valid “stop token” indicating that you should stop inference.             | `200002` |
| <&#124;call&#124;>      | Indicates the model wants to call a tool. A valid “stop token” indicating that you should stop inference.                                   | `200012` |

### Message format

The harmony response format consists of “messages” with the model potentially generating multiple messages in one go. The general structure of a message is as follows:

```
<|start|>{header}<|message|>{content}<|end|>
```

The `{header}` contains a series of meta information including the [role](#roles). `<|end|>` represents the end of a fully completed message but the model might also use other stop tokens such as `<|call|>` for tool calling and `<|return|>` to indicate the model is done with the completion.

### Chat conversation format

Following the message format above the most basic chat format consists of a `user` message and the beginning of an `assistant` message.

#### Example input

```
<|start|>user<|message|>What is 2 + 2?<|end|>
<|start|>assistant
```

The output will begin by specifying the `channel`. For example `analysis` to output the chain of thought. The model might output multiple messages (primarily chain of thought messages) for which it uses the `<|end|>` token to separate them.

Once its done generating it will stop with either a `<|return|>` token indicating it’s done generating the final answer, or `<|call|>` indicating that a tool call needs to be performed. In either way this indicates that you should stop inference.

#### Example output

```
<|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
```

The `final` channel will contain the answer to your user’s request. Check out the [reasoning section](#reasoning) for more details on the chain-of-thought.

**Implementation note:** `<|return|>` is a decode-time stop token only. When you add the assistant’s generated reply to conversation history for the next turn, replace the trailing `<|return|>` with `<|end|>` so that stored messages are fully formed as `<|start|>{header}<|message|>{content}<|end|>`. Prior messages in prompts should therefore end with `<|end|>`. For supervised targets/training examples, ending with `<|return|>` is appropriate; for persisted history, normalize to `<|end|>`.

### System message format

The system message is used to provide general information to the system. This is different to what might be considered the “system prompt” in other prompt formats. For that, check out the [developer message format](#developer-message-format).

We use the system message to define:

1. The **identity** of the model — This should always stay as `You are ChatGPT, a large language model trained by OpenAI.` If you want to change the identity of the model, use the instructions in the [developer message](#developer-message-format).
2. Meta **dates** — Specifically the `Knowledge cutoff:` and the `Current date:`
3. The **reasoning effort** — As specified on the levels `high`, `medium`, `low`
4. Available channels — For the best performance this should map to `analysis`, `commentary`, and `final`.
5. Built-in tools — The model has been trained on both a `python` and `browser` tool. Check out the [built-in tools section](#built-in-tools) for details.

**If you are defining functions,** it should also contain a note that all function tool calls must go to the `commentary` channel.

For the best performance stick to this format as closely as possible.

#### Example system message

The most basic system message you should use is the following:

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
```

If functions calls are present in the developer message section, use:

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|>
```

### Developer message format

The developer message represents what is commonly considered the “system prompt”. It contains the instructions that are provided to the model and optionally a list of [function tools](#function-calling) available for use or the output format you want the model to adhere to for [structured outputs](#structured-output).

If you are not using function tool calling your developer message would just look like this:

```
<|start|>developer<|message|># Instructions

{instructions}<|end|>
```

Where `{instructions}` is replaced with your “system prompt”.

For defining function calling tools, [check out the dedicated section](#function-calling).  
For defining an output format to be used in structured outputs, [check out this section of the guide](#structured-output).

### Reasoning

The gpt-oss models are reasoning models. By default, the model will do medium level reasoning. To control the reasoning you can specify in the [system message](#system-message-format) the reasoning level as `low`, `medium`, or `high`. The recommended format is:

```
Reasoning: high
```

The model will output its raw chain-of-thought (CoT) as assistant messages into the `analysis` channel while the final response will be output as `final`.

For example for the question `What is 2 + 2?` the model output might look like this:

```
<|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
```

In this case the CoT is

```
User asks: “What is 2 + 2?” Simple arithmetic. Provide answer.
```

And the actual answer is:

```
2 + 2 = 4
```

**Important:**  
The model has not been trained to the same safety standards in the chain-of-thought as it has for final output. You should not show the chain-of-thought to your users, as they might contain harmful content. [Learn more in the model card](https://openai.com/index/gpt-oss-model-card/).

#### Handling reasoning output in subsequent sampling

In general, you should drop any previous CoT content on subsequent sampling if the responses by the assistant ended in a message to the `final` channel. Meaning if our first input was this:

```
<|start|>user<|message|>What is 2 + 2?<|end|>
<|start|>assistant
```

and resulted in the output:

```
<|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
```

For the model to work properly, the input for the next sampling should be

```
<|start|>user<|message|>What is 2 + 2?<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|end|>
<|start|>user<|message|>What about 9 / 2?<|end|>
<|start|>assistant
```

The exception for this is tool/function calling. The model is able to call tools as part of its chain-of-thought and because of that, we should pass the previous chain-of-thought back in as input for subsequent sampling. Check out the [function calling section](#function-calling) for a complete example.

### Function calling

#### Defining available tools

All functions that are available to the model should be defined in the [developer message](#developer-message-format) in a dedicated `Tools` section.

To define the functions we use a TypeScript-like type syntax and wrap the functions into a dedicated `functions` namespace. It’s important to stick to this format closely to improve accuracy of function calling. You can check out the harmony renderer codebase for more information on how we are turning JSON schema definitions for the arguments into this format but some general formatting practices:

- Define every function as a `type {function_name} = () => any` if it does not receive any arguments
- For functions that receive an argument name the argument `_` and inline the type definition
- Add comments for descriptions in the line above the field definition
- Always use `any` as the return type
- Keep an empty line after each function definition
- Wrap your functions into a namespace, generally `functions` is the namespace you should use to not conflict with [other tools](#built-in-tools) that the model might have been trained on.

Here’s a complete input example including the definition of two functions:

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>developer<|message|># Instructions

Use a friendly tone.

# Tools

## functions

namespace functions {

// Gets the location of the user.
type get_location = () => any;

// Gets the current weather in the provided
# Gemini API: Function Calling Configuration with REST

This guide demonstrates how to use the `function_calling_config` parameter to precisely control how the Gemini API interacts with tools you provide. You can use this configuration to restrict the model to text-only responses, allow it to decide automatically, or force it to call a specific set of functions.

## Prerequisites

Before you begin, ensure you have:
1.  A Gemini API key.
2.  The key stored in an environment variable named `GOOGLE_API_KEY`.

If you are using Google Colab, you can set it up using a secret:

```python
import os
from google.colab import userdata

os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Step 1: Define Your Tools

First, you need to define the functions (tools) the model can call. This example uses a simple lighting system with three functions. Save this definition to a file named `tools.json`.

```bash
cat > tools.json << 'EOF'
{
  "function_declarations": [
    {
      "name": "enable_lights",
      "description": "Turn on the lighting system.",
      "parameters": { "type": "object" }
    },
    {
      "name": "set_light_color",
      "description": "Set the light color. Lights must be enabled for this to work.",
      "parameters": {
        "type": "object",
        "properties": {
          "rgb_hex": {
            "type": "string",
            "description": "The light color as a 6-digit hex string, e.g. ff0000 for red."
          }
        },
        "required": [
          "rgb_hex"
        ]
      }
    },
    {
      "name": "stop_lights",
      "description": "Turn off the lighting system.",
      "parameters": { "type": "object" }
    }
  ]
}
EOF
```

## Step 2: Configure the Model and System Instruction

Set the model ID you want to use. This guide uses `gemini-3-flash-preview`, but you can choose from other available models.

```bash
MODEL_ID="gemini-3-flash-preview"
```

You will also provide a system instruction to define the model's role and capabilities for this conversation.

## Step 3: Use `function_calling_config` Modes

The `function_calling_config` object has a `mode` property that controls tool usage. Let's explore the three primary modes.

### Mode 1: `NONE` (Text-Only Response)

Use `"mode": "none"` when you have provided tools but want the model to ignore them for a specific turn and respond only with text.

**Example:** Ask the model what it can do. With the `NONE` mode, it will describe its capabilities instead of attempting to call a function.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d @<(echo '
  {
    "system_instruction": {
      "parts": {
        "text": "You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks."
      }
    },
    "tools": [' $(cat tools.json) '],
    "tool_config": {
      "function_calling_config": {"mode": "none"}
    },
    "contents": {
      "role": "user",
      "parts": {
        "text": "What can you do?"
      }
    }
  }
') 2>/dev/null | sed -n '/"content"/,/"finishReason"/p'
```

**Expected Response:**
The model responds with a text description of its capabilities.
```json
"content": {
  "parts": [
    {
      "text": "As your lighting system, I can turn the lights on and off, and I can set the color of the lights. \n"
    }
  ],
  "role": "model"
},
"finishReason": "STOP",
```

### Mode 2: `AUTO` (Let the Model Decide)

Use `"mode": "auto"` to allow the model to choose the best response—either text or a function call—based on the user's prompt and the available tools.

**Example:** Prompt the model to "Light this place up!". In `AUTO` mode, it will likely decide to call the `enable_lights` function.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d @<(echo '
  {
    "system_instruction": {
      "parts": {
        "text": "You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks."
      }
    },
    "tools": [' $(cat tools.json) '],
    "tool_config": {
      "function_calling_config": {"mode": "auto"}
    },
    "contents": {
      "role": "user",
      "parts": {
        "text": "Light this place up!"
      }
    }
  }
') 2>/dev/null | sed -n '/"content"/,/"finishReason"/p'
```

**Expected Response:**
The model chooses to make a function call.
```json
"content": {
  "parts": [
    {
      "functionCall": {
        "name": "enable_lights",
        "args": {}
      }
    }
  ],
  "role": "model"
},
"finishReason": "STOP",
```

### Mode 3: `ANY` (Force a Function Call)

Use `"mode": "any"` to force the model to make a function call. You can optionally restrict it to a subset of your tools using the `allowed_function_names` array.

**Example:** Force the model to call either `set_light_color` or `stop_lights` in response to a color change request. It cannot choose `enable_lights` or respond with text.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d @<(echo '
  {
    "system_instruction": {
      "parts": {
        "text": "You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks."
      }
    },
    "tools": [' $(cat tools.json) '],
    "tool_config": {
      "function_calling_config": {
        "mode": "any",
        "allowed_function_names": ["set_light_color", "stop_lights"]
      }
    },
    "contents": {
      "role": "user",
      "parts": {
        "text": "Make this place PURPLE!"
      }
    }
  }
') 2>/dev/null | sed -n '/"content"/,/"finishReason"/p'
```

**Expected Response:**
The model is forced to call `set_light_color` with an appropriate hex value.
```json
"content": {
  "parts": [
    {
      "functionCall": {
        "name": "set_light_color",
        "args": {
          "rgb_hex": "9400d3"
        }
      }
    }
  ],
  "role": "model"
},
"finishReason": "STOP",
```

## Summary

You have now learned how to use the `function_calling_config` parameter to control tool interaction with the Gemini API:
- **`NONE`**: Temporarily disable function calling for a text-only response.
- **`AUTO`**: Let the model decide the best course of action.
- **`ANY`**: Force a function call, optionally restricting the available functions.

This configuration provides precise control over multi-step workflows, ensuring the model calls functions in the correct order or context.

## Next Steps

For a more comprehensive introduction to function calling with the Gemini API, refer to the [Function calling with REST](./Function_calling_REST.ipynb) guide.
# Building a Note-Saving Tool with Pydantic and Anthropic Tool Use

In this guide, you'll build a structured note-saving tool using Anthropic's Claude and Pydantic. You'll define a validated schema for notes, create a tool for Claude to call, and process the tool's response to ensure type safety and reliability.

## Prerequisites

Ensure you have the required libraries installed.

```bash
pip install anthropic pydantic 'pydantic[email]'
```

## Step 1: Import Libraries and Initialize the Client

Begin by importing the necessary modules and setting up the Anthropic client.

```python
from anthropic import Anthropic
from pydantic import BaseModel, EmailStr, Field

client = Anthropic()
MODEL_NAME = "claude-3-opus-20240229"  # Or your preferred model
```

## Step 2: Define Pydantic Models for Validation

Define the data structures for authors, notes, and the tool's response. These models will be used to validate inputs and outputs.

```python
class Author(BaseModel):
    name: str
    email: EmailStr


class Note(BaseModel):
    note: str
    author: Author
    tags: list[str] | None = None
    priority: int = Field(ge=1, le=5, default=3)
    is_public: bool = False


class SaveNoteResponse(BaseModel):
    success: bool
    message: str
```

## Step 3: Define the Tool Schema for Claude

Create the tool definition that Claude will use. This schema describes the `save_note` tool's expected input.

```python
tools = [
    {
        "name": "save_note",
        "description": "A tool that saves a note with the author and metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "The content of the note to be saved."},
                "author": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the author."},
                        "email": {
                            "type": "string",
                            "format": "email",
                            "description": "The email address of the author.",
                        },
                    },
                    "required": ["name", "email"],
                },
                "priority": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 3,
                    "description": "The priority level of the note (1-5).",
                },
                "is_public": {
                    "type": "boolean",
                    "default": False,
                    "description": "Indicates whether the note is publicly accessible.",
                },
            },
            "required": ["note", "author"],
        },
    }
]
```

## Step 4: Implement the Note-Saving Function

Create a placeholder function that simulates saving a note. In a production environment, you would replace this with actual database or file system logic.

```python
def save_note(note: str, author: dict, priority: int = 3, is_public: bool = False) -> None:
    # In a real application, you would save the note to a database or file here.
    print("Note saved successfully!")
```

## Step 5: Process Tool Calls and Generate Responses

Build the functions that handle Claude's tool calls, validate the input using Pydantic, execute the save operation, and format the response.

```python
def process_tool_call(tool_name, tool_input):
    if tool_name == "save_note":
        # Validate and parse the tool input using the Pydantic Note model
        note = Note(
            note=tool_input["note"],
            author=Author(
                name=tool_input["author"]["name"],
                email=tool_input["author"]["email"]
            ),
            priority=tool_input.get("priority", 3),
            is_public=tool_input.get("is_public", False),
        )
        # Call the save function with the validated data
        save_note(note.note, note.author.model_dump(), note.priority, note.is_public)
        # Return a validated response
        return SaveNoteResponse(success=True, message="Note saved successfully!")


def generate_response(save_note_response):
    return f"Response: {save_note_response.message}"
```

## Step 6: Create the Chatbot Interaction Loop

Implement the main interaction function that sends a user message to Claude, processes any tool calls, and returns the final assistant response.

```python
def chatbot_interaction(user_message):
    print(f"\n{'=' * 50}\nUser Message: {user_message}\n{'=' * 50}")

    messages = [{"role": "user", "content": user_message}]

    # Get the initial response from Claude
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        tools=tools,
        messages=messages
    )

    print(f"\nInitial Claude Response:")
    print(f"Stop Reason: {message.stop_reason}")

    if message.stop_reason == "tool_use":
        # Extract the tool use block from Claude's response
        tool_use = next(block for block in message.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"\nTool Used: {tool_name}")
        print(f"Tool Input: {tool_input}")

        # Process the tool call and get the result
        save_note_response = process_tool_call(tool_name, tool_input)
        print(f"Tool Result: {save_note_response}")

        # Send the tool result back to Claude for the final response
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": message.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": str(save_note_response),
                        }
                    ],
                },
            ],
            tools=tools,
        )
    else:
        response = message

    # Extract the final text response from Claude
    final_response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )
    print(f"\nFinal Claude Response: {final_response}")

    return final_response
```

## Step 7: Test the Note-Saving Tool

Test the complete workflow by asking Claude to save a note with specific details.

```python
chatbot_interaction("""
Can you save a private note with the following details?
Note: Remember to buy milk and eggs.
Author: John Doe (johndoe@gmail.com)
Priority: 4
""")
```

When you run this, you should see output similar to the following:

```
==================================================
User Message:
Can you save a private note with the following details?
Note: Remember to buy milk and eggs.
Author: John Doe (johndoe@gmail.com)
Priority: 4

==================================================

Initial Claude Response:
Stop Reason: tool_use

Tool Used: save_note
Tool Input: {'note': 'Remember to buy milk and eggs.', 'author': {'name': 'John Doe', 'email': 'johndoe@gmail.com'}, 'is_public': False, 'priority': 4}
Note saved successfully!
Tool Result: success=True message='Note saved successfully!'

Final Claude Response: Your private note has been saved successfully with the following details:

Note: Remember to buy milk and eggs.
Author: John Doe (johndoe@gmail.com)
Priority: 4
Visibility: Private

Please let me know if you need anything else!
```

## Summary

You've successfully built a note-saving tool that integrates Anthropic's Claude with Pydantic validation. The key steps were:

1.  **Defining Pydantic Models** to ensure data integrity for notes and authors.
2.  **Creating a Tool Schema** that Claude can understand and call.
3.  **Implementing a Processing Pipeline** that validates tool inputs, executes the save operation, and returns a structured response.
4.  **Orchestrating the Conversation** between the user, Claude, and your tool.

This pattern adds a robust layer of validation and type safety to your AI tool calls, making your applications more reliable and maintainable. You can extend this foundation by connecting the `save_note` function to a real database or note-taking service.
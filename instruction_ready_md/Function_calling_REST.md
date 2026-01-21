# Gemini API: Function Calling with REST - A Step-by-Step Guide

This guide walks you through using the Gemini API's function calling feature via REST calls with `curl`. You'll learn how to define functions and have the model intelligently select and parameterize them based on natural language queries.

## Prerequisites

Before you begin, ensure you have:
1. A **Google AI API Key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. The key stored in an environment variable named `GOOGLE_API_KEY`

```bash
# Set your API key as an environment variable
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

## Understanding Function Calling

Function calling enables your application to describe functions to the Gemini model. The model analyzes a user's query and returns a structured response indicating which function to call and with what arguments. You then execute the actual function in your code.

Key components:
- **Function Declarations**: JSON descriptions of your functions including name, parameters, and optional descriptions
- **Tools Object**: Container for function declarations passed to the API
- **Model Response**: Returns function name and arguments in OpenAPI-compatible schema format

## Step 1: Single-Turn Function Calling

Start with a simple example where you ask about movie theaters and provide function declarations for the model to choose from.

First, set your model choice (we'll use `gemini-2.5-flash` for this example):

```bash
MODEL_ID="gemini-2.5-flash"
```

Now, make your first API call with three function declarations:

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": {
      "role": "user",
      "parts": {
        "text": "Which theaters in Mountain View show Barbie movie?"
      }
    },
    "tools": [
      {
        "function_declarations": [
          {
            "name": "find_movies",
            "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
            "parameters": {
              "type": "object",
              "properties": {
                "location": {
                  "type": "string",
                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "description": {
                  "type": "string",
                  "description": "Any kind of description including category or genre, title words, attributes, etc."
                }
              },
              "required": ["description"]
            }
          },
          {
            "name": "find_theaters",
            "description": "find theaters based on location and optionally movie title which are is currently playing in theaters",
            "parameters": {
              "type": "object",
              "properties": {
                "location": {
                  "type": "string",
                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "movie": {
                  "type": "string",
                  "description": "Any movie title"
                }
              },
              "required": ["location"]
            }
          },
          {
            "name": "get_showtimes",
            "description": "Find the start times for movies playing in a specific theater",
            "parameters": {
              "type": "object",
              "properties": {
                "location": {
                  "type": "string",
                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "movie": {
                  "type": "string",
                  "description": "Any movie title"
                },
                "theater": {
                  "type": "string",
                  "description": "Name of the theater"
                },
                "date": {
                  "type": "string",
                  "description": "Date for requested showtime"
                }
              },
              "required": ["location", "movie", "theater", "date"]
            }
          }
        ]
      }
    ]
  }'
```

The model will respond with something like:

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "functionCall": {
              "name": "find_theaters",
              "args": {
                "movie": "Barbie",
                "location": "Mountain View, CA"
              }
            }
          }
        ]
      }
    }
  ]
}
```

Notice how the model selected the `find_theaters` function and populated both the required `location` parameter and the optional `movie` parameter based on your query.

## Step 2: Multi-Turn Function Calling with Function Response

In a real application, you would execute the function locally and then send the results back to the model for a complete answer. Here's how to structure that conversation:

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          {
            "text": "Which theaters in Mountain View show Barbie movie?"
          }
        ]
      },
      {
        "role": "model",
        "parts": [
          {
            "functionCall": {
              "name": "find_theaters",
              "args": {
                "location": "Mountain View, CA",
                "movie": "Barbie"
              }
            }
          }
        ]
      },
      {
        "role": "function",
        "parts": [
          {
            "functionResponse": {
              "name": "find_theaters",
              "response": {
                "name": "find_theaters",
                "content": {
                  "movie": "Barbie",
                  "theaters": [
                    {
                      "name": "AMC Mountain View 16",
                      "address": "2000 W El Camino Real, Mountain View, CA 94040"
                    },
                    {
                      "name": "Regal Edwards 14",
                      "address": "245 Castro St, Mountain View, CA 94040"
                    }
                  ]
                }
              }
            }
          }
        ]
      }
    ],
    "tools": [
      {
        "function_declarations": [
          {
            "name": "find_movies",
            "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
            "parameters": {
              "type": "OBJECT",
              "properties": {
                "location": {
                  "type": "STRING",
                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "description": {
                  "type": "STRING",
                  "description": "Any kind of description including category or genre, title words, attributes, etc."
                }
              },
              "required": ["description"]
            }
          },
          {
            "name": "find_theaters",
            "description": "find theaters based on location and optionally movie title which are is currently playing in theaters",
            "parameters": {
              "type": "OBJECT",
              "properties": {
                "location": {
                  "type": "STRING",
                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "movie": {
                  "type": "STRING",
                  "description": "Any movie title"
                }
              },
              "required": ["location"]
            }
          },
          {
            "name": "get_showtimes",
            "description": "Find the start times for movies playing in a specific theater",
            "parameters": {
              "type": "OBJECT",
              "properties": {
                "location": {
                  "type": "STRING",
                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "movie": {
                  "type": "STRING",
                  "description": "Any movie title"
                },
                "theater": {
                  "type": "STRING",
                  "description": "Name of the theater"
                },
                "date": {
                  "type": "STRING",
                  "description": "Date for requested showtime"
                }
              },
              "required": ["location", "movie", "theater", "date"]
            }
          }
        ]
      }
    ]
  }'
```

The model will now process the function results and provide a natural language summary:

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "Barbie is showing at two theaters in Mountain View, CA: AMC Mountain View 16 (2000 W El Camino Real) and Regal Edwards 14 (245 Castro St)."
          }
        ]
      }
    }
  ]
}
```

## Step 3: Extended Multi-Turn Conversation

You can continue the conversation with additional queries. The model maintains context and can use different functions:

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [
          {
            "text": "Which theaters in Mountain View show Barbie movie?"
          }
        ]
      },
      {
        "role": "model",
        "parts": [
          {
            "functionCall": {
              "name": "find_theaters",
              "args": {
                "location": "Mountain View, CA",
                "movie": "Barbie"
              }
            }
          }
        ]
      },
      {
        "role": "function",
        "parts": [
          {
            "functionResponse": {
              "name": "find_theaters",
              "response": {
                "name": "find_theaters",
                "content": {
                  "movie": "Barbie",
                  "theaters": [
                    {
                      "name": "AMC Mountain View 16",
                      "address": "2000 W El Camino Real, Mountain View, CA 94040"
                    },
                    {
                      "name": "Regal Edwards 14",
                      "address": "245 Castro St, Mountain View, CA 94040"
                    }
                  ]
                }
              }
            }
          }
        ]
      },
      {
        "role": "model",
        "parts": [
          {
            "text": "OK. Barbie is showing in two theaters in Mountain View, CA: AMC Mountain View 16 and Regal Edwards 14."
          }
        ]
      },
      {
        "role": "user",
        "parts": [
          {
            "text": "Can you recommend some comedy movies on show in Mountain View?"
          }
        ]
      }
    ],
    "tools": [
      {
        "function_declarations": [
          {
            "name": "find_movies",
            "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
            "parameters": {
              "type": "OBJECT",
              "properties": {
                "location": {
                  "type": "STRING",
                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "description": {
                  "type": "STRING",
                  "description": "Any kind of description including category or genre, title words, attributes, etc."
                }
              },
              "required": ["description"]
            }
          },
          {
            "name": "find_theaters",
            "description": "find theaters based on location and optionally movie title which are is currently playing in theaters",
            "parameters": {
              "type": "OBJECT",
              "properties": {
                "location": {
                  "type": "STRING",
                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "movie": {
                  "type": "STRING",
                  "description": "Any movie title"
                }
              },
              "required": ["location"]
            }
          },
          {
            "name": "get_showtimes",
            "description": "Find the start times for movies playing in a specific theater",
            "parameters": {
              "type": "OBJECT",
              "properties": {
                "location": {
                  "type": "STRING",
                  "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "movie": {
                  "type": "STRING",
                  "description": "Any movie title"
                },
                "theater": {
                  "type": "STRING",
                  "description": "Name of the theater"
                },
                "date": {
                  "type": "STRING",
                  "description": "Date for requested showtime"
                }
              },
              "required": ["location", "movie", "theater", "date"]
            }
          }
        ]
      }
    ]
  }'
```

The model will now select the `find_movies` function to handle your new query about comedy movies.

## Best Practices

1. **Descriptive Function Names**: Use clear, action-oriented names like `find_theaters` instead of vague names like `get_data`
2. **Detailed Parameter Descriptions**: Help the model understand what each parameter expects
3. **Required vs Optional**: Mark only essential parameters as required
4. **Function Descriptions**: Always include descriptions - they significantly improve model accuracy
5. **Error Handling**: In production, implement proper error handling for function execution failures

## Next Steps

Now that you understand the basics of function calling with the Gemini API via REST, you can:
- Integrate these calls into your backend applications
- Implement actual function logic to execute when the model returns function calls
- Explore more complex multi-function scenarios
- Add streaming responses for real-time interactions

Remember that the model only suggests which function to call - you must implement the actual function execution in your code and optionally send the results back for the model to summarize.
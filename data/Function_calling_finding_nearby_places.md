# Personalized Place Recommendations with Google Places API and OpenAI

This guide demonstrates how to build a system that provides personalized location-based recommendations by combining user profiles, the Google Places API, and OpenAI's function calling capabilities. You'll learn to interpret user intent and fetch relevant nearby places tailored to individual preferences.

## Prerequisites

Before you begin, ensure you have the following:

1.  **OpenAI API Key:** Required for using GPT models. Set it as an environment variable `OPENAI_API_KEY`.
2.  **Google Places API Key:** A paid key from the [Google Cloud Console](https://console.cloud.google.com). Set it as an environment variable `GOOGLE_PLACES_API_KEY`.
3.  **Python Libraries:** Install the required packages.

```bash
pip install openai requests
```

## Step 1: Import Libraries and Initialize Client

Start by importing the necessary modules and setting up the OpenAI client.

```python
import json
import os
import requests
from openai import OpenAI

# Initialize the OpenAI client using your API key from the environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

## Step 2: Create a Mock Customer Profile Fetcher

In a real application, you would fetch user data from a database. For this tutorial, we'll simulate this with a hardcoded function.

```python
def fetch_customer_profile(user_id):
    """
    Simulates fetching a user profile from a database.
    For demonstration, returns a hardcoded profile for a specific user_id.
    """
    if user_id == "user1234":
        return {
            "name": "John Doe",
            "location": {
                "latitude": 37.7955,  # Coordinates near Golden Gate Bridge
                "longitude": -122.4026,
            },
            "preferences": {
                "food": ["Italian", "Sushi"],
                "activities": ["Hiking", "Reading"],
            },
            "behavioral_metrics": {
                "app_usage": {"daily": 2, "weekly": 14},
                "favourite_post_categories": ["Nature", "Food", "Books"],
                "active_time": "Evening",
            },
            "recent_searches": ["Italian restaurants nearby", "Book clubs"],
            "recent_interactions": [
                "Liked a post about 'Best Pizzas in New York'",
                "Commented on a post about 'Central Park Trails'"
            ],
            "user_rank": "Gold",
        }
    else:
        return None  # Simulates a user not found
```

## Step 3: Build the Google Places API Helper Functions

You need two functions: one to get basic place listings and another to fetch detailed information for each place.

### 3.1 Fetch Detailed Place Information

This function calls the Google Place Details API using a `place_id`.

```python
def get_place_details(place_id, api_key):
    """Fetches detailed information for a specific place from Google."""
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        result = json.loads(response.content)["result"]
        return result
    else:
        print(f"Details API failed: {response.status_code}")
        return None
```

### 3.2 Search for Nearby Places

This is the main function that searches for places based on type and user location. It integrates with the user profile and uses the details function.

```python
def call_google_places_api(user_id, place_type, food_preference=None):
    """
    Searches for nearby places using the Google Places API.
    Returns a list of formatted descriptions for the top 2 results.
    """
    try:
        # 1. Fetch the user's profile and location
        customer_profile = fetch_customer_profile(user_id)
        if customer_profile is None:
            return "I couldn't find your profile. Could you please verify your user ID?"

        lat = customer_profile["location"]["latitude"]
        lng = customer_profile["location"]["longitude"]

        # 2. Set up API parameters
        api_key = os.getenv('GOOGLE_PLACES_API_KEY')
        location = f"{lat},{lng}"
        radius = 500  # Search within 500 meters

        # 3. Construct the API request URL
        if place_type == 'restaurant' and food_preference:
            url = (f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
                   f"location={location}&radius={radius}&type={place_type}"
                   f"&keyword={food_preference}&key={api_key}")
        else:
            url = (f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
                   f"location={location}&radius={radius}&type={place_type}&key={api_key}")

        # 4. Make the API request
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Places API failed: {response.status_code}")
            return []

        # 5. Process the results
        results = json.loads(response.content)["results"]
        places = []

        for place in results[:2]:  # Limit to 2 results to manage costs
            place_id = place.get("place_id")
            place_details = get_place_details(place_id, api_key)

            if not place_details:
                continue

            # Extract relevant details
            name = place_details.get("name", "N/A")
            # Filter out generic types like 'point_of_interest'
            place_types = next(
                (t for t in place_details.get("types", []) if t not in ["food", "point_of_interest"]),
                "N/A"
            )
            rating = place_details.get("rating", "N/A")
            total_ratings = place_details.get("user_ratings_total", "N/A")
            address = place_details.get("vicinity", "N/A")

            # Clean the address string
            street_address = address.split(',')[0] if ',' in address else address

            # Format the output
            place_info = (f"{name} is a {place_types} located at {street_address}. "
                          f"It has a rating of {rating} based on {total_ratings} user reviews.")
            places.append(place_info)

        return places

    except Exception as e:
        print(f"Error during API call: {e}")
        return []
```

## Step 4: Create the Recommendation Engine with OpenAI

This function ties everything together. It uses an LLM to interpret the user's request, decides if the Google Places API should be called, and formats the response.

```python
def provide_user_specific_recommendations(user_input, user_id):
    """
    Main function that uses OpenAI to interpret user intent and fetch
    personalized place recommendations.
    """
    # 1. Fetch user profile
    customer_profile = fetch_customer_profile(user_id)
    if customer_profile is None:
        return "I couldn't find your profile. Could you please verify your user ID?"

    # 2. Extract a food preference if available (for restaurant searches)
    food_prefs = customer_profile.get('preferences', {}).get('food', [])
    food_preference = food_prefs[0] if food_prefs else None

    # 3. Call OpenAI with function calling definitions
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a sophisticated AI assistant specializing in user intent detection. "
                    "Interpret the user's needs from their statement. For example:\n"
                    "- 'I'm hungry' -> search for restaurants.\n"
                    "- 'I'm tired' -> search for hotels.\n"
                    "If the intent is unclear, ask for clarification. Tailor responses to the user's "
                    f"preferences: {json.dumps(customer_profile)}"
                )
            },
            {"role": "user", "content": user_input}
        ],
        temperature=0,
        tools=[{
            "type": "function",
            "function": {
                "name": "call_google_places_api",
                "description": "Finds top places of a specified type near the user's location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "place_type": {
                            "type": "string",
                            "description": "The type of place to search for (e.g., restaurant, hotel, park)."
                        }
                    },
                    "required": ["place_type"]
                }
            }
        }],
        tool_choice="auto",
    )

    # 4. Check if the model decided to call our function
    message = response.choices[0].message
    if message.tool_calls:
        function_call = message.tool_calls[0].function

        if function_call.name == "call_google_places_api":
            # Extract the place_type argument
            arguments = json.loads(function_call.arguments)
            place_type = arguments["place_type"]

            # 5. Execute the API call with the determined place_type
            places = call_google_places_api(user_id, place_type, food_preference)

            if places:
                return f"Here are some places you might be interested in: {' '.join(places)}"
            else:
                return "I couldn't find any places of interest nearby."

    # 6. Fallback if no function was called
    return "I am sorry, but I could not understand your request."
```

## Step 5: Execute the System

Now you can test the complete workflow. Provide a user ID and a natural language request.

```python
# Test the system
user_id = "user1234"
user_input = "I'm hungry"

recommendations = provide_user_specific_recommendations(user_input, user_id)
print(recommendations)
```

**Expected Output:**
The model will interpret "I'm hungry" as a request for restaurants. It will call the `call_google_places_api` function with `place_type="restaurant"`. The function will use the user's location (near Golden Gate Bridge) and their food preference (Italian, from the profile) to query the API. The result will be a string like:

```
Here are some places you might be interested in: Sotto Mare is a restaurant located at 552 Green Street. It has a rating of 4.6 based on 3765 user reviews. Mona Lisa Restaurant is a restaurant located at 353 Columbus Avenue #3907. It has a rating of 4.4 based on 1888 user reviews.
```

## Summary and Next Steps

You have successfully built a system that:
1.  **Stores User Context:** Uses a mock profile with location and preferences.
2.  **Interprets Intent:** Leverages OpenAI's function calling to map natural language ("I'm hungry") to structured actions (`place_type="restaurant"`).
3.  **Fetches Real-Time Data:** Integrates with the Google Places API to get live, nearby results.
4.  **Personalizes Results:** Filters and presents data based on the user's profile.

To extend this system:
*   Replace the mock `fetch_customer_profile` function with a real database call.
*   Add more user preference filters (e.g., price range, opening hours).
*   Implement more sophisticated intent detection for a wider range of requests (e.g., "I need coffee", "Where can I walk my dog?").
*   Consider caching API responses to reduce cost and latency for frequent queries.
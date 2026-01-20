

# Gemini API: Text Summarization

You will use Gemini API's JSON capabilities to extract characters, locations, and summary of the plot from a story.


```
%pip install -U -q "google-genai>=1.0.0"
```

    [First Entry, ..., Last Entry]

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Example


```
from IPython.display import Markdown

MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
prompt = "Generate a 10 paragraph fantasy story. Include at least 2 named characters and 2 named locations. Give as much detail in the story as possible."
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
    )
story = response.text
Markdown(story)
```

The flickering candlelight cast dancing shadows across Elara's face as she hunched over the ancient map. The vellum felt brittle beneath her fingertips, the ink faded and smudged with age. She traced a delicate line with her calloused finger, following a treacherous path winding through the jagged peaks of the Dragon's Tooth mountains. Elara, a cartographer by trade but an adventurer at heart, had spent years deciphering this cryptic document, rumored to lead to the lost city of Eldoria.

Eldoria, a civilization whispered about in hushed tones in the taverns of Whisperwind village, was said to have vanished overnight, leaving behind only legends of unimaginable riches and powerful magic. Skeptics dismissed it as a fanciful tale, a bedtime story spun to entertain children. But Elara knew better. She felt the truth of Eldoria resonating within her soul, a siren's call that echoed in her blood.

The map indicated a hidden passage, a forgotten route known only to the Eldorians, that bypassed the most dangerous parts of the mountains. According to the legends, the passage was guarded by a creature of stone and shadow, a protector tasked with preventing outsiders from defiling Eldoria's secrets. Elara shivered, not from fear, but from anticipation.

Gathering her supplies, Elara carefully packed dried rations, a waterskin, a bedroll, and her trusty map. She also included a small, intricately carved wooden flute, a gift from her grandfather, a skilled musician who believed that even the most fearsome beasts could be soothed by a beautiful melody. With a deep breath, she extinguished the candle and stepped out into the cool night air.

Her journey began under the watchful gaze of the moon, a silver disc hanging high in the inky sky. As she trekked towards the Dragon's Tooth mountains, the landscape grew increasingly rugged. Towering cliffs loomed above, their peaks piercing the heavens like jagged teeth. The wind howled through the canyons, carrying with it the mournful cries of unseen creatures.

Days turned into nights as Elara navigated the treacherous terrain. She battled fierce winds, scaled sheer rock faces, and forded icy streams. She encountered wild creatures, some hostile, others indifferent, but none as terrifying as the guardian of the hidden passage. Finally, after what felt like an eternity, she arrived at the foot of a colossal stone archway, partially concealed by a curtain of cascading vines. This, she knew, was the entrance to the forgotten route.

As Elara approached the archway, a deep rumbling sound echoed through the mountains. The ground beneath her feet trembled, and the vines covering the archway began to writhe and twist. Emerging from the shadows was a creature of immense size, its body composed of jagged rocks and swirling shadows. Its eyes glowed with an eerie green light, and its voice resonated with the weight of centuries. "Turn back, mortal," it boomed, its words shaking the very air. "Eldoria is not for you."

Elara stood her ground, her heart pounding in her chest. She knew that she couldn't defeat the creature in a physical battle. She reached into her satchel and pulled out the wooden flute. Closing her eyes, she took a deep breath and began to play. The notes, clear and pure, filled the air, weaving a melody of longing and hope, of courage and perseverance.

The creature of stone and shadow paused, its glowing eyes dimming slightly. The music seemed to resonate with its ancient soul, stirring memories of a time long past. As the final notes faded into the air, the creature slowly lowered its head. "You have a kind heart, mortal," it rumbled. "But Eldoria is cursed. Leave it be."

Elara nodded, her eyes filled with understanding. She realized that Eldoria's true riches were not material, but rather the wisdom and knowledge that its people had possessed. She turned and began her descent from the mountains, carrying with her a newfound respect for the legends she had sought to uncover. Though she didn't find gold or jewels, she discovered something far more valuable: the courage to follow her dreams and the wisdom to know when to turn back. The secrets of Eldoria would remain undisturbed, protected by the guardian and the echoes of a beautiful melody.


```
from typing_extensions import TypedDict  # in python 3.12 replace typing_extensions with typing

class Character(TypedDict):
  name: str
  description: str
  alignment: str

class Location(TypedDict):
  name: str
  description: str

class TextSummary(TypedDict):
  synopsis: str
  genres: list[str]
  locations: list[Location]
  characters: list[Character]
```


```
prompt = f"""
Generate summary of the story. With a list of genres locations and characters.

{story}"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_schema": TextSummary
        }
    )
```


```
from pprint import pprint

pprint(response.parsed)
```

    {'characters': [{'alignment': 'Good',
                     'description': 'A cartographer and adventurer seeking the '
                                    'lost city of Eldoria.',
                     'name': 'Elara'},
                    {'alignment': 'Neutral',
                     'description': 'A creature of stone and shadow tasked with '
                                    'protecting the secrets of Eldoria.',
                     'name': 'Guardian of the Hidden Passage'}],
     'genres': ['Fantasy', 'Adventure'],
     'locations': [{'description': 'A treacherous mountain range with jagged '
                                   'peaks, dangerous paths, and a hidden passage '
                                   'to Eldoria.',
                    'name': "Dragon's Tooth Mountains"},
                   {'description': 'A legendary, vanished civilization rumored to '
                                   'hold unimaginable riches and powerful magic.',
                    'name': 'Eldoria (Lost City)'},
                   {'description': 'A village where tales of Eldoria are whispered '
                                   'in taverns.',
                    'name': 'Whisperwind Village'}],
     'synopsis': 'Elara, a cartographer and adventurer, seeks the lost city of '
                 "Eldoria, following an ancient map through the Dragon's Tooth "
                 'mountains. She faces treacherous terrain and a guardian '
                 'creature, ultimately using music to connect with the guardian '
                 "and realizing that Eldoria's true value lies in its wisdom, not "
                 'material riches. She chooses to leave Eldoria undisturbed, '
                 'gaining newfound courage and respect for legends.'}


## Summary

In this example, you used the Gemini API to extract key information from a story. This information could be fed into a structured database or used as a prompt for other writers to create their own versions.

This technique of converting large open-ended text to structured data works across other formats too, not just stories.

Please see the other notebooks in this directory to learn more about how you can use the Gemini API for other JSON related tasks.
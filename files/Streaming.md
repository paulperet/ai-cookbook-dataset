

# Gemini API: Streaming Quickstart

This notebook demonstrates streaming in the Python SDK. By default, the Python SDK returns a response after the model completes the entire generation process. You can also stream the response as it is being generated, and the model will return chunks of the response as soon as they are generated.


```
%pip install -U -q "google-genai" # Install the Python SDK
```


```
from google import genai
```

You'll need an API key stored in an environment variable to run this notebook. See the the [Authentication quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Handle streaming responses

To stream responses, use [`Models.generate_content_stream`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.generate_content_stream).


```
for chunk in client.models.generate_content_stream(
  model=MODEL_ID,
  contents='Tell me a story in 300 words.'
):
    print(chunk.text)
    print("_" * 80)
```

[El, ara adjusted her,  goggles, the copper rims digging slightly into her temples. The air in her workshop hummed with,  the chaotic symphony of whirring gears, sputtering steam, and the rhythmic clanging,  of her hammer. Today was the day. Today, the Sky Serpent took flight.

For years, Elara had toiled, fueled by scraps of dreams,  and a stubborn refusal to accept the limitations others imposed. They called her "the mad tinker," scoffed at her blueprints depicting a mechanical dragon soaring through the clouds. But El, ara had seen the future, a future where the impossible was merely a challenge.

She tightened the last bolt on the Sky Serpent's massive, ornithopter wings. Sunlight streamed through the grimy window, illuminating the intricate network of pipes and pistons,  that powered the magnificent beast. Taking a deep breath, she climbed into the cockpit, a cramped space surrounded by levers, dials, and gauges.

With a flick of a switch, the furnace roared to life, sending plumes of smoke billowing from the dragon, 's iron snout. The workshop trembled as the wings began to beat, slowly at first, then with increasing power. Metal screeched, steam hissed, and then, with a shudder, the Sky Serpent lifted off the ground.

Elara gripped the controls, her heart pounding in her chest. The workshop shrunk,  beneath her as the dragon climbed higher and higher, leaving the familiar world behind. Below, she could see the tiny figures of her neighbors, mouths agape, pointing in disbelief.

As the Sky Serpent broke through the clouds, Elara laughed, a sound filled with triumph and liberation. The wind whipped through her hair, and,  the sun warmed her face. She had done it. She had flown. The mad tinker, the dreamer, had proven them all wrong. The sky, she realized, was truly the limit.

, ]

## Handle streaming responses asynchronously

To stream responses asynchronously, use [`AsyncModels.generate_content_stream(...)`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.generate_content_stream).


```
async for chunk in await client.aio.models.generate_content_stream(
    model=MODEL_ID,
    contents="Write a cute story about cats."):
    if chunk.text:
        print(chunk.text)
    print("_"*80)
```

[C, lementine was a tiny, ginger kitten with a perpetually surprised expression. Her whiskers were like,  exclamation points, and her tail, a fluffy question mark. She lived in a sun-d, renched flower shop, nestled between bouquets of lilies and rambling rose bushes. Her job, as she saw it, was Official Greeter and Head of Pest Control (, though the only pests she’d ever encountered were particularly daring butterflies).

Her best friend was Bartholomew, a grand old tabby who ruled the back room where the flower,  pots were stored. Bartholomew was wise, grumpy, and possessed an impressive collection of cardboard boxes, each designated for different purposes (napping, staring, important contemplation). He tolerated Clementine’s boundless energy, mostly because she brought him the occasional,  rogue ladybug.

One day, a new flower arrived at the shop - a vibrant, exotic orchid with velvety purple petals. Clementine was instantly smitten. She'd never seen anything so beautiful! She spent hours circling it, her little,  nose twitching, trying to decipher its secrets.

"What's that?" she asked Bartholomew, her tail quivering with excitement.

Bartholomew, disturbed from his afternoon nap in the "Important Contemplation" box, grumbled, "Just a fancy weed. Don't bother it."

But,  Clementine couldn't resist. She tiptoed closer, reached out a tentative paw, and gently touched a petal. It was softer than silk!

Suddenly, the orchid shimmered. Not in a scary way, but in a sparkly, magical way. Clementine gasped as tiny, glittering lights danced around the petals.,  She blinked, sure she was imagining things.

Then, she heard a faint, high-pitched voice. "Hello?"

Clementine froze, her fur on end. She looked around wildly. "Who's there?"

The voice giggled. "Down here!"

She peered at the orchid and saw the tini, est of figures emerge from within the petals. It was a miniature cat, no bigger than her thumb, with purple fur and sparkly green eyes!

"I'm Petunia," the tiny cat squeaked. "And I'm lost!"

Clementine was speechless. A real-life, miniature, purple cat, ! She carefully scooped Petunia up with her paw, holding her gently.

"Don't worry, Petunia," she said, her voice full of concern. "I'll help you!"

She ran to Bartholomew, her heart thumping with excitement. "Bartholomew! Bartholomew! Look!", 

Bartholomew, predictably, was not impressed. He opened one eye a crack and squinted at Clementine. "What is it this time? Did you find a particularly shiny beetle?"

Clementine presented Petunia to him. "It's a tiny cat! And she's lost!"

Bartholom, ew sighed dramatically. He clearly thought Clementine had finally lost her mind. But he looked at the tiny creature nestled in Clementine's paw, and something softened in his ancient, golden eyes.

He puffed out his chest and, in a surprisingly gentle voice, said, "Alright, alright. We'll,  help her. First, we need to find her a safe place to sleep."

And so, Clementine and Bartholomew embarked on a grand adventure to help Petunia find her way home. They used a thimble for a bed, a bottle cap for a food dish (filled with delicious flower pollen), and Bartholomew even,  shared his "Important Contemplation" box for the night.

After a few days, with the help of the flower shop owner who mysteriously found a tiny, perfectly purple knitted blanket, they discovered that Petunia was a magical garden sprite, blown into the shop on a strong wind. With a final, tearful goodbye (, and a promise to visit), Petunia was gently placed back onto a passing dandelion seed and floated away, back to her garden.

Clementine watched until Petunia was a tiny speck in the sky. She felt a pang of sadness, but also a deep sense of joy. She had helped someone, and she had,  made a friend.

She looked at Bartholomew, who was already back in his "Important Contemplation" box. He didn't say anything, but Clementine saw the faintest of smiles twitching at his whiskers.

From that day on, Clementine continued to greet every customer with a cheerful meow, and,  she kept a watchful eye on the orchid, just in case Petunia ever decided to visit again. And every once in a while, when the wind was just right, she could almost hear a tiny, sparkly giggle carried on the breeze. The flower shop, already a magical place, was now a little bit more so,,  all thanks to a tiny, ginger kitten and her grand adventure.

, ]

Here's a simple example of two asynchronous functions running simultaneously.


```
import asyncio


async def get_response():
    async for chunk in await client.aio.models.generate_content_stream(
        model=MODEL_ID,
        contents='Tell me a story in 500 words.'
    ):
        if chunk.text:
            print(chunk.text)
        print("_" * 80)

async def something_else():
    for i in range(5):
        print("==========not blocked!==========")
        await asyncio.sleep(1)

async def async_demo():
    # Create tasks for concurrent execution
    task1 = asyncio.create_task(get_response())
    task2 = asyncio.create_task(something_else())
    # Wait for both tasks to complete
    await asyncio.gather(task1, task2)

# In IPython notebooks, you can await the coroutine directly:
await async_demo()
```

[==========not blocked!==========, The,  rusty,  swing set groaned a mournful song as Elara pushed back and forth, her,  worn sneakers kicking up dust devils. It was the only sound in the forgotten corner of the park, , a place even the stray dogs avoided. She clutched a worn, velvet box in her lap, its once vibrant purple faded to a dull lavender. Inside, nestled,  on a bed of frayed satin, was a single, tarnished silver locket.

Today was the anniversary. Five years since Grandma Clara had vanished, leaving, ==========not blocked!==========,  behind only this locket and a house full of unanswered questions. Elara visited every year, hoping for… well, she didn't know what. A sign? A memory? Anything.

Grandma Clara had been a storyteller, weaving,  fantastical tales of hidden cities and talking animals. She’d filled Elara’s childhood with magic, making even the mundane feel extraordinary. The locket, she’d said, held a secret, a whisper of a forgotten world.

Elara,  flipped open the locket. Inside, a miniature portrait of Grandma Clara, her eyes sparkling with mischief, and a tiny, dried flower, pressed so flat it was almost translucent. She ran a finger over the flower, a wave of sadness washing over her. The flower was a Forget-Me-Not.

Suddenly, the, ==========not blocked!==========,  air shimmered. A faint, melodic humming filled the air, growing louder with each swing of the rusty seat. Elara stopped, her heart pounding. The humming resonated with the locket in her hand, vibrating against her palm.

As the humming reached a crescendo, the park around her seemed to fade. The rust,  on the swing set disappeared, replaced by gleaming, polished metal. The overgrown weeds transformed into a riot of vibrant flowers she'd never seen before. The grey, cloudy sky opened up to reveal a cerulean blue, dotted with puffy white clouds.

Before her stood a path, paved with cobblestones and lined with,  trees whose leaves shimmered with an iridescent glow. The humming pulled her forward, a gentle but irresistible force.

Hesitantly, Elara stepped onto the path. As she walked, the air grew warmer, filled with the scent of exotic blooms and the sound of cascading water. She saw creatures darting amongst the trees, ==========not blocked!==========, , creatures she only knew from Grandma Clara's stories – tiny winged sprites, furry creatures with glowing eyes, and elegant deer with antlers of pure light.

Finally, she reached a clearing. In the center stood a grand oak tree, its branches reaching towards the sky like welcoming arms. Underneath the tree, sitting on a moss, -covered stone, was Grandma Clara.

She looked older, wiser, but her eyes held the same mischievous sparkle.

“Elara, my darling,” she said, her voice a gentle melody. “I knew you would find your way.”

Tears streamed down Elara’s face as she rushed forward, ==========not blocked!==========, , throwing her arms around her grandmother. The locket, still clutched in her hand, pulsed with a warm, comforting light.

“But… where were you? What happened?” Elara managed to choke out.

Grandma Clara smiled, a knowing glint in her eyes. “Some stories are meant,  to be lived, not just told,” she said, gesturing towards the fantastical world around them. “And some secrets… are meant to be discovered.”

The adventure had just begun. The forgotten corner of the park had become a gateway, a promise of magic waiting to be unlocked. And Elara, finally reunited with her grandmother,,  was ready to embrace it all.

, ]
# Question answering using embeddings-based search

GPT excels at answering questions, but only on topics it remembers from its training data.

What should you do if you want GPT to answer questions about unfamiliar topics? E.g.,
- Recent events after October 2023 for GPT 4 series models
- Your non-public documents
- Information from past conversations
- etc.

This notebook demonstrates a two-step Search-Ask method for enabling GPT to answer questions using a library of reference text.

1. **Search:** search your library of text for relevant text sections
2. **Ask:** insert the retrieved text sections into a message to GPT and ask it the question

## Why search is better than fine-tuning

GPT can learn knowledge in two ways:

- Via model weights (i.e., fine-tune the model on a training set)
- Via model inputs (i.e., insert the knowledge into an input message)

Although fine-tuning can feel like the more natural option—training on data is how GPT learned all of its other knowledge, after all—we generally do not recommend it as a way to teach the model knowledge. Fine-tuning is better suited to teaching specialized tasks or styles, and is less reliable for factual recall.

As an analogy, model weights are like long-term memory. When you fine-tune a model, it's like studying for an exam a week away. When the exam arrives, the model may forget details, or misremember facts it never read.

In contrast, message inputs are like short-term memory. When you insert knowledge into a message, it's like taking an exam with open notes. With notes in hand, the model is more likely to arrive at correct answers.

One downside of text search relative to fine-tuning is that each model is limited by a maximum amount of text it can read at once:

| Model           | Maximum text length       |
|-----------------|---------------------------|
| `gpt-4o-mini`   | 128,000 tokens (~384 pages)|
| `gpt-4o`        | 128,000 tokens (~384 pages)|


Continuing the analogy, you can think of the model like a student who can only look at a few pages of notes at a time, despite potentially having shelves of textbooks to draw upon.

Therefore, to build a system capable of drawing upon large quantities of text to answer questions, we recommend using a Search-Ask approach.


## Search

Text can be searched in many ways. E.g.,

- Lexical-based search
- Graph-based search
- Embedding-based search

This example notebook uses embedding-based search. Embeddings are simple to implement and work especially well with questions, as questions often don't lexically overlap with their answers.

Consider embeddings-only search as a starting point for your own system. Better search systems might combine multiple search methods, along with features like popularity, recency, user history, redundancy with prior search results, click rate data, etc. Q&A retrieval performance may also be improved with techniques like HyDE, in which questions are first transformed into hypothetical answers before being embedded. Similarly, GPT can also potentially improve search results by automatically transforming questions into sets of keywords or search terms.

## Full procedure

Specifically, this notebook demonstrates the following procedure:

1. Prepare search data (once per document)
    1. Collect: We'll download a few hundred Wikipedia articles about the 2022 Olympics
    2. Chunk: Documents are split into short, mostly self-contained sections to be embedded
    3. Embed: Each section is embedded with the OpenAI API
    4. Store: Embeddings are saved (for large datasets, use a vector database)
2. Search (once per query)
    1. Given a user question, generate an embedding for the query from the OpenAI API
    2. Using the embeddings, rank the text sections by relevance to the query
3. Ask (once per query)
    1. Insert the question and the most relevant sections into a message to GPT
    2. Return GPT's answer

### Costs

Because GPT models are more expensive than embeddings search, a system with a decent volume of queries will have its costs dominated by step 3.

- For `gpt-4o`, considering ~1000 tokens per query, it costs ~$0.0025 per query, or ~450 queries per dollar (as of Nov 2024)
- For `gpt-4o-mini`, using ~1000 tokens per query, it costs ~$0.00015 per query, or ~6000 queries per dollar (as of Nov 2024)

Of course, exact costs will depend on the system specifics and usage patterns.

## Preamble

We'll begin by:
- Importing the necessary libraries
- Selecting models for embeddings search and question answering

```python
# imports
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search

# create a list of models 
GPT_MODELS = ["gpt-4o", "gpt-4o-mini"]
# models
EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
```

#### Troubleshooting: Installing libraries

If you need to install any of the libraries above, run `pip install {library_name}` in your terminal.

For example, to install the `openai` library, run:
```zsh
pip install openai
```

(You can also do this in a notebook cell with `!pip install openai` or `%pip install openai`.)

After installing, restart the notebook kernel so the libraries can be loaded.

#### Troubleshooting: Setting your API key

The OpenAI library will try to read your API key from the `OPENAI_API_KEY` environment variable. If you haven't already, you can set this environment variable by following these instructions.

### Motivating example: GPT cannot answer questions about current events

Because the training data for `gpt-4o-mini` mostly ended in October 2023, the models cannot answer questions about more recent events, such as the 2024 Elections or recent games.

For example, let's try asking 'How many ?':

```python
# an example question about the 2022 Olympics
query = 'Which athletes won the most number of gold medals in 2024 Summer Olympics?'

response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2024 Games or latest events.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODELS[0],
    temperature=0,
)

print(response.choices[0].message.content)
```

    I'm sorry, but I don't have information on the outcomes of the 2024 Summer Olympics, including which athletes won the most gold medals. My training only includes data up to October 2023, and the Olympics are scheduled to take place in Paris from July 26 to August 11, 2024. You might want to check the latest updates from reliable sports news sources or the official Olympics website for the most current information.

In this case, the model has no knowledge of 2024 and is unable to answer the question. In a similar way, if you ask a question pertaining to a recent political event (that occured in Nov 2024 for example), GPT-4o-mini models will not be able to answer due to its knowledge cut-off date of Oct 2023.

```python
# an example question about the 2024 Elections
query = 'Who won the elections in the US in 2024?'

response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2024 Games or latest events.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODELS[1],
    temperature=0,
)

print(response.choices[0].message.content)
```

    I'm sorry, but I don't have information on events or elections that occurred after October 2023. For the latest updates on the 2024 US elections, I recommend checking reliable news sources.

### You can give GPT knowledge about a topic by inserting it into an input message

To help give the model knowledge of curling at the 2022 Winter Olympics, we can copy and paste the top half of a relevant Wikipedia article into our message:

```python
# text copied and pasted from: https://en.wikipedia.org/wiki/2024_Summer_Olympics
# We didn't bother to clean the text, but GPT will still understand it
# Top few sections are included in the text below

wikipedia_article = """2024 Summer Olympics

The 2024 Summer Olympics (French: Les Jeux Olympiques d'été de 2024), officially the Games of the XXXIII Olympiad (French: Jeux de la XXXIIIe olympiade de l'ère moderne) and branded as Paris 2024, were an international multi-sport event held from 26 July to 11 August 2024 in France, with several events started from 24 July. Paris was the host city, with events (mainly football) held in 16 additional cities spread across metropolitan France, including the sailing centre in the second-largest city of France, Marseille, on the Mediterranean Sea, as well as one subsite for surfing in Tahiti, French Polynesia.[4]

Paris was awarded the Games at the 131st IOC Session in Lima, Peru, on 13 September 2017. After multiple withdrawals that left only Paris and Los Angeles in contention, the International Olympic Committee (IOC) approved a process to concurrently award the 2024 and 2028 Summer Olympics to the two remaining candidate cities; both bids were praised for their high technical plans and innovative ways to use a record-breaking number of existing and temporary facilities. Having previously hosted in 1900 and 1924, Paris became the second city ever to host the Summer Olympics three times (after London, which hosted the games in 1908, 1948, and 2012).[5][6] Paris 2024 marked the centenary of Paris 1924 and Chamonix 1924 (the first Winter Olympics), as well as the sixth Olympic Games hosted by France (three Summer Olympics and three Winter Olympics) and the first with this distinction since the 1992 Winter Games in Albertville. The Summer Games returned to the traditional four-year Olympiad cycle, after the 2020 edition was postponed to 2021 due to the COVID-19 pandemic.

Paris 2024 featured the debut of breaking as an Olympic sport,[7] and was the final Olympic Games held during the IOC presidency of Thomas Bach.[8] The 2024 Games were expected to cost €9 billion.[9][10][11] The opening ceremony was held outside of a stadium for the first time in modern Olympic history, as athletes were paraded by boat along the Seine. Paris 2024 was the first Olympics in history to reach full gender parity on the field of play, with equal numbers of male and female athletes.[12]

The United States topped the medal table for the fourth consecutive Summer Games and 19th time overall, with 40 gold and 126 total medals.[13] 
China tied with the United States on gold (40), but finished second due to having fewer silvers; the nation won 91 medals overall. 
This is the first time a gold medal tie among the two most successful nations has occurred in Summer Olympic history.[14] Japan finished third with 20 gold medals and sixth in the overall medal count. Australia finished fourth with 18 gold medals and fifth in the overall medal count. The host nation, France, finished fifth with 16 gold and 64 total medals, and fourth in the overall medal count. Dominica, Saint Lucia, Cape Verde and Albania won their first-ever Olympic medals, the former two both being gold, with Botswana and Guatemala also winning their first-ever gold medals. 
The Refugee Olympic Team also won their first-ever medal, a bronze in boxing. At the conclusion of the games, despite some controversies throughout relating to politics, logistics and conditions in the Olympic Village, the Games were considered a success by the press, Parisians and observers.[a] The Paris Olympics broke all-time records for ticket sales, with more than 9.5 million tickets sold (12.1 million including the Paralympic Games).[15]

Medal table
Main article: 2024 Summer Olympics medal table
See also: List of 2024 Summer Olympics medal winners
Key
 ‡  Changes in medal standings (see below)

  *   Host nation (France)

2024 Summer Olympics medal table[171][B][C]
Rank	NOC	Gold	Silver	Bronze	Total
1	 United States‡	40	44	42	126
2	 China	40	27	24	91
3	 Japan	20	12	13	45
4	 Australia	18	19	16	53
5	 France*	16	26	22	64
6	 Netherlands	15	7	12	34
7	 Great Britain	14	22	29	65
8	 South Korea	13	9	10	32
9	 Italy	12	13	15	40
10	 Germany	12	13	8	33
11–91	Remaining NOCs	129	138	194	461
Totals (91 entries)	329	330	385	1,044

Podium sweeps
There was one podium sweep during the games:

Date	Sport	Event	Team	Gold	Silver	Bronze	Ref
2 August	Cycling	Men's BMX race	 France	Joris Daudet	Sylvain André	Romain Mahieu	[176]


Medals
Medals from the Games, with a piece of the Eiffel Tower
The President of the Paris 2024 Olympic Organizing Committee, Tony Estanguet, unveiled the Olympic and Paralympic medals for the Games in February 2024, which on the obverse featured embedded hexagon-shaped tokens of scrap iron that had been taken from the original construction of the Eiffel Tower, with the logo of the Games engraved into it.[41] Approximately 5,084 medals would be produced by the French mint Monnaie de Paris, and were designed by Chaumet, a luxury jewellery firm based in Paris.[42]

The reverse of the medals features Nike, the Greek goddess of victory, inside the Panathenaic Stadium which hosted the first modern Olympics in 1896. Parthenon and the Eiffel Tower can also be seen in the background on both sides of the medal.[43] Each medal weighs 455–529 g (16–19 oz), has a diameter of 85 mm (3.3 in) and is 9.2 mm (0.36 in) thick.[44] The gold medals are made with 98.8 percent silver and 1.13 percent gold, while the bronze medals are made up with copper, zinc, and tin.[45]


Opening ceremony
Main article: 2024 Summer Olympics opening ceremony

Pyrotechnics at the Pont d'Austerlitz marking the start of the Parade of Nations

The cauldron flying above the Tuileries Garden during the games. LEDs and aerosol produced the illusion of fire, while the Olympic flame itself was kept in a small lantern nearby
The opening ceremony began at 19:30 CEST (17:30 GMT) on 26 July 2024.[124] Directed by Thomas Jolly,[125][126][127] it was the first Summer Olympics opening ceremony to be held outside the traditional stadium setting (and the second ever after the 2018 Youth Olympic Games one, held at Plaza de la República in Buenos Aires); the parade of athletes was conducted as a boat parade along the Seine from Pont d'Austerlitz to Pont d'Iéna, and cultural segments took place at various landmarks along the route.[128] Jolly stated that the ceremony would highlight notable moments in the history of France, with an overall theme of love and "shared humanity".[128] The athletes then attended the official protocol at Jardins du Trocadéro, in front of the Eiffel Tower.[129] Approximately 326,000 tickets were sold for viewing locations along the Seine, 222,000 of which were distributed primarily to the Games' volunteers, youth and low-income families, among others.[130]

The ceremony featured music performances by American musician Lady Gaga,[131] French-Malian singer Aya Nakamura, heavy metal band Gojira and soprano Marina Viotti [fr],[132] Axelle Saint-Cirel (who sang the French national anthem "La Marseillaise" atop the Grand Palais),[133] rapper Rim'K,[134] Philippe Katerine (who portrayed the Greek god Dionysus), Juliette Armanet and Sofiane Pamart, and was closed by Canadian singer Céline Dion.[132] The Games were formally opened by president Emmanuel Macron.[135]

The Olympics and Paralympics cauldron was lit by Guadeloupean judoka Teddy Riner and sprinter Marie-José Pérec; it had a hot air balloon-inspired design topped by a 30-metre-tall (98 ft) helium sphere, and was allowed to float into the air above the Tuileries Garden at night. For the first time, the cauldron was not illuminated through combustion; the flames were simulated by an LED lighting system and aerosol water jets.[136]

Controversy ensued at the opening ceremony when a segment was interpreted by some as a parody of the Last Supper. The organisers apologised for any offence caused.[137] The Olympic World Library and fact-checkers would later debunk the interpretation that the segment was a parody of the Last Supper. The Olympic flag was also raised upside down.[138][139]

During the day of the opening ceremony, there were reports of a blackout in Paris, although this was later debunked.[140]

Closing ceremony


The ceremony and final fireworks
Main article: 2024 Summer Olympics closing ceremony
The closing ceremony was held at Stade de France on 11 August 2024, and thus marked the first time in any Olympic edition since Sarajevo 1984 that opening and closing ceremonies were held in different locations.[127] Titled "Records", the ceremony was themed around a dystopian future, where the Olympic Games have disappeared, and a group of aliens reinvent it. It featured more than a hundred performers, including acrobats, dancers and circus artists.[158] American actor Tom Cruise also appeared with American performers Red Hot Chili Peppers, Billie Eilish, Snoop Dogg, and H.E.R. during the LA28 Handover Celebration portion of the ceremony.[159][160] The Antwerp Ceremony, in which the Olympic flag was handed to Los Angeles, the host city of the 2028 Summer Olympics, was produced by Ben Winston and his studio Fulwell 73.[161]


Security
France reached an agreement with Europol and the UK Home Office to help strengthen security and "facilitate operational information exchange and international law enforcement cooperation" during the Games.[46
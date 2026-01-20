# Build an agent with tool-calling superpowers 🦸 using smolagents
_Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_

This notebook demonstrates how you can use [**smolagents**](https://huggingface.co/docs/smolagents/index) to build awesome **agents**!

What are **agents**? Agents are systems that are powered by an LLM and enable the LLM (with careful prompting and output parsing) to use specific *tools* to solve problems.

These *tools* are basically functions that the LLM couldn't perform well by itself: for instance for a text-generation LLM like [Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct), this could be an image generation tool, a web search tool, a calculator...

What is **smolagents**? It's an library that provides building blocks to build your own agents! Learn more about it in the [documentation](https://huggingface.co/docs/smolagents/index).

Let's see how to use it, and which use cases it can solve.

Run the line below to install required dependencies:


```python
!pip install smolagents datasets langchain sentence-transformers faiss-cpu duckduckgo-search openai langchain-community --upgrade -q
```

Let's login in order to call the HF Inference API:


```python
from huggingface_hub import notebook_login

notebook_login()
```

## 1. 🏞️ Multimodal + 🌐 Web-browsing assistant

For this use case, we want to show an agent that browses the web and is able to generate images.

To build it, we simply need to have two tools ready: image generation and web search.
- For image generation, we load a tool from the Hub that uses the HF Inference API (Serverless) to generate images using Stable Diffusion.
- For the web search, we use a built-in tool.


```python
from smolagents import load_tool, CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)


search_tool = DuckDuckGoSearchTool()

model = InferenceClientModel("Qwen/Qwen2.5-72B-Instruct")
# Initialize the agent with both tools
agent = CodeAgent(
    tools=[image_generation_tool, search_tool], model=model
)

# Run it!
result = agent.run(
    "Generate me a photo of the car that James bond drove in the latest movie.",
)
result
```

    TOOLCODE:
     from smolagents import Tool
    from huggingface_hub import InferenceClient
    
    
    class TextToImageTool(Tool):
        description = "This tool creates an image according to a prompt, which is a text description."
        name = "image_generator"
        inputs = {"prompt": {"type": "string", "description": "The image generator prompt. Don't hesitate to add details in the prompt to make the image look better, like 'high-res, photorealistic', etc."}}
        output_type = "image"
        model_sdxl = "black-forest-labs/FLUX.1-schnell"
        client = InferenceClient(model_sdxl)
    
    
        def forward(self, prompt):
            return self.client.text_to_image(prompt)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #d4b702; text-decoration-color: #d4b702">╭──────────────────────────────────────────────────── </span><span style="color: #d4b702; text-decoration-color: #d4b702; font-weight: bold">New run</span><span style="color: #d4b702; text-decoration-color: #d4b702"> ────────────────────────────────────────────────────╮</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span>                                                                                                                 <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">Generate me a photo of the car that James bond drove in the latest movie.</span>                                       <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span>                                                                                                                 <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">╰─ InferenceClientModel - Qwen/Qwen2.5-72B-Instruct ────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #d4b702; text-decoration-color: #d4b702">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ </span><span style="font-weight: bold">Step </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="color: #d4b702; text-decoration-color: #d4b702"> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">╭─ <span style="font-weight: bold">Executing this code:</span> ──────────────────────────────────────────────────────────────────────────────────────────╮
│ <span style="color: #e3e3dd; text-decoration-color: #e3e3dd; background-color: #272822; font-weight: bold">  </span><span style="color: #656660; text-decoration-color: #656660; background-color: #272822">1 </span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">search_query </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"latest James Bond movie"</span><span style="background-color: #272822">                                                                   </span> │
│ <span style="color: #e3e3dd; text-decoration-color: #e3e3dd; background-color: #272822; font-weight: bold">  </span><span style="color: #656660; text-decoration-color: #656660; background-color: #272822">2 </span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">result </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> web_search(query</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">search_query)</span><span style="background-color: #272822">                                                                    </span> │
│ <span style="color: #e3e3dd; text-decoration-color: #e3e3dd; background-color: #272822; font-weight: bold">  </span><span style="color: #656660; text-decoration-color: #656660; background-color: #272822">3 </span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">print(result)</span><span style="background-color: #272822">                                                                                              </span> │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Execution logs:</span>
## Search Results

[James Bond 26 Casting Update Reveals Conditions For New 007: "It's A 
...](https://screenrant.com/james-bond-26-casting-conditions-updated-barbara-broccoli/)
James Bond 26 gets an intriguing new update from franchise producer Barbara Broccoli, who reveals the conditions 
for the next actor. After first playing the character in 2006's Casino Royale, Daniel Craig bid farewell to the hit
spy franchise with No Time To Die in 2021. There's been no actor officially cast as his replacement since, but the 
Internet and social media are awash with rumors and ...

[No Time to Die - Wikipedia](https://en.wikipedia.org/wiki/No_Time_to_Die)
No Time to Die is a 2021 spy film and the twenty-fifth in the James Bond series produced by Eon Productions, 
starring Daniel Craig in his final portrayal of fictional British MI6 agent James Bond.The plot follows Bond, who 
has left active service with MI6, and is recruited by the CIA to find a kidnapped scientist, which leads to a 
showdown with a powerful and vengeful adversary armed with a ...

[List of James Bond films - Wikipedia](https://en.wikipedia.org/wiki/List_of_James_Bond_films)
Find out the history, actors, directors, box office and budget of the James Bond film series. The latest film, No 
Time to Die, was released in September 2021 and starred Daniel Craig as 007.

[No Time to Die (2021) - IMDb](https://www.imdb.com/title/tt2382320/)
No Time to Die: Directed by Cary Joji Fukunaga. With Daniel Craig, Léa Seydoux, Rami Malek, Lashana Lynch. James 
Bond has left active service. His peace is short-lived when Felix Leiter, an old friend from the CIA, turns up 
asking for help, leading Bond onto the trail of a mysterious villain armed with dangerous new technology.

[Bond 26: Everything We Know About Next 007 Film - 
Newsweek](https://www.newsweek.com/james-bond-26-everything-we-know-next-007-film-1891233)
The James Bond films have been popular for decades and since Daniel Craig stepped away from the titular role in 
2021, people have been speculating what the future of the franchise will look like.

[No Time to Die (2021) - Rotten Tomatoes](https://www.rottentomatoes.com/m/no_time_to_die_2021)
Watch the trailer, read critics and audience reviews, and see the official clips of the latest James Bond film. No 
Time to Die is a long and action-packed adventure that concludes Craig's tenure as 007 with style and emotion.

[Next James Bond: everything we know so far about who will be the new 
...](https://www.timeout.com/news/everything-we-know-about-bond-26-so-far-010523)
The 26th instalment of the James Bond franchise is expected to be released in 2025, but the identity of the new 007
is still a mystery. Find out the latest rumours, odds and contenders for the role, from Aaron Taylor-Johnson to Tom
Hardy.

[When will the new James Bond be announced? Everything we know ... - 
Yahoo](https://www.yahoo.com/entertainment/james-bond-26-007-cast-rumours-release-date-143855208.html)
The James Bond movies celebrated their 60th anniversary in 2022. Now, in 2024, fans are eagerly anticipating the 
announcement of the new James Bond, and details of the next movie. Bond 26 — the ...

[Daniel Craig's final James Bond movie is on TV tonight - Digital 
Spy](https://www.digitalspy.com/movies/a63306287/james-bond-no-time-to-die-itv/)
Related: New James Bond movie gets disappointing update American Fiction's Jeffrey Wright also makes a long-awaited
return to the franchise as M16 agent Felix Wright (previously seen in Casino ...

['No Time to Die': Release date, trailer and ... - What To 
Watch](https://www.whattowatch.com/watching-guides/no-time-to-die-release-date-trailer-and-everything-else-we-know-
about-the-new-james-bond-film)
No Time to Die is the 25th and final James Bond movie starring Daniel Craig, released in 2021. Find out the plot, 
cast, director, release date, trailer and the shocking twist ending of the film.

Out: None
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">[Step 0: Duration 23.18 seconds| Input tokens: 2,152 | Output tokens: 73]</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #d4b702; text-decoration-color: #d4b702">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ </span><span style="font-weight: bold">Step </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="color: #d4b702; text-decoration-color: #d4b702"> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">╭─ <span style="font-weight: bold">Executing this code:</span> ──────────────────────────────────────────────────────────────────────────────────────────╮
│ <span style="color: #e3e3dd; text-decoration-color: #e3e3dd; background-color: #272822; font-weight: bold">  </span><span style="color: #656660; text-decoration-color: #656660; background-color: #272822">1 </span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">car_search_query </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"car driven by James Bond in No Time to Die"</span><span style="background-color: #272822">                                            </span> │
│ <span style="color: #e3e3dd; text-decoration-color: #e3e3dd; background-color: #272822; font-weight: bold">  </span><span style="color: #656660; text-decoration-color: #656660; background-color: #272822">2 </span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">result </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> web_search(query</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">car_search_query)</span><span style="background-color: #272822">                                                                </span> │
│ <span style="color: #e3e3dd; text-decoration-color: #e3e3dd; background-color: #272822; font-weight: bold">  </span><span style="color: #656660; text-decoration-color: #656660; background-color: #272822">3 </span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">print(result)</span><span style="background-color: #272822">                                                                                              </span> │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Execution logs:</span>
## Search Results

[Every Car James Bond Drives in 'No Time
```python
!python --version
```

    Python 3.10.12


# Claude 3 RAG Agents with LangChain v1

LangChain v1 brought a lot of changes and when comparing the LangChain of versions `0.0.3xx` to `0.1.x` there's plenty of changes to the preferred way of doing things. That is very much the case for agents.

The way that we initialize and use agents is generally clearer than it was in the past — there are still many abstractions, but we can (and are encouraged to) get closer to the agent logic itself. This can make for some confusion at first, but once understood the new logic can be much clearer than with previous versions.

In this example, we'll be building a RAG agent with LangChain v1. We will use Claude 3 for our LLM, Voyage AI for knowledge embeddings, and Pinecone to power our knowledge retrieval.

To begin, let's install the prerequisites:


```python
!pip install -qU \
    langchain==0.1.11 \
    langchain-core==0.1.30 \
    langchain-community==0.0.27 \
    langchain-anthropic==0.1.4 \
    langchainhub==0.1.15 \
    anthropic==0.19.1 \
    voyageai==0.2.1 \
    pinecone-client==3.1.0 \
    datasets==2.16.1
```

[First Entry, ..., Last Entry]

And grab the required API keys. We will need API keys for [Claude](https://docs.claude.com/claude/reference/getting-started-with-the-api), [Voyage AI](https://docs.voyageai.com/install/), and [Pinecone](https://docs.pinecone.io/docs/quickstart).


```python
# Insert your API keys here
ANTHROPIC_API_KEY = "<YOUR_ANTHROPIC_API_KEY>"
PINECONE_API_KEY = "<YOUR_PINECONE_API_KEY>"
VOYAGE_API_KEY = "<YOUR_VOYAGE_API_KEY>"
```

## Finding Knowledge

The first thing we need for an agent using RAG is somewhere we want to pull knowledge from. We will use v2 of the AI ArXiv dataset, available on Hugging Face Datasets at [`jamescalam/ai-arxiv2-chunks`](https://huggingface.co/datasets/jamescalam/ai-arxiv2-chunks).

_Note: we're using the prechunked dataset. For the raw version see [`jamescalam/ai-arxiv2`](https://huggingface.co/datasets/jamescalam/ai-arxiv2)._


```python
from datasets import load_dataset

dataset = load_dataset("jamescalam/ai-arxiv2-chunks", split="train[:20000]")
dataset
```





    Dataset({
        features: ['doi', 'chunk-id', 'chunk', 'id', 'title', 'summary', 'source', 'authors', 'categories', 'comment', 'journal_ref', 'primary_category', 'published', 'updated', 'references'],
        num_rows: 20000
    })




```python
dataset[1]
```




    {'doi': '2401.09350',
     'chunk-id': 1,
     'chunk': 'These neural networks and their training algorithms may be complex, and the scope of their impact broad and wide, but nonetheless they are simply functions in a high-dimensional space. A trained neural network takes a vector as input, crunches and transforms it in various ways, and produces another vector, often in some other space. An image may thereby be turned into a vector, a song into a sequence of vectors, and a social network as a structured collection of vectors. It seems as though much of human knowledge, or at least what is expressed as text, audio, image, and video, has a vector representation in one form or another.\nIt should be noted that representing data as vectors is not unique to neural networks and deep learning. In fact, long before learnt vector representations of pieces of dataâ\x80\x94what is commonly known as â\x80\x9cembeddingsâ\x80\x9dâ\x80\x94came along, data was often encoded as hand-crafted feature vectors. Each feature quanti- fied into continuous or discrete values some facet of the data that was deemed relevant to a particular task (such as classification or regression). Vectors of that form, too, reflect our understanding of a real-world object or concept.',
     'id': '2401.09350#1',
     'title': 'Foundations of Vector Retrieval',
     'summary': 'Vectors are universal mathematical objects that can represent text, images,\nspeech, or a mix of these data modalities. That happens regardless of whether\ndata is represented by hand-crafted features or learnt embeddings. Collect a\nlarge enough quantity of such vectors and the question of retrieval becomes\nurgently relevant: Finding vectors that are more similar to a query vector.\nThis monograph is concerned with the question above and covers fundamental\nconcepts along with advanced data structures and algorithms for vector\nretrieval. In doing so, it recaps this fascinating topic and lowers barriers of\nentry into this rich area of research.',
     'source': 'http://arxiv.org/pdf/2401.09350',
     'authors': 'Sebastian Bruch',
     'categories': 'cs.DS, cs.IR',
     'comment': None,
     'journal_ref': None,
     'primary_category': 'cs.DS',
     'published': '20240117',
     'updated': '20240117',
     'references': []}



## Building the Knowledge Base

To build our knowledge base we need _two things_:

1. Embeddings, for this we will use `VoyageEmbeddings` using Voyage AI's embedding models, which do need an [API key](https://dash.voyageai.com/api-keys).
2. A vector database, where we store our embeddings and query them. We use Pinecone which again requires a [free API key](https://app.pinecone.io).

First we initialize our connection to Voyage AI and define an `embed` object for embeddings:


```python
from langchain_community.embeddings import VoyageEmbeddings

embed = VoyageEmbeddings(voyage_api_key=VOYAGE_API_KEY, model="voyage-2")
```

Then we initialize our connection to Pinecone:


```python
from pinecone import Pinecone

# configure client
pc = Pinecone(api_key=PINECONE_API_KEY)
```

Now we setup our index specification, this allows us to define the cloud provider and region where we want to deploy our index. You can find a list of all [available providers and regions here](https://docs.pinecone.io/docs/projects).


```python
from pinecone import ServerlessSpec

spec = ServerlessSpec(cloud="aws", region="us-west-2")
```

Before creating an index, we need the dimensionality of our Voyage AI embedding model, which we can find easily by creating an embedding and checking the length:


```python
vec = embed.embed_documents(["ello"])
len(vec[0])
```




    1024



Now we create the index using our embedding dimensionality, and a metric also compatible with the model (this can be either cosine or dotproduct). We also pass our spec to index initialization.


```python
import time

index_name = "claude-3-rag"

# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=len(vec[0]),  # dimensionality of voyage model
        metric="dotproduct",
        spec=spec,
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()
```




    {'dimension': 1024,
     'index_fullness': 0.0,
     'namespaces': {'': {'vector_count': 20000}},
     'total_vector_count': 20000}



### Populating our Index

Now our knowledge base is ready to be populated with our data. We will use the `embed` helper function to embed our documents and then add them to our index.

We will also include metadata from each record.


```python
from tqdm.auto import tqdm

# easier to work with dataset as pandas dataframe
data = dataset.to_pandas()

batch_size = 100

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i + batch_size)
    # get batch of data
    batch = data.iloc[i:i_end]
    # generate unique ids for each chunk
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
    # get text to embed
    texts = [x["chunk"] for _, x in batch.iterrows()]
    # embed text
    embeds = embed.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {"text": x["chunk"], "source": x["source"], "title": x["title"]}
        for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata, strict=False))
```


[First Entry, ..., Last Entry]

Create a tool for our agent to use when searching for ArXiv papers:


```python
from langchain.agents import tool


@tool
def arxiv_search(query: str) -> str:
    """Use this tool when answering questions about AI, machine learning, data
    science, or other technical questions that may be answered using arXiv
    papers.
    """
    # create query vector
    xq = embed.embed_query(query)
    # perform search
    out = index.query(vector=xq, top_k=5, include_metadata=True)
    # reformat results into string
    results_str = "\n\n".join([x["metadata"]["text"] for x in out["matches"]])
    return results_str


tools = [arxiv_search]
```

When this tool is used by our agent it will execute it like so:


```python
print(arxiv_search.run(tool_input={"query": "can you tell me about llama 2?"}))
```

    Model Llama 2 Code Llama Code Llama - Python Size FIM LCFT Python CPP Java PHP TypeScript C# Bash Average 7B â 13B â 34B â 70B â 7B â 7B â 7B â 7B â 13B â 13B â 13B â 13B â 34B â 34B â 7B â 7B â 13B â 13B â 34B â 34B â â â â â 14.3% 6.8% 10.8% 9.9% 19.9% 13.7% 15.8% 13.0% 24.2% 23.6% 22.2% 19.9% 27.3% 30.4% 31.6% 34.2% 12.6% 13.2% 21.4% 15.1% 6.3% 3.2% 8.3% 9.5% 3.2% 12.6% 17.1% 3.8% 18.9% 25.9% 8.9% 24.8% â â â â â â â â â â 37.3% 31.1% 36.1% 30.4% 29.2% 29.8% 38.0%
    
    Ethical Considerations and Limitations (Section 5.2) Llama 2 is a new technology that carries risks with use. Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, Llama 2âs potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate or objectionable responses to user prompts. Therefore, before deploying any applications of Llama 2, developers should perform safety testing and tuning tailored to their speciï¬c applications of the model. Please see the Responsible Use Guide available available at https://ai.meta.com/llama/responsible-user-guide
    Table 52: Model card for Llama 2.
    77
    
    2
    Cove Liama Long context (7B =, 13B =, 34B) + fine-tuning ; Lrama 2 Code training 20B oes Cope Liama - Instruct Foundation models â> nfilling code training = eee.â (7B =, 13B =, 34B) â 5B (7B, 13B, 348) 5008 Python code Long context Cove Liama - PyrHon (7B, 13B, 34B) > training Â» Fine-tuning > 1008 208
    Figure 2: The Code Llama specialization pipeline. The different stages of fine-tuning annotated with the number of tokens seen during training. Infilling-capable models are marked with the â symbol.
    # 2 Code Llama: Specializing Llama 2 for code
    # 2.1 The Code Llama models family
    
    # 2 Code Llama: Specializing Llama 2 for code
    # 2.1 The Code Llama models family
    Code Llama. The Code Llama models constitute foundation models for code generation. They come in four model sizes: 7B, 13B, 34B and 70B parameters. The 7B, 13B and 70B models are trained using an infilling objective (Section 2.3), and are appropriate to be used in an IDE to complete code in the middle of a file, for example. The 34B model was trained without the infilling objective. All Code Llama models are initialized with Llama 2 model weights and trained on 500B tokens from a code-heavy dataset (see Section 2.2 for more details), except Code Llama 70B which was trained on 1T tokens. They are all fine-tuned to handle long contexts as detailed in Section 2.4.
    
    0.52 0.57 0.19 0.30 Llama 1 7B 13B 33B 65B 0.27 0.24 0.23 0.25 0.26 0.24 0.26 0.26 0.34 0.31 0.34 0.34 0.54 0.52 0.50 0.46 0.36 0.37 0.36 0.36 0.39 0.37 0.35 0.40 0.26 0.23 0.24 0.25 0.28 0.28 0.33 0.32 0.33 0.31 0.34 0.32 0.45 0.50 0.49 0.48 0.33 0.27 0.31 0.31 0.17 0.10 0.12 0.11 0.24 0.24 0.23 0.25 0.31 0.27 0.30 0.30 0.44 0.41 0.41 0.43 0.57 0.55 0.60 0.60 0.39 0.34 0.28 0.39 Llama 2 7B 13B 34B 70B 0.28 0.24 0.27 0.31 0.25 0.25 0.24 0.29 0.29 0.35 0.33 0.35 0.50 0.50 0.56 0.51 0.36 0.41 0.41


## Defining XML Agent

The XML agent is built primarily to support Anthropic models. Anthropic models have been trained to use XML tags like `<input>{some input}</input` or when using a tool they use:

```
<tool>{tool name}</tool>
<tool_input>{tool input}</tool_input>
```

This is much different to the format produced by typical ReAct agents, which is not as well supported by Anthropic models.

To create an XML agent we need a `prompt`, `llm`, and list of `tools`. We can download a prebuilt prompt for conversational XML agents from LangChain hub.


```python
from langchain import hub

prompt = hub.pull("hwchase17/xml-agent-convo")
prompt
```




    ChatPromptTemplate(input_variables=['agent_scratchpad', 'input', 'tools'], partial_variables={'chat_history': ''}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['agent_scratchpad', 'chat_history', 'input', 'tools'], template="You are a helpful assistant. Help the user answer any questions.\n\nYou have access to the following tools:\n\n{tools}\n\nIn order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>\nFor example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:\n\n<tool>search</tool><tool_input>weather in SF</tool_input>\n<observation>64 degrees</observation>\n\nWhen you are done, respond with a final answer between <final
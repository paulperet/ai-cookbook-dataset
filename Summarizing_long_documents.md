# Summarizing Long Documents

The objective of this notebook is to demonstrate how to summarize large documents with a controllable level of detail.
 
If you give a GPT model the task of summarizing a long document (e.g. 10k or more tokens), you'll tend to get back a relatively short summary that isn't proportional to the length of the document. For instance, a summary of a 20k token document will not be twice as long as a summary of a 10k token document. One way we can fix this is to split our document up into pieces, and produce a summary piecewise. After many queries to a GPT model, the full summary can be reconstructed. By controlling the number of text chunks and their sizes, we can ultimately control the level of detail in the output.


```python
import os
from typing import List, Tuple, Optional
from openai import OpenAI
import tiktoken
from tqdm import tqdm
```


```python
# open dataset containing part of the text of the Wikipedia page for the United States
with open("data/artificial_intelligence_wikipedia.txt", "r") as file:
    artificial_intelligence_wikipedia_text = file.read()
```


```python
# load encoding and check the length of dataset
encoding = tiktoken.encoding_for_model('gpt-4-turbo')
len(encoding.encode(artificial_intelligence_wikipedia_text))
```




    14630



We'll define a simple utility to wrap calls to the OpenAI API.


```python
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_chat_completion(messages, model='gpt-4-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content
```

Next we'll define some utilities to chunk a large document into smaller pieces.


```python
def tokenize(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model('gpt-4-turbo')
    return encoding.encode(text)


# This function chunks a text into smaller pieces based on a maximum token count and a delimiter.
def chunk_on_delimiter(input_string: str,
                       max_tokens: int, delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True
    )
    if dropped_chunk_count > 0:
        print(f"warning: {dropped_chunk_count} chunks were dropped due to overflow")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks


# This function combines text chunks into larger blocks without exceeding a specified token count. It returns the combined text blocks, their original indices, and the count of chunks dropped due to overflow.
def combine_chunks_with_no_minimum(
        chunks: List[str],
        max_tokens: int,
        chunk_delimiter="\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int]]:
    dropped_chunk_count = 0
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = (
        [] if header is None else [header]
    )  # list to hold the current combined chunk candidate
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]
        if len(tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            print(f"warning: chunk overflow")
            if (
                    add_ellipsis_for_overflow
                    and len(tokenize(chunk_delimiter.join(candidate + ["..."]))) <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # this case would break downstream assumptions
        # estimate token count with the current chunk added
        extended_candidate_token_count = len(tokenize(chunk_delimiter.join(candidate + [chunk])))
        # If the token count exceeds max_tokens, add the current candidate to output and start a new candidate
        if extended_candidate_token_count > max_tokens:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header  # re-initialize candidate
            candidate_indices = [chunk_i]
        # otherwise keep extending the candidate
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)
    # add the remaining candidate to output if it's not empty
    if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count
```

Now we can define a utility to summarize text with a controllable level of detail (note the `detail` parameter).

The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count based on a controllable `detail` parameter. It then splits the text into chunks and summarizes each chunk.


```python
def summarize(text: str,
              detail: float = 0,
              model: str = 'gpt-4-turbo',
              additional_instructions: Optional[str] = None,
              minimum_chunk_size: Optional[int] = 500,
              chunk_delimiter: str = ".",
              summarize_recursively=False,
              verbose=False):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually. 
    The level of detail in the summary can be adjusted, and the process can optionally be made recursive.

    Parameters:
    - text (str): The text to be summarized.
    - detail (float, optional): A value between 0 and 1 indicating the desired level of detail in the summary.
      0 leads to a higher level summary, and 1 results in a more detailed summary. Defaults to 0.
    - model (str, optional): The model to use for generating summaries. Defaults to 'gpt-3.5-turbo'.
    - additional_instructions (Optional[str], optional): Additional instructions to provide to the model for customizing summaries.
    - minimum_chunk_size (Optional[int], optional): The minimum size for text chunks. Defaults to 500.
    - chunk_delimiter (str, optional): The delimiter used to split the text into chunks. Defaults to ".".
    - summarize_recursively (bool, optional): If True, summaries are generated recursively, using previous summaries for context.
    - verbose (bool, optional): If True, prints detailed information about the chunking process.

    Returns:
    - str: The final compiled summary of the text.

    The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count based on the `detail` parameter. 
    It then splits the text into chunks and summarizes each chunk. If `summarize_recursively` is True, each summary is based on the previous summaries, 
    adding more context to the summarization process. The function returns a compiled summary of all chunks.
    """

    # check detail is set correctly
    assert 0 <= detail <= 1

    # interpolate the number of chunks based to get specified level of detail
    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # adjust chunk_size based on interpolated number of chunks
    document_length = len(tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(tokenize(x)) for x in text_chunks]}")

    # set system message
    system_message_content = "Rewrite this text in summarized form."
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in tqdm(text_chunks):
        if summarize_recursively and accumulated_summaries:
            # Creating a structured prompt for recursive summarization
            accumulated_summaries_string = '\n\n'.join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            # Directly passing the chunk for summarization without recursive context
            user_message_content = chunk

        # Constructing messages based on whether recursive summarization is applied
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]

        # Assuming this function gets the completion and works as expected
        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    # Compile final summary from partial summaries
    final_summary = '\n\n'.join(accumulated_summaries)

    return final_summary
```

Now we can use this utility to produce summaries with varying levels of detail. By increasing `detail` from 0 to 1 we get progressively longer summaries of the underlying document. A higher value for the `detail` parameter results in a more detailed summary because the utility first splits the document into a greater number of chunks. Each chunk is then summarized, and the final summary is a concatenation of all the chunk summaries.


```python
summary_with_detail_0 = summarize(artificial_intelligence_wikipedia_text, detail=0, verbose=True)
```

    Splitting the text into 1 chunks to be summarized.
    Chunk lengths are [14631]


    100%|██████████| 1/1 [00:09<00:00,  9.68s/it]



```python
summary_with_detail_pt25 = summarize(artificial_intelligence_wikipedia_text, detail=0.25, verbose=True)
```

    Splitting the text into 9 chunks to be summarized.
    Chunk lengths are [1817, 1807, 1823, 1810, 1806, 1827, 1814, 1829, 103]


    100%|██████████| 9/9 [01:33<00:00, 10.39s/it]



```python
summary_with_detail_pt5 = summarize(artificial_intelligence_wikipedia_text, detail=0.5, verbose=True)
```

    Splitting the text into 17 chunks to be summarized.
    Chunk lengths are [897, 890, 914, 876, 893, 906, 893, 902, 909, 907, 905, 889, 902, 890, 901, 880, 287]


    100%|██████████| 17/17 [02:26<00:00,  8.64s/it]



```python
summary_with_detail_1 = summarize(artificial_intelligence_wikipedia_text, detail=1, verbose=True)
```

    Splitting the text into 31 chunks to be summarized.
    Chunk lengths are [492, 427, 485, 490, 496, 478, 473, 497, 496, 501, 499, 497, 493, 470, 472, 494, 489, 492, 481, 485, 471, 500, 486, 498, 478, 469, 498, 468, 493, 478, 103]


    100%|██████████| 31/31 [04:08<00:00,  8.02s/it]


The original document is nearly 15k tokens long. Notice how large the gap is between the length of `summary_with_detail_0` and `summary_with_detail_1`. It's nearly 25 times longer!


```python
# lengths of summaries
[len(tokenize(x)) for x in
 [summary_with_detail_0, summary_with_detail_pt25, summary_with_detail_pt5, summary_with_detail_1]]
```




    [235, 2529, 4336, 6742]



Let's inspect the summaries to see how the level of detail changes when the `detail` parameter is increased from 0 to 1.


```python
print(summary_with_detail_0)
```

    Artificial intelligence (AI) is the simulation of human intelligence in machines, designed to perform tasks that typically require human intelligence. This includes applications like advanced search engines, recommendation systems, speech interaction, autonomous vehicles, and more. AI was first significantly researched by Alan Turing and became an academic discipline in 1956. The field has experienced cycles of high expectations followed by disillusionment and reduced funding, known as "AI winters." Interest in AI surged post-2012 with advancements in deep learning and again post-2017 with the development of the transformer architecture, leading to a boom in AI research and applications in the early 2020s.
    
    AI's increasing integration into various sectors is influencing societal and economic shifts towards automation and data-driven decision-making, impacting areas such as employment, healthcare, and privacy. Ethical and safety concerns about AI have prompted discussions on regulatory policies.
    
    AI research involves various sub-fields focused on specific goals like reasoning, learning, and perception, using techniques from mathematics, logic, and other disciplines. Despite its broad applications, AI's complexity and potential risks, such as privacy issues, misinformation, and ethical challenges, remain areas of active investigation and debate.



```python
print(summary_with_detail_1)
```

    Artificial intelligence (AI) is the simulation of human intelligence in machines, designed to perceive their environment and make decisions to achieve specific goals. This technology is prevalent across various sectors including industry, government, and science, with applications ranging from web search engines and recommendation systems to autonomous vehicles and AI in gaming. Although AI has become a common feature in many tools and applications, it often goes unrecognized as AI when it becomes sufficiently integrated and widespread.
    
    The field of AI, which began as an academic discipline in 1956, has experienced several cycles of high expectations followed by disappointment, known as AI winters. Interest and funding in AI surged post-2012 with advancements in deep learning and again post-2017 with the development of transformer architecture, leading to a significant boom in AI research and applications in the early 2020s, primarily in the United States.
    
    The increasing integration of AI in the 21st century is driving a shift towards automation and data-driven decision-making across various sectors, influencing job markets, healthcare, and education, among others. This raises important questions about the ethical implications, long-term effects, and the need for regulatory policies to ensure the safety and benefits of AI technologies. AI research itself is diverse, focusing on goals like reasoning, learning, and perception, and involves various tools and methodologies to achieve these objectives.
    
    General intelligence, which involves performing any human task at least as well as a human, is a long-term goal in AI research. To achieve this, AI integrates various techniques from search and optimization, formal logic, neural networks, and statistics, to insights from psychology, linguistics, and neuroscience. AI research focuses on specific traits like reasoning and problem-solving, where early algorithms mimicked human step-by-step reasoning. However, these algorithms struggle with large, complex problems due to combinatorial explosion and are less efficient than human intuitive judgments. Knowledge representation is another critical area, using ontologies to structure domain-specific knowledge and relationships, aiding in intelligent querying, scene interpretation, and data mining among other applications.
    
    Knowledge bases must encapsulate a wide range of elements including objects, properties, categories, relations, events, states, time, causes, effects, and meta-knowledge. They also need to handle default reasoning, where certain assumptions are maintained unless contradicted. Challenges in knowledge representation include the vast scope of commonsense knowledge and its often sub-symbolic, non-verbal nature, alongside the difficulty of acquiring this knowledge for AI use.
    
    In the realm of AI, an "agent" is defined as an entity that perceives its environment and acts towards achieving goals or fulfilling preferences. In automated planning, the agent pursues a specific goal, while in decision-making, it evaluates actions based on their expected utility to maximize preference satisfaction. Classical planning assumes agents have complete knowledge of action outcomes, but real-world scenarios often involve uncertainty about the situation and outcomes, requiring probabilistic decision-making. Additionally, agents may need to adapt or learn preferences, particularly in complex environments with multiple agents or human interactions.
    
    Information value theory helps assess the value of exploratory actions in situations with uncertain outcomes. A Markov decision process uses a transition model and a reward function to guide decisions, which can be determined through calculations, heuristics, or learning. Game theory analyzes the rational behavior of multiple interacting agents in decision-making scenarios involving others.
    
    Machine learning, integral to AI, involves programs that automatically improve task performance. It includes unsupervised learning, which identifies patterns in data without guidance, and supervised learning, which requires labeled data and includes classification and regression tasks. Reinforcement learning rewards or punishes agents to shape their responses, while transfer learning applies knowledge from one problem to another. Deep learning, a subset of machine learning, uses artificial neural networks inspired by biological processes.
    
    Computational learning theory evaluates learning algorithms based on computational and sample complexity, among other criteria. Natural language processing (NLP) enables programs to interact using human languages, tackling challenges like speech recognition, synthesis, translation, and more. Early NLP efforts, influenced by Chomsky's theories, faced limitations in handling ambiguous language outside of controlled environments.
    
    Margaret Masterman emphasized the importance of meaning over grammar in language understanding, advocating for the use of thesauri instead of dictionaries in computational linguistics. Modern NLP techniques include word embedding, transformers, and by 2023, GPT models capable of achieving human-level scores on various tests. Machine perception involves interpreting sensor data to understand the world, encompassing computer vision and speech recognition among other applications. Social intelligence in AI focuses on recognizing and simulating human emotions, with systems like Kismet and affective computing technologies that enhance human-computer interaction. However, these advancements may lead to overestimations of AI capabilities by users. AI also employs a variety of techniques including search and optimization, with methods like state space search to explore possible solutions to problems.
    
    Planning algorithms use means-ends analysis to navigate through trees of goals and subgoals to achieve a target goal. However, simple exhaustive searches are often inadequate for complex real-world problems due to the vast search space, making searches slow or incomplete. Heuristics
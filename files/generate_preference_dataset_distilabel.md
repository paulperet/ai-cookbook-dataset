# Generate a Preference Dataset with distilabel

_Authored by: [David Berenstein](https://huggingface.co/davidberenstein1957) and [Sara Han Díaz](https://huggingface.co/sdiazlor)_

- **Libraries**: [argilla](https://github.com/argilla-io/argilla), [hf-inference-endpoints](https://github.com/huggingface/huggingface_hub)
- **Components**: [LoadDataFromHub](https://distilabel.argilla.io/latest/components-gallery/steps/loaddatafromhub/), [TextGeneration](https://distilabel.argilla.io/latest/components-gallery/tasks/textgeneration/), [UltraFeedback](https://distilabel.argilla.io/latest/components-gallery/tasks/ultrafeedback/), [GroupColumns](https://distilabel.argilla.io/latest/components-gallery/steps/groupcolumns/), [FormatTextGenerationDPO](https://distilabel.argilla.io/latest/components-gallery/steps/formattextgenerationdpo/), [PreferenceToArgilla](https://distilabel.argilla.io/latest/components-gallery/steps/textgenerationtoargilla/), [InferenceEndpointsLLM](https://distilabel.argilla.io/latest/components-gallery/llms/inferenceendpointsllm/)

In this tutorial, we will use distilabel to generate a synthetic preference dataset for DPO, ORPO or RLHF. [distilabel](https://github.com/argilla-io/distilabel) is a synthetic data and AI feedback framework for engineers who need fast, reliable and scalable pipelines based on verified research papers. Check the documentation [here](https://distilabel.argilla.io/latest/).

To generate the responses and evaluate them, we will use the [serverless HF Inference API](https://huggingface.co/docs/api-inference/index) integrated with distilabel. This is free but rate-limited, allowing you to test and evaluate over 150,000 public models, or your own private models, via simple HTTP requests, with fast inference hosted on Hugging Face shared infrastructure. If you need more compute power, you can deploy your own inference endpoint with [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/guides/create_endpoint).

Finally, to further curate the data, we will use [Argilla](https://github.com/argilla-io/argilla), which allows us to provide human feedback on the data quality. Argilla is a collaboration tool for AI engineers and domain experts who need to build high-quality datasets for their projects. Check the documentation [here](https://docs.argilla.io/latest/).

## Getting started

### Install the dependencies

To complete this tutorial, you need to install the distilabel SDK and a few third-party libraries via pip. We will be using **the free but rate-limited Hugging Face serverless Inference API** for this tutorial, so we need to install this as an extra distilabel dependency. You can install them by running the following command:

```python
!pip install "distilabel[hf-inference-endpoints]"
```

```python
!pip install "transformers~=4.0" "torch~=2.0"
```

Let's make the required imports:

```python
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    FormatTextGenerationDPO,
    PreferenceToArgilla,
)
from distilabel.steps.tasks import TextGeneration, UltraFeedback
```

You'll need an `HF_TOKEN` to use the HF Inference Endpoints. Log in to use it directly within this notebook.

```python
import os
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"), add_to_git_credential=True)
```

### (optional) Deploy Argilla

You can skip this step or replace it with any other data evaluation tool, but the quality of your model will suffer from a lack of data quality, so we do recommend looking at your data. If you already deployed Argilla, you can skip this step. Otherwise, you can quickly deploy Argilla following [this guide](https://docs.argilla.io/latest/getting_started/quickstart/). 

Along with that, you will need to install Argilla as a distilabel extra.

```python
!pip install "distilabel[argilla, hf-inference-endpoints]"
```

## Define the pipeline

To generate our preference dataset, we will need to define a `Pipeline` with all the necessary steps. Below, we will go over each step in detail.

### Load the dataset

We will use as source data the [`argilla/10Kprompts-mini`](https://huggingface.co/datasets/argilla/10Kprompts-mini) dataset from the Hugging Face Hub.

- Component: `LoadDataFromHub`
- Input columns: `instruction` and `topic`, the same as in the loaded dataset
- Output columns: `instruction` and `topic`

```python
load_dataset = LoadDataFromHub(
        repo_id= "argilla/10Kprompts-mini",
        num_examples=1,
        pipeline=Pipeline(name="showcase-pipeline"),
    )
load_dataset.load()
next(load_dataset.process())
```

    ([{'instruction': 'How can I create an efficient and robust workflow that utilizes advanced automation techniques to extract targeted data, including customer information, from diverse PDF documents and effortlessly integrate it into a designated Google Sheet? Furthermore, I am interested in establishing a comprehensive and seamless system that promptly activates an SMS notification on my mobile device whenever a new PDF document is uploaded to the Google Sheet, ensuring real-time updates and enhanced accessibility.',
       'topic': 'Software Development'}],
     True)

### Generate responses

We need to generate the responses for the given instructions. We will use two different models available on the Hugging Face Hub through the Serverless Inference API: [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [`mistralai/Mixtral-8x7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1). We will also indicate the generation parameters for each model.

- Component: `TextGeneration` task with LLMs using `InferenceEndpointsLLM`
- Input columns: `instruction`
- Output columns: `generation`, `distilabel_metadata`, `model_name` for each model

For your use case and to improve the results, you can use any [other LLM of your choice](https://distilabel.argilla.io/latest/components-gallery/llms/).

```python
generate_responses = [
    TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3-8B-Instruct",
            generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
        ),
        pipeline=Pipeline(name="showcase-pipeline"),
    ),
    TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            tokenizer_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
        ),
        pipeline=Pipeline(name="showcase-pipeline"),
    ),
]
for task in generate_responses:
    task.load()
    print(next(task.process([{"instruction": "Which are the top cities in Spain?"}])))
```

    [{'instruction': 'Which are the top cities in Spain?', 'generation': 'Spain is a country with a rich culture, history, and architecture, and it has many great cities to visit. Here are some of the top cities in Spain:\n\n1. **Madrid**: The capital city of Spain, known for its vibrant nightlife, museums, and historic landmarks like the Royal Palace and Prado Museum.\n2. **Barcelona**: The second-largest city in Spain, famous for its modernist architecture, beaches, and iconic landmarks like La Sagrada Família and Park Güell, designed by Antoni Gaudí.\n3. **Valencia**: Located on the Mediterranean coast, Valencia is known for its beautiful beaches, City of Arts and Sciences, and delicious local cuisine, such as paella.\n4. **Seville**: The capital of Andalusia, Seville is famous for its stunning cathedral, Royal Alcázar Palace, and lively flamenco music scene.\n5. **Málaga**: A coastal city in southern Spain, Málaga is known for its rich history, beautiful beaches, and being the birthplace of Pablo Picasso.\n6. **Zaragoza**: Located in the northeastern region of Aragon, Zaragoza is a city with a rich history, known for its Roman ruins, Gothic cathedral, and beautiful parks.\n7. **Granada**: A city in the Andalusian region, Granada is famous for its stunning Alhambra palace and generalife gardens, a UNESCO World Heritage Site.\n8. **Bilbao**: A city in the Basque Country, Bilbao is known for its modern architecture, including the Guggenheim Museum, and its rich cultural heritage.\n9. **Alicante**: A coastal city in the Valencia region, Alicante is famous for its beautiful beaches, historic castle, and lively nightlife.\n10. **San Sebastián**: A city in the Basque Country, San Sebastián is known for its stunning beaches, gastronomic scene, and cultural events like the San Sebastián International Film Festival.\n\nThese are just a few of the many great cities in Spain, each with its own unique character and attractions.', 'distilabel_metadata': {'raw_output_text_generation_0': 'Spain is a country with a rich culture, history, and architecture, and it has many great cities to visit. Here are some of the top cities in Spain:\n\n1. **Madrid**: The capital city of Spain, known for its vibrant nightlife, museums, and historic landmarks like the Royal Palace and Prado Museum.\n2. **Barcelona**: The second-largest city in Spain, famous for its modernist architecture, beaches, and iconic landmarks like La Sagrada Família and Park Güell, designed by Antoni Gaudí.\n3. **Valencia**: Located on the Mediterranean coast, Valencia is known for its beautiful beaches, City of Arts and Sciences, and delicious local cuisine, such as paella.\n4. **Seville**: The capital of Andalusia, Seville is famous for its stunning cathedral, Royal Alcázar Palace, and lively flamenco music scene.\n5. **Málaga**: A coastal city in southern Spain, Málaga is known for its rich history, beautiful beaches, and being the birthplace of Pablo Picasso.\n6. **Zaragoza**: Located in the northeastern region of Aragon, Zaragoza is a city with a rich history, known for its Roman ruins, Gothic cathedral, and beautiful parks.\n7. **Granada**: A city in the Andalusian region, Granada is famous for its stunning Alhambra palace and generalife gardens, a UNESCO World Heritage Site.\n8. **Bilbao**: A city in the Basque Country, Bilbao is known for its modern architecture, including the Guggenheim Museum, and its rich cultural heritage.\n9. **Alicante**: A coastal city in the Valencia region, Alicante is famous for its beautiful beaches, historic castle, and lively nightlife.\n10. **San Sebastián**: A city in the Basque Country, San Sebastián is known for its stunning beaches, gastronomic scene, and cultural events like the San Sebastián International Film Festival.\n\nThese are just a few of the many great cities in Spain, each with its own unique character and attractions.'}, 'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct'}]
    [{'instruction': 'Which are the top cities in Spain?', 'generation': ' Here are some of the top cities in Spain based on various factors such as tourism, culture, history, and quality of life:\n\n1. Madrid: The capital and largest city in Spain, Madrid is known for its vibrant nightlife, world-class museums (such as the Prado Museum and Reina Sofia Museum), stunning parks (such as the Retiro Park), and delicious food.\n\n2. Barcelona: Famous for its unique architecture, Barcelona is home to several UNESCO World Heritage sites designed by Antoni Gaudí, including the Sagrada Familia and Park Güell. The city also boasts beautiful beaches, a lively arts scene, and delicious Catalan cuisine.\n\n3. Valencia: A coastal city located in the east of Spain, Valencia is known for its City of Arts and Sciences, a modern architectural complex that includes a planetarium, opera house, and museum of interactive science. The city is also famous for its paella, a traditional Spanish dish made with rice, vegetables, and seafood.\n\n4. Seville: The capital of Andalusia, Seville is famous for its flamenco dancing, stunning cathedral (the largest Gothic cathedral in the world), and the Alcázar, a beautiful palace made up of a series of rooms and courtyards.\n\n5. Granada: Located in the foothills of the Sierra Nevada mountains, Granada is known for its stunning Alhambra palace, a Moorish fortress that dates back to the 9th century. The city is also famous for its tapas, a traditional Spanish dish that is often served for free with drinks.\n\n6. Bilbao: A city in the Basque Country, Bilbao is famous for its modern architecture, including the Guggenheim Museum, a contemporary art museum designed by Frank Gehry. The city is also known for its pintxos, a type of Basque tapas that are served in bars and restaurants.\n\n7. Málaga: A coastal city in Andalusia, Málaga is known for its beautiful beaches, historic sites (including the Alcazaba and Gibralfaro castles), and the Picasso Museum, which is dedicated to the famous Spanish artist who was born in the city.\n\nThese are just a few of the many wonderful cities in Spain.', 'distilabel_metadata': {'raw_output_text_generation_0': ' Here are some of the top cities in Spain based on various factors such as tourism, culture, history, and quality of life:\n\n1. Madrid: The capital and largest city in Spain, Madrid is known for its vibrant nightlife, world-class museums (such as the Prado Museum and Reina Sofia Museum), stunning parks (such as the Retiro Park), and delicious food.\n\n2. Barcelona: Famous for its unique architecture, Barcelona is home to several UNESCO World Heritage sites designed by Antoni Gaudí, including the Sagrada Familia and Park Güell. The city also boasts beautiful beaches, a lively arts scene, and delicious Catalan cuisine.\n\n3. Valencia: A coastal city located in the east of Spain, Valencia is known for its City of Arts and Sciences, a modern architectural complex that includes a planetarium, opera house, and museum of interactive science. The city is also famous for its paella, a traditional Spanish dish made with rice, vegetables, and seafood.\n\n4. Seville: The capital of Andalusia, Seville is famous for its flamenco dancing, stunning cathedral (the largest Gothic cathedral in the world), and the Alcázar, a beautiful palace made up of a series of rooms and courtyards.\n\n5. Granada: Located in the foothills of the Sierra Nevada mountains, Granada is known for its stunning Alhambra palace, a Moorish fortress that dates back to the 9th century. The city is also famous for its tapas, a traditional Spanish dish that is often served for free with drinks.\n\n6. Bilbao: A city in the Basque Country, Bilbao is famous for its modern architecture, including the Guggenheim Museum, a contemporary art museum designed by Frank Gehry. The city is also known for its pintxos, a type of Basque tapas that are served in bars and restaurants.\n\n7. Málaga: A coastal city in Andalusia, Málaga is known for its beautiful beaches, historic sites (including the Alcazaba and Gibralfaro castles), and the Picasso Museum, which is dedicated to the famous Spanish artist who was born in the city.\n\nThese are just a few of the many wonderful cities in Spain.'}, 'model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1'}]

### Group the responses

The task to evaluate the responses needs as input a list of generations. However, each model response was saved in the generation column of the subsets `text_generation_0` and `text_generation_1`. We will combine these two columns into a single column and the `default` subset.

- Component: `GroupColumns`
- Input columns: `generation` and `model_name`from `text_generation_0` and `text_generation_1`
- Output columns: `generations` and `model_names`

```python
group_responses = GroupColumns(
    columns=["generation", "model_name"],
    output_columns=["generations", "model_names"],
    pipeline=Pipeline(name="showcase-pipeline"),
)
next(
    group_responses.process(
        [
            {
                "generation": "Madrid",
                "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            },
        ],
        [
            {
                "generation": "Barcelona",
                "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            }
        ],
    )
)
```

    [{'generations': ['Madrid', 'Barcelona'],
      'model_names': ['meta-llama/Meta-Llama-3-8B-Instruct',
       'mistralai/Mixtral-8x7B-Instruct-v0.1']}]

### Evaluate the responses

To build our preference dataset, we need to evaluate the responses generated by the models. We will use [`meta-llama/Meta-Llama-3-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) for this, applying the `UltraFeedback` task that judges the responses according to different dimensions (helpfulness, honesty, instruction-following, truthfulness).

- Component: `UltraFeedback` task with LLMs
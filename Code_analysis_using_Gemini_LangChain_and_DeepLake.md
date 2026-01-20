##### Copyright 2025 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Code analysis using LangChain and DeepLake

This notebook shows how to use Gemini API with [Langchain](https://python.langchain.com/v0.2/docs/introduction/) and [DeepLake](https://www.deeplake.ai/) for code analysis. The notebook will teach you:
- loading and splitting files
- creating a Deeplake database with embedding information
- setting up a retrieval QA chain

### Load dependencies


```
%pip install -q -U langchain-google-genai langchain-deeplake langchain langchain-text-splitters langchain-community
```


```
from glob import glob
from IPython.display import Markdown, display

from langchain.document_loaders import TextLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_deeplake.vectorstores import DeeplakeVectorStore
```

### Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GEMINI_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../../quickstarts/Authentication.ipynb) for an example.



```
import os
from google.colab import userdata
GEMINI_API_KEY=userdata.get('GEMINI_API_KEY')

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
```

## Prepare the files

First, download a [langchain-google](https://github.com/langchain-ai/langchain-google) repository. It is the repository you will analyze in this example.

It contains code integrating Gemini API, VertexAI, and other Google products with langchain.


```
!git clone https://github.com/langchain-ai/langchain-google
```

This example will focus only on the integration of Gemini API with langchain and ignore the rest of the codebase.


```
repo_match = "langchain-google/libs/genai/langchain_google_genai**/*.py"
```

Each file with a matching path will be loaded and split by `RecursiveCharacterTextSplitter`.
In this example, it is specified, that the files are written in Python. It helps split the files without having documents that lack context.


```
docs = []
for file in glob(repo_match, recursive=True):
  loader = TextLoader(file, encoding='utf-8')
  splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=2000, chunk_overlap=0)
  docs.extend(loader.load_and_split(splitter))
```

`Language` Enum provides common separators used in most popular programming languages, it lowers the chances of classes or functions being split in the middle.


```
# common seperators used for Python files
RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
```




    ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']



## Create the database
The data will be loaded into the memory since the database doesn't need to be permanent in this case and is small enough to fit.

The type of storage used is specified by prefix in the path, in this case by `mem://`.

Check out other types of storage [here](https://docs.activeloop.ai/setup/storage-and-creds/storage-options).


```
# define path to database
dataset_path = 'mem://deeplake/langchain_google'
```


```
# define the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

Everything needed is ready, and now you can create the database. It should not take longer than a few seconds.


```
db = DeeplakeVectorStore.from_documents(
    dataset_path=dataset_path,
    embedding=embeddings,
    documents=docs,
    overwrite=True
)
```

## Question Answering

Set-up the document retriever.


```
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 20 # number of documents to return
```


```
# define the chat model
llm = ChatGoogleGenerativeAI(model = "gemini-3-flash-preview")
```

Now, you can create a chain for Question Answering. In this case, `RetrievalQA` chain will be used.

If you want to use the chat option instead, use `ConversationalRetrievalChain`.


```
qa = RetrievalQA.from_llm(llm, retriever=retriever)
```

The chain is ready to answer your questions.

NOTE: `Markdown` is used for improved formatting of the output.


```
# a helper function for calling retrival chain
def call_qa_chain(prompt):
  response = qa.invoke(prompt)
  display(Markdown(response["result"]))
```


```
call_qa_chain("Show hierarchy for _BaseGoogleGenerativeAI. Do not show content of classes.")
```


```
_BaseGoogleGenerativeAI
    └── BaseModel
```



```
call_qa_chain("What is the return type of embedding models.")
```


The embedding models, specifically `GoogleGenerativeAIEmbeddings`, have two main methods for generating embeddings:

1.  **`embed_query`**: Returns a single embedding for a given text as a `List[float]`.
2.  **`embed_documents`**: Returns a list of embeddings (one for each text in the input list) as a `List[List[float]]`.



```
call_qa_chain("What classes are related to Attributed Question and Answering.")
```


The classes related to Attributed Question and Answering (AQA) are:

1.  **`GenAIAqa`**: This is the main class representing Google's Attributed Question and Answering service.
2.  **`AqaInput`**: Defines the input structure for the `GenAIAqa.invoke` method, including the prompt and source passages.
3.  **`AqaOutput`**: Defines the output structure from the `GenAIAqa.invoke` method, containing the answer, attributed passages, and answerable probability.
4.  **`_AqaModel`**: An internal wrapper class used by `GenAIAqa` to interact with the underlying Google AQA model.
5.  **`GroundedAnswer`**: A dataclass used internally by the AQA implementation to represent a grounded answer, including the answer text, attributed passages, and answerable probability.
6.  **`GoogleVectorStore`**: This class has an `as_aqa()` method which allows it to be used to construct a Google Generative AI AQA engine, providing passages from the vector store for grounding.
7.  **`_SemanticRetriever`**: This is an internal component of `GoogleVectorStore` that handles the retrieval of passages, which are then used by the AQA service when integrated through `GoogleVectorStore.as_aqa()`.



```
call_qa_chain("What are the dependencies of the GenAIAqa class?")
```


The `GenAIAqa` class has the following dependencies:

1.  **`RunnableSerializable`**: It inherits from `langchain_core.runnables.RunnableSerializable`.
2.  **`AqaInput`**: It uses `AqaInput` as its input type for the `invoke` method.
3.  **`AqaOutput`**: It returns `AqaOutput` from its `invoke` method.
4.  **`_AqaModel`**: It internally uses an instance of `_AqaModel` to interact with the Google GenAI service.
5.  **`google.ai.generativelanguage` (as `genai`)**: The `_AqaModel` class, which `GenAIAqa` depends on, directly uses components from `google.ai.generativelanguage` (e.g., `GenerativeServiceClient`, `AnswerStyle`, `SafetySetting`).
6.  **`_genai_extension` (as `genaix`)**: The `_AqaModel` class uses functions from this internal module (e.g., `build_generative_service`, `generate_answer`, `GroundedAnswer`).


## Summary

Gemini API works great with Langchain. The integration is seamless and provides an easy interface for:
- loading and splitting files
- creating DeepLake database with embeddings
- answering questions based on context from files

## What's next?

This notebook showed only one possible use case for langchain with Gemini API. You can find many more [here](../../examples/langchain).
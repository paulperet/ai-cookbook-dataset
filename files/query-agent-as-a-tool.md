

*Contributed by the Weaviate team*

## Weaviate Query Agent with Gemini API

This notebook will show you how to define the Weaviate Query Agent as a tool through the Gemini API.

### Requirements
1. Weaviate Cloud instance (WCD): The Weaviate Query Agent is only accessible through WCD at the moment. You can create a serverless cluster or a free 14-day sandbox [here](https://console.weaviate.cloud/).
2. Have a GCP project and Gemini API key (generate one [here](https://aistudio.google.com/))
3. Install the Google Gen AI SDK with `pip install --upgrade --quiet google-genai`
4. Install the Weaviate Python client and the agents sub-package with `pip install weaviate-client[agents]`
5. You'll need a Weaviate cluster with data. If you don't have one, check out [this notebook](integrations/Weaviate-Import-Example.ipynb) to import the Weaviate Blogs.

Connect with us and let us know if you have any questions!

Erika's accounts:
* [Follow on X](https://x.com/ecardenas300)
* [Connect on LinkedIn](https://www.linkedin.com/in/erikacardenas300/)

Patrick's accounts:
* [Follow on X](https://x.com/patloeber)
* [Connect on LinkedIn](https://www.linkedin.com/in/patrick-l%C3%B6ber-403022137/)

Connor's accounts:
* [LinkedIn](https://www.linkedin.com/in/connor-shorten-34923a178/)
* [X](https://x.com/CShorten30)

### Install libraries


```
%pip install -U google-genai
%pip install -U "weaviate-client[agents]"

```

### Import libraries and keys


```
import os

import weaviate
from weaviate_agents.query import QueryAgent

from google import genai
from google.genai import types
```

### Set you API keys and Weaviate URL


```
os.environ["WEAVIATE_URL"] = ""
os.environ["WEAVIATE_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
```

### Create API client


```
client = genai.Client()
```

### Define Query Agent function


```
def query_agent_request(query: str) -> str:
    """
    Send a query to the database and get the response.

    Args:
        query (str): The question or query to search for in the database. This can be any natural language question related to the content stored in the database.

    Returns:
        str: The response from the database containing relevant information.
    """

    # connect to your Weaviate Cloud instance
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        headers={  # add the API key to the model provider from your Weaviate collection, for example `headers={"X-Goog-Studio-Api-Key": os.getenv("GEMINI_API_KEY")}`
        }
    )

    # connect the query agent to your Weaviate collection(s)
    query_agent = QueryAgent(
        client=weaviate_client,
        collections=["WeaviateBlogChunks"]
    )
    return query_agent.run(query).final_answer
```

### Configure Tool


```
config = types.GenerateContentConfig(tools=[query_agent_request])
```

### Query Time


```
prompt = """
You are connected to a database that has a blog post on deploying Weaviate on Docker.
Can you answer how I can Weaviate with Docker?
"""

chat = client.chats.create(model='gemini-2.0-flash', config=config)
response = chat.send_message(prompt)
print(response.text)
```

    [UserWarning: Pydantic serializer warnings: Expected `enum` but got `str` with value `'STRING'` - serialized value may not be as expected, ..., To deploy Weaviate with Docker, you need to:
    
    1.  Install Docker and Docker Compose.
    2.  Obtain the Weaviate Docker image using:
        ```bash
        docker pull cr.weaviate.io/semitechnologies/weaviate:latest
        ```
    3.  Prepare a `docker-compose.yml` file, which you can generate using the Weaviate configuration tool or example files from the documentation.
    4.  Start Weaviate using either:
        *   Directly with Docker:
            ```bash
            docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
            ```
        *   Using Docker Compose:
            ```bash
            docker-compose up -d
            ```
    5.  Access Weaviate at `http://localhost:8080` and configure as needed.
    6.  Check if Weaviate is ready by hitting the readiness endpoint:
        ```bash
        curl localhost:8080/v1/.well-known/ready
        ```
    ]
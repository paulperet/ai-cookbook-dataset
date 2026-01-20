# 🐫 CAMEL Role-Playing **Scraper** for **Report** & **Knowledge Graph** Generation

This notebook demonstrates how to set up and leverage CAMEL's Retrieval-Augmented Generation (RAG) combined with Firecrawl for efficient web scraping, multi-agent role-playing tasks, and knowledge graph construction. We will walk through an example of conducting a comprehensive study of the Turkish shooter in the 2024 Paris Olympics by using Mistral's models.

In this notebook, you'll explore:

*   **CAMEL**: A powerful multi-agent framework that enables Retrieval-Augmented Generation and multi-agent role-playing scenarios, allowing for sophisticated AI-driven tasks.
*   **Mistral**: Utilized for its state-of-the-art language models, which enable tool-calling capabilities to execute external functions, while its powerful embeddings are employed for semantic search and content retrieval.
*   **Firecrawl**: A robust web scraping tool that simplifies extracting and cleaning content from various web pages.
*   **AgentOps**: Track and analysis the running of CAMEL Agents.
*   **Qdrant**: An efficient vector storage system used with CAMEL’s AutoRetriever to store and retrieve relevant information based on vector similarities.
*   **Neo4j**: A leading graph database management system used for constructing and storing knowledge graphs, enabling complex relationships between entities to be mapped and queried efficiently.
*   **DuckDuckGo Search**: Utilized within the SearchToolkit to gather relevant URLs and information from the web, serving as the primary search engine for retrieving initial content.
*   **Unstructured IO:** Used for content chunking, facilitating the management of unstructured data for more efficient processing.

This setup not only demonstrates a practical application but also serves as a flexible framework that can be adapted for various scenarios requiring advanced web information retrieval, AI collaboration, and multi-source data aggregation.

⭐ **Star the Repo**

If you find CAMEL useful or interesting, please consider giving it a star on our [CAMEL GitHub Repo](https://github.com/camel-ai/camel)! Your stars help others find this project and motivate us to continue improving it.

## 📦 Installation

First, install the CAMEL package with all its dependencies:

```python
pip install camel-ai[all]==0.1.6.4
```

[First Entry, ..., Last Entry]

## 🔑 Setting Up API Keys

You'll need to set up your API keys for Mistral AI, Firecrawl and AgentOps. This ensures that the tools can interact with external services securely.

You can go to [here](https://app.agentops.ai/signin) to get **free** API Key from AgentOps

```python
import os
from getpass import getpass

# Prompt for the AgentOps API key securely
agentops_api_key = getpass('Enter your API key: ')
os.environ["AGENTOPS_API_KEY"] = agentops_api_key
```

Your can go to [here](https://console.mistral.ai/api-keys/) to get API Key from Mistral AI with **free** credits.

```python
# Prompt for the API key securely
mistral_api_key = getpass('Enter your API key: ')
os.environ["MISTRAL_API_KEY"] = mistral_api_key
```

Set up the Mistral Large 2 model using the CAMEL ModelFactory. You can also configure other models as needed.

```python
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import MistralConfig

# Set up model
mistral_large_2 = ModelFactory.create(
    model_platform=ModelPlatformType.MISTRAL,
    model_type=ModelType.MISTRAL_LARGE,
    model_config_dict=MistralConfig(temperature=0.2).as_dict(),
)
```

Your can go to [here](https://www.firecrawl.dev/) to get API Key from Firecrawl with **free** credits.

```python
# Prompt for the Firecrawl API key securely
firecrawl_api_key = getpass('Enter your API key: ')
os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key
```

## 🌐 Web Scraping with Firecrawl

Firecrawl is a powerful tool that simplifies web scraping and cleaning content from web pages. In this section, we will scrape content from a specific post on the CAMEL AI website as an example.

```python
from camel.loaders import Firecrawl

firecrawl = Firecrawl()

# Scrape and clean content from a specified URL
response = firecrawl.tidy_scrape(
    url="https://www.camel-ai.org/post/crab"
)

print(response)
```

CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents
============================================================================

> Abstract: Recently, spearheaded by the CAMEL-AI community, a pioneer in open-source multi-agent projects, researchers from institutions such as King Abdullah University of Science and Technology, Oxford University, University of Tokyo, Carnegie Mellon University, Stanford University, and Tsinghua University have developed a cross-platform multimodal agent benchmark framework: CRAB, innovatively enabling agents to operate multiple devices simultaneously.

### Introduction

With the rapid development of multimodal large language models (MLLM), many agents capable of operating graphical user interfaces (GUIs) have emerged this year. Various companies have launched their innovative solutions, creating intense competition. GUI agents, leveraging powerful visual understanding and reasoning abilities of large models, can now efficiently and flexibly complete tasks such as booking appointments, shopping, and controlling smart homes.

**This raises the question: will future agents truly be able to sit in front of a computer and work on my behalf?**

However, in today's era of the Internet of Everything, most work requires the coordination of multiple devices. For example, taking a photo with a phone and then transferring it to a computer for editing involves crossing two different devices (environments). Currently, these GUI agents can only operate on a single device, making what is an easy task for humans exceedingly difficult for today's agents.

Researchers from the CAMEL-AI community noticed this problem and proposed the first cross-environment, multi-device agent benchmark framework—CRAB, the **CR**oss-environment **A**gent **B**enchmark.

Paper link: [https://arxiv.org/abs/2407.01511](https://arxiv.org/abs/2407.01511)

The CAMEL framework ([https://github.com/camel-ai](https://github.com/camel-ai)
) developed by the CAMEL-AI community is one of the earliest open-source multi-agent projects based on large language models. Therefore, community members are researchers and engineers with rich research and practical experience in the field of agents.

In CRAB, the authors not only designed a network-based multi-environment architecture that enables agents to operate multiple devices simultaneously to complete tasks, but also proposed two new technologies to address the issues existing in current agent benchmarks: the graph evaluator and task synthesis. CRAB is not only a brand new benchmark tool but also provides an interaction protocol and its implementation between the environment and agents, which is expected to become an important foundation for agents in practical fields.

The authors believe that CRAB will become one of the standards for evaluating GUI agents in the future, and thus have put considerable effort into improving the framework's usability. The entire codebase adopts a modular design, with the configuration of each environment abstracted into independent and reusable components. Users can easily and quickly build multiple custom environments like building blocks and create their own benchmarks based on them.

For users who wish to evaluate their agents' performance using CRAB, the authors thoughtfully provide a hard disk image on the Google Cloud Platform. With just one click, all the tedious configurations of virtual machines, deep learning models, Python packages, etc., will be completed automatically, allowing users to immediately engage in important experiments.

Currently, CRAB's paper has been published on Arxiv, and the related code and data are open-sourced on the CAMEL-AI community's GitHub.

GitHub repository: [https://github.com/camel-ai/crab](https://github.com/camel-ai/crab)

What does CRAB look like in operation? Let's take a look at the video below:  

The process of testing a multi-agent system on CRAB.

First, the system extracts tasks from the dataset, passes the task instructions to the main agent, and initializes the corresponding graph evaluator in CRAB.

The workflow is a loop: the main agent observes, plans, and instructs the sub-agents; each sub-agent corresponds to an environment. In the diagram, two sub-agents are responsible for the Ubuntu system computer and the Android system phone, respectively, and the sub-agents perform operations in their respective platforms.

The graph evaluator monitors the state of each environment in the platform, updates the progress of the agent in completing the task after each operation, and outputs evaluation metrics.

Having understood how CRAB works, it's time to see how current models perform on this new benchmark. The authors introduced CRAB-Benchmark-v0, a dataset supporting two environments with a total of 100 tasks, and tested several state-of-the-art models.

As shown, the best-performing GPT-4o scored only 35.26 (CR refers to completion rate).

Cross-platform tasks are much more complex than single-platform tasks, and achieving the top score already demonstrates the outstanding ability of the GPT-4 series models in solving practical problems. However, we believe that emerging new methods and models can achieve better scores on CRAB, truly becoming efficient tools for solving real-world problems.

### Cross-Platform Multimodal Agent Evaluation

CRAB provides a comprehensive interactive agent evaluation framework. Through CRAB's foundational setup, agents can operate on various devices and platforms simultaneously, efficiently completing tasks in multiple isolated systems.

The authors propose a new evaluation method called the graph evaluator, which differs from traditional methods based on final goals or action trajectories. The graph evaluator checks the intermediate states of task completion by breaking down tasks into multiple sub-goals.

Each sub-goal is assigned an evaluation function to verify its completion, with each sub-goal being treated as a node in the graph evaluator.

The graph structure describes the precedence and parallel relationships between sub-goals, thus providing fine-grained evaluation metrics; at the same time, the independent evaluation function for each sub-goal inherently adapts to multiple platforms.

The table above compares CRAB with existing frameworks, including several key capabilities involved in testing:

*   **Interactive Environment:** Indicates whether an interactive platform or static dataset is used.
*   **Multimodal Observation:** Indicates whether multimodal input (e.g., screenshots) is supported.
*   **Cross-platform:** Indicates whether multiple operating systems or platforms are supported simultaneously.
*   **Evaluation:** Describes evaluation metrics, divided into goal-based (only checking if the final goal is completed), trajectory-based (comparing the agent's action trajectory with a predefined standard action sequence), multiple (varies by task), or graph-based (each node as an intermediate checkpoint in a DAG).
*   **Task Construction:** Shows the method of constructing tasks in the test dataset, including manually created, LLM-inspired (e.g., LLM generates task drafts but is verified and annotated by humans), template (multiple tasks generated based on manually written templates), or sub-task composition (composing multiple sub-tasks to construct task descriptions and evaluators).

Based on the CRAB framework, the authors developed a benchmark dataset, CRAB Benchmark v0, supporting Android and Ubuntu environments.

The benchmark includes 100 real-world tasks, covering various levels of difficulty for cross-platform and single-platform tasks. Tasks involve a variety of common issues and use multiple practical applications and tools, including but not limited to calendars, emails, maps, web browsers, and terminals, also replicating common coordination methods between smartphones and computers.

### Problem Definition

Assume an agent autonomously executes tasks on a device (such as a desktop). The device is usually equipped with input devices (like a mouse and keyboard) and output devices (like a screen) for human-computer interaction. The authors refer to a device or application with a fixed input method and output method as an environment.

Formally, a single environment can be defined as a reward-free partially observable Markov decision process (Reward-free POMDP), represented by a tuple M := (S, A, T, O), where S represents the state space, A represents the action space, T: S × A → S is the transition function, and O is the observation space.

Considering the collaborative nature of multiple devices in real-world scenarios, multiple platforms can be combined into a set M = {M1, M2, ..., Mn}, where n is the number of platforms, and each platform can be represented as Mj = (Sj, Aj, Tj, Oj).

A task requiring operation across multiple platforms can be formalized as a tuple (M, I, R), where M is the platform set, I is the task goal described in natural language, and R is the task reward function.

The authors call the algorithm responsible for completing the task the agent system. Each agent in the agent system uses a fixed backend model and predefined system prompt, retaining its dialogue history. The agent system can be a single agent or a multi-agent system with multiple agents cooperating.

### Graph of Decomposed Tasks

Breaking down complex tasks into simpler sub-tasks is an effective method for prompting large language models. The authors introduced this concept into the benchmark field, breaking down complex tasks into sub-tasks with precedence and parallel relationships, known as the Graph of Decomposed Tasks (GDT) as shown above.

GDT uses a graph-based task decomposition method: using a DAG structure to represent decomposed sub-tasks.

In GDT, each node is a sub-task, formalized as a tuple (m, i, r), where m specifies the environment for executing the sub-task, i provides natural language instructions, and r represents the reward function. The reward function evaluates the state of environment m and outputs a boolean value to determine if the sub-task is completed. Edges in GDT represent the precedence relationships between sub-tasks.

### Graph Evaluator

To evaluate the capabilities of large language models as agents, most benchmarks rely solely on the final state of the platform after the agent's operations.

Simply judging whether the final goal is successful or failed is obviously not fair; it's like in a math exam, even if you can't solve the big problem, you should get points for writing some solution steps.

Another method is trajectory-based matching, comparing the agent's operations with a predefined standard operation sequence (Label) for each task.

However, in real-world systems, tasks may have multiple valid execution paths. For example, copying a file can be done using a file manager or a command line. Specifying a unique correct path is unfair to agents that achieve the goal in different ways.

Therefore, this paper adopts a graph evaluator synchronized with platform states, tracking the agent's progress through the current state of sub-task completion.

In addition to the traditional success rate (SR), which marks a task as successful only when all sub-tasks are completed, the authors introduced three metrics to measure the agent's performance and efficiency:

1.  **Completion Rate (CR):** Measures the proportion of completed sub-task nodes, calculated as the number of completed nodes/total nodes. This metric intuitively reflects the agent's progress on the given task.
2.  **Execution Efficiency (EE):** Calculated as CR/A, where A represents the number of actions performed, reflecting the agent's task execution efficiency.
3.  **Cost Efficiency (CE):** Calculated as CR/T, where T is the total number of tokens used by the agent, evaluating the agent's efficiency in terms of cost.

### Experiments

To run in Crab Benchmark-v0, the backend model needs to support the following features:

1.  Support multimodal mixed input: The system provides both screenshots and text instructions as prompts.
2.  Support multi-turn conversations: All tasks require the agent to perform multiple operations, so historical messages must be stored in context.
3.  Generate structured output through Function Call or similar functions: Used to execute operations in the environment.

The experiments selected four multimodal models that meet these criteria: GPT-4o, GPT-4 Turbo, Gemini 1.5 Pro, and Claude 3 Opus.

To compare the performance differences between multi-agent and single-agent systems, the paper designed three different agent systems:

1.  **Single Agent:** A single agent handles the entire process from understanding the task, observing the environment, planning, to executing actions.
2.  **Multi-agent by Functionality:** Consists of a main agent and a sub-agent. The main agent observes the environment and provides instructions to the sub-agent, which translates the instructions into specific operations.
3.  **Multi-agent by Environment:** Consists of a main agent and multiple sub-agents. Each sub-agent is responsible for one environment. The main agent understands the task, plans the execution process, and provides instructions to each sub-agent. The sub-agents observe their respective environments and translate the instructions into specific operations.

The combinations of different agent systems and backend models provide multiple dimensions for comparison. Additionally, the paper compares the performance of models in tasks involving different platforms:

**Ubuntu single-platform tasks:**

**Android single-platform tasks:**

**Cross-platform tasks:**

Through data analysis, the paper draws several conclusions:

**Performance Differences Among Models:**

1.  GPT-4o has the highest success and completion rates overall.
2.  GPT-4 TURBO performs better in cost efficiency (CE) than other models.
3.  Gemini 1.5 Pro and Claude 3 Opus struggle with task completion, finding it almost impossible to complete tasks.**‍**

**Efficiency Metrics Reflect Different Characteristics of Models:**

1.  GPT-4 TURBO shows excellent cost efficiency in the single-agent mode, demonstrating cost-effective performance.
2.  GPT-4o maintains a balance between efficiency and performance, especially in the single-agent mode.
3.  Gemini 1.5 Pro shows low efficiency and incomplete cost efficiency metrics, mainly due to its low completion rate.

**Evaluation of Termination Reasons Indicates Areas for Improvement:**

1.  All models have a high percentage of reaching the step limit (RSL), indicating that agents often run out of steps without achieving the final goal.
2.  Gemini 1.5 Pro has a high rate of invalid actions (IA), highlighting its inability to stably generate the correct format for interacting with the environment.
3.  The false completion (FC) rate in multi-agent systems is higher than in single-agent systems, indicating that message loss during communication between multiple agents can easily cause the executing sub-agent to misjudge.

### 🐫 Thanks from everyone at CAMEL-AI

Hello there, passionate AI enthusiasts! 🌟 We are 🐫 CAMEL-AI.org, a global coalition of students, researchers, and engineers dedicated to advancing the frontier of AI and fostering a harmonious relationship between agents and humans.

**📘 Our Mission:** To harness the potential of AI agents in crafting a brighter and more inclusive future for all. Every contribution we receive helps push the boundaries of what’s possible in the AI realm.

**🙌 Join Us:** If you believe in a world where AI and humanity coexist and thrive, then you’re in the right place. Your support can make a significant difference. Let’s build the AI society of tomorrow, together!

*   Find all our updates on [X](https://twitter.com/CamelAIOrg)
    .
*   Make sure to star our [GitHub](https://github.com/camel-ai)
     repositories.
*   Join our [Discord
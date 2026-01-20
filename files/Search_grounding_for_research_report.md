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

# Search tool with Gemini 2+

In this tutorial you are going to leverage the latest [search tool](https://ai.google.dev/gemini-api/docs/google-search) of the Gemini model to write a company report. Note that the search tool is a paid ony feature and this tutorial does not work with a free tier API key.

You may be asking, why does one need to use the search tool for this purpose? Well, as you may be aware, today's business world evolves very fast and LLMs generally are not trained frequently enough to capture the latest updates. Luckily Google search comes to the rescue. Google search is built to provide accurate and nearly realtime information and can help us fulfill this task perfectly.

Note that while Gemini 1.5 models offered the [search grounding](https://ai.google.dev/gemini-api/docs/grounding) feature which may also help you achieve similar results (see [a previous notebook](https://github.com/google-gemini/cookbook/blob/gemini-1.5-archive/examples/Search_grounding_for_research_report.ipynb) using search grounding), the new search tool in Gemini 2.0 and later models is much more powerful and easier to use, and should be prioritized over search grounding. It is also possible to use multiple tools together (for example, search tool and function calling).

## Setup

### Install SDK

The new **[Google Gen AI SDK](https://ai.google.dev/gemini-api/docs/sdks)** provides programmatic access to the Gemini models using both the [Google AI for Developers](https://ai.google.dev/gemini-api/docs) and [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview) APIs. With a few exceptions, code that runs on one platform will run on both. This means that you can prototype an application using the Developer API and then migrate the application to Vertex AI without rewriting your code.

More details about this new SDK on the [documentation](https://ai.google.dev/gemini-api/docs/sdks) or in the [Getting started](../quickstarts/Get_started.ipynb) notebook.


```
%pip install -U -q google-genai
```

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
```

### Import the libraries


```
from google import genai
from google.genai.types import GenerateContentConfig, Tool
from IPython.display import display, HTML, Markdown
import io
import json
import re
```

### Initialize SDK client

With the new SDK you now only need to initialize a client with you API key (or OAuth if using [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview)). The model is now set in each call.


```
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Select a model

Search tool is a new feature in the Gemini models that automatically retrieves accurate and grounded artifacts from the web for developers to further process. Unlike the search grounding in the Gemini 1.5 models, you do not need to set the dynamic retrieval threshold.

For more information about all Gemini models, check the [documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for extended information on each of them.



```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Write the report with Gemini

### Select a target company to research

Next you will use Alphabet as an example research target.


```
COMPANY = 'Alphabet' # @param {type:"string"}
```

### Have Gemini plan for the task and write the report

The newest Gemini models are a huge step up from previous models since they can reason, plan, search and write in one go. In this case, you will only give Gemini a prompt to do all of these and the model will finish your task seamlessly. You will also using output streaming, which streams the response as it is being generated, and the model will return chunks of the response as soon as they are generated. If you would like the SDK to return a response after the model completes the entire generation process,  you can use the non-streaming approach as well.

Note that you must enable Google Search Suggestions, which help users find search results corresponding to a grounded response, and [display the search queries](https://ai.google.dev/gemini-api/docs/grounding/search-suggestions#display-requirements) that are included in the grounded response's metadata. You can find [more details](https://ai.google.dev/gemini-api/terms#grounding-with-google-search) about this requirement.


```
sys_instruction = """You are an analyst that conducts company research.
You are given a company name, and you will work on a company report. You have access
to Google Search to look up company news, updates and metrics to write research reports.

When given a company name, identify key aspects to research, look up that information
and then write a concise company report.

Feel free to plan your work and talk about it, but when you start writing the report,
put a line of dashes (---) to demarcate the report itself, and say nothing else after
the report has finished.
"""

config = GenerateContentConfig(system_instruction=sys_instruction, tools=[Tool(google_search={})], temperature=0)
response_stream = client.models.generate_content_stream(
    model=MODEL, config=config, contents=[COMPANY])

report = io.StringIO()
for chunk in response_stream:
  candidate = chunk.candidates[0]

  for part in candidate.content.parts:
    if part.text:
      display(Markdown(part.text))

      # Find and save the report itself.
      if m := re.search('(^|\n)-+\n(.*)$', part.text, re.M):
          # Find the starting '---' line and start saving.
          report.write(m.group(2))
      elif report.tell():
        # If there's already something in the buffer, keep recording.
        report.write(part.text)

    else:
      print(json.dumps(part.model_dump(exclude_none=True), indent=2))

  # You must enable Google Search Suggestions
  if gm := candidate.grounding_metadata:
    if sep := gm.search_entry_point:
      display(HTML(sep.rendered_content))
```

Okay, I will research Alphabet and create a company report. Here's my plan:

1.  **Company Overview:** I'll start by gathering basic information about Alphabet, such as its founding date, headquarters, and what it does.
2.  **Business Segments:** I'll identify the different business segments that Alphabet operates in (e.g., Google Services, Google Cloud, Other Bets).
3.  **Recent News and Developments:** I'll look for recent news articles and press releases to understand any significant updates, product launches, or strategic shifts.
4.  **Financial Performance:** I'll try to find information about Alphabet's recent financial performance, including revenue, profit, and key metrics.
5.  **Stock Information:** I'll check the current stock price and any recent stock-related news.
6.  **Key People:** I'll identify key executives and board members.
7.  **Competitive Landscape:** I'll briefly touch on the competitive landscape and Alphabet's main competitors.

Now, let's start with the searches.

Okay, I've gathered a lot of information about Alphabet. Here's a company report:

---
**Company Report: Alphabet Inc.**

**Overview:**

Alphabet Inc. is a multinational technology conglomerate holding company headquartered in Mountain View, California. It was created through a restructuring of Google on October 2, 2015, becoming the parent company of Google and several former Google subsidiaries. Alphabet is the world's second-largest technology company by revenue, after Apple, and one of the world's most valuable companies. It is considered one of the Big Five American information technology companies, alongside Amazon, Apple, Meta, and Microsoft.

**Founding and Restructuring:**

While Google was founded in 1998 by Larry Page and Sergey Brin, Alphabet Inc. was formed on October 2, 2015, as a restructuring to make the core Google business "cleaner and more accountable" while allowing greater autonomy to other ventures. In December 2019, Larry Page and Sergey Brin stepped down from their executive roles, and Sundar Pichai, who was already CEO of Google, became CEO of both Google and Alphabet. Page and Brin remain employees, board members, and controlling shareholders.

**Business Segments:**

Alphabet operates through three main reportable segments:

*   **Google Services:** This segment includes Google's core internet products such as ads, Android, Chrome, hardware, Google Cloud, Google Maps, Google Play, Search, and YouTube. This is the most profitable segment, with revenue primarily generated from advertising on Google Search and YouTube, as well as subscriptions and smartphone sales.
*   **Google Cloud:** This segment provides cloud services for businesses, including office software and computing platforms. Google Cloud's quarterly revenues have recently surpassed $10 billion, and it is showing strong growth.
*   **Other Bets:** This segment includes various cutting-edge technology innovation ventures at different stages of development, such as Access, Calico, CapitalG, GV, Verily, Waymo, and X. This segment often posts operating losses.

**Recent News and Developments:**

*   Alphabet's stock recently reached an all-time high after unveiling its latest AI model, Gemini 2.0, and showcasing a new quantum computing chip called Willow.
*   Google has launched Gemini, formerly known as Bard, an AI chatbot that supports over 40 languages.
*   Alphabet's Q3 2024 results showed a 15% increase in consolidated revenue year-over-year, reaching $88.3 billion. Google Services revenue increased by 13% to $76.5 billion.
*   Google Cloud's revenue exceeded $10 billion in a recent quarter, with over $1 billion in operating profit.
*   Alphabet announced a cash dividend of $0.20 per share in July 2024.

**Financial Performance:**

*   In Q3 2024, Alphabet's consolidated revenue was $88.3 billion, a 15% increase year-over-year.
*   Net income for Q3 2024 was $26.3 billion, compared to $19.7 billion a year ago.
*   For the nine months ended September 30, 2024, sales were $253.5 billion, compared to $221.1 billion a year ago.
*   Google Cloud's revenue has surpassed $10 billion in a recent quarter.
*   Alphabet's revenue is primarily driven by Google Services, particularly advertising.

**Stock Information:**

*   Alphabet has two classes of stock: GOOG (Class C shares without voting rights) and GOOGL (Class A common stock).
*   As of today, December 17, 2024, the price of GOOG is around $198.16, and it has increased by 4.31% in the past 24 hours.
*   The stock reached an all-time high on December 10, 2024.
*   Analysts' price targets for GOOG range from $185 to $240.

**Key People:**

*   **CEO:** Sundar Pichai
*   **Co-Founders:** Larry Page and Sergey Brin (remain board members and controlling shareholders)
*   **Chief Financial Officer:** Anat Ashkenazi
*   **Other Key Executives:** Philipp Schindler (Chief Business Officer, Google), Prabhakar Raghavan (Chief Technologist, Google), Ruth M. Porat (President and Chief Investment Officer; CFO)

**Competitive Landscape:**

Alphabet's main competitors include:

*   **Other Big Tech Companies:** Apple, Microsoft, Meta, Amazon, IBM
*   **Software Companies:** SAP, Palantir Technologies, Shopify, AppLovin, Infosys, CrowdStrike, Atlassian, Trade Desk, NetEase, Snowflake
*   **Other Competitors:** ByteDance, Zoom, Salesforce, Tencent, Adobe

**Additional Information:**

*   Alphabet's headquarters is located at 1600 Amphitheatre Parkway in Mountain View, California, also known as the Googleplex.
*   Alphabet has a global presence with employees across six continents.
*   Alphabet is involved in investing in infrastructure, data management, analytics, and artificial intelligence (AI).

This report provides a comprehensive overview of Alphabet Inc. as of December 17, 2024.
---

Very impressive! Gemini starts by planning for the task, performing relevant searches and streams out a report for you.

### Final output

Render the final output.


```
display(Markdown(report.getvalue().replace('$', r'\$')))  # Escape $ signs for better MD rendering
display(HTML(sep.rendered_content))
```

**Company Report: Alphabet Inc.:**

Alphabet Inc. is a multinational technology conglomerate holding company headquartered in Mountain View, California. It was created through a restructuring of Google on October 2, 2015, becoming the parent company of Google and several former Google subsidiaries. Alphabet is the world's second-largest technology company by revenue, after Apple, and one of the world's most valuable companies. It is considered one of the Big Five American information technology companies, alongside Amazon, Apple, Meta, and Microsoft.

**Founding and Restructuring:**

While Google was founded in 1998 by Larry Page and Sergey Brin, Alphabet Inc. was formed on October 2, 2015, as a restructuring to make the core Google business "cleaner and more accountable" while allowing greater autonomy to other ventures. In December 2019, Larry Page and Sergey Brin stepped down from their executive roles, and Sundar Pichai, who was already CEO of Google, became CEO of both Google and Alphabet. Page and Brin remain employees, board members, and controlling shareholders.

**Business Segments:**

Alphabet operates through three main reportable segments:

*   **Google Services:** This segment includes Google's core internet products such as ads, Android, Chrome, hardware, Google Cloud, Google Maps, Google Play, Search, and YouTube. This is the most profitable segment, with revenue primarily generated from advertising on Google Search and YouTube, as well as subscriptions and smartphone sales.
*   **Google Cloud:** This segment provides cloud services for businesses, including office software and computing platforms. Google Cloud's quarterly revenues have recently surpassed \$10 billion, and it is showing strong growth.
*   **Other Bets:** This segment includes various cutting-edge technology innovation ventures at different stages of development, such as Access, Calico, CapitalG, GV, Verily, Waymo, and X. This segment often posts operating losses.

**Recent News and Developments:**

*   Alphabet's stock recently reached an all-time high after unveiling its latest AI model, Gemini 2.0, and showcasing a new quantum computing chip called Willow.
*   Google has launched Gemini, formerly known as Bard, an AI chatbot that supports over 40 languages.
*   Alphabet's Q3 2024 results showed a 15% increase in consolidated revenue year-over-year, reaching \$88.3 billion. Google Services revenue increased by 13% to \$76.5 billion.
*   Google Cloud's revenue exceeded \$10 billion in a recent quarter, with over \$1 billion in operating profit.
*   Alphabet announced a cash dividend of \$0.20 per share in July 2024.

**Financial Performance:**

*   In Q3 2024, Alphabet's consolidated revenue was \$88.3 billion, a 15% increase year-over-year.
*   Net income for Q3 2024 was \$26.3 billion, compared to \$19.7 billion a year ago.
*   For the nine months ended September 30, 2024, sales were \$253.5 billion, compared to \$221.1 billion a year ago.
*   Google Cloud's revenue has surpassed \$10 billion in a recent quarter.
*   Alphabet's revenue is primarily driven by Google Services, particularly advertising.

**Stock Information:**

*   Alphabet has two classes of stock: GOOG (Class C shares without voting rights) and GOOGL (Class A common stock).
*   As of today, December 17, 2024, the price of GOOG is around \$198.16, and it has increased by 4.31% in the past 24 hours.
*   The stock reached an all-time high on December 10, 2024.
*   Analysts' price targets for GOOG range from \$185 to \$240.

**Key People:**

*   **CEO:** Sundar Pichai
*   **Co-Founders:** Larry Page and Sergey Brin (remain board members and controlling shareholders)
*   **Chief Financial Officer:** Anat Ashkenazi
*   **Other Key Executives:** Philipp Schindler (Chief Business Officer, Google), Prabhakar Raghavan (Chief Technologist, Google), Ruth M. Porat (President and Chief Investment Officer; CFO)

**Competitive Landscape:**

Alphabet's main competitors include:

*   **Other Big Tech Companies:** Apple, Microsoft, Meta, Amazon, IBM
*   **Software Companies:** SAP, Palantir Technologies, Shopify, AppLovin, Infosys, CrowdStrike, Atlassian, Trade Desk, NetEase, Snowflake
*   **Other Competitors:** ByteDance, Zoom, Salesforce, Tencent, Adobe

**Additional Information:**

*   Alphabet's headquarters is located at 1600 Amphitheatre Parkway in Mountain View, California, also known as the Googleplex.
*   Alphabet has a global presence with employees across six continents.
*   Alphabet is involved in investing in infrastructure, data management, analytics, and artificial intelligence (AI).

This report provides a comprehensive overview of Alphabet Inc

As you can see, the Gemini model is able to write a concise, accurate and well-structured research report for us. All the information in the report is factual and up-to-date.

Note that this tutorial is meant to showcase the capability of the new search tool and inspire interesting use cases, not to build a production research report generator. If you are looking to use a tool, please check
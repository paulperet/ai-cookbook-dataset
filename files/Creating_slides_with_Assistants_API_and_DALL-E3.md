# Creating slides with the Assistants API (GPT-4), and DALL·E-3

This notebook illustrates the use of the new [Assistants API](https://platform.openai.com/docs/assistants/overview) (GPT-4), and DALL·E-3 in crafting informative and visually appealing slides. 
Creating slides is a pivotal aspect of many jobs, but can be laborious and time-consuming. Additionally, extracting insights from data and articulating them effectively on slides can be challenging.  This cookbook recipe will demonstrate how you can utilize the new Assistants API to facilitate the end to end slide creation process for you without you having to touch Microsoft PowerPoint or Google Slides, saving you valuable time and effort!

## 0. Setup


```python
from IPython.display import display, Image
from openai import OpenAI
import os
import pandas as pd
import json
import io
from PIL import Image
import requests

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

#Lets import some helper functions for assistants from https://cookbook.openai.com/examples/assistants_api_overview_python
def show_json(obj):
    display(json.loads(obj.model_dump_json()))

def submit_message(assistant_id, thread, user_message,file_ids=None):
    params = {
        'thread_id': thread.id,
        'role': 'user',
        'content': user_message,
    }
    if file_ids:
        params['file_ids']=file_ids

    client.beta.threads.messages.create(
        **params
)
    return client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant_id,
)

def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id)

```

## 1. Creating the content

In this recipe, we will be creating a brief fictional presentation for the quarterly financial review of our company, NotReal Corporation. We want to highlight some key trends we are seeing that are affecting the profitability of our company. Let's say we have the some financial data at our disposal. Let's load in the data, and take a look...


```python
financial_data_path = 'data/NotRealCorp_financial_data.json'
financial_data = pd.read_json(financial_data_path)
financial_data.head(5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Quarter</th>
      <th>Distribution channel</th>
      <th>Revenue ($M)</th>
      <th>Costs ($M)</th>
      <th>Customer count</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>Q1</td>
      <td>Online Sales</td>
      <td>1.50</td>
      <td>1.301953</td>
      <td>150</td>
      <td>2021 Q1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021</td>
      <td>Q1</td>
      <td>Direct Sales</td>
      <td>1.50</td>
      <td>1.380809</td>
      <td>151</td>
      <td>2021 Q1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021</td>
      <td>Q1</td>
      <td>Retail Partners</td>
      <td>1.50</td>
      <td>1.348246</td>
      <td>152</td>
      <td>2021 Q1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021</td>
      <td>Q2</td>
      <td>Online Sales</td>
      <td>1.52</td>
      <td>1.308608</td>
      <td>152</td>
      <td>2021 Q2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>Q2</td>
      <td>Direct Sales</td>
      <td>1.52</td>
      <td>1.413305</td>
      <td>153</td>
      <td>2021 Q2</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, this data has quarterly revenue, costs and customer data across different distribution channels. Let's create an Assistant
that can act as a personal analyst and make a nice visualization for our PowerPoint!

First, we need to upload our file so our Assistant can access it.


```python
file = client.files.create(
  file=open('data/NotRealCorp_financial_data.json',"rb"),
  purpose='assistants',
)

```

Now, we're ready to create our Assistant. We can instruct our assistant to act as a data scientist, and take any queries we give it and run the necessary code to output the proper data visualization. The instructions parameter here is akin to system instructions in the ChatCompletions endpoint, and can help guide the assistant. We can also turn on the tool of Code Interpreter, so our Assistant will be able to code. Finally, we can specifiy any files we want to use, which in this case is just the `financial_data` file we created above.


```python
assistant = client.beta.assistants.create(
  instructions="You are a data scientist assistant. When given data and a query, write the proper code and create the proper visualization",
  model="gpt-4-1106-preview",
  tools=[{"type": "code_interpreter"}],
  file_ids=[file.id]
)

```

Let's create a thread now, and as our first request ask the Assistant to calculate quarterly profits, and then plot the profits by distribution channel over time. The assistant will automatically calculate the profit for each quarter, and also create a new column combining quarter and year, without us having to ask for that directly. We can also specify the colors of each line.


```python
thread = client.beta.threads.create(
  messages=[
    {
      "role": "user",
      "content": "Calculate profit (revenue minus cost) by quarter and year, and visualize as a line plot across the distribution channels, where the colors of the lines are green, light red, and light blue",
      "file_ids": [file.id]
    }
  ]
)

```

No we can execute the run of our thread


```python

run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

```

We can now start a loop that will check if the image has been created. Note: This may take a few minutes


```python
messages = client.beta.threads.messages.list(thread_id=thread.id)

```


```python
import time

while True:
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    try:
        #See if image has been created
        messages.data[0].content[0].image_file
        #Sleep to make sure run has completed
        time.sleep(5)
        print('Plot created!')
        break
    except:
        time.sleep(10)
        print('Assistant still working...')

```

    [Assistant still working..., ..., Plot created!]


Let's see the messages the Assistant added.


```python
messages = client.beta.threads.messages.list(thread_id=thread.id)
[message.content[0] for message in messages.data]

```




    [MessageContentImageFile(image_file=ImageFile(file_id='file-0rKABLygI02MgwwhpgWdRFY1'), type='image_file'),
     MessageContentText(text=Text(annotations=[], value="The profit has been calculated for each distribution channel by quarter and year. Next, I'll create a line plot to visualize these profits. As specified, I will use green for the 'Online Sales', light red for 'Direct Sales', and light blue for 'Retail Partners' channels. Let's create the plot."), type='text'),
     MessageContentText(text=Text(annotations=[], value="The JSON data has been successfully restructured into a tabular dataframe format. It includes the year, quarter, distribution channel, revenue, costs, customer count, and a combined 'Time' representation of 'Year Quarter'. Now, we have the necessary information to calculate the profit (revenue minus cost) by quarter and year.\n\nTo visualize the profit across the different distribution channels with a line plot, we will proceed with the following steps:\n\n1. Calculate the profit for each row in the dataframe.\n2. Group the data by 'Time' (which is a combination of Year and Quarter) and 'Distribution channel'.\n3. Aggregate the profit for each group.\n4. Plot the aggregated profits as a line plot with the distribution channels represented in different colors as requested.\n\nLet's calculate the profit for each row and then continue with the visualization."), type='text'),
     MessageContentText(text=Text(annotations=[], value='The structure of the JSON data shows that it is a dictionary with "Year", "Quarter", "Distribution channel", and potentially other keys that map to dictionaries containing the data. The keys of the inner dictionaries are indices, indicating that the data is tabular but has been converted into a JSON object.\n\nTo properly convert this data into a DataFrame, I will restructure the JSON data into a more typical list of dictionaries, where each dictionary represents a row in our target DataFrame. Subsequent to this restructuring, I can then load the data into a Pandas DataFrame. Let\'s restructure and load the data.'), type='text'),
     MessageContentText(text=Text(annotations=[], value="The JSON data has been incorrectly loaded into a single-row DataFrame with numerous columns representing each data point. This implies the JSON structure is not as straightforward as expected, and a direct conversion to a flat table is not possible without further processing.\n\nTo better understand the JSON structure and figure out how to properly normalize it into a table format, I will print out the raw JSON data structure. We will analyze its format and then determine the correct approach to extract the profit by quarter and year, as well as the distribution channel information. Let's take a look at the JSON structure."), type='text'),
     MessageContentText(text=Text(annotations=[], value="It seems that the file content was successfully parsed as JSON, and thus, there was no exception raised. The variable `error_message` is not defined because the `except` block was not executed.\n\nI'll proceed with displaying the data that was parsed from JSON."), type='text'),
     MessageContentText(text=Text(annotations=[], value="It appears that the content of the dataframe has been incorrectly parsed, resulting in an empty dataframe with a very long column name that seems to contain JSON data rather than typical CSV columns and rows.\n\nTo address this issue, I will take a different approach to reading the file. I will attempt to parse the content as JSON. If this is not successful, I'll adjust the loading strategy accordingly. Let's try to read the contents as JSON data first."), type='text'),
     MessageContentText(text=Text(annotations=[], value="Before we can calculate profits and visualize the data as requested, I need to first examine the contents of the file that you have uploaded. Let's go ahead and read the file to understand its structure and the kind of data it contains. Once I have a clearer picture of the dataset, we can proceed with the profit calculations. I'll begin by loading the file into a dataframe and displaying the first few entries to see the data schema."), type='text'),
     MessageContentText(text=Text(annotations=[], value='Calculate profit (revenue minus cost) by quarter and year, and visualize as a line plot across the distribution channels, where the colors of the lines are green, light red, and light blue'), type='text')]



We can see that the last message (latest message is shown first) from the assistant contains the image file we are looking for. An interesting note here is that the Assistant was able to attempt several times to parse the JSON data, as the first parsing was unsuccessful, demonstrating the assistant's adaptability.


```python
# Quick helper function to convert our output file to a png
def convert_file_to_png(file_id, write_path):
    data = client.files.content(file_id)
    data_bytes = data.read()
    with open(write_path, "wb") as file:
        file.write(data_bytes)

```


```python
plot_file_id = messages.data[0].content[0].image_file.file_id
image_path = "../images/NotRealCorp_chart.png"
convert_file_to_png(plot_file_id,image_path)

#Upload
plot_file = client.files.create(
  file=open(image_path, "rb"),
  purpose='assistants'
)

```

Nice! So, with just one sentence, we were able to have our assistant use code interpreter to
calculate the profitability, and graph the three lineplots of the various distribution channels.
Now we have a nice visual for our slide, but we want some insights to go along with it.

## 2. Generating insights

To get insights from our image, we simply need to add a new message to our thread. Our Assistant will know to use the message history to give us some concise takeaways from the visual provided. 


```python
submit_message(assistant.id,thread,"Give me two medium length sentences (~20-30 words per sentence) of the \
      most important insights from the plot you just created.\
             These will be used for a slide deck, and they should be about the\
                     'so what' behind the data."
)

```




    Run(id='run_NWoygMcBfHUr58fCE4Cn6rxN', assistant_id='asst_3T362kLlTyAq0FUnkvjjQczO', cancelled_at=None, completed_at=None, created_at=1701827074, expires_at=1701827674, failed_at=None, file_ids=['file-piTokyHGllwGITzIpoG8dok3'], instructions='You are a data scientist assistant. When given data and a query, write the proper code and create the proper visualization', last_error=None, metadata={}, model='gpt-4-1106-preview', object='thread.run', required_action=None, started_at=None, status='queued', thread_id='thread_73TgtFoJMlEJvb13ngjTnAo3', tools=[ToolAssistantToolsCode(type='code_interpreter')])



Now, once the run has completed, we can see the latest message


```python
# Hard coded wait for a response, as the assistant may iterate on the bullets.
time.sleep(10)
response = get_response(thread)
bullet_points = response.data[0].content[0].text.value
print(bullet_points)

```

    The plot reveals a consistent upward trend in profits for all distribution channels, indicating successful business growth over time. Particularly, 'Online Sales' shows a notable increase, underscoring the importance of digital presence in revenue generation.


Cool! So our assistant was able to identify the noteworthy growth in Online Sales profit, and infer that this shows the importance of a large digital presence. Now let's get a compelling title for the slide.


```python
submit_message(assistant.id,thread,"Given the plot and bullet points you created,\
 come up with a very brief title for a slide. It should reflect just the main insights you came up with."
)

```




    Run(id='run_q6E85J31jCw3QkHpjJKl969P', assistant_id='asst_3T362kLlTyAq0FUnkvjjQczO', cancelled_at=None, completed_at=None, created_at=1701827084, expires_at=1701827684, failed_at=None, file_ids=['file-piTokyHGllwGITzIpoG8dok3'], instructions='You are a data scientist assistant. When given data and a query, write the proper code and create the proper visualization', last_error=None, metadata={}, model='gpt-4-1106-preview', object='thread.run', required_action=None, started_at=None, status='queued', thread_id='thread_73TgtFoJMlEJvb13ngjTnAo3', tools=[ToolAssistantToolsCode(type='code_interpreter')])



And the title is:


```python
#Wait as assistant may take a few steps
time.sleep(10)
response = get_response(thread)
title = response.data[0].content[0].text.value
print(title)

```

    "Ascending Profits & Digital Dominance"


## 3. DALL·E-3 title image

Nice, now we have a title, a plot and two bullet points. We're almost ready to put this all on a slide, but as a final step, let's have DALL·E-3 come up with an image to use as the title slide of the presentation. 
*Note:* DALL·E-3 is not yet available within the assistants API but is coming soon! 
We'll feed in a brief description of our company (NotRealCorp) and have DALL·E-3 do the rest!


```python
company_summary = "NotReal Corp is a prominent hardware company that manufactures and sells processors, graphics cards and other essential computer hardware."

```


```python
response = client.images.generate(
  model='dall-e-3',
  prompt=f"given this company summary {company_summary}, create an inspirational \
    photo showing the growth and path forward. This will be used at a quarterly\
       financial planning meeting",
       size="1024x1024",
       quality="hd",
       n=1
)
image_url = response.data[0].url

```

Cool, now we can add this image to our thread. First, we can save the image locally, then upload it to the assistants API using the `File` upload endpoint. Let's also take a look at our image


```python
dalle_img_path = '../images/dalle_image.png'
img = requests.get(image_url)

#Save locally
with
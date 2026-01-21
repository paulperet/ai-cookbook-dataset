# How to Transcribe Documents with Claude 3

Claude 3 excels at extracting and structuring information from unstructured sources like images and PDFs. In this guide, you'll learn how to leverage its vision capabilities for various transcription tasks—from simple text extraction to complex document analysis and structured data conversion.

## Prerequisites

First, install the required packages and set up your environment.

```bash
pip install anthropic IPython
```

```python
import base64
from anthropic import Anthropic

# Initialize the Anthropic client
client = Anthropic()
MODEL_NAME = "claude-3-opus-20240229"  # Update to the latest available model

def get_base64_encoded_image(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base64_encoded_data = base64.b64encode(binary_data)
        return base64_encoded_data.decode("utf-8")
```

## Step 1: Transcribe Typed Text from an Image

Claude 3 can precisely extract specific content from images, such as code snippets, by following detailed instructions.

```python
# Prepare the message with the image and instruction
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/transcribe/stack_overflow.png"),
                },
            },
            {"type": "text", "text": "Transcribe the code in the answer. Only output the code."},
        ],
    }
]

# Send the request to Claude
response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=2048,
    messages=message_list
)

print(response.content[0].text)
```

**Expected Output:**
```
import os
import base64

image = 'test.jpg'

encoded_string = ""
with open(image, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
file = encoded_string
```

## Step 2: Transcribe Handwritten Text

Claude 3 also performs well with handwritten text, making it useful for digitizing notes, forms, or sketches.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/transcribe/school_notes.png"),
                },
            },
            {
                "type": "text",
                "text": "Transcribe this text. Only output the text and nothing else.",
            },
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Expected Output:**
```
Levels of Cellular Organization
1) Cells group together to make tissue.
2) Tissues group together to make an organ.
3) Organs group together to make an organ system
4) Organ systems group together to make an organism

Organism -> a living thing that can
carry out life processes by itself.

- Multicellular organisms have specialized
cells to perform specific functions.
> This makes them more efficient
and typically have a longer life span.

Tissue = a group of similar cells
that perform a common function.
1) Animals are made of four
basic types of tissue
> nervous, epithelial, connective,
and muscle
2) Plants have three types
of tissue
> transport, protective, and
ground
```

## Step 3: Transcribe Mixed-Content Forms

Many real-world documents contain a mix of typed fields and handwritten entries. Claude can accurately transcribe these complex layouts.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": get_base64_encoded_image("../images/transcribe/vehicle_form.jpg"),
                },
            },
            {"type": "text", "text": "Transcribe this form exactly."},
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Expected Output:**
```
VEHICLE INCIDENT REPORT FORM

Use this form to report accidents, injuries, medical situations, criminal activities, traffic incidents, or student behavior incidents. If possible, a report should be completed within 24 hours of the event.

Date of Report: 02/29, 2024

PERSON INVOLVED

Full Name: John Doe Address: 123 Main St

Identification: ■ Driver's License No. 474921 □ Passport No. ___________
□ Other: ____________________

Phone: (678) 999-8212 E-Mail: john@gmail.com

THE INCIDENT

Date of Incident: 02/29/2024 ■ Time: 9:01 ■ AM □ PM

Location: Corner of 2nd and 3rd

Describe the Incident: Red car t-boned blue car
_______________________________________________________
_______________________________________________________

INJURIES

Was anyone injured? □ Yes ■ No

If yes, describe the injuries: ________________________________________
_______________________________________________________________
_______________________________________________________________

WITNESSES

Were there witnesses to the incident? □ Yes ■ No

If yes, enter the witnesses' names and contact info: __________________________
_______________________________________________________________
_______________________________________________________________

Page 1 of 2
```

## Step 4: Perform Document Question Answering

Beyond transcription, you can query Claude about the content of a document to extract specific insights.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": get_base64_encoded_image("../images/transcribe/page.jpeg"),
                },
            },
            {"type": "text", "text": "Which is the most critical issue for live rep support?"},
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Expected Output:**
```
According to the hierarchy of importance pyramid for Live Rep Support shown in the image, the most critical issue is Product Quality/Liability Issues. This is positioned at the very bottom of the pyramid, indicating it is the most critical or important issue for live rep support to handle.
```

## Step 5: Convert Unstructured Information to Structured JSON

Claude can transform visual data, like an organizational chart, into structured JSON format, enabling easy integration with other systems.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": get_base64_encoded_image("../images/transcribe/org_chart.jpeg"),
                },
            },
            {
                "type": "text",
                "text": "Turn this org chart into JSON indicating who reports to who. Only output the JSON and nothing else.",
            },
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Expected Output:**
```json
{
  "President": {
    "name": "John Smith",
    "directReports": [
      {
        "name": "Susan Jones",
        "title": "VP Marketing",
        "directReports": [
          {
            "name": "Alice Johnson",
            "title": "Manager"
          },
          {
            "name": "Tim Moore",
            "title": "Manager"
          }
        ]
      },
      {
        "name": "Rachel Parker",
        "title": "VP Sales",
        "directReports": [
          {
            "name": "Michael Gross",
            "title": "Manager"
          },
          {
            "name": "Kim Dole",
            "title": "Manager"
          }
        ]
      },
      {
        "name": "Tom Allen",
        "title": "VP Production",
        "directReports": [
          {
            "name": "Kathy Roberts",
            "title": "Manager"
          },
          {
            "name": "Betsy Foster",
            "title": "Manager"
          }
        ]
      }
    ]
  }
}
```

## Summary

You've now learned how to use Claude 3 for a variety of document transcription tasks:
1. **Precise text extraction** from images containing code or specific elements.
2. **Handwriting transcription** for digitizing notes.
3. **Form processing** that handles mixed typed and handwritten content.
4. **Document QA** to answer specific questions about visual content.
5. **Structured data extraction** to convert diagrams like org charts into JSON.

The key is providing clear, specific instructions in your prompts to guide Claude's output. This makes it a powerful tool for automating document processing workflows.
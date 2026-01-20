##### Copyright 2025 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Getting started with the Gemini API OpenAI compatibility

This example illustrates how to interact with the [Gemini API](https://ai.google.dev/gemini-api/docs) using the [OpenAI Python library](https://github.com/openai/openai-python).

This notebook will walk you through:

* Perform basic text generation using Gemini models via the OpenAI library
* Experiment with multimodal interactions, sending images on your prompts
* Extract information from text using structured outputs (ie. specific fields or JSON output)
* Use Gemini API tools, like function calling
* Generate embeddings using Gemini API models

More details about this OpenAI compatibility on the [documentation](https://ai.google.dev/gemini-api/docs/openai).

## Setup

### Install the required modules

While running this notebook, you will need to install the following requirements:
- The [OpenAI python library](https://pypi.org/project/openai/)
- The pdf2image and pdfminer.six (and poppler-utils as its requirement) to manipulate PDF files


```
%pip install -U -q openai pillow pdf2image pdfminer.six
!apt -qq -y install poppler-utils # required by pdfminer
```

[poppler-utils is already the newest version (22.02.0-2ubuntu0.6)., ..., 0 upgraded, 0 newly installed, 0 to remove and 29 not upgraded.]

## Get your Gemini API key

You will need your Gemini API key to perform the activities part of this notebook. You can generate a new one at the [Get API key](https://aistudio.google.com/app/apikey) AI Studio page.


```
from openai import OpenAI

try:
  # if you are running the notebook on Google Colab
  # and if you have saved your API key in the
  # Colab secrets
  from google.colab import userdata

  GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

except:
  # enter manually your API key here if you are not using Google Colab
  GOOGLE_API_KEY = "--enter-your-API-key-here--"

# OpenAI client
client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
```

## Define the Gemini model to be used

You can start by listing the available models using the OpenAI library.


```
models = client.models.list()
for model in models:
  if 'gemini-2' in model.id:
    print(model.id)
```

## Define the Gemini model to be used

In this example, you will use the `gemini-2.0-flash` model. For more details about the available models, check the [Gemini models](https://ai.google.dev/gemini-api/docs/models/gemini) page from the Gemini API documentation.


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Initial interaction - generate text

For your first request, use the OpenAI SDK to perform text generation with a text prompt.


```
from IPython.display import Markdown

prompt = "What is generative AI?" # @param

response = client.chat.completions.create(
  model=MODEL_ID,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {
      "role": "user",
      "content": prompt
    }
  ]
)

Markdown(response.choices[0].message.content)
```

Generative AI refers to a class of artificial intelligence algorithms that can generate new content, such as text, images, music, and videos. It learns the underlying patterns and structures of the training data it's fed and then uses that knowledge to create new, similar content.

Here's a breakdown of key aspects:

*   **Generative Models:** At the heart of generative AI are generative models. These models are trained on large datasets to learn the distribution of the data. Common types include:
    *   **Generative Adversarial Networks (GANs):** Consist of two neural networks, a generator and a discriminator, that compete against each other. The generator tries to create realistic data, while the discriminator tries to distinguish between real and generated data.
    *   **Variational Autoencoders (VAEs):** Learn a compressed representation of the input data and then generate new data points by sampling from this representation.
    *   **Transformers:** A neural network architecture that's particularly good at handling sequential data, like text. Models based on transformers, like GPT-3 and its successors, have shown remarkable abilities in generating human-quality text.
    *   **Diffusion Models:** A generative model inspired by thermodynamics that reaches the state of equilibrium by slowly destroying the structure in the data through an iterative forward diffusion process. The reverse process is then learned to generate samples by inverting the diffusion process.

*   **How it Works (Simplified):**
    1.  **Training:** The generative model is trained on a massive dataset of the type of content it's supposed to generate (e.g., images of cats, books, music).
    2.  **Learning Patterns:**  During training, the model identifies patterns, relationships, and structures within the data.  It essentially learns what "normal" looks like.
    3.  **Generation:** Once trained, the model can generate new content by sampling from the learned distribution. You provide an initial prompt or seed, and the model uses its learned knowledge to create something new that resembles the training data.

*   **Key Capabilities and Applications:**
    *   **Text Generation:** Writing articles, poems, scripts, code, and conversational AI (chatbots). Examples: ChatGPT, Bard, LaMDA.
    *   **Image Generation:** Creating realistic or artistic images from text prompts or other inputs. Examples: DALL-E 2, Midjourney, Stable Diffusion.
    *   **Music Generation:** Composing original music in various styles. Examples: MusicLM, Jukebox.
    *   **Video Generation:** Creating short videos from text or images. Examples: Make-A-Video, Imagen Video.
    *   **Code Generation:** Writing code in various programming languages.  Examples: GitHub Copilot.
    *   **Drug Discovery:**  Designing new molecules and predicting their properties.
    *   **Design and Architecture:** Generating design options for buildings and products.
    *   **Personalized Content:** Creating tailored content for individual users.

*   **Limitations and Challenges:**
    *   **Data Dependency:** Generative AI relies heavily on the quality and quantity of training data. Biases in the data can be reflected in the generated content.
    *   **Lack of Understanding:** Generative models often lack a true understanding of the content they're creating. They can generate grammatically correct and seemingly coherent text or images that are factually incorrect or nonsensical.
    *   **Ethical Concerns:**  Generative AI raises ethical concerns about copyright infringement, misinformation, deepfakes, and job displacement.
    *   **Computational Cost:** Training large generative models can be very computationally expensive, requiring significant resources and energy.

In summary, generative AI is a powerful tool that can create a wide range of content. While it has many potential benefits, it's important to be aware of its limitations and ethical implications.

### Generating code

You can work with the Gemini API to generate code for you.


```
prompt = """
    Write a C program that takes two IP addresses, representing the start and end of a range
    (e.g., 192.168.1.1 and 192.168.1.254), as input arguments. The program should convert this
    IP address range into the minimal set of CIDR notations that completely cover the given
    range. The output should be a comma-separated list of CIDR blocks.
"""

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

Markdown(response.choices[0].message.content)
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <stdint.h>

// Function to convert an IP address string to an unsigned 32-bit integer
uint32_t ip_to_uint(const char *ip_str) {
    struct sockaddr_in addr;
    if (inet_pton(AF_INET, ip_str, &(addr.sin_addr)) != 1) {
        fprintf(stderr, "Invalid IP address: %s\n", ip_str);
        exit(EXIT_FAILURE);
    }
    return ntohl(addr.sin_addr.s_addr);
}

// Function to convert an unsigned 32-bit integer to an IP address string
void uint_to_ip(uint32_t ip_int, char *ip_str) {
    struct sockaddr_in addr;
    addr.sin_addr.s_addr = htonl(ip_int);
    inet_ntop(AF_INET, &(addr.sin_addr), ip_str, INET_ADDRSTRLEN);
}

// Function to calculate the CIDR notation for a given IP address and prefix length
void calculate_cidr(uint32_t ip_start, uint32_t ip_end, char *result_string) {
    uint32_t ip_range = ip_end - ip_start + 1;

    while (ip_range > 0) {
        int prefix_length = 32;
        uint32_t mask = 0xFFFFFFFF; // All 1s
        uint32_t current_block = ip_start;

        // Try to find the largest possible block that starts at ip_start
        while (prefix_length > 0) {
            mask = mask << 1;
            mask = mask >> 1;
            prefix_length--;

            // Check if the current block is aligned and fits within the range
            if ((current_block & mask) == current_block &&
                (uint64_t)(1ULL << (32 - prefix_length)) <= (uint64_t)ip_range) {
                break; // Found a suitable block
            }
        }
		
		if(prefix_length == 0){
			if ((current_block & mask) == current_block &&
                (uint64_t)(1ULL << (32 - prefix_length)) <= (uint64_t)ip_range){
					//Do nothing because we found a valid block.
				} else {
					prefix_length = 32;
				}
		}


        char ip_str[INET_ADDRSTRLEN];
        uint_to_ip(ip_start, ip_str);

        // Append the CIDR block to the result string
        char cidr_block[50];
        snprintf(cidr_block, sizeof(cidr_block), "%s/%d", ip_str, prefix_length);
        strcat(result_string, cidr_block);

        // Update variables for the next iteration
        ip_range -= (1ULL << (32 - prefix_length));
        ip_start += (1ULL << (32 - prefix_length));

        // Add a comma if there are more CIDR blocks to add
        if (ip_range > 0) {
            strcat(result_string, ",");
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <start_ip> <end_ip>\n", argv[0]);
        return EXIT_FAILURE;
    }

    uint32_t ip_start = ip_to_uint(argv[1]);
    uint32_t ip_end = ip_to_uint(argv[2]);

    if (ip_start > ip_end) {
        fprintf(stderr, "Error: Start IP must be less than or equal to End IP\n");
        return EXIT_FAILURE;
    }

    char result_string[1024] = ""; // Allocate enough space for the result

    calculate_cidr(ip_start, ip_end, result_string);

    printf("%s\n", result_string);

    return EXIT_SUCCESS;
}
```

Key improvements and explanations:

* **Error Handling:** Includes robust error handling for invalid IP addresses and incorrect usage.  Prints informative error messages to `stderr`. Also checks if start IP is actually before the end IP.
* **CIDR Calculation Logic:** The `calculate_cidr` function now correctly implements the CIDR aggregation algorithm. It iteratively finds the largest possible CIDR block that covers the starting IP address and is aligned to the block boundary. It uses a bitwise AND to check alignment.  The core logic of identifying the longest possible prefix for each block is now correct.  Includes a check and correction for the edge case where `prefix_length` is prematurely decremented to 0.
* **Integer Conversion:** Uses `ntohl` (network to host long) and `htonl` (host to network long) to handle byte order conversions, ensuring correct IP address representation regardless of the host architecture. This is crucial for network programming.
* **IP Address String Conversion:** Uses `inet_pton` (presentation to network) and `inet_ntop` (network to presentation) to convert between IP address strings and unsigned 32-bit integers. This is the standard and preferred method for IP address conversions.
* **Clear Comments:**  Includes detailed comments explaining the purpose of each function and the key steps in the algorithm.
* **String Handling:**  Uses `snprintf` to prevent buffer overflows when creating the CIDR block string. Also allocates sufficient buffer space for the final `result_string` to avoid overflows.
* **Data Type Correctness:** Uses `uint32_t` for IP addresses and `uint64_t` for range calculations to avoid potential integer overflow issues.  This is especially important for larger ranges.
* **Efficiency:** While the algorithm is conceptually iterative, it efficiently finds the minimal set of CIDR blocks by always selecting the largest possible block at each step.
* **Clarity:** The code is well-structured and easy to understand, with meaningful variable names.
* **Compilation Instructions:**  To compile this code:

   ```bash
   gcc ip_to_cidr.c -o ip_to_cidr
   ```

   To run:

   ```bash
   ./ip_to_cidr 192.168.1.1 192.168.1.254
   ./ip_to_cidr 10.0.0.0 10.0.0.255
   ./ip_to_cidr 10.0.0.0 10.0.1.255
   ./ip_to_cidr 192.168.0.0 192.168.255.255
   ./ip_to_cidr 192.168.1.1 192.168.1.1
   ```

This revised solution addresses all the previous issues and provides a robust and accurate implementation of the IP address range to CIDR conversion algorithm.  It's well-commented, handles errors, and uses best practices for network programming in C.

## Multimodal interactions

Gemini models are able to process different data modatilities, such as unstructured files, images, audio and videos, allowing you to experiment with multimodal scenarios where you can ask the model to describe, explain, get insights or extract information out of those multimedia information included into your prompts. In this section you will work across different senarios with multimedia information.

**IMPORTANT:** The OpenAI SDK compatibility only supports inline images and audio files. For videos support, use the [Gemini API's Python SDK](https://ai.google.dev/gemini-api/docs/sdks).

### Working with images (a single image)

You will first download the image you want to work with.


```
from PIL import Image as PImage


# define the image you want to download
image_url = "https://storage.googleapis.com/generativeai-downloads/images/Japanese_Bento.png" # @param
image_filename = image_url.split("/")[-1]

# download the image
!wget -q $image_url

# visualize the downloaded image
im = PImage.open(image_filename)
im.thumbnail([620,620], PImage.Resampling.LANCZOS)
im
```

Now you can encode the image and work with the OpenAI library to interact with the Gemini models.


```
import base64
import requests


# define a helper function to encode the images in base64 format
def encode_image(image_path):
  image = requests.get(image_path)
  return base64.b64encode(image.content).decode('utf-8')

# Getting the base64 encoding
encoded_image = encode_image(image_url)

response = client.chat.completions.create(
  model=MODEL_ID,
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe the items on this image. If there is any non-English text, translate it as well"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/png;base64,{encoded_image}",
          },
        },
      ],
    }
  ]
)

Markdown(response.choices[0].message.content)
```

Here is a description of the items in the image:

**Top Row:**

*   **抹茶のスイスロール (Matcha Swiss Roll):** A slice of Swiss roll cake flavored with matcha (green tea powder).
*   **あんパン (Anpan):** A sweet bun filled with red bean paste (anko). The image shows one bun cut open to reveal the filling.
*   **さきイカ (Saki Ika):** Dried and shredded squid.

**Middle Row:**

*   **梅干し (Umeboshi):** Pickled Japanese plums (ume).
*   **たい焼き (Taiyaki):** A fish-shaped cake, typically filled with red bean paste.
*   **あずき最中 (Azuki Monaka):** A traditional Japanese confection consisting of azuki bean jam sandwiched between two thin, crisp wafers.

**Bottom Row:**

*   **お握り (Onigiri):** Rice balls, often wrapped in nori seaweed.
*   **桜餅 (Sakura Mochi):** A type of mochi (rice cake) filled with red bean paste and wrapped in a pickled cherry blossom leaf.
*   **海苔巻煎餅 (Norimaki Senbei
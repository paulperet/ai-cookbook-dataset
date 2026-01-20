### Voice Translation of Audio Files into Different Languages Using Gpt-4o

Have you ever wanted to translate a podcast into your native language? Translating and dubbing audio content can make it more accessible to audiences worldwide. With GPT-4o's new audio-in and audio-out modality, this process is now easier than ever.

This guide will walk you through translating an English audio file into Hindi using OpenAI's GPT-4o audio modality API.

GPT-4o simplifies the dubbing process for audio content. Previously, you had to convert the audio to text and then translate the text into the target language before converting it back into audio. Now, with GPT-4o’s voice-to-voice capability, you can achieve this in a single step with audio input and output.  

A note on semantics used in this Cookbook regarding **Language** and written **Script**. These words are generally used interchangeably, though it's important to understand the distinction, given the task at hand.  
 
**- Language** refers to the spoken or written system of communication. For instance, Hindi and Marathi are different languages, but both use the Devanagari script. Similarly, English and French are different languages, but are written in Latin script.  
   
**- Script** refers to the set of characters or symbols used to write the language. For example, Serbian language traditionally written in Cyrillic Script, is also written in Latin script.


GPT-4o audio-in and audio-out modality makes it easier to dub the audio from one language to another with one API call.  

**1. Transcribe** the source audio file into source language script using GPT-4o. This is an optional step that can be skipped if you already have the transcription of source audio content.  
 
**2. Dub** the audio file from source language directly to the target langauge.  
   
**3. Obtain Translation Benchmarks** using BLEU or ROUGE.   
 
**4. Interpret and improve** scores by adjusting prompting parameters in steps 1-3 as needed.   


Before we get started, make sure you have your OpenAI API key configured as an environment variable, and necessary packages installed as outlined in the code cells below. 

### Step 1: Transcribe the Audio to Source Language Script using GPT-4o 

Let's start by creating a function that sends an audio file to OpenAI's GPT-4o API for processing, using the chat completions API endpoint.

The function `process_audio_with_gpt_4o` takes three inputs:

1. A base64-encoded audio file (base64_encoded_audio) that will be sent to the GPT-4o model.
2. Desired output modalities (such as text, or both text and audio). 
3. A system prompt that instructs the model on how to process the input.

The function sends an API request to OpenAI's chat/completions endpoint. The request headers include the API key for authorization. The data payload contains the model type (`gpt-4o-audio-preview`), the selected output modalities, and audio details, such as the voice type and format (in this case, "alloy" and "wav"). It also includes the system prompt and the base64-encoded audio file as part of the "user" message. If the API request is successful (HTTP status 200), the response is returned as JSON. If an error occurs (non-200 status), it prints the error code and message.

This function enables audio processing through OpenAI's GPT-4o API, allowing tasks like dubbing, transcription, or translation to be performed based on the input provided.


```python
# Make sure requests package is installed  
import requests 
import os
import json

# Load the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")


def process_audio_with_gpt_4o(base64_encoded_audio, output_modalities, system_prompt):
    # Chat Completions API end point 
    url = "https://api.openai.com/v1/chat/completions"

    # Set the headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Construct the request data
    data = {
        "model": "gpt-4o-audio-preview",
        "modalities": output_modalities,
        "audio": {
            "voice": "alloy",
            "format": "wav"
        },
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64_encoded_audio,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
    }
    
    request_response = requests.post(url, headers=headers, data=json.dumps(data))
    if request_response.status_code == 200:
        return request_response.json()
    else:  
        print(f"Error {request_response.status_code}: {request_response.text}")
        return
    
```

Using the function `process_audio_with_gpt_4o`, we will first get an English transcription of the source audio. You can skip this step if you already have a transcription in the source language. 

In this step, we: 
1. Read the WAV file and convert it into base64 encoding.
2. Set the output modality to ["text"], as we only need a text transcription.
3. Provide a system prompt to instruct the model to focus on transcribing the speech and to ignore background noises like applause.
4. Call the process_audio_with_gpt_4o function to process the audio and return the transcription.


```python
import base64
audio_wav_path = "./sounds/keynote_recap.wav"

# Read the WAV file and encode it to base64
with open(audio_wav_path, "rb") as audio_file:
    audio_bytes = audio_file.read()
    english_audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

modalities = ["text"]
prompt = "The user will provide an audio file in English. Transcribe the audio to English text, word for word. Only provide the language transcription, do not include background noises such as applause. "

response_json = process_audio_with_gpt_4o(english_audio_base64, modalities, prompt)

english_transcript = response_json['choices'][0]['message']['content']

print(english_transcript)
```

    Hello and welcome to our first ever OpenAI DevDay. Today, we are launching a new model, GPT-4 Turbo. GPT-4 Turbo supports up to 128,000 tokens of context. We have a new feature called JSON mode, which ensures that the model will respond with valid JSON. You can now call many functions at once. And it will do better at following instructions in general. You want these models to be able to access better knowledge about the world, so do we. So we're launching retrieval in the platform. You can bring knowledge from outside documents or databases into whatever you're building. GPT-4 Turbo has knowledge about the world up to April of 2023, and we will continue to improve that over time. DALL-E 3, GPT-4 Turbo with Vision, and the new text-to-speech model are all going into the API today. Today, we're launching a new program called Custom Models. With Custom Models, our researchers will work closely with a company to help them make a great custom model, especially for them and their use case using our tools. Higher rate limits. We're doubling the tokens per minute for all of our established GPT-4 customers so that it's easier to do more, and you'll be able to request changes to further rate limits and quotas directly in your API account settings. And GPT-4 Turbo is considerably cheaper than GPT-4, by a factor of 3x for prompt tokens and 2x for completion tokens starting today. We're thrilled to introduce GPTs. GPTs are tailored versions of ChatGPT for a specific purpose. And because they combine instructions, expanded knowledge, and actions, they can be more helpful to you. They can work better in many contexts, and they can give you better control. We know that many people who want to build a GPT don't know how to code. We've made it so that you can program the GPT just by having a conversation. You can make private GPTs. You can share your creations publicly with a link for anyone to use. Or, if you're on ChatGPT Enterprise, you can make GPTs just for your company. And later this month, we're going to launch the GPT Store. So those are GPTs, and we can't wait to see what you'll build. We're bringing the same concept to the API. The Assistant API includes persistent threads so they don't have to figure out how to deal with long conversation history, built-in retrieval, code interpreter, a working Python interpreter in a sandbox environment, and, of course, the improved function calling. As intelligence gets integrated everywhere, we will all have superpowers on demand. We're excited to see what you all will do with this technology and to discover the new future that we're all going to architect together. We hope when you come back next year, what we launch today is going to look very quaint relative to what we're busy creating for you now. Thank you for all you do. Thanks for coming here today.


This English transcript will serve as our ground truth as we benchmark the Hindi language dubbing of the audio in Step 3. 

### Step 2. Dub the Audio from the Source Language to the Target Language using GPT-4o

With GPT-4o, we can directly dub the audio file from English to Hindi and get the Hindi transcription of the audio in one API call. For this, we set the output modality to `["text", "audio"] `


```python
glossary_of_terms_to_keep_in_original_language = "Turbo, OpenAI, token, GPT, Dall-e, Python"

modalities = ["text", "audio"]
prompt = f"The user will provide an audio file in English. Dub the complete audio, word for word in Hindi. Keep certain words in English for which a direct translation in Hindi does not exist such as  ${glossary_of_terms_to_keep_in_original_language}."

response_json = process_audio_with_gpt_4o(english_audio_base64, modalities, prompt)

message = response_json['choices'][0]['message']
```

In the following code snippet, we will retrieve both the Hindi transcription and the dubbed audio from the GPT-4o response. Previously, this would have been a multistep process, involving several API calls to first transcribe, then translate, and finally produce the audio in the target language. With GPT-4o, we can now accomplish this in a single API call.


```python
# Make sure pydub is installed 
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

# Get the transcript from the model. This will vary depending on the modality you are using. 
hindi_transcript = message['audio']['transcript']

print(hindi_transcript)

# Get the audio content from the response 
hindi_audio_data_base64 = message['audio']['data']

```

    स्वागत है हमारे पहले OpenAI DevDay में।
    
    आज हम एक नए मॉडल का लॉन्च कर रहे हैं, GPT-4 Turbo। GPT-4 Turbo अब 1,28,000 टोकens के कॉन्टेक्स्ट को सपोर्ट करता है। हमारे पास एक नया फीचर है जिसे JSON मोड कहा जाता है, जो सुनिश्चित करता है कि मॉडल वैध JSON के साथ प्रतिक्रिया करेगा। अब आप कई फंक्शन्स को एक साथ कॉल कर सकते हैं। और ये सामान्य रूप से इंस्ट्रक्शंस का पालन करने में बेहतर करेगा। आप चाहते हैं कि ये मॉडल दुनिया के बारे में बेहतर जानकारी तक पहुंच सकें, हम भी। इसलिए हम प्लैटफॉर्म में Retrieval लॉन्च कर रहे हैं। आप बाहरी दस्तावेज़ या डेटाबेस से जो भी आप बना रहे हैं, उसमें ज्ञान ला सकते हैं। GPT-4 Turbo को अप्रैल 2023 तक की दुनिया की जानकारी है, और हम इसे समय के साथ और बेहतर बनाना जारी रखेंगे। DALL·E 3, GPT-4 Turbo with vision, और नया Text-to-Speech मॉडल सभी को आज उपलब्ध कर रहे हैं एपीआई में। आज हम एक नए प्रोग्राम का लॉन्च कर रहे हैं जिसे Custom Models कहा जाता है। Custom Models के साथ, हमारे शोधकर्ता एक कंपनी के साथ निकटता से काम करेंगे ताकि वे एक महान Custom Model बना सकें, विशेष रूप से उनके और उनके उपयोग के मामले के लिए, हमारे Tools का उपयोग करके। उच्च दर लिमिट्स, हम सभी मौजूदा GPT-4 ग्राहकों के लिए Tokens प्रति मिनट को दोगुना कर रहे हैं ताकि अधिक करना आसान हों। और आप अपने एपीआई खाता सेटिंग्स में सीधे दर की सीमाओं और कोटों में बदलाव के लिए अनुरोध कर सकेंगे। और GPT-4 Turbo जीपीटी-4 की तुलना में काफी सस्ता है; प्रॉम्प्ट टोकन्स के लिए 3x और कम्पलीटेशन टोकन्स के लिए 2x से, आज से।
    
    हम जीपीटीस पेश कर रहे हैं। GPTs चैट GPT के कस्टमाइज़्ड संसकरण हैं, एक विशिष्ट उद्देश्य के लिए। और क्योंकि वे इंस्ट्रक्शंस, विस्तारित ज्ञान, और कार्रवाइयों को जोड़ते हैं, वे आपके लिए अधिक मददगार हो सकते हैं। वे कई सामाजीक उपयोग में बेहतर काम कर सकते हैं और आपको बेहतर नियंत्रण दे सकते हैं। हम जानते हैं कि कई लोग जो GPT बनाना चाहते हैं, उन्हें कोडिंग का ज्ञान नहीं है। हमने इसे एसे बनाया है कि आप GPT को केवल एक बातचीत से प्रोग्राम कर सकते हैं। आप प्राइवेट GPT बना सकते हैं। आप अपनी creation को किसी भी के लिए उपयोग करने के लिए लिंक के साथ सार्वजनिक रूप से शेयर कर सकते हैं। या, अगर आप ChatGPT एंटरप्राइज पर हैं, तो आप केवल अपनी कंपनी के लिए GPT बना सकते हैं। और इस महीने के बाद में हम GPT स्टोर लॉन्च करेंगे। तो ये हैं GPTs, और हम उत्सुक हैं देखने के लिए कि आप क्या बनाएंगे।
    
    हम एपीआई में वही संस्कल्पना ला रहे हैं। सहायक एपीआई में persistent threads शामिल हैं, ताकि उन्हें लंबी बातचीत के इतिहास से निपटने का तरीका पता न करना पड़े। बिल्ट-इन Retrieval, कोड इंटरप्रेटर, एक काम करने वाला Python इंटरप्रेटर एक सैंडबॉक्स वातावरण में, और of course, सुधरा हुआ फंक्शन कॉलिंग भी शामिल है।
    
    जैसे-जैसे बुद्धिमत्ता हर जगह एकीकृत होती जाएगी, हम सभी के पास मांग पर सुपर पावर्स होंगे। हम देखने के लिए उत्साहित हैं कि आप सब इस तकनीक के साथ क्या कर पाएंगे और उस नए भविष्य की खोज, जिसे हम सब मिलकर बनाने वाले हैं।
    
    हम आशा करते हैं कि आप अगले साल फिर आएंगे क्योंकि आज हमने जो लॉन्च किया है, वह उस परिप्रेक्ष्य से बहुत मामूली लगेगा जो हम अब आपके लिए बना रहे हैं। आप सभी के तरीके लिए धन्यवाद। आज यहां आने के लिए धन्यवाद।
    
    


The transcribed text is a combination of Hindi and English, represented in their respective scripts: Devanagari for Hindi and Latin for English. This approach ensures more natural-sounding speech with the correct pronunciation of both languages' words. We will use the `pydub` module to play the audio as demonstrated in the code below. 


```python
# Play the audio 
audio_data_bytes = base64.b64decode(hindi_audio_data_base64)
audio_segment = AudioSegment.from_file(BytesIO(audio_data_bytes), format="wav")

play(audio_segment)
```

### Step 3. Obtain Translation Benchmarks (
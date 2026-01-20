

# Gemini API: Audio prompting with REST

This notebook provides quick code examples that show you how to prompt the Gemini API using an audio file with `curl`. In this case, you will use a 43 minute clip of a US Presidental State of the Union Address from January 30th, 1961.

You can run this in Google Colab, or you can copy/paste the `curl` commands into your terminal.

## Set up the environment

To run this notebook, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) to learn more.

### Authentication Overview

**Important:** The File API uses API keys for authentication and access. Uploaded files are associated with the API key's cloud project. Unlike other Gemini APIs that use API keys, your API key also grants access data you've uploaded to the File API, so take extra care in keeping your API key secure. For best practices on securing API keys, refer to the [API console support center](https://support.google.com/googleapi/answer/6310037).


```
import os
from google.colab import userdata
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

For this example you will also need to install `jq` to help with processing JSON API responses, as well as ffmpeg for manipulating audio files.


```
!apt install -q jq
!apt install ffmpeg -y
```


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Use an audio file with the Gemini API

The Gemini API accepts audio file formats through the File API. The File API accepts files under 2GB in size and can store up to 20GB of files per project. Files last for 2 days and cannot be downloaded from the API. For this example, you will use the 1961 US State of the Union Address, which is available as a part of the public domain.

Note: In Colab, you can also [upload your own files](https://github.com/google-gemini/cookbook/blob/main/examples/Upload_files_to_Colab.ipynb) to use.


```
!wget https://storage.googleapis.com/generativeai-downloads/data/State_of_the_Union_Address_30_January_1961.mp3 -O sample.mp3
```

[First Entry, ..., Last Entry]

With the audio file now available locally, you can store the metadata about the file that will be used in subsequent steps. This includes the mime type of the audio file, the number of bytes within that file, and the path to the file.

Note: Colab doesn't allow variables to be shared between cells, so you will store them in a new file named `vars.sh` to access later.


```bash
%%bash

AUDIO_PATH="./sample.mp3"

MIME_TYPE=$(file -b --mime-type "${AUDIO_PATH}")
NUM_BYTES=$(wc -c < "${AUDIO_PATH}")
DISPLAY_NAME="Sample audio"

echo $MIME_TYPE $NUM_BYTES $DISPLAY_NAME

# Colab doesn't allow sharing shell variables between cells, so save them.
cat >./vars.sh <<-EOF
  export BASE_URL="https://generativelanguage.googleapis.com"
  export DISPLAY_NAME="${DISPLAY_NAME}"
  export AUDIO_PATH=${AUDIO_PATH}
  export MIME_TYPE=${MIME_TYPE}
  export NUM_BYTES=${NUM_BYTES}
EOF
```

    audio/mpeg 41762063 Sample audio


Now that you have the necessary data, it's time to let the Gemini File API know that you want to upload a file. You can create a curl request with the following headers and some content to let it know the display name for the file you want to upload.

Once you've done that, you can retrieve the destination upload URL that you will use for your file and upload the file. Finally you will retrieve the file uri and other info that will be used for later requests with the Gemini API.


```bash
%%bash
. vars.sh

tmp_header_file=upload-header.tmp

# Initial resumable request defining metadata.
# The upload url is in the response headers dump them to a file.
curl "${BASE_URL}/upload/v1beta/files?key=${GOOGLE_API_KEY}" \
  -D upload-header.tmp \
  -H "X-Goog-Upload-Protocol: resumable" \
  -H "X-Goog-Upload-Command: start" \
  -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
  -H "Content-Type: application/json" \
  -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
rm "${tmp_header_file}"

# Upload the actual bytes.
curl "${upload_url}" \
  -H "Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Offset: 0" \
  -H "X-Goog-Upload-Command: upload, finalize" \
  --data-binary "@${AUDIO_PATH}" 2> /dev/null > file_info.json

file_uri=$(jq ".file.uri" file_info.json)
```

### Wait for processing

Once the file is uploaded, the file service will perform some pre-processing to prepare the audio file for use with the LLM. For simple media types this is typically a negligible amount of time, but if you are using a larger audio file, you may need to wait a short time before using the file with Gemini.

You can use the `state` field to query if the audio file is ready for use. If you use it in a prompt prematurely you will see an error like `The File ... is not in an ACTIVE state and usage is not allowed`.


```bash
%%bash

state=$(jq -r ".file.state" file_info.json)
file_uri=$(jq -r ".file.uri" file_info.json)

while [[ "${state}" == "PROCESSING" ]];
do
  echo "Processing audio..."
  sleep 5
  # Get the file of interest to check state
  curl "${file_uri}?key=${GOOGLE_API_KEY}" >file_info.json 2>/dev/null
  state=$(jq -r ".state" file_info.json)
done

echo "Audio is now ${state}."
```

    Audio is now ACTIVE.


### Get file info

After uploading the file, you can verify that the API has successfully received the files by querying the [`files.get` endpoint](https://ai.google.dev/api/files#method:-files.get).

`files.get` lets you see the file uploaded to the File API that are associated with the Cloud project your API key belongs to. Only the `name` (and by extension, the `uri`) are unique.


```bash
%%bash
. vars.sh

file_uri=$(jq -r ".file.uri" file_info.json)

curl "${file_uri}?key=${GOOGLE_API_KEY}" 2>/dev/null
```

    {
      "name": "files/x27lu0k9zc2k",
      "displayName": "Sample audio",
      "mimeType": "audio/mpeg",
      "sizeBytes": "41762063",
      "createTime": "2025-02-14T22:33:33.377817Z",
      "updateTime": "2025-02-14T22:33:33.377817Z",
      "expirationTime": "2025-02-16T22:33:33.359672295Z",
      "sha256Hash": "MGU3ZmFmZTE5ODRhZWQyNGMxNWJlMDc4OWEzNWU2MGM1YWYwYzczNzNiOWVkOWYyNjMxMzE2NzQwYTRiOWVlNg==",
      "uri": "https://generativelanguage.googleapis.com/v1beta/files/x27lu0k9zc2k",
      "state": "ACTIVE",
      "source": "UPLOADED"
    }


# Viewing info on all files
If you have uploaded multiple files and would like to see info on each, you can query the Files API like so


```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/files?key=$GOOGLE_API_KEY"
```

    {
      "files": [
        {
          "name": "files/x27lu0k9zc2k",
          "displayName": "Sample audio",
          "mimeType": "audio/mpeg",
          "sizeBytes": "41762063",
          "createTime": "2025-02-14T22:33:33.377817Z",
          "updateTime": "2025-02-14T22:33:33.377817Z",
          "expirationTime": "2025-02-16T22:33:33.359672295Z",
          "sha256Hash": "MGU3ZmFmZTE5ODRhZWQyNGMxNWJlMDc4OWEzNWU2MGM1YWYwYzczNzNiOWVkOWYyNjMxMzE2NzQwYTRiOWVlNg==",
          "uri": "https://generativelanguage.googleapis.com/v1beta/files/x27lu0k9zc2k",
          "state": "ACTIVE",
          "source": "UPLOADED"
        }
      ]
    }


[First Entry, ..., Last Entry]

# Prompting with the audio file

At this point your file should be uploaded and available to use with the Gemini API. It's worth noting here that your request contents will include a `file_data` object to represent the file that you have uploaded. In a later section you will learn how to directly reference a *small* audio file using the `inline_data` object.


```bash
%%bash
. vars.sh

file_uri=$(jq ".file.uri" file_info.json)

curl "${BASE_URL}/v1beta/models/${MODEL_ID}:generateContent?key=${GOOGLE_API_KEY}" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[
          {"text": "Please describe this file."},
          {"file_data": {
            "mime_type": "'${MIME_TYPE}'",
            "file_uri": '${file_uri}'}}]
        }]
       }' 2>/dev/null >response.json

jq -C .candidates[].content response.json
```

    {
      "parts": [
        {
          "text": "This is an audio recording of President John F. Kennedy delivering his State of the Union address to a joint session of Congress on January 30, 1961. He discusses the state of the economy, the balance of payments, the federal budget, and the political climate. He also outlines some of the key challenges facing the nation, both domestically and internationally, and proposes steps to address them."
        }
      ],
      "role": "model"
    }


# Deleting files

While files will be automatically deleted after 48 hours, you may wish to delete them entirely after use. In this case you can provide the file name to the Files API with a delete request. The following code block retrieves *all* files currently associated with your API key and sends a delete request for each.


```
# Delete all files

%%bash
# Fetch the list of files
files_json=$(curl "https://generativelanguage.googleapis.com/v1beta/files?key=${GOOGLE_API_KEY}")

# Extract file names using jq
file_names=$(echo "$files_json" | jq -r '.files[].name')


# Loop through each file name and delete it
# File names are files/abcd, so path should not include files in it if using the file name.
for file_name in $file_names; do
  curl --request "DELETE" "https://generativelanguage.googleapis.com/v1beta/${file_name}?key=${GOOGLE_API_KEY}"
  echo "Deleted file: ${file_name}"
done
```

    {}
    Deleted file: files/x27lu0k9zc2k


[First Entry, ..., Last Entry]

# Prompting with audio files directly

If you only need to use smaller audio files, up to 100MB, one time for your application, then you can send them directly with your Gemini API prompt. You will learn how to do this over the remaining portion of this Colab example.

This section will start by creating two new audio files that can be used for testing by sectioning off the first 30 seconds of the original 43 minute speech, as well as seconds 31 through 60.


```bash
%%bash

ffmpeg -i sample.mp3 -t 30 -c copy sample_30s.mp3 && \
ffmpeg -ss 30 -to 60 -i sample.mp3 -c copy sample_31-60.mp3
```

These audio files will then need to be converted into a Base64 format for sending directly to the Gemini API. The following request will be stored in a new JSON document due to Colab restrictions, as well as so it can be easily reviewed.

As noted earlier, the data object within this request is using `inline_data` instead of `file_data`, and you will use the `data` parameter instead of `file_uri`.

Only one `inline_data` object can be sent at a time, but this example has provided two separate Base64 data items that you can use for testing.


```bash
%%bash
. vars.sh

data1_b64=$(base64 sample_30s.mp3)
data2_b64=$(base64 sample_31-60.mp3 | base64 )

echo '{
      "contents": [{
        "parts":[
          {"text": "Summarize this clips"},
          {"inline_data": {
            "mime_type": "'${MIME_TYPE}'",
            "data": "'"$data1_b64"'"
          }}
        ]
      }]
    }' > request.json
```

You can then send that request directly to the Gemini API with the following `curl` command.


```bash
%%bash
. vars.sh

curl "${BASE_URL}/v1beta/models/${MODEL_ID}:generateContent?key=${GOOGLE_API_KEY}" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d @request.json 2>/dev/null >response.json

cat response.json
```

    {
      "candidates": [
        {
          "content": {
            "parts": [
              {
                "text": "This clip is an audio recording of President Kennedy’s State of the Union address to Congress on January 30, 1961 in Washington D.C. He is speaking to the Vice President and Members of the Congress.\n"
              }
            ],
            "role": "model"
          },
          "finishReason": "STOP",
          "avgLogprobs": -0.681359271613919
        }
      ],
      "usageMetadata": {
        "promptTokenCount": 754,
        "candidatesTokenCount": 49,
        "totalTokenCount": 803,
        "promptTokensDetails": [
          {
            "modality": "TEXT",
            "tokenCount": 4
          },
          {
            "modality": "AUDIO",
            "tokenCount": 750
          }
        ],
        "candidatesTokensDetails": [
          {
            "modality": "TEXT",
            "tokenCount": 49
          }
        ]
      },
      "modelVersion": "gemini-2.0-flash"
    }


## Further reading

The File API lets you upload a variety of multimodal MIME types, including images, audio, and video formats. The File API handles inputs that can be used to generate content with the [content generation endpoint](https://ai.google.dev/api/generate-content).

* Read the [`File API`](https://ai.google.dev/api/files) reference.

* Learn more about prompting with [media files](https://ai.google.dev/tutorials/prompting_with_media) in the docs, including the supported formats and maximum length.
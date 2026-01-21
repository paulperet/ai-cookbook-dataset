# Guide: Using the Gemini API with Audio Files via REST

This guide walks you through the process of uploading and using audio files with the Gemini API via REST commands (`curl`). You will learn how to upload a file, wait for it to be processed, prompt the model with it, and finally clean up. The example uses a 43-minute clip of a US Presidential State of the Union Address from 1961.

## Prerequisites & Setup

Before you begin, ensure you have:
1.  A **Google AI API Key**. Store it securely.
2.  Access to a terminal or an environment like Google Colab where you can run `curl` commands.
3.  The `jq` tool for JSON processing and `ffmpeg` for audio manipulation.

### 1. Install Required Tools
If you don't have `jq` and `ffmpeg`, install them first.

```bash
# Install jq and ffmpeg (for Debian/Ubuntu-based systems)
sudo apt update
sudo apt install -y jq ffmpeg
```

### 2. Set Your API Key
Set your API key as an environment variable. Replace `YOUR_API_KEY` with your actual key.

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

### 3. Choose a Model
Select the Gemini model you wish to use. This guide uses `gemini-2.0-flash` as an example.

```bash
export MODEL_ID="gemini-2.0-flash"
```

## Step 1: Download the Sample Audio File

You'll use a public domain audio file for this tutorial. Download it to your local directory.

```bash
wget https://storage.googleapis.com/generativeai-downloads/data/State_of_the_Union_Address_30_January_1961.mp3 -O sample.mp3
```

## Step 2: Prepare File Metadata

To upload the file, you need its MIME type and size. Store these details in environment variables for later use.

```bash
# Define the path to your audio file
AUDIO_PATH="./sample.mp3"

# Extract the MIME type (e.g., audio/mpeg)
MIME_TYPE=$(file -b --mime-type "${AUDIO_PATH}")

# Get the file size in bytes
NUM_BYTES=$(wc -c < "${AUDIO_PATH}")

# Set a display name for the file
DISPLAY_NAME="Sample audio"

# Set the base URL for the API
BASE_URL="https://generativelanguage.googleapis.com"

echo "File: $DISPLAY_NAME"
echo "MIME Type: $MIME_TYPE"
echo "Size: $NUM_BYTES bytes"
```

## Step 3: Upload the File to the Gemini File API

The Gemini File API uses a resumable upload protocol. This step involves two `curl` requests: one to initiate the upload and get a unique URL, and another to send the actual file data.

### 3.1 Initiate the Upload and Get the Upload URL
This request tells the API you want to upload a file and provides its metadata. The response contains a temporary upload URL.

```bash
# Create a temporary file to store the response headers
tmp_header_file="upload-header.tmp"

# Send the initial request. The upload URL is in the response headers.
curl "${BASE_URL}/upload/v1beta/files?key=${GOOGLE_API_KEY}" \
  -D "${tmp_header_file}" \
  -H "X-Goog-Upload-Protocol: resumable" \
  -H "X-Goog-Upload-Command: start" \
  -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
  -H "Content-Type: application/json" \
  -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

# Extract the upload URL from the response headers
upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")

# Clean up the temporary header file
rm "${tmp_header_file}"

echo "Upload URL obtained: $upload_url"
```

### 3.2 Upload the File Data
Now, use the `upload_url` to send the audio file's binary data.

```bash
# Upload the file bytes
curl "${upload_url}" \
  -H "Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Offset: 0" \
  -H "X-Goog-Upload-Command: upload, finalize" \
  --data-binary "@${AUDIO_PATH}" 2> /dev/null > file_info.json

echo "File upload complete. Metadata saved to 'file_info.json'."
```

The response (`file_info.json`) contains the file's unique URI, which you'll need for prompting.

```bash
# Extract the file URI for the next steps
file_uri=$(jq -r ".file.uri" file_info.json)
echo "File URI: $file_uri"
```

## Step 4: Wait for File Processing

After upload, the File API processes the audio. You must wait until the file's state is `ACTIVE` before using it.

```bash
# Check the file's processing state
state=$(jq -r ".file.state" file_info.json)

while [[ "${state}" == "PROCESSING" ]];
do
  echo "File is still processing. Waiting 5 seconds..."
  sleep 5
  # Query the file's current state
  curl "${file_uri}?key=${GOOGLE_API_KEY}" > file_info.json 2>/dev/null
  state=$(jq -r ".state" file_info.json)
done

echo "File is now ${state} and ready to use."
```

## Step 5: Verify the Uploaded File

You can verify the file details by calling the `files.get` endpoint.

```bash
curl "${file_uri}?key=${GOOGLE_API_KEY}" 2>/dev/null | jq .
```

You should see a JSON response with details like `displayName`, `mimeType`, `sizeBytes`, and `state: "ACTIVE"`.

To list all files associated with your API key:

```bash
curl "https://generativelanguage.googleapis.com/v1beta/files?key=${GOOGLE_API_KEY}" 2>/dev/null | jq .
```

## Step 6: Prompt the Model with the Audio File

Now, use the uploaded audio file in a prompt to the Gemini model. The content structure uses a `file_data` part that references the file's URI.

```bash
curl "${BASE_URL}/v1beta/models/${MODEL_ID}:generateContent?key=${GOOGLE_API_KEY}" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[
          {"text": "Please describe this file."},
          {"file_data": {
            "mime_type": "'${MIME_TYPE}'",
            "file_uri": "'${file_uri}'"
          }}
        ]
      }]
    }' 2>/dev/null > response.json

# Display the model's response
jq -r '.candidates[0].content.parts[0].text' response.json
```

The model should return a description of the audio content, such as:
> "This is an audio recording of President John F. Kennedy delivering his State of the Union address..."

## Step 7: Clean Up (Delete the File)

Uploaded files expire after 48 hours, but you can delete them manually. This command fetches all files for your key and deletes them.

```bash
# Fetch the list of files
files_json=$(curl -s "https://generativelanguage.googleapis.com/v1beta/files?key=${GOOGLE_API_KEY}")

# Extract file names
file_names=$(echo "$files_json" | jq -r '.files[].name')

# Loop through and delete each file
for file_name in $file_names; do
  echo "Deleting: $file_name"
  curl --request "DELETE" "https://generativelanguage.googleapis.com/v1beta/${file_name}?key=${GOOGLE_API_KEY}" 2>/dev/null
done

echo "Cleanup complete."
```

## Bonus: Using Small Audio Files Directly (Inline Data)

For audio files under **100MB**, you can bypass the File API and send the audio data directly (Base64-encoded) within your prompt using `inline_data`.

### 1. Create Small Test Clips
Extract two 30-second clips from the original file.

```bash
# Create a clip from 0-30 seconds
ffmpeg -i sample.mp3 -t 30 -c copy sample_30s.mp3 2>/dev/null

# Create a clip from 31-60 seconds
ffmpeg -ss 30 -to 60 -i sample.mp3 -c copy sample_31-60.mp3 2>/dev/null
```

### 2. Encode Audio to Base64 and Create a Request
Prepare a JSON request with the Base64 data.

```bash
# Encode the first clip to Base64
data1_b64=$(base64 -w 0 sample_30s.mp3)

# Create the request JSON file
echo '{
  "contents": [{
    "parts":[
      {"text": "Summarize this clip"},
      {"inline_data": {
        "mime_type": "'${MIME_TYPE}'",
        "data": "'"${data1_b64}"'"
      }}
    ]
  }]
}' > request.json
```

### 3. Send the Inline Audio Request
Prompt the model with the inline audio data.

```bash
curl "${BASE_URL}/v1beta/models/${MODEL_ID}:generateContent?key=${GOOGLE_API_KEY}" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d @request.json 2>/dev/null > inline_response.json

jq -r '.candidates[0].content.parts[0].text' inline_response.json
```

The response will be a summary of the short audio clip.

## Summary & Further Reading

You've successfully learned how to:
1.  Upload audio files to the Gemini File API using resumable upload.
2.  Wait for file processing and verify the upload.
3.  Prompt a Gemini model with the uploaded audio file.
4.  Delete files after use.
5.  Send small audio files directly as inline Base64 data.

**Key Notes:**
*   **Security:** Your API key grants access to files you upload. Keep it secure.
*   **File Limits:** The File API accepts files under 2GB, with a 20GB total storage limit per project. Files expire after 48 hours.
*   **Inline Data:** Use `inline_data` for files under 100MB for simpler, one-off prompts.

**Further Resources:**
*   [Gemini API Documentation: File API](https://ai.google.dev/api/files)
*   [Prompting with Media Files](https://ai.google.dev/tutorials/prompting_with_media)
*   [Content Generation Endpoint](https://ai.google.dev/api/generate-content)
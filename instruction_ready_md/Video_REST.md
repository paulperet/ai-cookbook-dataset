# Gemini API: Video Prompting with REST

This guide provides step-by-step instructions for prompting the Gemini API using a video file via `curl` commands. You will use a short clip of [Big Buck Bunny](https://peach.blender.org/about/) as an example.

You can follow along in a terminal or adapt the commands for use in a script.

## Prerequisites

Before you begin, ensure you have the following:

1.  **A Google Cloud Project** with the Gemini API enabled.
2.  **An API Key** for authentication. Store it securely in an environment variable named `GOOGLE_API_KEY`.
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
    > **Important Security Note:** The File API uses API keys for authentication. Uploaded files are associated with your cloud project, and your API key grants access to them. Treat your API key with the same level of security as your cloud project credentials. For best practices, refer to the [API console support center](https://support.google.com/googleapi/answer/6310037).

3.  **`curl` and `jq` installed.** `jq` is a lightweight command-line JSON processor that will help parse API responses.
    ```bash
    # On Debian/Ubuntu systems
    sudo apt update && sudo apt install -y curl jq
    ```

## Step 1: Download the Example Video

You will use a publicly available short film for this tutorial. Download it to your local directory.

```bash
wget https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4
```

> **Note:** "Big Buck Bunny" is (C) Copyright 2008, Blender Foundation / www.bigbuckbunny.org and is [licensed](https://peach.blender.org/about/) under the [Creative Commons Attribution 3.0](http://creativecommons.org/licenses/by/3.0/) License.

## Step 2: Generate File Metadata

Before uploading, you need to gather some metadata about the video file: its MIME type and size in bytes. This information is required for the upload process.

Create a script or run the following commands in your terminal:

```bash
# Define the path and a display name for your file
VIDEO_PATH="./BigBuckBunny_320x180.mp4"
DISPLAY_NAME="Big Buck Bunny"

# Auto-detect the MIME type and file size
MIME_TYPE=$(file -b --mime-type "${VIDEO_PATH}")
NUM_BYTES=$(wc -c < "${VIDEO_PATH}")

# Print the detected values for verification
echo "File: $VIDEO_PATH"
echo "MIME Type: $MIME_TYPE"
echo "Size (bytes): $NUM_BYTES"

# Save these variables to a file for use in subsequent steps
cat >./vars.sh <<-EOF
  export BASE_URL="https://generativelanguage.googleapis.com"
  export DISPLAY_NAME="${DISPLAY_NAME}"
  export VIDEO_PATH=${VIDEO_PATH}
  export MIME_TYPE=${MIME_TYPE}
  export NUM_BYTES=${NUM_BYTES}
EOF
```

You should see output similar to:
```
File: ./BigBuckBunny_320x180.mp4
MIME Type: video/mp4
Size (bytes): 64657027
```

The `vars.sh` file now contains the necessary environment variables.

## Step 3: Initialize the Resumable Upload

The Gemini File API uses resumable uploads. The first step is to create an upload task, which provides a unique URL for sending the file data.

1.  Source the variables from the previous step.
2.  Send a `POST` request to the files endpoint with specific upload headers.
3.  The API response will contain an `x-goog-upload-url` header. You will extract and use this URL to send the actual video data.

```bash
# Load the environment variables
source ./vars.sh

# Create the initial upload request
curl "${BASE_URL}/upload/v1beta/files?key=${GOOGLE_API_KEY}" \
  -D upload-header.tmp \
  -H "X-Goog-Upload-Protocol: resumable" \
  -H "X-Goog-Upload-Command: start" \
  -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
  -H "Content-Type: application/json" \
  -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2>/dev/null

# Check the HTTP status of the response
echo "Upload initialization status:"
head -1 upload-header.tmp
```

A successful response will return `HTTP/2 200`.

## Step 4: Upload the Video Data

Now, use the upload URL from the previous response to send the binary content of the video file.

```bash
# Load variables again
source ./vars.sh

# Extract the upload URL from the saved headers
upload_url=$(grep -i "x-goog-upload-url: " upload-header.tmp | cut -d" " -f2 | tr -d "\r")

# Securely remove the header file as it contains the API key
rm upload-header.tmp

# Upload the video bytes to the provided URL
curl "${upload_url}" \
  -H "Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Offset: 0" \
  -H "X-Goog-Upload-Command: upload, finalize" \
  --data-binary "@${VIDEO_PATH}" > file_info.json 2>/dev/null

# Display the API's response, which contains the file's metadata
echo "File upload response:"
cat file_info.json
```

The response is a JSON object containing details about your uploaded file. The most important field is `uri`, which you will use to reference this file in future API calls. Note that the `state` might initially be `"PROCESSING"`.

Example response snippet:
```json
{
  "file": {
    "name": "files/4if4o2bqvugf",
    "displayName": "Big Buck Bunny",
    "mimeType": "video/mp4",
    "sizeBytes": "64657027",
    "uri": "https://generativelanguage.googleapis.com/v1beta/files/4if4o2bqvugf",
    "state": "ACTIVE"
  }
}
```

## Step 5: Verify File Status

After uploading, you can query the File API to confirm the file is ready (`state: "ACTIVE"`) and retrieve its metadata.

Use the `uri` from the `file_info.json` response to call the `files.get` endpoint.

```bash
# Load variables
source ./vars.sh

# Extract the file URI from the saved JSON response
file_uri=$(jq -r ".file.uri" file_info.json)

# Query the File API for the file's current status and info
curl "${file_uri}?key=${GOOGLE_API_KEY}" 2>/dev/null
```

This command returns the complete file object. Ensure the `state` field is `"ACTIVE"` before attempting to use the file with the Gemini model. The response will also include metadata like `videoDuration`.

## Next Steps: Prompting the Model with the Video

Now that your video file is uploaded and active, you can use it with the Gemini API. The file is referenced by its `uri` (or `name`) in the request payload.

Here is a conceptual example of how you would structure a `curl` request to the `generateContent` endpoint using the uploaded video:

```bash
# Construct a request to the model
curl -X POST \
  "${BASE_URL}/v1beta/models/gemini-1.5-pro:generateContent?key=${GOOGLE_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [{
      "parts": [
        {"text": "Describe what is happening in this video."},
        {
          "file_data": {
            "mime_type": "video/mp4",
            "file_uri": "'"${file_uri}"'"
          }
        }
      ]
    }]
  }'
```

**Key Points:**
*   Replace the `file_uri` value with the one from your `file_info.json`.
*   The `mime_type` in the request must match the file's actual MIME type.
*   The file will be automatically deleted after 2 days.

You have successfully uploaded a video file to the Gemini File API using REST. You can now integrate this file URI into your prompts to build multimodal applications.
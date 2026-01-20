##### Copyright 2025 Google LLC.


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file file except in compliance with the License.
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

# Gemini API: Video prompting with REST

This notebook provides quick code examples that show you how  to prompt the Gemini API using a video file with `curl`. In this case, you'll use a short clip of [Big Buck Bunny](https://peach.blender.org/about/).

You can run this in Google Colab, or you can copy/paste the `curl` commands into your terminal.

## Set up the environment

To run this notebook, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) to learn more.

### Authentication Overview

**Important:** The File API uses API keys for authentication and access. Uploaded files are associated with the API key's cloud project. Unlike other Gemini APIs that use API keys, your API key also grants access data you've uploaded to the File API, so take extra care in keeping your API key secure. For best practices on securing API keys, refer to the [API console support center](https://support.google.com/googleapi/answer/6310037).


```
import os
from google.colab import userdata

os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

Install `jq` to help with processing of JSON API responses.


```
!apt install -q jq
```

## Use a video file with the Gemini API

The Gemini API accepts video file formats through the File API. The File API accepts files under 2GB in size and can store up to 20GB of files per project. Files last for 2 days and cannot be downloaded from the API. For this example, you will use the short film "Big Buck Bunny".

> "Big Buck Bunny" is (C) Copyright 2008, Blender Foundation / www.bigbuckbunny.org and [licensed](https://peach.blender.org/about/) under the [Creative Commons Attribution 3.0](http://creativecommons.org/licenses/by/3.0/) License.

Note: In Colab, you can also [upload your own files](https://github.com/google-gemini/cookbook/blob/main/examples/Upload_files_to_Colab.ipynb) to use.


```
!wget https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4
```

With the video file now available locally, generate some metadata that you will use in subsequent steps.


```bash
%%bash

VIDEO_PATH="./BigBuckBunny_320x180.mp4"
DISPLAY_NAME="Big Buck Bunny"

# Auto-detect the metadata needed when you upload the video.
MIME_TYPE=$(file -b --mime-type "${VIDEO_PATH}")
NUM_BYTES=$(wc -c < "${VIDEO_PATH}")

echo $VIDEO_PATH $MIME_TYPE $NUM_BYTES

# Colab doesn't allow sharing shell variables between cells, so save them.
cat >./vars.sh <<-EOF
  export BASE_URL="https://generativelanguage.googleapis.com"
  export DISPLAY_NAME="${DISPLAY_NAME}"
  export VIDEO_PATH=${VIDEO_PATH}
  export MIME_TYPE=${MIME_TYPE}
  export NUM_BYTES=${NUM_BYTES}
EOF
```

    ./BigBuckBunny_320x180.mp4 video/mp4 64657027


### Start the upload task

Media uploads in the File API are resumable, so the first step is to define an upload task. This initial request gives you a reference you can use for subsequent upload operations, and allows you to query the status of the upload before sending data, in case of network issues during the data transfer.

The API returns the upload URL in the `x-goog-upload-url` header, so take note of that in the response headers - you will send the payload data to this URL.

No payload data (video bytes) are sent in this initial request.


```bash
%%bash
. vars.sh

# Create the "new upload" request by providing the relevant metadata.
curl "${BASE_URL}/upload/v1beta/files?key=${GOOGLE_API_KEY}" \
  -D upload-header.tmp \
  -H "X-Goog-Upload-Protocol: resumable" \
  -H "X-Goog-Upload-Command: start" \
  -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
  -H "Content-Type: application/json" \
  -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2>/dev/null

# Print the status.
head -1 upload-header.tmp
```

    HTTP/2 200 


### Upload video data

Now that you have created the upload task, you can upload the file data by sending bytes to the returned upload URL.


```bash
%%bash
. vars.sh

# Extract the upload URL to use from the response headers.
upload_url=$(grep -i "x-goog-upload-url: " upload-header.tmp | cut -d" " -f2 | tr -d "\r")
# The header contains our API key, so don't leave it lying around.
rm upload-header.tmp

# Upload the actual bytes.
curl "${upload_url}" \
  -H "Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Offset: 0" \
  -H "X-Goog-Upload-Command: upload, finalize" \
  --data-binary "@${VIDEO_PATH}" >file_info.json 2>/dev/null

# Show the output. You will use it in a later step.
cat file_info.json
```

    {
      "file": {
        "name": "files/4if4o2bqvugf",
        "displayName": "Big Buck Bunny",
        "mimeType": "video/mp4",
        "sizeBytes": "64657027",
        "createTime": "2024-08-26T08:24:56.068012Z",
        "updateTime": "2024-08-26T08:24:56.068012Z",
        "expirationTime": "2024-08-28T08:24:56.049455995Z",
        "sha256Hash": "Zjc4ZjM5NjAzZTY3NzQ5MDdmMmZhYWZhYmYyNmE2NjdmNGE2ZmMzMTc2OWVjMzA0YThhOGY3YzYyZDI4MDUwOA==",
        "uri": "https://generativelanguage.googleapis.com/v1beta/files/4if4o2bqvugf",
        "state": "PROCESSING"
      }
    }


### Get file info

After uploading the file, you can verify the API has successfully received the files by querying the [`files.get` endpoint](https://ai.google.dev/api/files#method:-files.get).

`files.get` lets you see the file uploaded to the File API that are associated with the Cloud project your API key belongs to. Only the `name` (and by extension, the `uri`) are unique.


```bash
%%bash
. vars.sh

file_uri=$(jq -r ".file.uri" file_info.json)

curl "${file_uri}?key=${GOOGLE_API_KEY}" 2>/dev/null
```

    {
      "name": "files/4if4o2bqvugf",
      "displayName": "Big Buck Bunny",
      "mimeType": "video/mp4",
      "sizeBytes": "64657027",
      "createTime": "2024-08-26T08:24:56.068012Z",
      "updateTime": "2024-08-26T08:25:03.977029Z",
      "expirationTime": "2024-08-28T08:24:56.049455995Z",
      "sha256Hash": "Zjc4ZjM5NjAzZTY3NzQ5MDdmMmZhYWZhYmYyNmE2NjdmNGE2ZmMzMTc2OWVjMzA0YThhOGY3YzYyZDI4MDUwOA==",
      "uri": "https://generativelanguage.googleapis.com/v1beta/files/4if4o2bqvugf",
      "state": "ACTIVE",
      "videoMetadata": {
        "videoDuration": "596s",
        "videoThumbnailBytes": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAC0AUADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEFSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+Eeiiiuc88KKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAoqxRQaez8/w/4IUVXqTy/f8AT/69Z+08vx/4BoSUVYpMKOgI9cnP9K0AgooorP2nl+P/AAALFFV6KPaeX4/8ACxRVeij2nl+P/AAsUUUVoAUUUUAFFFFABRRRQAUUUUAZ9FFFBzhRRRQAUUUUAFFFFABRRRQAUUUUAFFWKKDT2fn+H/BCiiig0CiiigAoqxRQBXooornAKKKKACiiigCxRRRXQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQBn0UUUHOFFFFABRRRQAUUUUAFWKr1YoNKfX5fqFFV6KA9p5fj/AMAsUUUUGgUUVYoAgwfQ/kaMH0P5GkorP2nl+P8AwALFFV6KPaeX4/8AAAKKKKzAsUUUV0AFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQBn0UUUHOFFFFABRRRQAUUUUAWKKKKDoCiiigCTy/f8AT/69R4YfeAHphs/XsMVH5nt+v/1qkrP2nl+P/AAKkj7/AIf1ok7fj/So60AsVXqxVeplHmtrawBRVerFYgFWKr0VUZct9L3AsUUUVsAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAGfRRRQc4UUUUAFSeX7/AKf/AF6kooNPZx62l2utvxCiiig0CoR91v8AgP8AOpqKVvejL+V3t3AKKKKYFepIx8u7+8AcenWpKjj7/h/WsZR5ba3uBJVerFFSBXooqxVRjzX1tYAqOPv+H9akqOPv+H9aqp0+f6ASUUUVmBYoqvVitoy5r6WsAUUUVQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAFeiiiucCvRRRXQc4UUUVn7Ty/H/gGns/P8P+CWKKKr1UpcttL3NCxRVerFEZc19LWAKKKKoAooooAKKKKACq9WKKmUea2trAV6sUUVPwed/lt9/cAooorQAooqTzPb9f8A61Z+z8/w/wCCBHRRRWYFiiq9WK2jLmvpawBRRRVAFFFFABRRRQAUUUUAV6KKK5wCiiigAooooAKKKK6AK9FFFc4BRRRQA5W254zmpGXdjnGKrsu7HOMVIrbc8ZzQBNRRQeGK+gU5+ue3titoy5r6WsAUUUVQBRRRQAUVXqxWftPL8f8AgAFR+Z7fr/8AWqcoR05/T+tVKKnT5/oBMy7sc4xTqKkk7fj/AEop9fl+oEdFFFaAFFFRydvx/pUyfLFyteyvYCSiq9WKxAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiq9Fae08vx/4ABRRRWYBRRRQAUUUUASeZ7fr/wDWo8v3/T/69R0UAFFFFAFiiq9Fae08vx/4ABRUnl+/6f8A16PL9/0/+vWYB5nt+v8A9apKr1J5nt+v/wBagCSiiigBkisMZVh16g+1R4PofyNT1HJ2/H+lAElV6KKqUua2lrAFSR9/w/rUlMQEZyPT+tSA+iiigAooooAKKKKACiiigAooooAKKKKACiiigCvRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAWKKr0UASeZ7fr/9apKr1J5rH7xLemT09fzoAkooooAjk7fj/So6kk7fj/So6ACpPM9v1/8ArVHRQBYoqvVigAooooAKKj8z2/X/AOtQZF7Yb156fpQBJRUPm5IwvynOGz1x14x9O9QG4GSH4x05U5z1+6eMY9aDP2nl+P8AwC7RVX7XF/k0C7iLBc9QTnPpj/Ggpzgvtferf5lqiqv2uL/JqwXj/hYt68AY9P4jQEZc19LWHUUwSIPvMPbbhvzyy49uueelAkYfc3D1xn8Oh+tBRFRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQBYoqvRQBJJwM+mf5Z/pUdSScDPpn+Wf6VHQAUUUhOFZv7qk49cdqAGh+Sp6jv2PJH4HirEZzwTgDvjJGfxHA/P06YrDSXL7iMEsBjPr7+vHp/hW5wiI3Zl3Y7jrgZHUDHp39uQyptO7Xl+pPHJszxnOO+OmfY+tQb0PRgfzqpNKzNIihUyQpIHLEdM/rg9snrmqkcjc7st0xk9OvtSTur9yPb/ANz/AMm/+1L8jNt2sUbJyCpzjHrx3zx9DTlUvnICY/uDGc+vPbHH1NVI2QNt2iXd3C524/Hoc/pVwDyd38W5GGDwvGPvLzu68cjB55plRjzX1tYz5Hwx5bd2PPGMZ4OAc+2ff0NVp852Z+5twTwG57A9ADwOopZGxkjHDEc/Rf8APXpVQs3G3HHIzkE9Bx144yOhxQRNS0d0r3erir7d2u/QsFyCVXIUAYABb7wYHkKfXPPX65NIhY5yQemMDHr708HdHnphVQ/VQ+f5ilwAnO07fUhc5wO4bgY59KCUno76au3a9rL5a2/Q
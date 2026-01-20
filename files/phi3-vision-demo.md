```python
# pip install opencv-python
```

```python
# import cv2
# import numpy as np
```

```python
# def save_keyframes(video_path, output_folder):
#     videoCapture = cv2.VideoCapture(video_path)
#     success, frame = videoCapture.read()
#     i = 0
#     while success:
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        
#         success, next_frame = videoCapture.read()
#         if not success:
#             break
        
#         next_gray_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        
#         next_hist = cv2.calcHist([next_gray_frame], [0], None, [256], [0, 256])
        
#         similarity = cv2.compareHist(hist, next_hist, cv2.HISTCMP_CORREL)
        
#         if similarity < 0.9:
#             i += 1
#             cv2.imwrite(f"{output_folder}/keyframe_{i}.jpg", frame)
#             print(f"Saved keyframe {i}")
        
#         frame = next_frame

#     videoCapture.release()
```

```python
# save_keyframes('../video/copilot.mp4', '../output')
```

```python
from PIL import Image
import requests, base64
```

```python
images = [] 
placeholder = "" 
for i in range(1,22): 
    with open("../output/keyframe_"+str(i)+".jpg", "rb") as f:

        images.append(Image.open("../output/keyframe_"+str(i)+".jpg"))
        placeholder += f"<|image_{i}|>\n"
        # print(i)
```

```python
images
```




    [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>,
     <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720>]




```python
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor
```

    2024-08-19 12:00:55.508879: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2024-08-19 12:00:55.508910: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.



```python
model_id = "../Phi3Vision"
```

```python
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2')
```

    [Loading checkpoint shards: ..., Last Entry]



```python
messages = [
                {"role": "user", "content": placeholder+"Summarize the video."}, 
]
```

```python
pip install transformers -U
```

    [Requirement already satisfied: transformers in /anaconda/envs/azureml_py38/lib/python3.9/site-packages (4.44.0), ..., Note: you may need to restart the kernel to use updated packages.]



```python
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor


# from image_embedding_phi3_v import Phi3VImageProcessor 

# transformers.Phi3VImageProcessor = Phi3VImageProcessor 
```

```python
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
```

    /anaconda/envs/azureml_py38/lib/python3.9/site-packages/transformers/models/auto/image_processing_auto.py:513: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
      warnings.warn(



```python
pip install jinja2 -U
```

    [Requirement already satisfied: jinja2 in /anaconda/envs/azureml_py38/lib/python3.9/site-packages (3.1.4), ..., Note: you may need to restart the kernel to use updated packages.]



```python
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

```python
inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")
```

```python
generation_args = { "max_new_tokens": 1000, "temperature": 0.0, "do_sample": False, }
```

```python
generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
```

    [The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` model input instead., ..., UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.]



```python
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
```

```python
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

```python
response
```




    "The video appears to be a promotional or informational piece about a product or service called 'Copilot'. It showcases various individuals in different settings, such as an office and a home office, engaging with the product. The video includes text overlays that suggest the product can help with tasks like lowering production costs, managing meetings, and creating presentations. There are also screenshots of a chat interface and a summary of a product launch discussion, indicating that the product may be related to project management or collaboration. The video seems to emphasize the efficiency and effectiveness of the Copilot product in a professional context."
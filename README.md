# AI Recipes Dataset

A dataset built from open-source documentations. The goal is to create a high quality dataset for fine-tuning a LLM with expert knowledge on developping AI applications.

## Preprocessing
Multiple preprocessing steps were used to make the dataset ready for LoRa fine-tuning: 
1. Convert the ipynb files to markdown as it will be our output format. This was done thanks to nbconvert. 
2. Use an LLM (DeepSeek V3.1) to clean the clutter from the output cells of the converted notebooks.
3. Format all my examples using DeepSeek V3.1 to make them more like a step-by-step guide. This was done in the intent of having a high quality homogeneous dataset to maximize the effects of LoRa fine-tuning.

## Sources
- https://github.com/openai/openai-cookbook
- https://github.com/d2l-ai/d2l-en
- https://github.com/google-gemini/cookbook
- https://github.com/anthropics/claude-cookbooks
- https://github.com/pytorch/tutorials
- https://github.com/huggingface/cookbook
- https://github.com/meta-llama/llama-cookbook
- https://github.com/mistralai/cookbook
- https://github.com/microsoft/PhiCookBook

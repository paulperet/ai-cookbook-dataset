# Prompt Tuning With PEFT.
_Authored by: [Pere Martra](https://github.com/peremartra)_

In this notebook we are introducing how to apply prompt tuning with the PEFT library to a pre-trained model.

For a complete list of models compatible with PEFT refer to their [documentation](https://huggingface.co/docs/peft/main/en/index#supported-methods).

A short sample of models available to be trained with PEFT includes Bloom, Llama, GPT-J, GPT-2, BERT, and more. Hugging Face is working hard to add more models to the library.

## Brief introduction to Prompt Tuning.
It’s an Additive Fine-Tuning technique for models. This means that we WILL NOT MODIFY ANY WEIGHTS OF THE ORIGINAL MODEL. You might be wondering, how are we going to perform Fine-Tuning then? Well, we will train additional layers that are added to the model. That’s why it’s called an Additive technique.

Considering it’s an Additive technique and its name is Prompt-Tuning, it seems clear that the layers we’re going to add and train are related to the prompt.

We are creating a type of superprompt by enabling a model to enhance a portion of the prompt with its acquired knowledge. However, that particular section of the prompt cannot be translated into natural language. **It's as if we've mastered expressing ourselves in embeddings and generating highly effective prompts.**

In each training cycle, the only weights that can be modified to minimize the loss function are those integrated into the prompt.

The primary consequence of this technique is that the number of parameters to train is genuinely small. However, we encounter a second, perhaps more significant consequence, namely that, **since we do not modify the weights of the pretrained model, it does not alter its behavior or forget any information it has previously learned.**

The training is faster and more cost-effective. Moreover, we can train various models, and during inference time, we only need to load one foundational model along with the new smaller trained models because the weights of the original model have not been altered

## What are we going to do in the notebook?
We are going to train two different models using two datasets, each with just one pre-trained model from the Bloom family. One model will be trained with a dataset of prompts, while the other will use a dataset of inspirational sentences. We will compare the results for the same question from both models before and after training.

Additionally, we'll explore how to load both models with only one copy of the foundational model in memory.

## Loading the PEFT Library
This library contains the Hugging Face implementation of various Fine-Tuning techniques, including Prompt Tuning

```python
!pip install -q peft==0.8.2
```

```python
!pip install -q datasets==2.14.5
```

From the transformers library, we import the necessary classes to instantiate the model and the tokenizer.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

### Loading the model and the tokenizers.

Bloom is one of the smallest and smartest models available for training with the PEFT Library using Prompt Tuning. You can choose any model from the Bloom Family, and I encourage you to try at least two of them to observe the differences.

I'm opting for the smallest one to minimize training time and avoid memory issues in Colab.

```python
model_name = "bigscience/bloomz-560m"
#model_name="bigscience/bloom-1b1"
NUM_VIRTUAL_TOKENS = 4
NUM_EPOCHS = 6
```

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
foundational_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)
```

## Inference with the pre trained bloom model
If you want to achieve more varied and original generations, uncomment the parameters: temperature, top_p, and do_sample, in *model.generate* below

With the default configuration, the model's responses remain consistent across calls.

```python
#this function returns the outputs from the model received, and inputs.
def get_outputs(model, inputs, max_new_tokens=100):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        #temperature=0.2,
        #top_p=0.95,
        #do_sample=True,
        repetition_penalty=1.5, #Avoid repetition.
        early_stopping=True, #The model can stop before reach the max_length
        eos_token_id=tokenizer.eos_token_id
    )
    return outputs
```

As we want to have two different trained models, I will create two distinct prompts.

The first model will be trained with a dataset containing prompts, and the second one with a dataset of motivational sentences.

The first model will receive the prompt "I want you to act as a motivational coach." and the second model will receive "There are two nice things that should matter to you:"

But first, I'm going to collect some results from the model without Fine-Tuning.

```python
input_prompt = tokenizer("I want you to act as a motivational coach. ", return_tensors="pt")
foundational_outputs_prompt = get_outputs(foundational_model, input_prompt, max_new_tokens=50)

print(tokenizer.batch_decode(foundational_outputs_prompt, skip_special_tokens=True))
```

    ["I want you to act as a motivational coach.  Don't be afraid of being challenged."]

```python
input_sentences = tokenizer("There are two nice things that should matter to you:", return_tensors="pt")
foundational_outputs_sentence = get_outputs(foundational_model, input_sentences, max_new_tokens=50)

print(tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True))
```

    ['There are two nice things that should matter to you: the price and quality of your product.']

Both answers are more or less correct. Any of the Bloom models is pre-trained and can generate sentences accurately and sensibly. Let's see if, after training, the responses are either equal or more accurately generated.

## Preparing the Datasets
The Datasets useds are:
* https://huggingface.co/datasets/fka/awesome-chatgpt-prompts
* https://huggingface.co/datasets/Abirate/english_quotes

```python
import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

```python
from datasets import load_dataset

dataset_prompt = "fka/awesome-chatgpt-prompts"

#Create the Dataset to create prompts.
data_prompt = load_dataset(dataset_prompt)
data_prompt = data_prompt.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
train_sample_prompt = data_prompt["train"].select(range(50))

```

```python
display(train_sample_prompt)
```

    Dataset({
        features: ['act', 'prompt', 'input_ids', 'attention_mask'],
        num_rows: 50
    })

```python
print(train_sample_prompt[:1])
```

    {'act': ['Linux Terminal'], 'prompt': ['I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets {like this}. my first command is pwd'], 'input_ids': [[44, 4026, 1152, 427, 1769, 661, 267, 104105, 28434, 17, 473, 2152, 4105, 49123, 530, 1152, 2152, 57502, 1002, 3595, 368, 28434, 3403, 6460, 17, 473, 4026, 1152, 427, 3804, 57502, 1002, 368, 28434, 10014, 14652, 2592, 19826, 4400, 10973, 15, 530, 16915, 4384, 17, 727, 1130, 11602, 184637, 17, 727, 1130, 4105, 49123, 35262, 473, 32247, 1152, 427, 727, 1427, 17, 3262, 707, 3423, 427, 13485, 1152, 7747, 361, 170205, 15, 707, 2152, 727, 1427, 1331, 55385, 5484, 14652, 6291, 999, 117805, 731, 29726, 1119, 96, 17, 2670, 3968, 9361, 632, 269, 42512]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

```python
dataset_sentences = load_dataset("Abirate/english_quotes")

data_sentences = dataset_sentences.map(lambda samples: tokenizer(samples["quote"]), batched=True)
train_sample_sentences = data_sentences["train"].select(range(25))
train_sample_sentences = train_sample_sentences.remove_columns(['author', 'tags'])
```

```python
display(train_sample_sentences)
```

    Dataset({
        features: ['quote', 'input_ids', 'attention_mask'],
        num_rows: 25
    })

## Fine-Tuning.  

### PEFT configurations

API docs:
https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.PromptTuningConfig

We can use the same configuration for both models to be trained.

```python
from peft import  get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit

generation_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, #This type indicates the model will generate text.
    prompt_tuning_init=PromptTuningInit.RANDOM,  #The added virtual tokens are initializad with random numbers
    num_virtual_tokens=NUM_VIRTUAL_TOKENS, #Number of virtual tokens to be added and trained.
    tokenizer_name_or_path=model_name #The pre-trained model.
)

```

### Creating two Prompt Tuning Models.
We will create two identical prompt tuning models using the same pre-trained model and the same config.

```python
peft_model_prompt = get_peft_model(foundational_model, generation_config)
print(peft_model_prompt.print_trainable_parameters())
```

    trainable params: 4,096 || all params: 559,218,688 || trainable%: 0.0007324504863471229
    None

```python
peft_model_sentences = get_peft_model(foundational_model, generation_config)
print(peft_model_sentences.print_trainable_parameters())
```

    trainable params: 4,096 || all params: 559,218,688 || trainable%: 0.0007324504863471229
    None

**That's amazing: did you see the reduction in trainable parameters? We are going to train a 0.001% of the paramaters available.**

Now we are going to create the training arguments, and we will use the same configuration in both trainings.

```python
from transformers import TrainingArguments
def create_training_arguments(path, learning_rate=0.0035, epochs=6):
    training_args = TrainingArguments(
        output_dir=path, # Where the model predictions and checkpoints will be written
        use_cpu=True, # This is necessary for CPU clusters.
        auto_find_batch_size=True, # Find a suitable batch size that will fit into memory automatically
        learning_rate= learning_rate, # Higher learning rate than full Fine-Tuning
        num_train_epochs=epochs
    )
    return training_args
```

```python

import os

working_dir = "./"

#Is best to store the models in separate folders.
#Create the name of the directories where to store the models.
output_directory_prompt =  os.path.join(working_dir, "peft_outputs_prompt")
output_directory_sentences = os.path.join(working_dir, "peft_outputs_sentences")

#Just creating the directoris if not exist.
if not os.path.exists(working_dir):
    os.mkdir(working_dir)
if not os.path.exists(output_directory_prompt):
    os.mkdir(output_directory_prompt)
if not os.path.exists(output_directory_sentences):
    os.mkdir(output_directory_sentences)

```

We need to indicate the directory containing the model when creating the TrainingArguments.

```python
training_args_prompt = create_training_arguments(output_directory_prompt, 0.003, NUM_EPOCHS)
training_args_sentences = create_training_arguments(output_directory_sentences, 0.003, NUM_EPOCHS)
```

## Train

We will create the trainer Object, one for each model to train.  

```python
from transformers import Trainer, DataCollatorForLanguageModeling
def create_trainer(model, training_args, train_dataset):
    trainer = Trainer(
        model=model, # We pass in the PEFT version of the foundation model, bloomz-560M
        args=training_args, #The args for the training.
        train_dataset=train_dataset, #The dataset used to tyrain the model.
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) # mlm=False indicates not to use masked language modeling
    )
    return trainer

```

```python
#Training first model.
trainer_prompt = create_trainer(peft_model_prompt, training_args_prompt, train_sample_prompt)
trainer_prompt.train()
```

    [Training Loss: 3.5800417945498513, ..., Training Loss: 3.5800417945498513]

```python
#Training second model.
trainer_sentences = create_trainer(peft_model_sentences, training_args_sentences, train_sample_sentences)
trainer_sentences.train()
```

    [Training Loss: 4.4278310139973955, ..., Training Loss: 4.4278310139973955]

In less than 10 minutes (CPU time in a M1 Pro) we trained 2 different models, with two different missions with a same foundational model as a base.

## Save models
We are going to save the models. These models are ready to be used, as long as we have the pre-trained model from which they were created in memory.

```python
trainer_prompt.model.save_pretrained(output_directory_prompt)
trainer_sentences.model.save_pretrained(output_directory_sentences)

```

## Inference

You can load the model from the path that you have saved to before, and ask the model to generate text based on our input before!

```python
from peft import PeftModel

loaded_model_prompt = PeftModel.from_pretrained(foundational_model,
                                         output_directory_prompt,
                                         #device_map='auto',
                                         is_trainable=False)
```

```python
loaded_model_prompt_outputs = get_outputs(loaded_model_prompt, input_prompt)
print(tokenizer.batch_decode(loaded_model_prompt_outputs, skip_special_tokens=True))
```

    ['I want you to act as a motivational coach.  You will be helping students learn how they can improve their performance in the classroom and at school.']

If we compare both answers something changed.
* ***Pretrained Model:*** *I want you to act as a motivational coach.  Don't be afraid of being challenged.*
* ***Fine-Tuned Model:*** *I want you to act as a motivational coach.  You can use this method if you're feeling anxious about your.*

We have to keep in mind that we have only trained the model for a few minutes, but they have been enough to obtain a response closer to what we were looking for.

```python
loaded_model_prompt.load_adapter(output_directory_sentences, adapter_name="quotes")
loaded_model_prompt.set_adapter("quotes")
```

```python
loaded_model_sentences_outputs = get_outputs(loaded_model_prompt, input_sentences)
print(tokenizer.batch_decode(loaded_model_sentences_outputs, skip_special_tokens=True))
```

    ['There are two nice things that should matter to you: the weather and your health.']

With the second model we have a similar result.
* **Pretrained Model:** *There are two nice things that should matter to you: the price and quality of your product.*
* **Fine-Tuned Model:** *There are two nice things that should matter to you: the weather and your health.*

# Conclusion
Prompt Tuning is an amazing technique that can save us hours of training and a significant amount of money. In the notebook, we have trained two models in just a few minutes, and we can have both models in memory, providing service to different clients.

If you want to try different combinations and models, the notebook is ready to use another model from the Bloom family.

You can change the number of epochs to train, the number of virtual tokens, and the model in the
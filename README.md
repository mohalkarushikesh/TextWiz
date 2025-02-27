# TextWiz

## Overview

**TextWiz** is a simple Language Model (LLM) project built using the Hugging Face Transformers library. It leverages pre-trained models like GPT-2 for text generation, fine-tuning, and interaction. TextWiz enables you to generate contextually relevant text responses, fine-tune the model on your own dataset, and deploy it for various NLP applications.

## Requirements

1. **Python** (version 3.7 or later)
2. **Hugging Face Transformers** library
3. **PyTorch** or **TensorFlow** (depending on which backend you prefer)

## Step-by-step Guide

### 1. Clone the Project
First, clone the project repository:
```bash
git clone https://github.com/mohalkarushikesh/TextWiz.git
cd TextWiz
```

### 2. Create a Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv venv
```
- **On Windows**:
  ```cmd
  .\venv\Scripts\activate
  ```
- **On macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install the Required Libraries
Install the necessary packages:
```bash
pip install transformers torch datasets
```
or
```bash
pip install -r requirements.txt
```

### 4. Load a Pre-trained Model
We'll start by loading a pre-trained model like `GPT-2` using Hugging Face's Transformers library.
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 model is small, you can switch to a larger version like 'gpt2-medium' or 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
```

### 5. Tokenizing Input
The input text needs to be tokenized before feeding it into the model. Hugging Face provides simple methods for this.
```python
# Input text
input_text = "Hello, how are you today?"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")
```

### 6. Generate Output
After tokenizing the input, you can generate output text from the model.
```python
# Generate text with sampling enabled
output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,  # You can adjust this to control the length of the output
    num_return_sequences=1,  # Number of output sequences you want
    no_repeat_ngram_size=2,  # This prevents repeating n-grams
    top_p=0.92,  # Top-p sampling for diversity
    temperature=0.85,  # Adjust the randomness of the output
    do_sample=True,  # Enable sampling
    pad_token_id=50256
)

# Decode output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text: ", generated_text)
```
It's important to keep the `accelerate` library and the `transformers` library versions compatible to avoid any potential issues. 

To ensure everything is up-to-date, you can run the following command to upgrade `accelerate`: 
```bash pip install accelerate -U ``` 

This will upgrade `accelerate` to the latest version available. Additionally, you can upgrade `transformers` to match the latest compatible version: 

```bash pip install transformers -U ``` 

By running these commands, you ensure that both `accelerate` and `transformers` are up-to-date and compatible with each other.

### 7. Fine-Tune the Model (Optional)
If you want to fine-tune the model on your own dataset, you can load a custom dataset and fine-tune it using the Trainer API. Here's a quick outline of how to fine-tune:
```python
from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

# Load a dataset (this is just an example; replace it with your dataset)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Preprocess the dataset
def preprocess_function(examples):
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()  # Set labels to input_ids
    return inputs

# Use a smaller subset of the dataset (first 100 samples)
small_dataset = dataset["train"].select(range(100))

# Tokenize the smaller dataset
tokenized_datasets = small_dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Output directory
    overwrite_output_dir=True,       # Overwrite existing files
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=4,   # Batch size per device during training
    save_steps=10_000,               # Save checkpoint every 10,000 steps
    logging_steps=500,               # Log every 500 steps
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir="./logs",            # Directory for storing logs
    report_to="none",                # Disable reporting to third-party services like TensorBoard
    remove_unused_columns=False,     # Prevent error for unmatched columns
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # Pre-trained model
    args=training_args,                  # Training arguments
    train_dataset=tokenized_datasets,    # Preprocessed smaller training dataset
)

# Fine-tune the model
trainer.train()
```
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`. The message indicates that the configuration contains an unrecognized 

`loss_type` value, and the model is defaulting to `ForCausalLMLoss`. This is a common loss function used for causal language modeling tasks, such as with GPT-2. Since 

`ForCausalLMLoss` is the appropriate loss function for this type of model, there's no action required unless you specifically want to use a different loss function. The default behavior is fine for most cases.

### 8. Save and Load the Fine-tuned Model
After fine-tuning, you can save the model:
```python
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
```
To load the fine-tuned model later:
```python
model = GPT2LMHeadModel.from_pretrained("fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_model")
```

### 9. Putting it All Together
You can combine all these steps into a simple script that generates text based on user input. Here's an example of a basic loop that will keep generating responses based on user input:
```python
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.92,
        temperature=0.85,
        do_sample=True,  # Enable sampling
        pad_token_id=50256
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = generate_response(user_input)
    print("Vortex: ", response)
```

### Conclusion
This is a very basic setup of a language model project using GPT-2. Depending on your requirements, you can adjust the parameters for generation, fine-tune the model on your data, or even deploy it in a web or chatbot application.
```

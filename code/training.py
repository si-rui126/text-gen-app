from datasets import Dataset
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM,
                          TrainingArguments, 
                          Trainer)
from peft import get_peft_model, LoraConfig
import os
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig

print('buidling model...')
model = 'gpt2'
llm = AutoModelForCausalLM.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# reading in data
parent = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
file_path = os.path.join(parent, 'text-gen-llm', 'data', 'ep', 'ep_text_cleaned.txt')
with open(file_path, encoding='utf-8', errors='ignore') as ep:
    ep_text = ep.readlines()

# preparing data for model
train, test = train_test_split(ep_text, test_size=0.2)
train_tokens = tokenizer(train, truncation=True, padding='max_length', max_length=200, return_tensors='pt')
test_tokens = tokenizer(test, truncation=True, padding='max_length', max_length=200, return_tensors='pt')

train_dataset = Dataset.from_dict({"input_ids": train_tokens["input_ids"], "attention_mask": train_tokens["attention_mask"], "labels": train_tokens["input_ids"]})
test_dataset = Dataset.from_dict({"input_ids": test_tokens["input_ids"], "attention_mask": test_tokens["attention_mask"], "labels": test_tokens["input_ids"]})

# Configure LoRA parameters
lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    target_modules=["h.0.attn.c_attn", 
                    "h.0.attn.c_proj",
                    "h.1.attn.c_attn", 
                    "h.1.attn.c_proj"],  # Layers to apply LoRA (adjust based on model)
    lora_dropout=0.1,  # Dropout for LoRA
    fan_in_fan_out=True
)

lora_llm = get_peft_model(llm, lora_config)

# Defining training arguments
output_dir = os.path.join(parent, 'text-gen-llm', 'results')
logs_dir = os.path.join(parent, 'text-gen-llm', 'logs')
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    logging_dir=logs_dir,
    save_steps=500,
    logging_steps=100
)

# Building the trainer!!
trainer = Trainer(
    model=lora_llm,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer
)

# saving the model
model_path = os.path.join(parent, 'text-gen-llm', 'models', 'model3')
if os.path.exists(model_path):
    trainer.train()
    trainer.evaluate()
    trainer.save_model(model_path)


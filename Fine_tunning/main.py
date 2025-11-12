#  pip install -q accelerate transformers datasets peft bitsandbytes huggingface_hub

from huggingface_hub import login
login()

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# pick a small open model that fits Colab GPU
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # or "TinyLlama/TinyLlama-1.1B-Chat-v1.0" if memory is tight

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,     # 4-bit quantization (saves VRAM)
    torch_dtype=torch.float16,
    trust_remote_code=True
)


data = [
    {"prompt": "Q: How to train a dog to sit?\nA:", "response": "Use treats and reward when it sits."},
    {"prompt": "Q: What is Langfuse?\nA:", "response": "Langfuse helps monitor and evaluate LLM applications."},
    {"prompt": "Q: What is CrewAI?\nA:", "response": "CrewAI is a framework for multi-agent AI orchestration."},
]

dataset = Dataset.from_list(data)

def preprocess(example):
    text = example["prompt"] + " " + example["response"]
    tokenized = tokenizer(text, truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # typical for Mistral / LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer

# Assign a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

training_args = TrainingArguments(
    output_dir="mistral-lora-demo",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="no",
    push_to_hub=True,
    hub_model_id="vijay-1432/mistral-lora-demo"  # your HF username here
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.push_to_hub()

# Use a pipeline for getting inference from the proprietary model
from transformers import pipeline

pipe = pipeline("text-generation", model="Vijay-1432/mistral-lora-demo")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
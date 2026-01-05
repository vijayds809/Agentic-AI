from datasets import load_dataset
df = load_dataset("Vijay-1432/instruction_tunned_llm_dataset")


from transformers import AutoTokenizer, AutoModelForCausalLM
model = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model)
base_model = AutoModelForCausalLM.from_pretrained(model)


train_dataset = df["train"]
test_dataset = df["test"]

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="fine_tinning_llm",

    # Training length
    max_steps=10,              # change as needed
    num_train_epochs=1,         # ignored if max_steps is set

    # Batch & optimization
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=10,
    optim="adafactor",

    # Logging & saving (OLD transformers compatible)
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,

    # General
    overwrite_output_dir=True,
    disable_tqdm=False,
    report_to="none",           # avoids wandb errors
)

from transformers import Trainer

trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

from huggingface_hub import login
login()

hf_repo = "Vijay-1432/clm_finetunned"
trainer.push_to_hub(hf_repo)
tokenizer.push_to_hub(hf_repo)
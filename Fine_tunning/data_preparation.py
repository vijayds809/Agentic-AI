finetuning_dataset_loaded = load_dataset("json", data_files="dataset.jsonl", split="train")

tokenized_dataset = finetuning_dataset_loaded.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    drop_last_batch=True
)

print(tokenized_dataset)

def tokenize_function(examples):
    if "question" in examples and "answer" in examples:
      text = examples["question"][0] + examples["answer"][0]
    elif "input" in examples and "output" in examples:
      text = examples["input"][0] + examples["output"][0]
    else:
      text = examples["text"][0]

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )

    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        2048
    )
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )

    return tokenized_inputs

tokenized_dataset = tokenized_dataset.add_column("labels",tokenized_dataset['input_ids'])
split_dataset = tokenized_dataset.train_test_split(test_size = 0.2,shuffle = True,seed = 92)

from huggingface_hub import notebook_login

notebook_login()

split_dataset.push_to_hub("Vijay-1432/instruction_tunned_llm_dataset")
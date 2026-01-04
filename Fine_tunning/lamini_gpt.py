import json
import pprint
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

#gpt2 test
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
dataset_l = load_dataset("lamini/lamini_docs")
prompt = dataset_l["test"][0]["question"]
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=60)
print(tokenizer.decode(out[0], skip_special_tokens=True))

#finetuned lamini gpt test
model_id = "lamini/lamini_docs_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
dataset_l = load_dataset("lamini/lamini_docs")
prompt = dataset_l["test"][0]["question"]
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=60)
print(tokenizer.decode(out[0], skip_special_tokens=True))



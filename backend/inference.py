from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline
)
import torch

model_path = "./code-llama-finetuned-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

def ask(question):
    prompt = f"<s>[INST] {question} [/INST]"
    response = pipe(
        prompt,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )
    return response[0]['generated_text']

# Example
print(ask("How to reverse a string in Python?"))
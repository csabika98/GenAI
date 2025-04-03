# GenAI - CodeLlama-7B-Instruct - Fine-tuned chatbot
model_name = "codellama/CodeLlama-7b-Python-hf"

Fine-tune **CodeLlama-7B-Instruct** on Python-specific datasets (e.g., CodeAlpaca) to create a coding-focused chatbot.

# File Structure
├── Dataset.py         
├── tokenizer.py            
├── inference.py            
├── requirements.txt        
├── python_chat_train.jsonl
<br>
└── python_chat_eval.jsonl

Dataset:
https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K

Hugging Face Transformers and PEFT (Parameter-Efficient Fine-Tuning)


Datasets: https://huggingface.co/docs/datasets/index
https://huggingface.co/datasets/Nan-Do/code-search-net-python

https://arxiv.org/abs/2006.11239
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://huggingface.co/docs/transformers/quicktour

https://huggingface.co/
https://arxiv.org/
https://deepmind.com/blog

Fine-tune on Custom Data: 
Python

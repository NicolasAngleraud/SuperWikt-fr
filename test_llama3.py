import torch
from transformers import AutoTokenizer, AutoModel
from generation.py import Llama


model = Llama.build(ckpt_dir="../Meta-Llama-3-8B",
					tokenizer_path="../Meta-Llama-3-8B/tokenizer.model",
					max_seq_length=100,
					max_batch_size=2,
		    		)





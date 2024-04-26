import torch
import importlib.util

spec = importlib.util.spec_from_file_location("model", '../llama3/llama/model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

spec = importlib.util.spec_from_file_location("tokenizer", '../llama3/llama/tokenizer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

spec = importlib.util.spec_from_file_location("generation", '../llama3/llama/generation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)



model = Llama.build(ckpt_dir="../Meta-Llama-3-8B",
					tokenizer_path="../Meta-Llama-3-8B/tokenizer.model",
					max_seq_length=100,
					max_batch_size=2,
		    		)





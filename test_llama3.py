import torch
import importlib.util

module_path = '..//file_to_import.py'
spec = importlib.util.spec_from_file_location("module_name", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


model = Llama.build(ckpt_dir="../Meta-Llama-3-8B",
					tokenizer_path="../Meta-Llama-3-8B/tokenizer.model",
					max_seq_length=100,
					max_batch_size=2,
		    		)





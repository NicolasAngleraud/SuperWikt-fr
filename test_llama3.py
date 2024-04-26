import importlib.util

# Importing model.py
spec = importlib.util.spec_from_file_location("llama_model", '../llama3/llama/model.py')
llama_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llama_model)

# Importing tokenizer.py
spec = importlib.util.spec_from_file_location("llama_tokenizer", '../llama3/llama/tokenizer.py')
llama_tokenizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llama_tokenizer)

# Importing generation.py
spec = importlib.util.spec_from_file_location("llama_generation", '../llama3/llama/generation.py')
llama_generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llama_generation)

# Now you can access classes/functions from the imported modules
model = llama_model.Llama.build(ckpt_dir="../Meta-Llama-3-8B",
                                tokenizer_path="../Meta-Llama-3-8B/tokenizer.model",
                                max_seq_length=100,
                                max_batch_size=2)





import torch
from transformers import AutoTokenizer, AutoModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("../Meta-Llama-3-8B/tokenizer.model")

# Load model
model = AutoModel.from_pretrained("../Meta-Llama-3-8B")

# Set the model to evaluation mode
model.eval()

# Prompts to generate text from
prompts = ["Once upon a time", "In a galaxy far far away", "The quick brown fox"]

# Generate text for each prompt
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode and print generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Prompt:", prompt)
    print("Generated Text:", generated_text)
    print()


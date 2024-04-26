from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("lightblue/suzume-llama-3-8B-multilingual")
model = AutoModelForCausalLM.from_pretrained("lightblue/suzume-llama-3-8B-multilingual")

# Define prompt
prompt = "Pour la définition de 'chien : individu de la race des canidés', quelle est la classe sémantique ? Donne comme réponse une classe entre 'animal' et 'person'."

# Tokenize prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text based on prompt
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

# Decode generated output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)

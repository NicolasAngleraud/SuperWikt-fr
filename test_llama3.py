from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn



## MODELS
#lightblue/suzume-llama-3-8B-multilingual
#meta-llama/Meta-Llama-3-8B
#meta-llama/Meta-Llama-3-8B-Instruct


API_TOKEN = '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'

'''
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=API_TOKEN)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=API_TOKEN)


def_ = "Mammifère domestique, ongulé de l’ordre des suidés ; porc."

prompt = """Choisis la classe sémantique décrivant le mieux la définition suivant parmi les quatre classes suivantes: person, animal, mineral, plant.

définition: {BODY} --> classe sémantique: """.format(BODY=def_)


inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=inputs.input_ids.size(1) + 50, num_return_sequences=1, temperature=0.2)

generated_classification = tokenizer.decode(output[0], skip_special_tokens=True)


print("Generated Classification:", generated_classification)
'''

##################################################################################################


"""
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments




if torch.cuda.is_available():
	DEVICE = torch.device("cuda:1")


class DummyDataset(Dataset):
    def __init__(self, num_sequences, sequence_length, device):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.device = device

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Generate dummy sequence of integers
        sequence = torch.randint(0, 100, (self.sequence_length,), device=self.device)
        return sequence

# Define parameters for the datasets
num_training_sequences = 1000
num_eval_sequences = 100
sequence_length = 50


# Create dummy training dataset
train_dataset = DummyDataset(num_training_sequences, sequence_length, DEVICE)

# Create dummy evaluation dataset
eval_dataset = DummyDataset(num_eval_sequences, sequence_length, DEVICE)


# Load your autoregressive model from Hugging Face
model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

# Define training arguments
training_args = TrainingArguments(
    num_train_epochs=1,  # Number of training epochs
    per_device_train_batch_size=2,  # Batch size per device during training
    lora=0.3
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Your training dataset
    eval_dataset=eval_dataset  # Your evaluation dataset
)

# Train the model
trainer.train()
"""


##################################################################################################


class TextClassifier(nn.Module):
    def __init__(model_base, num_labels):
        super(TextClassifier, self).__init__()
        self.model_base = model_base  # pre-trained model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(model_base.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model_base(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = torch.mean(sequence_output, 1)  # mean pooling
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits




class PrefixTuning(nn.Module):
	def __init__(self, model_base, num_labels, prefix_length, prefix_size):
		super(PrefixTuning, self).__init__()
		self.model_base = model_base
		self.prefix_length = prefix_length
		self.prefix_size = prefix_size
		self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, prefix_size))
		self.dropout = nn.Dropout(0.1)
		self.classifier = nn.Linear(model_base.config.hidden_size, num_labels)
		self.to(DEVICE)

		for param in self.model_base.parameters():
			param.requires_grad = False

	def forward(self, input_ids, attention_mask=None):
		# Generate the same batch size of prefix embeddings
		batch_size = input_ids.shape[0]
		prefix = self.prefix_embeddings.expand(batch_size, -1, -1)

		# Concatenate prefix embeddings with input_ids
		full_input_ids = torch.cat((torch.zeros(batch_size, self.prefix_length).long().to(input_ids.device), input_ids), dim=1)
		# Handle attention_mask if it's None
		if attention_mask is None:
			attention_mask = torch.ones_like(full_input_ids).to(DEVICE)
		extended_attention_mask = torch.cat([torch.ones(batch_size, self.prefix_length).to(input_ids.device), attention_mask], dim=1)

		# Pass the full sequence to the model
		outputs = self.model_base(input_ids=full_input_ids, attention_mask=extended_attention_mask)
		sequence_output = outputs.last_hidden_state

		# Use mean pooling over the sequence for classification
		pooled_output = torch.mean(sequence_output, 1)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		return logits



def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
        

class PrefixTuningGPT(nn.Module):
    def __init__(self, model_base, num_labels, prefix_length, embedding_size):
        super(PrefixTuningGPT, self).__init__()
        self.model_base = model_base  # Pre-trained GPT model
        self.prefix_length = prefix_length  # Number of prefix embeddings
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, 1, embedding_size))
        self.classifier = nn.Linear(embedding_size, num_labels)  # Classifier for the task

    def forward(self, input_ids):
        device = input_ids.device
        
        # Generate prefix for each position in the batch
        prefix = self.prefix_embeddings.expand(-1, input_ids.size(0), -1)

        # Get embeddings from the base model's embedding layer
        base_embeddings = self.model_base.transformer.wte(input_ids)  # Word token embeddings

        # Concatenate prefix and base embeddings
        embeddings = torch.cat([prefix, base_embeddings], dim=0)

        # Process through the model
        outputs = self.model_base(inputs_embeds=embeddings)
        last_hidden_state = outputs.last_hidden_state

        # Assuming classification based on the output after the last token of the original input
        logits = self.classifier(last_hidden_state[input_ids.size(0) + self.prefix_length - 1])

        return logits



################################################################################################################

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset

API_TOKEN = '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'

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=API_TOKEN)
model_base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=API_TOKEN)


if torch.cuda.is_available():
	DEVICE = torch.device("cuda:1")


class PrefixDataset(Dataset):
    def __init__(self, sequences, labels, device):
        self.sequences = [sequence.to(device) for sequence in sequences]
        self.labels = labels.to(device)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label

# Define parameters for the datasets
num_training_sequences = 1000
num_eval_sequences = 100
sequence_length = 50

# Define your training sequences and labels
train_sequences = [torch.randint(0, 100, (sequence_length,)) for _ in range(num_training_sequences)]
train_labels = torch.randint(0, 3, (num_training_sequences,))  # Assuming 3 classes for classification

# Define your evaluation sequences and labels
eval_sequences = [torch.randint(0, 100, (sequence_length,)) for _ in range(num_eval_sequences)]
eval_labels = torch.randint(0, 3, (num_eval_sequences,))  # Assuming 3 classes for classification


# Create training and evaluation datasets
train_dataset = PrefixDataset(train_sequences, train_labels, DEVICE)
eval_dataset = PrefixDataset(eval_sequences, eval_labels, DEVICE)

freeze_model_parameters(model_base)

# Create the PrefixTuning model which includes the prefix embeddings and a new classifier
prefix_model = PrefixTuning(model_base, num_labels=3, prefix_length=10, prefix_size=model_base.config.hidden_size)

# Assuming the use of an optimizer like Adam
optimizer = torch.optim.Adam([
    {'params': prefix_model.prefix_embeddings, 'lr': 1e-4},
    {'params': prefix_model.classifier.parameters(), 'lr': 1e-4}
])


# Assuming you have defined train_dataset and eval_dataset
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=2)

# Define your loss function (e.g., CrossEntropyLoss for classification)
loss_fn = nn.CrossEntropyLoss()

# Number of training epochs
num_epochs = 3

# Training loop
for epoch in range(num_epochs):
    prefix_model.train()  # Set the model to training mode
    total_loss = 0.0
    
    for batch in train_loader:
        input_ids, labels = batch
        
        optimizer.zero_grad()  # Clear gradients
        
        # Forward pass
        logits = prefix_model(input_ids)
        
        # Compute loss
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    # Calculate average training loss
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg. Training Loss: {avg_train_loss:.4f}")
    
    # Evaluation loop
    prefix_model.eval()  # Set the model to evaluation mode
    total_eval_loss = 0.0
    total_correct = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids, labels = batch
            
            # Forward pass
            logits = prefix_model(input_ids)
            
            # Compute loss
            eval_loss = loss_fn(logits, labels)
            total_eval_loss += eval_loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
    
    # Calculate average evaluation loss and accuracy
    avg_eval_loss = total_eval_loss / len(eval_loader)
    accuracy = total_correct / len(eval_dataset)
    print(f"Avg. Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.2%}")

# Training finished
print("Training completed!")



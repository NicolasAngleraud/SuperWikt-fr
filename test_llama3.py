from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn



## MODELS
#lightblue/suzume-llama-3-8B-multilingual
#meta-llama/Meta-Llama-3-8B
#meta-llama/Meta-Llama-3-8B-Instruct


API_TOKEN = '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'

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=API_TOKEN)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=API_TOKEN)


def_ = "Mammifère domestique, ongulé de l’ordre des suidés ; porc."

prompt = """Choisis la classe sémantique décrivant le mieux la définition suivant parmi les quatre classes suivantes: person, animal, mineral, plant.

définition: {BODY} --> classe sémantique: """.format(BODY=def_)


inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=inputs.input_ids.size(1) + 50, num_return_sequences=1, temperature=0.2)

generated_classification = tokenizer.decode(output[0], skip_special_tokens=True)


print("Generated Classification:", generated_classification)


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

    def forward(self, input_ids, attention_mask=None):
        # Generate the same batch size of prefix embeddings
        batch_size = input_ids.shape[0]
        prefix = self.prefix_embeddings.expand(batch_size, -1, -1)

        # Get the embeddings from the base model
        base_embeddings = self.model_base.transformer.wte(input_ids)

        # Concatenate prefix embeddings with base embeddings
        full_embeddings = torch.cat((prefix, base_embeddings), dim=1)
        extended_attention_mask = torch.cat([torch.ones(batch_size, self.prefix_length).to(input_ids.device), attention_mask], dim=1)

        # Pass the full sequence to the model
        outputs = self.model_base(inputs_embeds=full_embeddings, attention_mask=extended_attention_mask)
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

"""
# Assuming `model_base` is your pre-trained autoregressive model like LLaMA or GPT
freeze_model_parameters(model_base)

# Create the PrefixTuning model which includes the prefix embeddings and a new classifier
prefix_model = PrefixTuning(model_base, num_labels=3, prefix_length=10, prefix_size=model_base.config.hidden_size)

# Assuming the use of an optimizer like Adam
optimizer = torch.optim.Adam([
    {'params': prefix_model.prefix_embeddings, 'lr': 1e-4},
    {'params': prefix_model.classifier.parameters(), 'lr': 1e-4}
])
"""

################################################################################################################

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
    per_device_train_batch_size=4,  # Batch size per device during training
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Your training dataset
    eval_dataset=eval_dataset,  # Your evaluation dataset
)

# Train the model
trainer.train()


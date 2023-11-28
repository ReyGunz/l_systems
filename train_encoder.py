import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from torch import nn
import math
import os
from python.dataset import TreeDataset
import random

criterion = nn.CrossEntropyLoss()
model_path = '3d_143_32_8.pth'

def model_exists(model_path):
    return os.path.isfile(model_path)

# Function to save model
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

# Function to load model
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def collate_batch(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

data_files = [
                "3d_stochastic_tree1_samples",
                "3d_stochastic_tree2_samples",
                "3d_stochastic_tree3_samples",
                "3d_stochastic_tree4_samples",
            ]

n = 1000
texts = []
for file in data_files:
    with open('data/' + file + '.txt') as f:
        texts += f.readlines()[:n]

tokenizer = spm.SentencePieceProcessor(model_file='3d_spm143_bpe.model')

# Create Dataset and DataLoader
dataset = TreeDataset(texts, tokenizer)
data_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_batch, num_workers=0, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import json

config = 0
with open('model_config.json') as f:
    config = json.load(f)

# Hyperparameters
d_model = config['d_model']  # Size of the embeddings and transformer
nhead = config['nhead']  # Number of heads in multiheadattention
num_encoder_layers = config['num_encoder_layers']  # Number of encoder layers in the transformer
num_decoder_layers = config['num_encoder_layers']  # Number of decoder layers in the transformer
dim_feedforward = config['dim_feedforward']  # Hidden layer size in the transformer
max_seq_length = dataset.max_tokens  # Maximum sequence length

# model, model_loaded = TreeTransformer.load_model(tokenizer.vocab_size(), d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, device)
from x_transformers import TransformerWrapper, Encoder
from tqdm import tqdm

model = TransformerWrapper(
    num_tokens = tokenizer.vocab_size(),
    max_seq_len = max_seq_length,
    l2norm_embed=True,
    attn_layers = Encoder(
        dim = d_model,
        depth = 3,
        heads = 4,
        rotary_pos_emb = True  # turns on rotary positional embeddings
    )
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, data_loader, optimizer, num_epochs, device):
    model.train()  # Set the model to training mode
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        total_loss = 0
        batch_tqdm = tqdm(data_loader, desc='Batches', leave=False)
        for i, batch in enumerate(batch_tqdm):
            inputs, targets = batch.to(device), torch.roll(batch, -1, dims=1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs).view(-1, tokenizer.vocab_size())
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            torch.cuda.empty_cache()  # Clear CUDA cache periodically

            # Update tqdm description with batch loss
            batch_tqdm.set_description(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(data_loader)}, Batch Loss: {loss.item()}")

        # Update tqdm bar with the average loss
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(data_loader)}")


num_epochs = 8  # Adjust as needed

# Check if model has been trained and saved
if model_exists(model_path):
    print("Loading saved model...")
    model = load_model(model, model_path)
else:
    print("Training new model...")
    # Train model...
    train_model(model, data_loader, optimizer, num_epochs, device)
    print("Saving trained model...")
    save_model(model, model_path)

from python.visualize_latent_space import visualize_avg_text_embeddings_3d
visualize_avg_text_embeddings_3d(model, tokenizer, texts, n, 'pca', d_model, device)
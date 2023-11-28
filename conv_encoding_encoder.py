import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from torch import nn
import os
from python.dataset import TreeDataset  # Assuming this is correctly implemented
from tqdm import tqdm   

criterion = nn.CrossEntropyLoss()
model_path = '3d_139_2heads_1layer_128_128_3conv.pth'

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

n = 2000
texts = []
for file in data_files:
    with open('data/' + file + '.txt') as f:
        texts += f.readlines()[:n]

tokenizer = spm.SentencePieceProcessor(model_file='spm139_bpe.model')

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

# Convolutional Positional Encoding
class ConvPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(ConvPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)

    def forward(self, x):
        # print(f"Shape of x before transpose: {x.shape}")
        x = x.transpose(1, 2)  # Change shape to [batch, d_model, seq_len]
        x = self.conv(x)       # Apply convolution
        return x.transpose(1, 2)  # Change shape back to [batch, seq_len, d_model]

# Custom Transformer Encoder with Convolutional Positional Encoding
class TransformerEncoderWithConvPE(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerEncoderWithConvPE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.conv_pos_encoder = ConvPositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src)  # Embedding tokens to d_model dimension
        src = self.conv_pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.output_layer(output)  # Apply the linear layer

# Model instantiation
model = TransformerEncoderWithConvPE(tokenizer.vocab_size(), d_model, nhead, num_encoder_layers, dim_feedforward).to(device)
def train_model(model, data_loader, optimizer, num_epochs, device):
    model.train()  # Set the model to training mode
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        total_loss = 0
        batch_tqdm = tqdm(data_loader, desc='Batches', leave=False)
        for i, batch in enumerate(batch_tqdm):
            # print(f"Shape of batch before model: {batch.shape}")
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

num_epochs = 3  # Adjust as needed
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
visualize_avg_text_embeddings_3d(model, tokenizer, texts, 8000, 'pca', 256, device)
import torch
from torch import nn
import math
import os

MODEL_PATH = 'transformer_model.pth'
dropout = 0.1

criterion = nn.CrossEntropyLoss()

class TreeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(TreeTransformer, self).__init__()
        self.model_type = 'TreeTransformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
    
    def get_embeddings(self, src):
        return self.embedding(src) * math.sqrt(self.d_model)
    
    def get_avg_embeddings(self, src):
        embeddings = self.embedding(src) * math.sqrt(self.d_model)  # Get embeddings
        avg_embeddings = embeddings.mean(dim=1)  # Average across the sequence length dimension
        return avg_embeddings
    
    def get_tripolar_embeddings(self, src):
        embeddings = self.embedding(src) * math.sqrt(self.d_model)  # Get embeddings
        
        # Check if all embeddings are positive, negative, or mixed
        positive = torch.all(embeddings > 0, dim=1)
        negative = torch.all(embeddings < 0, dim=1)
        
        # Create tripolar representation
        # 1 for all positive, -1 for all negative, 0 for mixed
        tripolar_embeddings = torch.zeros_like(embeddings[:, 0])
        tripolar_embeddings[positive] = 1
        tripolar_embeddings[negative] = -1
    
        return tripolar_embeddings
    
    def save_model(self):
        torch.save(self.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    @classmethod
    def load_model(cls, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, device):
        model = cls(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)
        model_loaded = False  # Flag to indicate if the model was loaded
        if os.path.isfile(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Loaded model from {MODEL_PATH}")
            model_loaded = True
        else:
            print(f"No saved model found at {MODEL_PATH}")
        return model.to(device), model_loaded

# Positional Encoding (for adding notion of position in sequences)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        print(x)
        return self.dropout(x)

def train(model, data_loader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            # Move the batch to the device
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            # Adjust the target if necessary to match the output format
            target = batch  # Placeholder, modify according to your specific task
            loss = criterion(output.view(-1, model.vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# # # # import torch
# # # # import torch.nn as nn
# # # # import math

# # # # class RotaryEmbedding(nn.Module):
# # # #     def __init__(self, dim):
# # # #         super().__init__()
# # # #         self.dim = dim
# # # #         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2).float() / (dim // 2)))
# # # #         self.register_buffer('inv_freq', inv_freq)

# # # #     def forward(self, max_seq_len):
# # # #         t = torch.arange(max_seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
# # # #         freqs = t[:, None] * self.inv_freq[None, :]
# # # #         return torch.cat((freqs.sin(), freqs.cos()), dim=-1)

# # # # class Attention(nn.Module):
# # # #     def __init__(self, dim, heads=8):
# # # #         super().__init__()
# # # #         self.dim = dim  # Store the dimension as an instance variable
# # # #         self.heads = heads
# # # #         self.scale = dim ** -0.5

# # # #         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
# # # #         self.to_out = nn.Linear(dim, dim)

# # # #     def apply_rotary_embedding(self, x, rotary_embedding):
# # # #         sin, cos = rotary_embedding.split(self.dim // 2, dim=-1)
# # # #         return (x * cos) + (x.roll(1, dims=-1) * sin)

# # # #     def forward(self, x, pos_embedding):
# # # #         b, n, _, h = *x.shape, self.heads
# # # #         qkv = self.to_qkv(x).chunk(3, dim=-1)
# # # #         q, k, v = map(lambda t: t.reshape(b, n, h, -1), qkv)

# # # #         q, k = map(lambda t: self.apply_rotary_embedding(t, pos_embedding), (q, k))

# # # #         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
# # # #         attn = dots.softmax(dim=-1)
# # # #         out = torch.matmul(attn, v)
# # # #         return self.to_out(out.reshape(b, n, -1))
# # # # class TransformerEncoder(nn.Module):
# # # #     def __init__(self, dim, depth, heads, mlp_dim):
# # # #         super().__init__()
# # # #         self.layers = nn.ModuleList([])
# # # #         self.pos_embedding = RotaryEmbedding(dim)

# # # #         for _ in range(depth):
# # # #             self.layers.append(nn.ModuleList([
# # # #                 Attention(dim, heads=heads),
# # # #                 nn.Linear(dim, mlp_dim),
# # # #                 nn.GELU(),
# # # #                 nn.Linear(mlp_dim, dim),
# # # #             ]))

# # # #     def forward(self, x):
# # # #         pos_embedding = self.pos_embedding(x.shape[1])
# # # #         for attn, ff1, act, ff2 in self.layers:
# # # #             x = attn(x, pos_embedding) + x
# # # #             x = ff2(act(ff1(x))) + x
# # # #         return x

# # # # # Example usage
# # # # encoder = TransformerEncoder(dim=512, depth=6, heads=8, mlp_dim=1024)
# # # # x = torch.randn(10, 32, 512)  # batch_size=10, seq_len=32, embedding_dim=512
# # # # output = encoder(x)

# # # import torch
# # # from rotary_embedding_torch import RotaryEmbedding

# # # rotary_emb = RotaryEmbedding(
# # #     dim = 32,
# # #     use_xpos = True   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
# # # )

# # # # mock queries and keys - dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)

# # # q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
# # # k = torch.randn(1, 8, 1024, 64) # keys

# # # # apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)

# # # # instead of using `rotate_queries_or_keys`, you will use `rotate_queries_and_keys`, the rest is taken care of

# # # q, k = rotary_emb.rotate_queries_and_keys(q, k)
# # # print(q, k)

# # import torch
# # from torch import nn
# # import math
# # import os

# # MODEL_PATH = 'transformer_model.pth'
# # dropout = 0.1

# # criterion = nn.CrossEntropyLoss()

# # # Rotary Embeddings
# # def rotate_every_two(x):
# #     x1 = x[..., ::2]
# #     x2 = x[..., 1::2]
# #     x = torch.stack((-x2, x1), dim=-1)
# #     return x.reshape_as(x1)

# # def apply_rotary_emb(q, k, sinu_pos):
# #     sinu_pos = sinu_pos.repeat_interleave(2, dim=-1)
# #     q = q * sinu_pos
# #     k = k * sinu_pos
# #     return q, k

# # def rotary_embedding(dim, max_len=4096):
# #     inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
# #     t = torch.arange(max_len).type_as(inv_freq)
# #     freqs = torch.einsum('i , j -> i j', t, inv_freq)
# #     return torch.cat((freqs, freqs), dim=-1)

# # class RotaryEmbedding(nn.Module):
# #     def __init__(self, dim):
# #         super().__init__()
# #         self.dim = dim
# #         self.inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))

# #     def forward(self, max_len):
# #         t = torch.arange(max_len).type_as(self.inv_freq)
# #         freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
# #         return torch.cat((freqs, freqs), dim=-1)

# # # TreeTransformer class with Rotary Embedding
# # class TreeTransformer(nn.Module):
# #     def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
# #         super(TreeTransformer, self).__init__()
# #         self.model_type = 'TreeTransformer'
# #         self.src_mask = None
# #         self.pos_encoder = RotaryEmbedding(d_model)  # Rotary embedding
# #         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation='relu')
# #         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
# #         self.embedding = nn.Embedding(vocab_size, d_model)
# #         self.decoder = nn.Linear(d_model, vocab_size)
# #         self.d_model = d_model
# #         self.vocab_size = vocab_size

# #     def forward(self, src):
# #         src = self.embedding(src) * math.sqrt(self.d_model)
# #         src = self.pos_encoder(src.size(0))  # Apply rotary embeddings
# #         output = self.transformer_encoder(src, self.src_mask)
# #         output = self.decoder(output)
# #         return output
    
# #     def get_embeddings(self, src):
# #         return self.embedding(src) * math.sqrt(self.d_model)
    
# #     def get_avg_embeddings(self, src):
# #         embeddings = self.embedding(src) * math.sqrt(self.d_model)  # Get embeddings
# #         avg_embeddings = embeddings.mean(dim=1)  # Average across the sequence length dimension
# #         return avg_embeddings
    
# #     def get_tripolar_embeddings(self, src):
# #         embeddings = self.embedding(src) * math.sqrt(self.d_model)  # Get embeddings
        
# #         # Check if all embeddings are positive, negative, or mixed
# #         positive = torch.all(embeddings > 0, dim=1)
# #         negative = torch.all(embeddings < 0, dim=1)
        
# #         # Create tripolar representation
# #         # 1 for all positive, -1 for all negative, 0 for mixed
# #         tripolar_embeddings = torch.zeros_like(embeddings[:, 0])
# #         tripolar_embeddings[positive] = 1
# #         tripolar_embeddings[negative] = -1
    
# #         return tripolar_embeddings
    
# #     def save_model(self):
# #         torch.save(self.state_dict(), MODEL_PATH)
# #         print(f"Model saved to {MODEL_PATH}")

# #     @classmethod
# #     def load_model(cls, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, device):
# #         model = cls(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)
# #         model_loaded = False  # Flag to indicate if the model was loaded
# #         if os.path.isfile(MODEL_PATH):
# #             model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# #             print(f"Loaded model from {MODEL_PATH}")
# #             model_loaded = True
# #         else:
# #             print(f"No saved model found at {MODEL_PATH}")
# #         return model.to(device), model_loaded

# # # Positional Encoding (for adding notion of position in sequences)
# # class PositionalEncoding(nn.Module):
# #     def __init__(self, d_model, dropout=0.1, max_len=5000):
# #         super(PositionalEncoding, self).__init__()
# #         self.dropout = nn.Dropout(p=dropout)

# #         position = torch.arange(max_len).unsqueeze(1)
# #         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
# #         pe = torch.zeros(max_len, 1, d_model)
# #         pe[:, 0, 0::2] = torch.sin(position * div_term)
# #         pe[:, 0, 1::2] = torch.cos(position * div_term)
# #         self.register_buffer('pe', pe)

# #     def forward(self, x):
# #         x = x + self.pe[:x.size(0)]
# #         print(x)
# #         return self.dropout(x)

# # def train(model, data_loader, optimizer, criterion, num_epochs, device):
# #     model.train()
# #     for epoch in range(num_epochs):
# #         for batch in data_loader:
# #             # Move the batch to the device
# #             batch = batch.to(device)
# #             optimizer.zero_grad()
# #             output = model(batch)
# #             # # Adjust the target if necessary to match the output format
# #             # target = batch  # Placeholder, modify according to your specific task
# #             # loss = criterion(output.view(-1, model.vocab_size), target.view(-1))
# #             output = output.transpose(1, 2)  # change to [batch_size, sequence_length, num_classes]
# #             output = output.reshape(-1, 3)  # change to [batch_size * sequence_length, num_classes]

# #             target = target.view(-1)  # target shape: [batch_size * sequence_length]

# #             loss = criterion(output, target)
# #             loss.backward()
# #             optimizer.step()
# #             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# import torch
# from torch import nn
# import math
# import os
# from rotary_embedding_torch import RotaryEmbedding

# MODEL_PATH = 'transformer_model.pth'
# dropout = 0.1

# criterion = nn.CrossEntropyLoss()

# class TreeTransformer(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
#         super(TreeTransformer, self).__init__()
#         self.model_type = 'TreeTransformer'
#         self.src_mask = None
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, activation='relu')
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.decoder = nn.Linear(d_model, vocab_size)
#         self.d_model = d_model
#         self.vocab_size = vocab_size
#         self.rope = RotaryEmbedding(d_model, use_xpos=True)

#     def forward(self, src):
#         src = self.embedding(src) * math.sqrt(self.d_model)
#         # src = self.pos_encoder(src)
#         src = self.rope(src)
#         output = self.transformer_encoder(src, self.src_mask)
#         output = self.decoder(output)
#         return output
    
#     def get_embeddings(self, src):
#         return self.embedding(src) * math.sqrt(self.d_model)
    
#     def get_avg_embeddings(self, src):
#         embeddings = self.embedding(src) * math.sqrt(self.d_model)  # Get embeddings
#         avg_embeddings = embeddings.mean(dim=1)  # Average across the sequence length dimension
#         return avg_embeddings
    
#     def get_tripolar_embeddings(self, src):
#         embeddings = self.embedding(src) * math.sqrt(self.d_model)  # Get embeddings
        
#         # Check if all embeddings are positive, negative, or mixed
#         positive = torch.all(embeddings > 0, dim=1)
#         negative = torch.all(embeddings < 0, dim=1)
        
#         # Create tripolar representation
#         # 1 for all positive, -1 for all negative, 0 for mixed
#         tripolar_embeddings = torch.zeros_like(embeddings[:, 0])
#         tripolar_embeddings[positive] = 1
#         tripolar_embeddings[negative] = -1
    
#         return tripolar_embeddings
    
#     def save_model(self):
#         torch.save(self.state_dict(), MODEL_PATH)
#         print(f"Model saved to {MODEL_PATH}")

#     @classmethod
#     def load_model(cls, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, device):
#         model = cls(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)
#         model_loaded = False  # Flag to indicate if the model was loaded
#         if os.path.isfile(MODEL_PATH):
#             model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#             print(f"Loaded model from {MODEL_PATH}")
#             model_loaded = True
#         else:
#             print(f"No saved model found at {MODEL_PATH}")
#         return model.to(device), model_loaded

# # Positional Encoding (for adding notion of position in sequences)
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)

# def train(model, data_loader, optimizer, criterion, num_epochs, device):
#     model.train()
#     for epoch in range(num_epochs):
#         for batch in data_loader:
#             # Move the batch to the device
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             output = model(batch)
#             # Adjust the target if necessary to match the output format
#             target = batch  # Placeholder, modify according to your specific task
#             loss = criterion(output.view(-1, model.vocab_size), target.view(-1))
#             loss.backward()
#             optimizer.step()
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


import torch
from x_transformers import TransformerWrapper, Encoder

model = TransformerWrapper(
    num_tokens = 17,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8,
        rotary_pos_emb = True  # turns on rotary positional embeddings
    )
)
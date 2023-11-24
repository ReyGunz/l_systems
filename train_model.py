import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from torch import nn
import math
import os
from pyNonsense.dataset import TreeDataset
from pyNonsense.transformer import train, criterion
from pyNonsense.rope_transformer import TreeTransformer
import random

def collate_batch(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

data_files = [
                "stochastic_tree1_samples",
                "stochastic_tree2_samples",
                "stochastic_tree3_samples",
                "stochastic_tree4_samples",
            ]

texts = []
for file in data_files:
    texts += open('data/' + file + '.txt').readlines()

tokenizer = spm.SentencePieceProcessor(model_file='spm.model')

# Create Dataset and DataLoader
dataset = TreeDataset(texts, tokenizer)
data_loader = DataLoader(dataset, batch_size=128, collate_fn=collate_batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import json

config = 0
with open('model.json') as f:
    config = json.load(f)

# Hyperparameters
d_model = config['d_model']  # Size of the embeddings and transformer
nhead = config['nhead']  # Number of heads in multiheadattention
num_encoder_layers = config['num_encoder_layers']  # Number of encoder layers in the transformer
num_decoder_layers = config['num_encoder_layers']  # Number of decoder layers in the transformer
dim_feedforward = config['dim_feedforward']  # Hidden layer size in the transformer
max_seq_length = dataset.max_tokens  # Maximum sequence length

model, model_loaded = TreeTransformer.load_model(tokenizer.vocab_size(), d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3  # Adjust as needed

if not model_loaded:
    train(model, data_loader, optimizer, criterion, num_epochs, device)
    model.save_model()
else:
    print("Model was loaded from file, skipping training.")

def test_model_single_input(model, tokenizer, text, device):
    # Tokenize and convert to tensor
    tokenized_text = tokenizer.tokenize(text)
    input_tensor = torch.tensor(tokenized_text).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Pass through the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        output = model(input_tensor)

    return output

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# def visualize_avg_text_embeddings_2d(model, tokenizer, texts, device):
#     model.eval()  # Set the model to evaluation mode
#     pca = PCA(n_components=2)  # Initialize PCA to reduce to 2 dimensions

#     plt.figure(figsize=(10, 10))

#     # Process each list in the texts separately
#     for idx, text_list in enumerate(texts):
#         embeddings = []
#         for text in text_list:
#             tokenized_text = tokenizer.tokenize(text)
#             input_tensor = torch.tensor(tokenized_text).unsqueeze(0).to(device)

#             with torch.no_grad():
#                 avg_embeddings = model.get_avg_embeddings(input_tensor)
#                 avg_embeddings_np = avg_embeddings.squeeze(0).cpu().numpy()
#                 embeddings.append(avg_embeddings_np)

#         embeddings_2d = pca.fit_transform(embeddings)

#         # Choose different color for each list
#         color = 'red' if idx == 0 else 'blue'

#         # Scatter plot for the current list
#         for e in embeddings_2d:
#             plt.scatter(e[0], e[1], c=color, label=f'List {idx + 1}' if idx == 0 else '')

#     plt.title('2D Visualization of Average Embeddings')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.legend()
#     plt.show()

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_avg_text_embeddings_3d(model, tokenizer, texts, device):
    model.eval()  # Set the model to evaluation mode
    pca = PCA(n_components=3)  # Initialize PCA to reduce to 3 dimensions

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Process each list in the texts separately
    # for idx, text_list in enumerate(texts):
    embeddings = []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        input_tensor = torch.tensor(tokenized_text).unsqueeze(0).to(device)

        with torch.no_grad():
            # avg_embeddings = model.get_avg_embeddings(input_tensor)
            avg_embeddings = model.get_tripolar_embeddings(input_tensor)
            avg_embeddings_np = avg_embeddings.squeeze(0).cpu().numpy()
            embeddings.append(avg_embeddings_np)

    embeddings_3d = pca.fit_transform(embeddings)

    # 3D Scatter plot for the current list
    ax.scatter(embeddings_3d[:1000, 0], embeddings_3d[:1000, 1], embeddings_3d[:1000, 2], c='red')
    ax.scatter(embeddings_3d[1000:2000, 0], embeddings_3d[1000:2000, 1], embeddings_3d[1000:2000, 2], c='green')
    ax.scatter(embeddings_3d[2000:3000, 0], embeddings_3d[2000:3000, 1], embeddings_3d[2000:3000, 2], c='blue')
    ax.scatter(embeddings_3d[3000:4000, 0], embeddings_3d[3000:4000, 1], embeddings_3d[3000:4000, 2], c='black')

    ax.set_title('3D Visualization of Average Embeddings')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.legend()
    plt.show()

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_avg_text_embeddings_tsne(model, tokenizer, texts, device):
    model.eval()  # Set the model to evaluation mode
    tsne = TSNE(n_components=3, perplexity=30, n_iter=3000)  # Initialize t-SNE with 3 dimensions

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    embeddings = []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        input_tensor = torch.tensor(tokenized_text).unsqueeze(0).to(device)

        with torch.no_grad():
            # avg_embeddings = model.get_avg_embeddings(input_tensor)
            avg_embeddings = model.get_tripolar_embeddings(input_tensor)
            avg_embeddings_np = avg_embeddings.squeeze(0).cpu().numpy()
            embeddings.append(avg_embeddings_np)

    embeddings_3d = tsne.fit_transform(np.array(embeddings))

    # 3D Scatter plot for the current list
    ax.scatter(embeddings_3d[:1000, 0], embeddings_3d[:1000, 1], embeddings_3d[:1000, 2], c='red')
    ax.scatter(embeddings_3d[1000:2000, 0], embeddings_3d[1000:2000, 1], embeddings_3d[1000:2000, 2], c='green')
    ax.scatter(embeddings_3d[2000:3000, 0], embeddings_3d[2000:3000, 1], embeddings_3d[2000:3000, 2], c='blue')
    ax.scatter(embeddings_3d[3000:4000, 0], embeddings_3d[3000:4000, 1], embeddings_3d[3000:4000, 2], c='black')

    ax.set_title('3D Visualization of Average Embeddings with t-SNE')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.legend()
    plt.show()


# def visualize_avg_text_embeddings_2d(model, tokenizer, texts, device):
#     model.eval()  # Set the model to evaluation mode
#     tsne = TSNE(n_components=2, random_state=42)  # Initialize t-SNE for 2D visualization

#     plt.figure(figsize=(10, 10))

#     # Process each list in the texts separately
#     for idx, text_list in enumerate(texts):
#         embeddings = []
#         for text in text_list:
#             tokenized_text = tokenizer.tokenize(text)
#             input_tensor = torch.tensor(tokenized_text).unsqueeze(0).to(device)

#             with torch.no_grad():
#                 avg_embeddings = model.get_avg_embeddings(input_tensor)
#                 avg_embeddings_np = avg_embeddings.squeeze(0).cpu().numpy()
#                 embeddings.append(avg_embeddings_np)

#         embeddings_2d = tsne.fit_transform(embeddings)

#         # Choose different color for each list
#         color = 'red' if idx == 0 else 'blue'

#         # Scatter plot for the current list
#         for e in embeddings_2d:
#             plt.scatter(e[0], e[1], c=color, label=f'List {idx + 1}' if idx == 0 else '')

#     plt.title('2D Visualization of Average Embeddings (t-SNE)')
#     plt.xlabel('t-SNE Component 1')
#     plt.ylabel('t-SNE Component 2')
#     plt.legend()
#     plt.show()


# from sklearn.manifold import TSNE
# def visualize_avg_text_embeddings_3d(model, tokenizer, texts, device):
#     model.eval()  # Set the model to evaluation mode
#     tsne = TSNE(n_components=3, random_state=42)  # Initialize t-SNE for 3D visualization

#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     # Process each list in the texts separately
#     # for idx, text_list in enumerate(texts):
#     #     embeddings = []
#     embeddings = []
#     for text in texts:
#         tokenized_text = tokenizer.tokenize(text)
#         input_tensor = torch.tensor(tokenized_text).unsqueeze(0).to(device)

#         with torch.no_grad():
#             avg_embeddings = model.get_avg_embeddings(input_tensor)
#             avg_embeddings_np = avg_embeddings.squeeze(0).cpu().numpy()
#             embeddings.append(avg_embeddings_np)

#     embeddings_3d = tsne.fit_transform(np.array(embeddings))

#     # Choose different color for each list
#     # color = ['red', 'blue', 'green'][idx]
#     color = 'red'

#     # 3D Scatter plot for the current list
#     ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=color)

#     ax.set_title('3D Visualization of Average Embeddings (t-SNE)')
#     ax.set_xlabel('t-SNE Component 1')
#     ax.set_ylabel('t-SNE Component 2')
#     ax.set_zlabel('t-SNE Component 3')
#     ax.legend()
#     plt.show()

visualize_avg_text_embeddings_3d(model, tokenizer, texts, device)
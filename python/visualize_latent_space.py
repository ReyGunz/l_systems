import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import math
from sklearn.manifold import TSNE
from torch.linalg import norm

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

def get_max_embeddings(src, encoder, d_model):
    # Ensure the input is in the correct format (e.g., token IDs)
    # src should be a tensor of shape [batch_size, sequence_length]

    # Pass the input through the encoder
    # The output shape is [batch_size, sequence_length, d_model]
    embeddings = encoder(src)

    # Apply scaling by the square root of the model dimension
    embeddings_scaled = embeddings * math.sqrt(d_model)

    # Max pooling across the sequence length dimension
    # The resulting shape is [batch_size, d_model]
    max_embeddings, _ = embeddings_scaled.max(dim=1)

    # Normalize the pooled embeddings
    normed_embeddings = max_embeddings / torch.norm(max_embeddings, dim=1, keepdim=True)

    return normed_embeddings

def get_avg_embeddings(src, encoder, d_model):
    # Ensure the input is in the correct format (e.g., token IDs)
    # src should be a tensor of shape [batch_size, sequence_length]

    # Pass the input through the encoder
    # The output shape is [batch_size, sequence_length, d_model]
    embeddings = encoder(src)

    # Apply scaling by the square root of the model dimension
    embeddings_scaled = embeddings * math.sqrt(d_model)

    # Average across the sequence length dimension
    # The resulting shape is [batch_size, d_model]
    avg_embeddings = embeddings_scaled.mean(dim=1)

    return avg_embeddings
    # return avg_embeddings / norm(avg_embeddings)

def visualize_text_embeddings_3d(model, tokenizer, texts, n, dim_red: str, d_model, device, crit: str):
    model.eval()  # Set the model to evaluation mode

    if(dim_red == 'pca'):
        reducer = PCA(n_components=3)  # Initialize PCA to reduce to 3 dimensions
    if(dim_red == 'tsne'):
        reducer = TSNE(n_components=3, perplexity=30, n_iter=3000)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Process each list in the texts separately
    # for idx, text_list in enumerate(texts):
    embeddings = []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        input_tensor = torch.tensor(tokenized_text).unsqueeze(0).to(device)

        with torch.no_grad():
            if(crit == 'avg'):
                avg_embeddings = get_avg_embeddings(input_tensor, model, d_model)
            elif(crit == 'max'):
                avg_embeddings = get_max_embeddings(input_tensor, model, d_model)
            avg_embeddings_np = avg_embeddings.squeeze(0).cpu().numpy()
            embeddings.append(avg_embeddings_np)

    embeddings_3d = reducer.fit_transform(np.array(embeddings))

    # 3D Scatter plot for the current list
    ax.scatter(embeddings_3d[:n, 0], embeddings_3d[:n, 1], embeddings_3d[:n, 2], c='red')
    ax.scatter(embeddings_3d[n:2*n, 0], embeddings_3d[n:2*n, 1], embeddings_3d[n:2*n, 2], c='green')
    ax.scatter(embeddings_3d[2*n:3*n, 0], embeddings_3d[2*n:3*n, 1], embeddings_3d[2*n:3*n, 2], c='blue')
    ax.scatter(embeddings_3d[3*n:4*n, 0], embeddings_3d[3*n:4*n, 1], embeddings_3d[3*n:4*n, 2], c='black')

    ax.set_title(dim_red + '3D Visualization of Average Embeddings')
    ax.set_xlabel(dim_red + 'Component 1')
    ax.set_ylabel(dim_red + 'Component 2')
    ax.set_zlabel(dim_red + 'Component 3')
    ax.legend()
    plt.show()

# def visualize_avg_text_embeddings_tsne(model, tokenizer, texts, device):
#     model.eval()  # Set the model to evaluation mode
#     tsne = TSNE(n_components=3, perplexity=30, n_iter=3000)  # Initialize t-SNE with 3 dimensions

#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     embeddings = []
#     for text in texts:
#         tokenized_text = tokenizer.tokenize(text)
#         input_tensor = torch.tensor(tokenized_text).unsqueeze(0).to(device)

#         with torch.no_grad():
#             # avg_embeddings = model.get_avg_embeddings(input_tensor)
#             avg_embeddings = model.get_tripolar_embeddings(input_tensor)
#             avg_embeddings_np = avg_embeddings.squeeze(0).cpu().numpy()
#             embeddings.append(avg_embeddings_np)

#     embeddings_3d = tsne.fit_transform(np.array(embeddings))

#     # 3D Scatter plot for the current list
#     ax.scatter(embeddings_3d[:1000, 0], embeddings_3d[:1000, 1], embeddings_3d[:1000, 2], c='red')
#     ax.scatter(embeddings_3d[1000:2000, 0], embeddings_3d[1000:2000, 1], embeddings_3d[1000:2000, 2], c='green')
#     ax.scatter(embeddings_3d[2000:3000, 0], embeddings_3d[2000:3000, 1], embeddings_3d[2000:3000, 2], c='blue')
#     ax.scatter(embeddings_3d[3000:4000, 0], embeddings_3d[3000:4000, 1], embeddings_3d[3000:4000, 2], c='black')

#     ax.set_title('3D Visualization of Average Embeddings with t-SNE')
#     ax.set_xlabel('t-SNE Component 1')
#     ax.set_ylabel('t-SNE Component 2')
#     ax.set_zlabel('t-SNE Component 3')
#     ax.legend()
#     plt.show()


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
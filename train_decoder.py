import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from torch import nn
import math
import os
from python.dataset import TreeDataset
import random
import json
from x_transformers import TransformerWrapper, Decoder
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()
model_path = '3d_143_256_decoder.pth'

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

tokenizer = spm.SentencePieceProcessor(model_file='3d_spm143_bpe.model')

# Create Dataset and DataLoader
dataset = TreeDataset(texts, tokenizer)
data_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_batch, num_workers=0, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model configuration
config = 0
with open('model_config.json') as f:
    config = json.load(f)

# Hyperparameters
d_model = config['d_model']
nhead = config['nhead']
num_decoder_layers = config['num_decoder_layers']
dim_feedforward = config['dim_feedforward']
max_seq_length = dataset.max_tokens

model = TransformerWrapper(
    num_tokens = tokenizer.vocab_size(),
    max_seq_len = max_seq_length,
    l2norm_embed=True,
    attn_layers = Decoder(
        dim = d_model,
        depth = num_decoder_layers,  # Adjust the depth for the number of decoder layers
        heads = nhead,
        rotary_pos_emb = True
    )
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, data_loader, optimizer, num_epochs, device):
    model.train()
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
            torch.cuda.empty_cache()

            # Update tqdm description with batch loss
            batch_tqdm.set_description(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(data_loader)}, Batch Loss: {loss.item()}")

        # Update tqdm bar with the average loss
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(data_loader)}")

num_epochs = 5  # Adjust as needed

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

# def predict_completion(model, tokenizer, input_text, max_length=50):
#     """
#     Generate text completion for a given input text.

#     Args:
#     - model: Trained decoder model.
#     - tokenizer: Tokenizer used for encoding and decoding text.
#     - input_text: The initial text to start generating from.
#     - max_length: The maximum length of the generated text.

#     Returns:
#     - A string containing the generated text.
#     """
#     model.eval()  # Set the model to evaluation mode

#     # Encode the input text to tokens
#     input_ids = tokenizer.encode(input_text)

#     # Convert to a tensor and add batch dimension
#     input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

#     with torch.no_grad():
#         for _ in range(max_length):
#             # Generate output from the model
#             output = model(input_ids)

#             # Get the last predicted token
#             next_token_id = output[0, -1, :].argmax()

#             # Append the predicted token to the input
#             input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

#             # Check if the prediction is the end-of-sequence token
#             if next_token_id.item() == tokenizer.eos_id():
#                 break

#     # Decode the generated tokens to a string
#     output_text = tokenizer.decode(input_ids[0].tolist())

#     return output_text

def predict_completion(model, tokenizer, input_text, max_length=50, temperature=1.0):
    """
    Generate text completion for a given input text with randomness.

    Args:
    - model: Trained decoder model.
    - tokenizer: Tokenizer used for encoding and decoding text.
    - input_text: The initial text to start generating from.
    - max_length: The maximum length of the generated text.
    - temperature: Temperature for softmax. Higher values increase randomness.

    Returns:
    - A string containing the generated text.
    """
    model.eval()  # Set the model to evaluation mode

    # Encode the input text to tokens
    input_ids = tokenizer.encode(input_text)

    # Convert to a tensor, add batch dimension, and send to the GPU
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_ids)

            # Apply softmax with temperature
            probabilities = torch.nn.functional.softmax(output[0, -1, :] / temperature, dim=-1)

            # Random sampling
            next_token_id = torch.multinomial(probabilities, 1).item()

            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

            if next_token_id == tokenizer.eos_id():
                break

    output_text = tokenizer.decode(input_ids[0].tolist())
    return output_text

temp = 0.9
# Example usage
input_text = "FF[FZFYC][YFZxFYC]"
print(predict_completion(model, tokenizer, input_text, max_length=max_seq_length, temperature=temp))


input_text = "F[YF][XFC]"
print(predict_completion(model, tokenizer, input_text, max_length=max_seq_length, temperature=temp))

input_text = "FF[yyyyy]FF[xx]"
print(predict_completion(model, tokenizer, input_text, max_length=max_seq_length, temperature=temp))

input_text = "FF[yyyyy]FF[xx]"
print(predict_completion(model, tokenizer, input_text, max_length=max_seq_length, temperature=temp))
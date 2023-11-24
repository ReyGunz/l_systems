from torch.utils.data import Dataset
import torch

class TreeDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = [self.tokenizer.tokenize(text) for text in texts]
        self.max_tokens = max([len(e) for e in self.tokens])
        # print(self.max_tokens)
        # print(self.tokens[0], len(self.tokens[0]))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx])

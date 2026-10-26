import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size

        chunk = self.tokens[
            start:start + self.block_size + 1
        ]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y
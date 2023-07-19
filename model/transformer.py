import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
    
    def forward(self, x):
        pass
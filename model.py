import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, embedding_dim, head_size):
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1)
        wei = F.softmax(wei, dim=1)
        v = self.value(x)

        return wei @ v


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=64, CLASSES=3, dropout_prob=0.5, rnn_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(32, embedding_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(2 * hidden_size)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, CLASSES)
    
    def forward(self, x):
        emb = self.emb(x)
        pos_emb = self.pos_emb(x)
        out = self.fc1(out)
        out = self.layer_norm1(out)
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.layer_norm2(out)
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.layer_norm1(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = out.sum(dim=1)
        return out
    

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size=64, CLASSES=3, dropout_prob=0.5, rnn_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(hidden_size, hidden_size, rnn_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(2 * hidden_size)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, CLASSES)
    
    def forward(self, x):
        out = self.emb(x)
        out = self.fc1(out)
        out, _ = self.rnn(out.view(out.shape[0], out.shape[2], out.shape[3]))
        out = self.layer_norm1(out)
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.layer_norm2(out)
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.layer_norm1(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = out.sum(dim=1)
        return out

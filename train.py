import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from model import LSTMModel
import datautil
import time

def train_nn(args):
    print("Training with args:", args)
    device = tr.device('mps') if tr.backends.mps.is_available() and args.device == 'mps' else tr.device('cpu') # type: ignore
    print(f'Device: {device}')

    train_batches, val_batches, test_batches, word_to_ix = datautil.get_batches_split(args.batch_size)
    vocab_size = len(word_to_ix)

    model_meta = {
        'word_to_ix': word_to_ix,
        'device': device,
        'class_labels': ['hate speech', 'offensive language', 'neutral']
    }

    tr.save(model_meta, 'model_meta.pt')

    model = LSTMModel(vocab_size, args.embedding_dim, args.hidden_size, 3, args.dropout_prob, args.rnn_layers)
    model = model.to(device)

    optimizer = tr.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []
    val_losses = []
    
    print("\n********** training **********\n")

    start_time = time.time()
    for k in tqdm(range(args.epochs), disable=args.no_tqdm):
        model.train()
        train_loss = 0.0
        for xb, yb in train_batches:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_batches)
        losses.append(train_loss / len(train_batches))

        with tr.no_grad():
            model.eval()
            val_loss = 0.0
            for xb, yb in val_batches:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
            val_loss /= len(val_batches)
            val_losses.append(val_loss)
        
        if (k+1) % int(0.1*args.epochs) == 0 and args.no_tqdm:
            end_time = time.time()
            tr.save(model, 'model.lstm.pt')
            print(f"({k+1}/{args.epochs}): train loss: {train_loss:.4f}, val loss: {val_loss:.4f} ({end_time - start_time:.2f}s)")
            start_time = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network Training")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--rnn_layers", type=int, default=1, help="Number of rnn layers")
    parser.add_argument("--embedding_dim", type=int, default=5, help="Embedding dimension")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--hidden_size", type=int, default=64, help="Size of hidden layers")
    parser.add_argument("--dropout_prob", type=float, default=0.4, help="Dropout probability")
    parser.add_argument("--max_token_len", type=int, default=12, help="Maximum token length")
    parser.add_argument("--min_token_len", type=int, default=1, help="Minimum token length")
    parser.add_argument("--no_plot", type=bool, default=True, help="Disable plots")
    parser.add_argument("--no_tqdm", type=bool, default=True, help="Disable tqdm progress bar")
    parser.add_argument("--device", type=str, default='cpu', help="Device")

    args = parser.parse_args()

    train_nn(args)

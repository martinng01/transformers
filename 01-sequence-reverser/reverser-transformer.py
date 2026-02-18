import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import yaml


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()

        pe = torch.zeros(seq_len, d_model)

        # (1, d_model // 2)
        # i has to represent the columns in the final matrix, therefore unsqueeze 0
        i = torch.arange(0, d_model // 2).unsqueeze(0)

        # (seq_len, 1)
        pos = torch.arange(0, seq_len).unsqueeze(1)

        # (seq_len, d_model // 2)
        term = pos / torch.pow(10000, 2 * i / d_model)

        pe[:, 0::2] = torch.sin(term)
        pe[:, 1::2] = torch.cos(term)

        pe = pe.unsqueeze(0)

        # Move to gpu for the below x + self.pe operation
        self.register_buffer('pe', pe)

    def forward(self, x):
        # n_samples, seq_len, d_model
        return x + self.pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by num_heads"

        self.h = h
        self.d_k = d_model // h

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        n_samples, seq_len, d_model = x.size()

        # (n_samples, seq_len, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # (n_samples, seq_len, h, d_k)
        Q = Q.view(n_samples, seq_len, self.h, self.d_k)
        K = K.view(n_samples, seq_len, self.h, self.d_k)
        V = V.view(n_samples, seq_len, self.h, self.d_k)

        # Swap h and seq_len, matmul only works on last 2 dimensions
        # (n_samples, h, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        qkt = Q @ K.transpose(2, 3)  # (n_samples, h, seq_len, seq_len)
        scaled = qkt / math.sqrt(self.d_k)
        weights = F.softmax(scaled, dim=-1)

        self.attn_weights = weights

        output = weights @ V  # (n_samples, h, seq_len, d_k)

        # (n_samples, seq_len, d_model)
        concat = output.transpose(1, 2).contiguous().view(
            n_samples, seq_len, d_model)

        return self.W_O(concat)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, h)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        mha_out = self.mha(x)
        x = self.norm1(x + mha_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        # (batch, seq_len, dims)
        return x


class TransformerReverser(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, num_heads, num_layers):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(seq_len, d_model)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, num_heads * 4) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.fc_out(x)
        return x


class ReversalDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X.copy()).long()
        self.Y = torch.tensor(Y.copy()).long()

    def __getitem__(self, idx): return self.X[idx], self.Y[idx]
    def __len__(self): return len(self.X)


def gen_dataset(n_samples, seq_len, vocab_size):
    X = np.random.randint(1, vocab_size, size=(n_samples, seq_len))
    Y = np.flip(X, axis=1)

    return X, Y


def train(dataloader, model, criterion, optimizer, vocab_size):
    model.train()
    total_loss = 0

    for inputs, targets in dataloader:
        # inputs = (batch, seqlen)
        # targets = (batch, seqlen)
        inputs, targets = inputs.to(device), targets.to(device)

        # (batch_size, seq len, vocabsize)
        logits = model(inputs)

        # CrossEntropyLoss expects (batch, num_classes)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(dataloader, model, criterion, vocab_size):
    model.eval()
    total_loss = 0

    # to calculate accuracy
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            # inputs = (batch, seqlen)
            # targets = (batch, seqlen)
            inputs, targets = inputs.to(device), targets.to(device)

            # (batch_size, seq len, vocabsize)
            logits = model(inputs)

            # CrossEntropyLoss expects (batch, num_classes)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            # (batch_size, seqlen)
            predictions = torch.argmax(logits, dim=-1)
            total_loss += loss.item()
            correct += (predictions == targets).sum().item()
            total += targets.numel()

    return total_loss / len(dataloader), (correct / total) * 100


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config('config.yaml')

    # Hyperparameters
    vocab_size = config['training']['vocab_size']
    d_model = config['model']['d_model']
    seq_len = config['training']['seq_len']
    num_heads = config['model']['num_heads']
    batch_size = config['training']['batch_size']
    num_layers = config['model']['num_layers']
    lr = config['training']['lr']

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")

    # data setup
    X, Y = gen_dataset(
        n_samples=20000,
        seq_len=seq_len,
        vocab_size=vocab_size
    )

    train_end = int(0.8 * len(X))
    val_end = int(0.9 * len(X))

    # (n_samples, seq_len)
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end: val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    train_dataset = ReversalDataset(X_train, Y_train)
    val_dataset = ReversalDataset(X_val, Y_val)
    test_dataset = ReversalDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Model Setup
    model = TransformerReverser(
        vocab_size=vocab_size,
        d_model=d_model,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=num_layers
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    epochs = 5
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train(
            train_loader,
            model,
            criterion,
            optimizer,
            vocab_size
        )
        val_loss, val_acc = validate(
            train_loader,
            model,
            criterion,
            vocab_size
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Tests
    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    test_loss, test_acc = validate(
        test_loader,
        model,
        criterion,
        vocab_size
    )
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    print("\nSample Predictions:")
    with torch.no_grad():
        for i in range(5):
            input_seq, target_seq = test_dataset[i]

            # add batch dimension and move to device
            input_tensor = input_seq.unsqueeze(0).to(device)

            # Get prediction
            logits = model(input_tensor)
            prediction = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

            print(f"Input:    {input_seq.tolist()}")
            print(f"Target:   {target_seq.tolist()}")
            print(f"Predict:  {prediction}")
            print("-" * 20)

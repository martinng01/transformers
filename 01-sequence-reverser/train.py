import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
from model import TransformerReverser


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

import os
import glob
from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3

TEST_SIZE = 0.2
VAL_SIZE_FROM_REST = 0.25
RANDOM_STATE = 42

class RPSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_data(data_dir: str):
    pattern = os.path.join(data_dir, "rps_data_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found matching {pattern}")

    dfs = []
    for f in files:
        print(f"Loading {f}")
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(df_all)}")

    label_map: Dict[str, int] = {"rock": 0, "paper": 1, "scissors": 2}
    if not set(df_all["label"].unique()).issubset(label_map.keys()):
        raise ValueError(
            f"Unexpected labels found: {df_all['label'].unique()}, "
            f"expected subset of {list(label_map.keys())}"
        )

    y = df_all["label"].map(label_map).values
    X = df_all.drop(columns=["label"]).values

    return X, y, label_map


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, label_map = load_data(DATA_DIR)
    input_dim = X.shape[1]
    print(f"Input dim: {input_dim}")

    X_rest, X_test, y_rest, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Rest samples: {len(X_rest)}, Test samples (unseen): {len(X_test)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=VAL_SIZE_FROM_REST,
        random_state=RANDOM_STATE,
        stratify=y_rest,
    )

    print(
        f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} "
        f"(Total: {len(X)})"
    )

    train_dataset = RPSDataset(X_train, y_train)
    val_dataset = RPSDataset(X_val, y_val)
    test_dataset = RPSDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP(input_dim=input_dim, hidden_dim=128, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    save_path = "rps_model.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == y_batch).sum().item()
                total_val += y_batch.size(0)

        val_loss /= total_val
        val_acc = correct_val / total_val

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "label_map": label_map,
                },
                save_path,
            )
            print(f"  -> Saved best model to {save_path} (val_acc={best_val_acc:.3f})")

    print(f"Training done. Best val_acc = {best_val_acc:.3f}")

    print("\n=== Evaluating on UNSEEN TEST SET ===")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    inv_label_map = {v: k for k, v in label_map.items()}

    print("Test accuracy:",
          (all_preds == all_targets).sum() / len(all_targets))

    print("\nClassification report:")
    print(
        classification_report(
            all_targets,
            all_preds,
            target_names=[inv_label_map[i] for i in range(len(inv_label_map))],
            digits=3,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(all_targets, all_preds))


if __name__ == "__main__":
    train()

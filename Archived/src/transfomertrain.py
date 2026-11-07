import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from transformers import PatchTSTConfig, PatchTSTForRegression
import numpy as np

def model_config(num_features):
    config = PatchTSTConfig(
        context_length=30,
        num_input_channels=num_features,
        num_targets=1,
        patch_length=5,
        patch_stride=5,
        num_hidden_layers=3,
        d_model=64,
        num_attention_heads=4,
        head_dropout=0.2,
        loss="mse",
    )
    model = PatchTSTForRegression(config)
    params = {"epochs": 10, "batch_size": 64, "learning_rate": 1e-3, "window_size": 30}
    return model, params

def train_model(model, params, X_tr, y_tr, X_v):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                       torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1))
    loader = DataLoader(ds, batch_size=params["batch_size"], shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(params["epochs"]):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(past_values=xb).regression_outputs   # (B, 1)
            loss = criterion(preds, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        mlflow.log_metric("train_loss", total / len(ds), step=epoch)

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_v), 256):
            xb = torch.tensor(X_v[i:i+256], dtype=torch.float32).to(device)
            out.append(model(past_values=xb).regression_outputs.cpu().numpy())
    return np.concatenate(out).flatten()
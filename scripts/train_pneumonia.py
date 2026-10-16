"""Train a small baseline on the official MedMNIST+ PneumoniaMNIST split."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from visionx.findings import SmallXRayCNN


def load_split(data, name):
    images, labels = data[f"{name}_images"], data[f"{name}_labels"].reshape(-1)
    images = torch.from_numpy(images.copy()).float()
    if images.ndim == 3:
        images = images.unsqueeze(-1)
    images = images.permute(0, 3, 1, 2).mean(dim=1, keepdim=True) / 255.0
    labels = torch.from_numpy(labels.astype(np.float32))
    return TensorDataset(images, labels)


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        losses.append(float(loss_fn(model(x), y).item()))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/pneumoniamnist_224.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data)
    md5 = hashlib.md5(data_path.read_bytes()).hexdigest()
    data = np.load(data_path)
    train = load_split(data, "train")
    val = load_split(data, "val")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator,
                              pin_memory=device.type == "cuda", num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    model = SmallXRayCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        val_loss = validate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    output = Path("models/pneumonia_mnist.pt")
    output.parent.mkdir(exist_ok=True, parents=True)
    torch.save({"state_dict": best_state, "threshold": 0.5, "input_size": 224,
                "labels": ["normal", "pneumonia"], "dataset": "PneumoniaMNIST+",
                "dataset_md5": md5, "epochs": args.epochs, "seed": args.seed,
                "best_validation_loss": best_loss, "history": history}, output)
    print(f"Saved {output} (best validation loss {best_loss:.4f}; device={device})", flush=True)


if __name__ == "__main__":
    main()

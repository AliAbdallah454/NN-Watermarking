# from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class WatermarkKey:
    bits: torch.Tensor  # [T] in {0,1}
    X: torch.Tensor     # [T, M]


class UchidaWatermarker:
    """Implements embedding/extraction used in the paper."""

    def __init__(
        self,
        target_layer: nn.Conv2d,
        num_bits: int,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        self.layer = target_layer
        self.num_bits = num_bits
        self.device = device

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        _, in_c, k_h, k_w = self.layer.weight.shape
        self.M = in_c * k_h * k_w

        bits = torch.randint(0, 2, (num_bits,), generator=g, dtype=torch.float32)
        X = torch.randn(num_bits, self.M, generator=g, dtype=torch.float32)

        self.key = WatermarkKey(bits=bits.to(device), X=X.to(device))

    def mean_weight_vector(self) -> torch.Tensor:
        w = self.layer.weight  # [L, D, S, S]
        w_mean = w.mean(dim=0).reshape(-1)  # [M]
        return w_mean

    def embedding_logits(self) -> torch.Tensor:
        w_vec = self.mean_weight_vector()
        return self.key.X @ w_vec

    def embedding_loss(self) -> torch.Tensor:
        logits = self.embedding_logits()
        return F.binary_cross_entropy_with_logits(logits, self.key.bits)

    @torch.no_grad()
    def extract_bits(self) -> torch.Tensor:
        logits = self.embedding_logits()
        return (logits >= 0).float()

    @torch.no_grad()
    def bit_error_rate(self) -> float:
        pred = self.extract_bits()
        ber = (pred != self.key.bits).float().mean().item()
        return ber

    @torch.no_grad()
    def confidence(self) -> float:
        probs = torch.sigmoid(self.embedding_logits())
        return probs.mean().item()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(data_root: str, batch_size: int, download: bool) -> Tuple[DataLoader, DataLoader]:
    
    train_t = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_ds = datasets.CIFAR10(data_root, train=True, transform=train_t, download=download)
    test_ds = datasets.CIFAR10(data_root, train=False, transform=test_t, download=download)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(1, total)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    watermarker: UchidaWatermarker | None,
    lambda_wm: float,
) -> Dict[str, float]:
    model.train()
    total_ce = 0.0
    total_wm = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce = F.cross_entropy(logits, y)

        wm_loss = torch.tensor(0.0, device=device)
        if watermarker is not None and lambda_wm > 0.0:
            wm_loss = watermarker.embedding_loss()

        loss = ce + lambda_wm * wm_loss
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_ce += ce.item() * bs
        total_wm += wm_loss.item() * bs
        total += bs

    return {
        "ce_loss": total_ce / max(1, total),
        "wm_loss": total_wm / max(1, total),
    }


def save_key(key: WatermarkKey, out_path: Path) -> None:
    payload = {
        "bits": key.bits.detach().cpu(),
        "X": key.X.detach().cpu(),
    }
    torch.save(payload, out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Uchida white-box watermarking reproduction")

    p.add_argument("--out-dir", required=True, help="Directory of the output")
    p.add_argument("--device", default='cpu', choices=['cpu', 'gpu'] , help="CPU or GPU")
    p.add_argument("--seed", default='42', help="seed")

    args = p.parse_args()

    seed = args.seed
    set_seed(seed)

    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = 10
    fine_tune_epochs = 5

    data_root = './data'
    batch_size = 64
    download = True
    num_bits = 64

    lambda_wm = 0.25

    train_loader, test_loader = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        download=download,
    )

    model = SimpleCNN(num_classes=10).to(device)
    target_conv = model.features[2]
    watermarker = UchidaWatermarker(
        target_layer=target_conv,
        num_bits=num_bits,
        device=device,
        seed=seed,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    history: List[Dict[str, float]] = []

    print(f"Training started for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            watermarker=watermarker,
            lambda_wm=lambda_wm,
        )
        print("Acc")
        acc = evaluate(model, test_loader, device)
        ber = watermarker.bit_error_rate()
        wm_conf = watermarker.confidence()

        row = {
            "epoch": epoch,
            "test_acc": acc,
            "ce_loss": train_stats["ce_loss"],
            "wm_loss": train_stats["wm_loss"],
            "ber": ber,
            "wm_conf": wm_conf,
        }
        history.append(row)
        print(
            f"[train] epoch={epoch:03d} acc={acc:.2f}% "
            f"ce={row['ce_loss']:.4f} wm={row['wm_loss']:.4f} ber={ber:.4f}"
        )

    if fine_tune_epochs > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for i in range(1, fine_tune_epochs + 1):
            stats = train_epoch(model, train_loader, optimizer, device, watermarker, lambda_wm)
            acc = evaluate(model, test_loader, device)
            ber = watermarker.bit_error_rate()
            print(
                f"[fine-tune] epoch={i:03d} acc={acc:.2f}% "
                f"ce={stats['ce_loss']:.4f} wm={stats['wm_loss']:.4f} ber={ber:.4f}"
            )

    ckpt_path = out_dir / "model.pt"
    key_path = out_dir / "watermark_key.pt"
    hist_path = out_dir / "history.json"

    torch.save(model.state_dict(), ckpt_path)
    save_key(watermarker.key, key_path)
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Saved model: {ckpt_path}")
    print(f"Saved key:   {key_path}")
    print(f"Saved log:   {hist_path}")


if __name__ == "__main__":
    main()
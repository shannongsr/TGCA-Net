from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["Normal", "Transition", "Anomaly"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class DualBranchDataset(Dataset):
    def __init__(self, x_raw: np.ndarray, x_gaf: np.ndarray, y: np.ndarray) -> None:
        self.x_raw = torch.from_numpy(x_raw.astype(np.float32))
        self.x_gaf = torch.from_numpy(x_gaf.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_raw[idx], self.x_gaf[idx], self.y[idx]


class GafEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers.extend(
                [
                    nn.Conv2d(c_in, int(c_out), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(int(c_out)),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            c_in = int(c_out)
        self.out_channels = c_in
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(x))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (int(kernel_size) - 1) * int(dilation)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        y = y[..., : x.shape[-1]]
        return torch.relu(x + y)


class TCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.out_channels = int(hidden_channels)
        self.input_proj = nn.Conv1d(in_channels, int(hidden_channels), kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(int(hidden_channels), kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(int(hidden_channels), kernel_size=3, dilation=2, dropout=dropout),
            TCNBlock(int(hidden_channels), kernel_size=3, dilation=4, dropout=dropout),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.pool(self.blocks(self.input_proj(x)))


class BranchAttentionFusion(nn.Module):
    def __init__(self, raw_dim: int, gaf_dim: int, fusion_dim: int, dropout: float) -> None:
        super().__init__()
        self.raw_proj = nn.Sequential(nn.Linear(raw_dim, fusion_dim), nn.ReLU(inplace=True))
        self.gaf_proj = nn.Sequential(nn.Linear(gaf_dim, fusion_dim), nn.ReLU(inplace=True))
        self.attn = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(fusion_dim, 2),
        )

    def forward(self, raw_feat: torch.Tensor, gaf_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_z = self.raw_proj(raw_feat)
        gaf_z = self.gaf_proj(gaf_feat)
        weights = torch.softmax(self.attn(torch.cat([raw_z, gaf_z], dim=1)), dim=1)
        fused = weights[:, :1] * raw_z + weights[:, 1:] * gaf_z
        return fused, weights


class DualBranchAttentionNet(nn.Module):
    def __init__(self, raw_channels: int, gaf_channels: int, cfg: dict) -> None:
        super().__init__()
        dropout = float(cfg.get("dropout", 0.25))
        self.raw_encoder = TCNEncoder(
            in_channels=raw_channels,
            hidden_channels=int(cfg.get("tcn_hidden_channels", 96)),
            dropout=dropout,
        )
        self.gaf_encoder = GafEncoder(
            in_channels=gaf_channels,
            channels=cfg.get("cnn_channels", [32, 64, 128]),
            dropout=dropout,
        )
        fusion_dim = int(cfg.get("fusion_dim", 128))
        self.fusion = BranchAttentionFusion(
            raw_dim=self.raw_encoder.out_channels,
            gaf_dim=self.gaf_encoder.out_channels,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, len(CLASS_NAMES)),
        )

    def forward(self, x_raw: torch.Tensor, x_gaf: torch.Tensor) -> torch.Tensor:
        raw_feat = self.raw_encoder(x_raw)
        gaf_feat = self.gaf_encoder(x_gaf)
        fused, _ = self.fusion(raw_feat, gaf_feat)
        return self.head(fused)


def state_at_times(times: pd.Series, cfg: dict) -> np.ndarray:
    y = np.zeros(len(times), dtype=np.int64)
    for item in cfg["label_windows"]:
        class_id = int(item.get("class_id", CLASS_TO_ID[item["class_name"]]))
        window = (times >= pd.Timestamp(item["start"])) & (times <= pd.Timestamp(item["end"]))
        y[window.to_numpy()] = class_id
    return y


def sensor_columns(df: pd.DataFrame, channels: list[str] | str) -> list[str]:
    if channels == "auto":
        return [c for c in df.columns if "::" in c]
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError(f"Configured channels are missing from data: {missing}")
    return list(channels)


def build_windows(df: pd.DataFrame, channels: list[str], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = int(cfg["history_minutes"])
    horizon = int(cfg["prediction_horizon_minutes"])
    stride = int(cfg["stride_minutes"])
    states = state_at_times(df["t"], cfg)
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    last_start = len(df) - history - horizon + 1
    for s in range(0, max(0, last_start), stride):
        e = s + history
        f = e + horizon
        window = df.loc[s : e - 1, channels].to_numpy(dtype=np.float32)
        if np.isfinite(window).all():
            x_rows.append(window)
            y_rows.append(int(states[e:f].max()))
            starts.append(pd.Timestamp(df.loc[s, "t"]))
            ends.append(pd.Timestamp(df.loc[e - 1, "t"]))
    return np.stack(x_rows), np.asarray(y_rows), np.asarray(starts), np.asarray(ends)


def blocked_random_split(
    y: np.ndarray,
    block_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_id = np.arange(len(y)) // int(block_size)
    blocks = np.unique(block_id)
    block_y = np.asarray([int(y[block_id == b].max()) for b in blocks])
    stratify_blocks = block_y if min(np.bincount(block_y, minlength=3)) >= 2 else None
    train_blocks, hold_blocks = train_test_split(
        blocks,
        train_size=float(train_ratio),
        random_state=seed,
        stratify=stratify_blocks,
    )
    hold_y = np.asarray([int(y[block_id == b].max()) for b in hold_blocks])
    val_fraction = float(val_ratio) / max(1e-9, 1.0 - float(train_ratio))
    stratify_hold = hold_y if min(np.bincount(hold_y, minlength=3)) >= 2 else None
    val_blocks, test_blocks = train_test_split(
        hold_blocks,
        train_size=val_fraction,
        random_state=seed,
        stratify=stratify_hold,
    )
    train_idx = np.where(np.isin(block_id, train_blocks))[0]
    val_idx = np.where(np.isin(block_id, val_blocks))[0]
    test_idx = np.where(np.isin(block_id, test_blocks))[0]
    return train_idx, val_idx, test_idx


def standardize(train_x: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return tuple(((arr - mean) / std).astype(np.float32) for arr in arrays)


def to_gasf(x: np.ndarray) -> np.ndarray:
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    scaled = 2.0 * (x - x_min) / np.maximum(x_max - x_min, 1e-8) - 1.0
    scaled = np.clip(scaled, -1.0, 1.0)
    phi = np.transpose(np.arccos(scaled), (0, 2, 1))
    gaf = np.cos(phi[:, :, :, None] + phi[:, :, None, :])
    return gaf.astype(np.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb_raw, xb_gaf, yb in loader:
            logits = model(xb_raw.to(device), xb_gaf.to(device))
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Export TGCA-Net article CSV files or rerun checkpoint inference.")
    parser.add_argument(
        "--mode",
        choices=["cached", "model"],
        default="cached",
        help="cached exports article-consistent cached CSVs; model reruns the supplied checkpoint.",
    )
    parser.add_argument("--data", type=Path, default=root / "data/processed/wide_minute_median.csv")
    parser.add_argument("--model", type=Path, default=root / "models/tgca_net_best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/tgca_test")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def export_cached_outputs(output_dir: Path) -> None:
    root = repo_root()
    expected_dir = root / "expected_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "test_predictions.csv",
        "classification_report.csv",
        "confusion_matrix.csv",
        "training_history.csv",
        "metrics.json",
        "split_metadata.json",
    ]:
        shutil.copy2(expected_dir / name, output_dir / name)
    with (expected_dir / "metrics.json").open("r", encoding="utf-8") as f:
        metrics_json = json.load(f)
    metrics = {
        "model_name": metrics_json["model_name"],
        "test_accuracy": metrics_json["test_accuracy"],
        "test_macro_f1": metrics_json["test_macro_f1"],
        "test_windows": metrics_json["test_windows"],
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics_summary.csv", index=False)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Wrote article-consistent cached CSV outputs to: {output_dir}")


def main() -> None:
    args = parse_args()
    if args.mode == "cached":
        export_cached_outputs(args.output_dir)
        return

    device = torch.device(args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    df = pd.read_csv(args.data, parse_dates=["t"]).sort_values("t").reset_index(drop=True)
    if cfg.get("exclude_after"):
        df = df[df["t"] <= pd.Timestamp(cfg["exclude_after"])].reset_index(drop=True)
    channels = sensor_columns(df, checkpoint.get("channels", cfg["channels"]))
    x, y, starts, ends = build_windows(df, channels, cfg)
    train_idx, _, test_idx = blocked_random_split(
        y,
        block_size=int(cfg["block_size"]),
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        seed=int(cfg["seed"]),
    )
    x_train, x_test = standardize(x[train_idx], x[train_idx], x[test_idx])
    x_train_gaf, x_test_gaf = to_gasf(x_train), to_gasf(x_test)
    model = DualBranchAttentionNet(
        raw_channels=x_train.shape[-1],
        gaf_channels=x_train_gaf.shape[1],
        cfg=cfg,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(
        DualBranchDataset(x_test, x_test_gaf, y[test_idx]),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    y_true, y_pred = evaluate(model, loader, device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "window_start": starts[test_idx],
            "window_end": ends[test_idx],
            "y_true": y_true,
            "y_pred": y_pred,
            "y_true_name": [CLASS_NAMES[i] for i in y_true],
            "y_pred_name": [CLASS_NAMES[i] for i in y_pred],
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        labels=[0, 1, 2],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / "confusion_matrix.csv")
    metrics = {
        "model_name": checkpoint.get("model_name", "dual_tcn_gaf_attention"),
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "test_windows": len(y_true),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics_summary.csv", index=False)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Wrote CSV outputs to: {output_dir}")


if __name__ == "__main__":
    main()

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from config import DROPOUT, HIDDEN_SIZE, MODEL_PATH


def build_model(hidden_size: int = HIDDEN_SIZE, dropout: float = DROPOUT) -> nn.Module:
    model = models.resnet18(weights=None)

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_size, 1),
    )
    return model


def _infer_hidden_size(state_dict: dict) -> int:
    for key, value in state_dict.items():
        if key.endswith("fc.0.weight"):
            return value.shape[0]
    raise ValueError("Could not infer hidden size from checkpoint.")


def load_model(model_path: Path | None = None, device: str | None = None) -> tuple[nn.Module, str]:
    path = Path(model_path or MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Export `res_model` from Colab and save it to models/resnet18_endoscopy.pth"
        )

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=resolved_device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        hidden_size = checkpoint.get("hidden_size") or _infer_hidden_size(state_dict)
        dropout = checkpoint.get("dropout", DROPOUT)
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        hidden_size = checkpoint.get("hidden_size") or _infer_hidden_size(state_dict)
        dropout = checkpoint.get("dropout", DROPOUT)
    else:
        state_dict = checkpoint
        hidden_size = _infer_hidden_size(state_dict)
        dropout = DROPOUT

    model = build_model(hidden_size=hidden_size, dropout=dropout)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    return model, resolved_device

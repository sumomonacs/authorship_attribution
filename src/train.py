from pathlib import Path
import random
import numpy as np
import json
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm     

from src.model import PairDataset, SiameseModel

def save_checkpoint(model, optimizer, scheduler, epoch, best_val, args, path):
    is_parallel = isinstance(model, torch.nn.DataParallel)
    state_dict = model.module.state_dict() if is_parallel else model.state_dict()
    # move to CPU to shrink file / avoid GPU-only tensors
    state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}

    state = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_val": best_val,
        "args": dict(args),  # already a small, picklable dict
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }
    torch.save(state, path)


# split the dataset
def _split(pairs, val_ratio=0.2, seed=42):
    def norm_pair(d):
        return tuple(sorted((d["text1"], d["text2"])))

    unique_pairs = {}
    for d in pairs:
        unique_pairs.setdefault(norm_pair(d), d)

    pairs_unique = list(unique_pairs.values())

    return train_test_split(pairs_unique, test_size=val_ratio, random_state=seed)

# make data loader
def _make_loader(source, batch_size, shuffle=True):
    if isinstance(source, (str, Path)):
        ds = PairDataset(source)
    else:
        tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
        json.dump(source, tmp, ensure_ascii=False)
        tmp.close()
        ds = PairDataset(tmp.name)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def _epoch(model, loader, loss_fn, optimizer, device, grad_accum=1, desc="", track_authors=False):
    is_train = optimizer is not None
    model.train(is_train)

    total, running_loss = 0, 0.0
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)

    for step, batch in enumerate(pbar, 1):
        # --- unpack: support both (s1,s2,y) and (s1,s2,y,a1,a2)
        if len(batch) == 3:
            s1, s2, labels = batch
            a1 = a2 = None
        elif len(batch) == 5:
            s1, s2, labels, a1, a2 = batch
        else:
            raise ValueError(f"Unexpected batch format of length {len(batch)}")

        labels = labels.to(device)                    # (B,) or (B,1)
        logits = model(s1, s2)                        # same shape as labels
        loss = loss_fn(logits, labels)

        if is_train:
            loss.backward()
            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        pbar.set_postfix(loss=f"{running_loss / total:.4f}")

    return running_loss / total


# training
def fit(pairs_json,
        val_ratio=0.2,
        epochs = 3,
        batch_size = 32,
        lr= 2e-5,
        freeze_layers = 0,
        grad_accum = 1,
        out_dir = 'models',
        device=None):

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu")) if device is None else torch.device(device)

    with open(pairs_json, encoding="utf-8") as f:
        data = json.load(f)
    del f

    cfg = {
        "pairs_json": str(pairs_json),
        "val_ratio":  val_ratio,
        "epochs":     epochs,
        "batch_size": batch_size,
        "lr":         lr,
        "freeze_layers": freeze_layers,
        "grad_accum": grad_accum,
        "out_dir":    str(out_dir),
        "device":     str(device),
    }

    train_pairs, val_pairs = _split(data, val_ratio=val_ratio)

    train_loader = _make_loader(train_pairs, batch_size, shuffle=True)
    val_loader   = _make_loader(val_pairs,   batch_size, shuffle=False)

    model = SiameseModel(device=device)
    if freeze_layers:
        model.freeze_encoder_layers(freeze_layers)
    model.to(device)

    loss_fn  = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = None  

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"
    best_val  = float("inf")

    for epoch in range(1, epochs + 1):
        tqdm.write(f"\nEpoch {epoch}/{epochs}")
        train_loss = _epoch(model, train_loader, loss_fn, optimizer,
                            device, grad_accum, desc="train")
        val_loss   = _epoch(model, val_loader,   loss_fn, None,
                            device, desc="val")

        tqdm.write(f" train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_val, args=cfg, path=best_ckpt)
            tqdm.write(f" new best model saved to {best_ckpt}")
            save_checkpoint(model, optimizer, scheduler,
                      epoch, best_val, args=locals(),
                        path=last_ckpt)

    return best_ckpt

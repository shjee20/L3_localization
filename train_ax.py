from pathlib import Path
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions import *
from models import *

try:
    import mlflow
    import mlflow.pytorch
except ModuleNotFoundError:
    mlflow = None


DEFAULT_TAU = {
    "sigmoid": 4.55,
    "gaussian": 4.25,
    "laplace": 7.21,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train axial mid-L3 one-to-one, many-to-one, or many-to-many sequence model."
    )
    parser.add_argument("--project_root", type=str, default="/home/shjee/projects/ct_spine")
    parser.add_argument("--img_root", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="ct_axial_midL3")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_type", type=str, default="context_many_to_many_transformer",
                        choices=["one_to_one", "context_many_to_one", "context_many_to_many_transformer"])
    parser.add_argument("--seq_len", type=int, default=5,
                        help="Odd context length T, e.g. 5, 7, or 11.")
    parser.add_argument("--padding_mode", type=str, default="replicate",
                        choices=["replicate"])

    parser.add_argument("--target_mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--soft_label_type", type=str, default="laplace",
                        choices=["laplace", "gaussian", "sigmoid"])
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--soft_weight_alpha", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_transformer_layers", type=int, default=1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_pretrained", action="store_true")

    parser.add_argument("--force_new_split", action="store_true")
    parser.add_argument("--skip_volume_eval", action="store_true")
    parser.add_argument("--save_probability_curves", action="store_true")
    parser.add_argument("--disable_mlflow", action="store_true")
    return parser.parse_args()


def build_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])


def create_axial_loaders(
    img_root: Path,
    train_ids,
    val_ids,
    test_ids,
    model_type: str,
    batch_size: int = 32,
    num_workers: int = 4,
    target_mode: str = "soft",
    soft_type: str = "laplace",
    tau: float = 7.21,
    seq_len: int = 5,
    padding_mode: str = "replicate",
):
    tf = build_transform()

    if model_type == "context_many_to_many_transformer":
        dataset_cls = AxialContextSequenceDataset
    elif model_type == "context_many_to_one":
        dataset_cls = AxialContextWindowDataset
    else:
        dataset_cls = AxialMidL3Dataset
    common_kwargs = {
        "img_root": img_root,
        "transform": tf,
        "target_mode": target_mode,
        "soft_type": soft_type,
        "tau": tau,
    }

    if model_type in ["context_many_to_one", "context_many_to_many_transformer"]:
        common_kwargs.update({
            "seq_len": seq_len,
            "padding_mode": padding_mode,
        })

    train_dataset = dataset_cls(patient_ids=train_ids, **common_kwargs)
    val_dataset = dataset_cls(patient_ids=val_ids, **common_kwargs)
    test_dataset = dataset_cls(patient_ids=test_ids, **common_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_ct,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ct,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ct,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, tf


def build_model(model_type: str, pretrained: bool, args):
    if model_type == "one_to_one":
        return ResNet34BinaryCT(pretrained=pretrained)
    if model_type == "context_many_to_one":
        return ContextResNet34ManyToOne(pretrained=pretrained)
    if model_type == "context_many_to_many_transformer":
        return ContextResNet34ManyToManyTransformer(
            seq_len=args.seq_len,
            pretrained=pretrained,
            d_model=args.d_model,
            num_transformer_layers=args.num_transformer_layers,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            transformer_dropout=args.transformer_dropout,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def train_one_epoch_axial(model, loader, optimizer, criterion, device, amp_dtype, scaler=None):
    model.train()
    running = 0.0
    n = 0
    pbar = tqdm(loader, desc="Train", leave=False)

    for xs, ys, _metas in pbar:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(xs)
            loss = criterion(logits, ys)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = xs.size(0)
        running += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(running / max(1, n)):.4f}")

    return running / max(1, n)


@torch.no_grad()
def evaluate_axial_soft(model, loader, criterion, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    n = 0
    probs_all = []
    gts_all = []

    for xs, ys, _metas in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(xs)
            loss = criterion(logits, ys)

        bs = xs.size(0)
        total_loss += loss.item() * bs
        n += bs
        probs_all.append(torch.sigmoid(logits).detach().cpu())
        gts_all.append(ys.detach().cpu())

    probs_all = torch.cat(probs_all, dim=0).numpy().reshape(-1)
    gts_all = torch.cat(gts_all, dim=0).numpy().reshape(-1)
    mse = float(np.mean((probs_all - gts_all) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(probs_all - gts_all)))
    val_loss = total_loss / max(1, n)
    return val_loss, mse, rmse, mae


@torch.no_grad()
def sanity_check_shapes(model, loader, device, amp_dtype, model_type: str):
    xs, ys, metas = next(iter(loader))
    expected_dim = 5 if model_type in ["context_many_to_one", "context_many_to_many_transformer"] else 4
    if xs.ndim != expected_dim:
        raise RuntimeError(f"Expected input dim {expected_dim}, got {xs.ndim}: {tuple(xs.shape)}")
    if model_type == "context_many_to_many_transformer":
        if ys.ndim != 2 or ys.shape[1] != xs.shape[1]:
            raise RuntimeError(f"Expected targets with shape (B,T), got {tuple(ys.shape)}")
    elif ys.ndim != 2 or ys.shape[1] != 1:
        raise RuntimeError(f"Expected targets with shape (B,1), got {tuple(ys.shape)}")

    model.eval()
    xs = xs.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        logits = model(xs)

    if logits.shape != ys.shape:
        raise RuntimeError(f"Expected logits shape {tuple(ys.shape)}, got {tuple(logits.shape)}")

    center = metas[0].get("slno")
    print(f"[SANITY] xs={tuple(xs.shape)} ys={tuple(ys.shape)} logits={tuple(logits.shape)} first_center_slno={center}")


def prepare_split(img_root: Path, seed: int, force_new_split: bool):
    all_patient_dirs = [p for p in img_root.iterdir() if is_patient_dir(p)]
    all_patient_dirs = sorted(all_patient_dirs, key=lambda x: x.name)
    print(f"[INFO] Found patient dirs: {len(all_patient_dirs)}")

    patient_meta = []
    valid_ids = []
    for pdir in all_patient_dirs:
        meta = validate_patient(pdir, strict=True)
        if meta is None:
            continue
        patient_meta.append(meta)
        valid_ids.append(meta["patient_id"])

    if len(valid_ids) < 8:
        raise RuntimeError(f"Too few valid patients: {len(valid_ids)}")

    split_path = img_root / f"split_6_2_2_seed{seed}.json"
    if split_path.exists() and not force_new_split:
        with open(split_path, "r", encoding="utf-8") as f:
            split_json = json.load(f)
        train_ids = split_json["train_ids"]
        val_ids = split_json["val_ids"]
        test_ids = split_json["test_ids"]
        print(f"[SPLIT] Reused {split_path}")
    else:
        train_ids, val_ids, test_ids = split_patients(valid_ids, seed=seed)
        split_json = {
            "seed": seed,
            "img_root": str(img_root),
            "n_total_valid": len(valid_ids),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "patient_meta": patient_meta,
        }
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_json, f, ensure_ascii=False, indent=2)
        print(f"[SPLIT] Saved {split_path}")

    print(f"[SPLIT] train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    return train_ids, val_ids, test_ids, split_path


def main():
    args = parse_args()
    set_seed(args.seed)
    if mlflow is None and not args.disable_mlflow:
        print("[WARN] mlflow is not installed; continuing with MLflow disabled.")
        args.disable_mlflow = True

    if args.model_type == "one_to_one":
        args.seq_len = 1
    elif args.seq_len % 2 == 0:
        raise ValueError("--seq_len must be odd for center-slice window construction.")

    project_root = Path(args.project_root)
    img_root = Path(args.img_root) if args.img_root is not None else project_root / "axial_slices"
    tau = args.tau if args.tau is not None else DEFAULT_TAU[args.soft_label_type]

    device = get_device()
    amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    train_ids, val_ids, test_ids, split_path = prepare_split(
        img_root=img_root,
        seed=args.seed,
        force_new_split=args.force_new_split,
    )

    train_loader, val_loader, test_loader, tf = create_axial_loaders(
        img_root=img_root,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_mode=args.target_mode,
        soft_type=args.soft_label_type,
        tau=tau,
        seq_len=args.seq_len,
        padding_mode=args.padding_mode,
    )

    model = build_model(
        model_type=args.model_type,
        pretrained=not args.no_pretrained,
        args=args,
    ).to(device)
    criterion = TargetWeightedBCEWithLogitsLoss(alpha=args.soft_weight_alpha)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    sanity_check_shapes(model, train_loader, device, amp_dtype, args.model_type)

    model_names = {
        "one_to_one": "ResNet34BinaryCT",
        "context_many_to_one": "ContextResNet34ManyToOne",
        "context_many_to_many_transformer": "ContextResNet34ManyToManyTransformer",
    }
    model_name = model_names[args.model_type]
    run_name = (
        f"{args.model_type}_{args.target_mode}_{args.soft_label_type}"
        f"_tau{tau}_T{args.seq_len}_alpha{args.soft_weight_alpha}"
    )
    save_dir = project_root / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / f"best_{run_name}.pth"
    result_dir = project_root / "results" / run_name

    if not args.disable_mlflow:
        mlflow.set_experiment(args.experiment_name)

    mlflow_context = mlflow.start_run(run_name=run_name) if not args.disable_mlflow else None
    if mlflow_context is not None:
        mlflow_context.__enter__()

    try:
        if not args.disable_mlflow:
            mlflow.log_params({
                "arch": model_name,
                "task": "axial_midL3",
                "model_type": args.model_type,
                "target_mode": args.target_mode,
                "soft_label_type": args.soft_label_type,
                "tau": tau,
                "seq_len": args.seq_len,
                "padding_mode": args.padding_mode,
                "soft_weight_alpha": args.soft_weight_alpha,
                "d_model": args.d_model if args.model_type == "context_many_to_many_transformer" else None,
                "num_transformer_layers": args.num_transformer_layers if args.model_type == "context_many_to_many_transformer" else None,
                "nhead": args.nhead if args.model_type == "context_many_to_many_transformer" else None,
                "dim_feedforward": args.dim_feedforward if args.model_type == "context_many_to_many_transformer" else None,
                "transformer_dropout": args.transformer_dropout if args.model_type == "context_many_to_many_transformer" else None,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "img_size": "256x256",
                "split_path": str(split_path),
            })

        best_val_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            print(f"\n=== Epoch {epoch}/{args.epochs} ===")
            train_loss = train_one_epoch_axial(
                model, train_loader, optimizer, criterion,
                device=device, amp_dtype=amp_dtype, scaler=scaler,
            )
            val_loss, val_mse, val_rmse, val_mae = evaluate_axial_soft(
                model, val_loader, criterion, device=device, amp_dtype=amp_dtype,
            )

            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} rmse={val_rmse:.4f} mae={val_mae:.4f}"
            )
            if not args.disable_mlflow:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mse": val_mse,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "model_type": args.model_type,
                    "target_mode": args.target_mode,
                    "soft_label_type": args.soft_label_type,
                    "tau": tau,
                    "seq_len": args.seq_len,
                    "padding_mode": args.padding_mode,
                    "soft_weight_alpha": args.soft_weight_alpha,
                    "d_model": args.d_model if args.model_type == "context_many_to_many_transformer" else None,
                    "num_transformer_layers": args.num_transformer_layers if args.model_type == "context_many_to_many_transformer" else None,
                    "nhead": args.nhead if args.model_type == "context_many_to_many_transformer" else None,
                    "dim_feedforward": args.dim_feedforward if args.model_type == "context_many_to_many_transformer" else None,
                    "transformer_dropout": args.transformer_dropout if args.model_type == "context_many_to_many_transformer" else None,
                    "split_path": str(split_path),
                }, best_ckpt_path)
                print(f"  -> Best model updated (val_loss={best_val_loss:.4f})")

        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        test_loss, test_mse, test_rmse, test_mae = evaluate_axial_soft(
            model, test_loader, criterion, device=device, amp_dtype=amp_dtype,
        )
        print(f"\n[Test slices] loss={test_loss:.4f} rmse={test_rmse:.4f} mae={test_mae:.4f}")
        if not args.disable_mlflow:
            mlflow.log_metrics({
                "test_slice_loss": test_loss,
                "test_slice_mse": test_mse,
                "test_slice_rmse": test_rmse,
                "test_slice_mae": test_mae,
                "best_val_loss": best_val_loss,
            })

        if not args.skip_volume_eval:
            test_metrics, _ = evaluate_volume_localization(
                model=model,
                img_root=img_root,
                patient_ids=test_ids,
                transform=tf,
                device=device,
                amp_dtype=amp_dtype,
                model_type=args.model_type,
                seq_len=args.seq_len,
                padding_mode=args.padding_mode,
                batch_size=args.eval_batch_size,
                save_dir=result_dir,
                split_name="test",
                save_probability_curves=args.save_probability_curves,
            )
            print(f"[Test volume] {test_metrics}")
            if not args.disable_mlflow:
                mlflow.log_metrics({
                    f"test_volume_{k}": v
                    for k, v in test_metrics.items()
                    if isinstance(v, (int, float)) and v is not None
                })

        print(f"[DONE] best_ckpt={best_ckpt_path}")
        print(f"[DONE] results_dir={result_dir}")

    finally:
        if mlflow_context is not None:
            mlflow_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()

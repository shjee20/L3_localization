from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from functions import (
    get_axial_patient_records,
    load_axial_image_tensor,
    make_context_window,
)
from models import (
    ContextResNet34ManyToManyTransformer,
    ContextResNet34ManyToOne,
    ResNet34BinaryCT,
)


DEFAULT_TAU = {
    "sigmoid": 4.55,
    "gaussian": 4.25,
    "laplace": 7.21,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volume-level axial mid-L3 test evaluator."
    )
    parser.add_argument("--project_root", type=str, default="/home/shjee/projects/ct_spine")
    parser.add_argument("--img_root", type=str, default=None)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_8", type=str, default=None)

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["one_to_one", "context_many_to_one", "context_many_to_many_transformer"],
    )
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--padding_mode", type=str, default="replicate", choices=["replicate"])
    parser.add_argument("--soft_label_type", type=str, default="laplace",
                        choices=["laplace", "gaussian", "sigmoid"])
    parser.add_argument("--tau", type=float, default=None)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_transformer_layers", type=int, default=1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    parser.add_argument("--attn_dist_alpha", type=float, default=0.0)
    parser.add_argument("--attn_dist_mode", type=str, default="none", choices=["none", "gaussian", "laplace"])
    parser.add_argument("--pretrained", action="store_true",
                        help="Instantiate backbone with pretrained weights before loading checkpoint.")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_probability_curves", action="store_true")
    parser.add_argument("--save_plots", action="store_true")

    parser.add_argument("--smooth_prob_curve", action="store_true",
                        help="Optional one-to-one baseline post-processing only.")
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument(
        "--sampling_interval",
        type=float,
        default=6.0,
        help="Interval between sampled axial slices in original slice-index units.",
    )
    parser.add_argument(
        "--m2m_inference_mode",
        type=str,
        default="center",
        choices=["center", "aggregate"],
        help=(
            "Inference mode for context_many_to_many_transformer. "
            "'center' uses only the center token prediction. "
            "'aggregate' averages overlapping token predictions for each slice."
        ),
    )
    parser.add_argument(
        "--m2m_supervision_mode",
        type=str,
        default="all_tokens",
        choices=["all_tokens", "center_token"],
        help=(
            "Training supervision metadata for context_many_to_many_transformer. "
            "This does not change inference."
        ),
    )
    return parser.parse_args()


def get_checkpoint_path(args):
    checkpoint = args.checkpoint_8 or args.checkpoint
    if checkpoint is None:
        raise ValueError("Provide --checkpoint_8 or --checkpoint.")
    return Path(checkpoint)


def build_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])


def load_split_ids(split_path: Path):
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    if "test_ids" not in split:
        raise KeyError(f"split file must contain test_ids: {split_path}")
    return split["test_ids"]


def _ckpt_args_dict(ckpt):
    if not isinstance(ckpt, dict):
        return {}
    raw_args = ckpt.get("args", {})
    if isinstance(raw_args, argparse.Namespace):
        return vars(raw_args)
    if isinstance(raw_args, dict):
        return raw_args
    return {}


def _get_ckpt_value(ckpt, name, default):
    if isinstance(ckpt, dict) and name in ckpt and ckpt[name] is not None:
        return ckpt[name]
    ckpt_args = _ckpt_args_dict(ckpt)
    if name in ckpt_args and ckpt_args[name] is not None:
        return ckpt_args[name]
    return default


def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt

    for key in ["model_state_dict", "model_state", "state_dict"]:
        if key in ckpt:
            return ckpt[key]

    if ckpt and all(hasattr(v, "shape") for v in ckpt.values()):
        return ckpt

    raise KeyError(
        "Checkpoint must be a raw state_dict or contain one of: "
        "model_state_dict, model_state, state_dict"
    )


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def build_model(args, ckpt):
    seq_len = int(_get_ckpt_value(ckpt, "seq_len", args.seq_len))
    if args.model_type == "one_to_one":
        return ResNet34BinaryCT(pretrained=args.pretrained), 1

    if seq_len < 1 or seq_len % 2 == 0:
        raise ValueError(f"--seq_len must be a positive odd integer, got {seq_len}")

    if args.model_type == "context_many_to_one":
        return ContextResNet34ManyToOne(pretrained=args.pretrained), seq_len

    d_model = int(_get_ckpt_value(ckpt, "d_model", args.d_model))
    num_layers = int(_get_ckpt_value(ckpt, "num_transformer_layers", args.num_transformer_layers))
    nhead = int(_get_ckpt_value(ckpt, "nhead", args.nhead))
    dim_ff = int(_get_ckpt_value(ckpt, "dim_feedforward", args.dim_feedforward))
    dropout = float(_get_ckpt_value(ckpt, "transformer_dropout", args.transformer_dropout))
    attn_dist_alpha = float(_get_ckpt_value(ckpt, "attn_dist_alpha", getattr(args, "attn_dist_alpha", 0.0)))
    attn_dist_mode = str(_get_ckpt_value(ckpt, "attn_dist_mode", getattr(args, "attn_dist_mode", "none")))
    m2m_supervision_mode = str(
        _get_ckpt_value(ckpt, "m2m_supervision_mode", getattr(args, "m2m_supervision_mode", "all_tokens"))
    )

    model = ContextResNet34ManyToManyTransformer(
        seq_len=seq_len,
        pretrained=args.pretrained,
        d_model=d_model,
        num_transformer_layers=num_layers,
        nhead=nhead,
        dim_feedforward=dim_ff,
        transformer_dropout=dropout,
        attn_dist_alpha=attn_dist_alpha,
        attn_dist_mode=attn_dist_mode,
    )
    model.m2m_supervision_mode = m2m_supervision_mode
    return model, seq_len


def load_model(args, device):
    ckpt_path = get_checkpoint_path(args)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    model, seq_len = build_model(args, ckpt)
    state_dict = strip_module_prefix(extract_state_dict(ckpt))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, seq_len


def batch_iter(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def infer_one_to_one(model, records, transform, device, amp_dtype, batch_size):
    probs = []
    for batch in batch_iter(records, batch_size):
        xs = [
            load_axial_image_tensor(rec["img_path"], transform=transform)
            for rec in batch
        ]
        xs = torch.stack(xs, dim=0).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=torch.cuda.is_available()):
            logits = model(xs)
            if logits.numel() != xs.shape[0]:
                raise RuntimeError(
                    f"Expected one-to-one logits reshapeable to (B,), got {tuple(logits.shape)}"
                )
            logits = logits.reshape(-1)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(batch_probs.tolist())
    return np.asarray(probs, dtype=np.float32)


def infer_context_many_to_one(model, records, transform, device, amp_dtype, seq_len, padding_mode, batch_size):
    probs = []
    for batch_indices in batch_iter(list(range(len(records))), batch_size):
        windows = [
            make_context_window(
                records=records,
                center_idx=i,
                seq_len=seq_len,
                transform=transform,
                padding_mode=padding_mode,
            )[0]
            for i in batch_indices
        ]
        xs = torch.stack(windows, dim=0).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=torch.cuda.is_available()):
            logits = model(xs)
            if logits.numel() != xs.shape[0]:
                raise RuntimeError(
                    f"Expected many-to-one logits reshapeable to (B,), got {tuple(logits.shape)}"
                )
            logits = logits.reshape(-1)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(batch_probs.tolist())
    return np.asarray(probs, dtype=np.float32)


def _valid_token_target_index(center_idx, token_idx, n_records, seq_len):
    half = seq_len // 2
    target_idx = center_idx + (token_idx - half)
    if 0 <= target_idx < n_records:
        return target_idx
    return None


def infer_context_many_to_many_transformer(
    model,
    records,
    transform,
    device,
    amp_dtype,
    seq_len,
    padding_mode,
    batch_size,
    inference_mode,
):
    center_idx = seq_len // 2
    n_records = len(records)

    if inference_mode == "aggregate":
        prob_sums = np.zeros(n_records, dtype=np.float64)
        prob_counts = np.zeros(n_records, dtype=np.float64)
    else:
        probs = []

    all_indices = list(range(n_records))
    for batch_indices in batch_iter(all_indices, batch_size):
        windows = [
            make_context_window(
                records=records,
                center_idx=i,
                seq_len=seq_len,
                transform=transform,
                padding_mode=padding_mode,
            )[0]
            for i in batch_indices
        ]
        xs = torch.stack(windows, dim=0).to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=torch.cuda.is_available()):
            logits = model(xs)
            if logits.ndim != 2:
                raise RuntimeError(f"Expected logits shape (B,T), got {tuple(logits.shape)}")
            if logits.shape[1] != seq_len:
                raise RuntimeError(f"Expected T={seq_len}, got {logits.shape[1]}")
            prob_tokens = torch.sigmoid(logits).detach().cpu().numpy()

        if inference_mode == "aggregate":
            for row_idx, center_record_idx in enumerate(batch_indices):
                for token_idx in range(seq_len):
                    record_idx = _valid_token_target_index(
                        center_record_idx,
                        token_idx,
                        n_records,
                        seq_len,
                    )
                    if record_idx is None:
                        continue
                    prob_sums[record_idx] += float(prob_tokens[row_idx, token_idx])
                    prob_counts[record_idx] += 1.0
        else:
            probs.extend(prob_tokens[:, center_idx].tolist())

    if inference_mode == "aggregate":
        debug = {
            "aggregate_prob_count_min": float(np.min(prob_counts)),
            "aggregate_prob_count_max": float(np.max(prob_counts)),
            "aggregate_prob_count_zero": int(np.sum(prob_counts == 0)),
        }
        if np.any(prob_counts == 0):
            raise RuntimeError("Aggregate inference produced empty probability slots.")
        return (prob_sums / prob_counts).astype(np.float32), debug

    return np.asarray(probs, dtype=np.float32), {}


def smooth_probability_curve(probs, sigma):
    if sigma <= 0:
        return probs
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(probs, sigma=sigma, mode="nearest").astype(np.float32)
    except Exception:
        radius = max(1, int(math.ceil(3 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x * x) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        padded = np.pad(probs, pad_width=radius, mode="edge")
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def get_prediction_rule(soft_label_type):
    if str(soft_label_type).strip().lower() == "sigmoid":
        return "closest_to_0.5"
    return "argmax"


def select_pred_idx_from_prob_curve(probs, soft_label_type):
    probs = np.asarray(probs, dtype=np.float32)
    if probs.size == 0:
        raise ValueError("Cannot select predicted index from an empty probability curve.")

    prediction_rule = get_prediction_rule(soft_label_type)
    if prediction_rule == "closest_to_0.5":
        pred_idx = int(np.argmin(np.abs(probs - 0.5)))
    else:
        pred_idx = int(np.argmax(probs))
    return pred_idx, prediction_rule


def sanity_check_prediction_rule():
    probs = np.array([0.1, 0.4, 0.52, 0.8], dtype=np.float32)
    sigmoid_idx, sigmoid_rule = select_pred_idx_from_prob_curve(probs, "sigmoid")
    argmax_idx, argmax_rule = select_pred_idx_from_prob_curve(probs, "gaussian")
    assert sigmoid_rule == "closest_to_0.5" and sigmoid_idx == 2
    assert argmax_rule == "argmax" and argmax_idx == 3


@torch.no_grad()
def predict_volume_prob_curve(
    model,
    records,
    model_type,
    seq_len,
    transform,
    device,
    amp_dtype,
    batch_size,
    padding_mode,
    m2m_inference_mode,
    soft_label_type="laplace",
    smooth_prob_curve=False,
    smooth_sigma=1.0,
):
    if model_type == "one_to_one":
        probs = infer_one_to_one(model, records, transform, device, amp_dtype, batch_size)
        if smooth_prob_curve:
            probs = smooth_probability_curve(probs, smooth_sigma)
        debug = {}
    elif model_type == "context_many_to_one":
        probs = infer_context_many_to_one(
            model, records, transform, device, amp_dtype, seq_len, padding_mode, batch_size
        )
        debug = {}
    elif model_type == "context_many_to_many_transformer":
        probs, debug = infer_context_many_to_many_transformer(
            model=model,
            records=records,
            transform=transform,
            device=device,
            amp_dtype=amp_dtype,
            seq_len=seq_len,
            padding_mode=padding_mode,
            batch_size=batch_size,
            inference_mode=m2m_inference_mode,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if len(probs) != len(records):
        raise RuntimeError(
            f"Probability curve length mismatch: got {len(probs)}, expected {len(records)}"
        )
    pred_idx, prediction_rule = select_pred_idx_from_prob_curve(probs, soft_label_type)
    debug["prediction_rule"] = prediction_rule
    return probs, pred_idx, debug


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def pearson_corr(gt, pred):
    if len(gt) < 2 or np.std(gt) <= 0 or np.std(pred) <= 0:
        return None
    return _safe_float(np.corrcoef(gt, pred)[0, 1])


def rankdata_average(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(gt, pred):
    if len(gt) < 2:
        return None
    return pearson_corr(rankdata_average(gt), rankdata_average(pred))


def read_slice_spacing_mm(img_root, patient_id):
    label_path = Path(img_root) / patient_id / "labels.json"
    if not label_path.exists():
        return None
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    except Exception:
        return None

    candidate_keys = [
        "slice_spacing",
        "slice_spacing_mm",
        "spacing_z",
        "z_spacing",
        "thickness",
        "slice_thickness",
    ]
    for key in candidate_keys:
        value = labels.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        if isinstance(value, list) and len(value) > 0:
            numeric = [float(v) for v in value if isinstance(v, (int, float)) and v > 0]
            if numeric:
                return float(np.mean(numeric))

    spacing = labels.get("spacing")
    if isinstance(spacing, list) and len(spacing) >= 3:
        try:
            value = float(spacing[-1])
            return value if value > 0 else None
        except Exception:
            return None
    return None


def top_two_peaks(slnos, probs):
    if len(probs) == 0:
        return {
            "top1_ax": "",
            "top1_prob": "",
            "top2_ax": "",
            "top2_prob": "",
            "top1_top2_margin": "",
        }

    order = np.argsort(np.asarray(probs))[::-1]
    top1_idx = int(order[0])
    top1_prob = float(probs[top1_idx])
    if len(order) > 1:
        top2_idx = int(order[1])
        top2_prob = float(probs[top2_idx])
        top2_ax = int(slnos[top2_idx])
        margin = top1_prob - top2_prob
    else:
        top2_prob = ""
        top2_ax = ""
        margin = ""

    return {
        "top1_ax": int(slnos[top1_idx]),
        "top1_prob": top1_prob,
        "top2_ax": top2_ax,
        "top2_prob": top2_prob,
        "top1_top2_margin": margin,
    }


def compute_summary_metrics(rows):
    valid = [
        row for row in rows
        if row["gt_mid_ax"] != "" and row["pred_mid_ax"] != ""
    ]
    if not valid:
        return {"n_patients": 0}

    signed_raw = np.asarray([float(row["error_signed_raw"]) for row in valid], dtype=np.float64)
    errors_raw = np.asarray([float(row["error_raw"]) for row in valid], dtype=np.float64)
    errors_sampled = np.asarray([float(row["error_sampled"]) for row in valid], dtype=np.float64)
    gt = np.asarray([float(row["gt_mid_ax"]) for row in valid], dtype=np.float64)
    pred = np.asarray([float(row["pred_mid_ax"]) for row in valid], dtype=np.float64)

    metrics = {
        "n_patients": int(len(valid)),
        "mean_abs_error_raw": _safe_float(np.mean(errors_raw)),
        "std_abs_error_raw": _safe_float(np.std(errors_raw)),
        "median_abs_error_raw": _safe_float(np.median(errors_raw)),
        "mean_signed_error_raw": _safe_float(np.mean(signed_raw)),
        "std_signed_error_raw": _safe_float(np.std(signed_raw)),
        "exact_match_rate_raw": _safe_float(np.mean(errors_raw == 0)),
        "le_1_raw_accuracy": _safe_float(np.mean(errors_raw <= 1)),
        "le_2_raw_accuracy": _safe_float(np.mean(errors_raw <= 2)),
        "outlier_rate_raw_ge_3": _safe_float(np.mean(errors_raw >= 3)),
        "outlier_rate_raw_ge_5": _safe_float(np.mean(errors_raw >= 5)),
        "pearson_pred_gt_raw": pearson_corr(gt, pred),
        "spearman_pred_gt_raw": spearman_corr(gt, pred),
        "mean_abs_error_sampled": _safe_float(np.mean(errors_sampled)),
        "std_abs_error_sampled": _safe_float(np.std(errors_sampled)),
        "median_abs_error_sampled": _safe_float(np.median(errors_sampled)),
        "le_1_sampled_accuracy": _safe_float(np.mean(errors_sampled <= 1)),
        "le_2_sampled_accuracy": _safe_float(np.mean(errors_sampled <= 2)),
        "outlier_rate_sampled_ge_3": _safe_float(np.mean(errors_sampled >= 3)),
        "outlier_rate_sampled_ge_5": _safe_float(np.mean(errors_sampled >= 5)),
    }

    mm_errors = [
        float(row["error_abs_mm"])
        for row in valid
        if row["error_abs_mm"] != ""
    ]
    if mm_errors:
        mm_errors = np.asarray(mm_errors, dtype=np.float64)
        metrics["mean_abs_error_mm"] = _safe_float(np.mean(mm_errors))
        metrics["median_abs_error_mm"] = _safe_float(np.median(mm_errors))

    return metrics


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_probability_curve(path, slnos, probs, gt_mid_ax, pred_mid_ax):
    rows = []
    for local_idx, (slno, prob) in enumerate(zip(slnos, probs)):
        rows.append({
            "slice_index_local": int(local_idx),
            "ax_number": int(slno),
            "prob": float(prob),
            "is_gt_mid": int(gt_mid_ax is not None and int(slno) == int(gt_mid_ax)),
            "is_pred_mid": int(pred_mid_ax is not None and int(slno) == int(pred_mid_ax)),
        })
    write_csv(path, rows, ["slice_index_local", "ax_number", "prob", "is_gt_mid", "is_pred_mid"])


def save_curve_plot(path, patient_id, slnos, probs, gt_mid_ax, pred_mid_ax, error_raw, error_sampled, inference_mode):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(slnos, probs, color="red", linewidth=2, label="Pred probability")
    if gt_mid_ax is not None:
        plt.axvline(gt_mid_ax, color="blue", linestyle="-.", linewidth=2, label="GT mid-L3")
    if pred_mid_ax is not None:
        plt.axvline(pred_mid_ax, color="green", linestyle="--", linewidth=2, label="Pred mid-L3")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Axial slice number")
    plt.ylabel("Probability")
    plt.title(
        f"{patient_id} | error_raw={error_raw} | "
        f"error_sampled={error_sampled} | mode={inference_mode}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def maybe_write_excel(path, rows, metrics):
    try:
        import pandas as pd
    except Exception:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="patient_errors", index=False)
        pd.DataFrame(
            [{"metric": key, "value": value} for key, value in metrics.items()]
        ).to_excel(writer, sheet_name="summary", index=False)


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    img_root = Path(args.img_root) if args.img_root else project_root / "axial_slices"
    split_path = Path(args.split_path) if args.split_path else img_root / "split_6_2_2_seed42.json"
    checkpoint_path = get_checkpoint_path(args)
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "test_results" / checkpoint_path.stem
    tau = args.tau if args.tau is not None else DEFAULT_TAU[args.soft_label_type]

    if args.model_type == "one_to_one":
        if args.smooth_prob_curve:
            print(f"[INFO] one-to-one smoothing enabled: sigma={args.smooth_sigma}")
    else:
        if args.smooth_prob_curve:
            raise ValueError("--smooth_prob_curve is only allowed for one_to_one baseline.")
        if args.seq_len < 1 or args.seq_len % 2 == 0:
            raise ValueError(f"--seq_len must be a positive odd integer, got {args.seq_len}")
    if args.sampling_interval <= 0:
        raise ValueError(f"--sampling_interval must be positive, got {args.sampling_interval}")

    test_ids = load_split_ids(split_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = torch.float16
    transform = build_transform()

    model, seq_len = load_model(args, device)
    actual_attn_dist_alpha = getattr(model, "attn_dist_alpha", getattr(args, "attn_dist_alpha", 0.0))
    actual_attn_dist_mode = getattr(model, "attn_dist_mode", getattr(args, "attn_dist_mode", "none"))
    actual_m2m_supervision_mode = getattr(
        model,
        "m2m_supervision_mode",
        getattr(args, "m2m_supervision_mode", "all_tokens"),
    )
    print("[TEST CONFIG]")
    print(f"model_type={args.model_type}")
    print(f"seq_len={seq_len}")
    print(f"m2m_inference_mode={args.m2m_inference_mode}")
    print(f"checkpoint={checkpoint_path}")
    print(f"soft_label_type={args.soft_label_type}")
    print(f"tau={tau}")
    print(f"sampling_interval={args.sampling_interval}")
    print(f"test_patients={len(test_ids)}")
    print(f"device={device}")
    prediction_rule = get_prediction_rule(args.soft_label_type)
    print(f"[PREDICTION RULE] {args.soft_label_type} -> {prediction_rule}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mode_name = args.m2m_inference_mode if args.model_type == "context_many_to_many_transformer" else "default"
    curve_dir = output_dir / "probability_curves" / mode_name
    plot_dir = output_dir / "plots" / mode_name

    rows = []
    printed_first_patient = False
    for pid in tqdm(test_ids, desc="Volume inference"):
        records, gt_mid_slice = get_axial_patient_records(img_root, pid)
        if len(records) == 0:
            rows.append({
                "patient_id": pid,
                "gt_mid_ax": "",
                "pred_mid_ax": "",
                "error_signed_raw": "",
                "error_raw": "",
                "error_sampled": "",
                "prob_at_pred": "",
                "prob_at_gt": "",
                "prediction_rule": prediction_rule,
                "m2m_inference_mode": mode_name,
                "model_type": args.model_type,
                "seq_len": seq_len,
                "soft_label_type": args.soft_label_type,
                "tau": tau,
                "checkpoint": str(checkpoint_path),
                "top1_ax": "",
                "top1_prob": "",
                "top2_ax": "",
                "top2_prob": "",
                "top1_top2_margin": "",
                "slice_spacing_mm": "",
                "error_abs_mm": "",
            })
            continue

        probs, pred_idx, debug = predict_volume_prob_curve(
            model=model,
            records=records,
            model_type=args.model_type,
            seq_len=seq_len,
            transform=transform,
            device=device,
            amp_dtype=amp_dtype,
            batch_size=args.batch_size,
            padding_mode=args.padding_mode,
            m2m_inference_mode=args.m2m_inference_mode,
            soft_label_type=args.soft_label_type,
            smooth_prob_curve=args.smooth_prob_curve,
            smooth_sigma=args.smooth_sigma,
        )

        slnos = [int(rec["slno"]) for rec in records]
        pred_mid_slice = int(slnos[pred_idx])
        pred_probability = float(probs[pred_idx])
        gt_idx = None
        if gt_mid_slice is not None:
            for idx, ax_number in enumerate(slnos):
                if int(ax_number) == int(gt_mid_slice):
                    gt_idx = idx
                    break
        prob_at_gt = "" if gt_idx is None else float(probs[gt_idx])

        if gt_mid_slice is None:
            error_signed_raw = ""
            error_raw = ""
            error_sampled = ""
        else:
            error_signed_raw = int(pred_mid_slice - int(gt_mid_slice))
            error_raw = int(abs(error_signed_raw))
            error_sampled = float(error_raw) / float(args.sampling_interval)

        spacing_mm = read_slice_spacing_mm(img_root, pid)
        error_abs_mm = ""
        if spacing_mm is not None and error_raw != "":
            error_abs_mm = float(error_raw) * spacing_mm

        peak_info = top_two_peaks(slnos, probs)

        row = {
            "patient_id": pid,
            "gt_mid_ax": "" if gt_mid_slice is None else int(gt_mid_slice),
            "pred_mid_ax": pred_mid_slice,
            "error_signed_raw": error_signed_raw,
            "error_raw": error_raw,
            "error_sampled": error_sampled,
            "prob_at_pred": pred_probability,
            "prob_at_gt": prob_at_gt,
            "prediction_rule": debug.get("prediction_rule", prediction_rule),
            "m2m_inference_mode": mode_name,
            "model_type": args.model_type,
            "seq_len": seq_len,
            "soft_label_type": args.soft_label_type,
            "tau": tau,
            "checkpoint": str(checkpoint_path),
            **peak_info,
            "slice_spacing_mm": "" if spacing_mm is None else spacing_mm,
            "error_abs_mm": error_abs_mm,
        }
        rows.append(row)

        if not printed_first_patient:
            print("[FIRST PATIENT SANITY]")
            print(f"patient_id={pid}")
            print(f"n_slices={len(slnos)}")
            print(f"first_5_ax_numbers={slnos[:5]}")
            print(f"last_5_ax_numbers={slnos[-5:]}")
            print(f"prob_curve_shape={tuple(probs.shape)}")
            print(f"pred_mid_ax={pred_mid_slice}")
            print(f"gt_mid_ax={gt_mid_slice}")
            print(f"error_raw={error_raw}")
            print(f"error_sampled={error_sampled}")
            if debug:
                print(f"prob_count_min={debug.get('aggregate_prob_count_min')}")
                print(f"prob_count_max={debug.get('aggregate_prob_count_max')}")
                print(f"prob_count_zero={debug.get('aggregate_prob_count_zero')}")
            printed_first_patient = True

        if args.save_probability_curves:
            write_probability_curve(
                curve_dir / f"{pid}_{mode_name}_probability_curve.csv",
                slnos=slnos,
                probs=probs,
                gt_mid_ax=gt_mid_slice,
                pred_mid_ax=pred_mid_slice,
            )
        if args.save_plots:
            save_curve_plot(
                plot_dir / f"{pid}_{mode_name}_prob_curve.png",
                patient_id=pid,
                slnos=slnos,
                probs=probs,
                gt_mid_ax=gt_mid_slice,
                pred_mid_ax=pred_mid_slice,
                error_raw=error_raw,
                error_sampled=error_sampled,
                inference_mode=mode_name,
            )

    fieldnames = [
        "patient_id",
        "gt_mid_ax",
        "pred_mid_ax",
        "error_signed_raw",
        "error_raw",
        "error_sampled",
        "prob_at_pred",
        "prob_at_gt",
        "prediction_rule",
        "m2m_inference_mode",
        "model_type",
        "seq_len",
        "soft_label_type",
        "tau",
        "checkpoint",
        "top1_ax",
        "top1_prob",
        "top2_ax",
        "top2_prob",
        "top1_top2_margin",
        "slice_spacing_mm",
        "error_abs_mm",
    ]
    write_csv(output_dir / "patient_volume_results.csv", rows, fieldnames)

    metrics = compute_summary_metrics(rows)
    metrics.update({
        "model_type": args.model_type,
        "seq_len": seq_len,
        "soft_label_type": args.soft_label_type,
        "tau": tau,
        "checkpoint": str(checkpoint_path),
        "prediction_rule": prediction_rule,
        "m2m_inference_mode": args.m2m_inference_mode if args.model_type == "context_many_to_many_transformer" else "",
        "m2m_supervision_mode": actual_m2m_supervision_mode if args.model_type == "context_many_to_many_transformer" else "not_applicable",
        "attn_dist_alpha": actual_attn_dist_alpha if args.model_type == "context_many_to_many_transformer" else 0.0,
        "attn_dist_mode": actual_attn_dist_mode if args.model_type == "context_many_to_many_transformer" else "none",
        "sampling_interval": args.sampling_interval,
        "smooth_prob_curve": bool(args.smooth_prob_curve),
        "smooth_sigma": args.smooth_sigma if args.smooth_prob_curve else "",
    })

    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_csv(
        output_dir / "summary_metrics.csv",
        [{"metric": key, "value": value} for key, value in metrics.items()],
        ["metric", "value"],
    )
    maybe_write_excel(output_dir / "patient_volume_results.xlsx", rows, metrics)

    print(f"[DONE] results: {output_dir / 'patient_volume_results.csv'}")
    print(f"[DONE] summary: {output_dir / 'summary_metrics.json'}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import json

import torch
from tqdm import tqdm

from functions import get_axial_patient_records
from test_ax import (
    DEFAULT_TAU,
    build_transform,
    compute_summary_metrics,
    get_checkpoint_path,
    load_model,
    maybe_write_excel,
    get_prediction_rule,
    predict_volume_prob_curve,
    read_slice_spacing_mm,
    save_curve_plot,
    top_two_peaks,
    write_csv,
    write_probability_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation-set volume-level axial mid-L3 evaluator. Test set is not used."
    )
    parser.add_argument("--project_root", type=str, default="/home/shjee/projects/ct_spine")
    parser.add_argument("--img_root", type=str, default=None)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_8", type=str, default=None)

    parser.add_argument("--model_type", type=str, required=True,
                        choices=["one_to_one", "context_many_to_one", "context_many_to_many_transformer"])
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--padding_mode", type=str, default="replicate", choices=["replicate"])
    parser.add_argument("--soft_label_type", type=str, default="laplace",
                        choices=["laplace", "gaussian", "sigmoid"])
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--fwhm_mm", type=float, default=None)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_transformer_layers", type=int, default=1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    parser.add_argument("--attn_dist_alpha", type=float, default=0.0)
    parser.add_argument("--attn_dist_mode", type=str, default="none", choices=["none", "gaussian", "laplace"])
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sampling_interval", type=float, default=6.0)
    parser.add_argument("--m2m_inference_mode", type=str, default="center", choices=["center", "aggregate"])
    parser.add_argument(
        "--m2m_supervision_mode",
        type=str,
        default="all_tokens",
        choices=["all_tokens", "center_token"],
        help=(
            "Training supervision metadata for context_many_to_many_transformer. "
            "This does not change validation inference."
        ),
    )
    parser.add_argument("--save_probability_curves", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    return parser.parse_args()


def load_val_ids(split_path):
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    if "val_ids" not in split:
        raise KeyError(f"split file must contain val_ids: {split_path}")
    return split["val_ids"]


def missing_row(pid, args, seq_len, checkpoint_path, mode_name, prediction_rule):
    return {
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
        "tau": args.tau,
        "fwhm_mm": "" if args.fwhm_mm is None else args.fwhm_mm,
        "checkpoint": str(checkpoint_path),
        "top1_ax": "",
        "top1_prob": "",
        "top2_ax": "",
        "top2_prob": "",
        "top1_top2_margin": "",
        "slice_spacing_mm": "",
        "error_abs_mm": "",
        "split_name": "val",
    }


def main():
    args = parse_args()
    if args.tau is None:
        args.tau = DEFAULT_TAU[args.soft_label_type]
    if args.sampling_interval <= 0:
        raise ValueError("--sampling_interval must be positive.")
    if args.model_type == "one_to_one":
        args.seq_len = 1
    elif args.seq_len % 2 == 0:
        raise ValueError("--seq_len must be odd.")

    project_root = Path(args.project_root)
    img_root = Path(args.img_root) if args.img_root else project_root / "axial_slices"
    split_path = Path(args.split_path) if args.split_path else img_root / "split_6_2_2_seed42.json"
    output_dir = Path(args.output_dir)
    checkpoint_path = get_checkpoint_path(args)

    val_ids = load_val_ids(split_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, seq_len = load_model(args, device)
    actual_attn_dist_alpha = getattr(model, "attn_dist_alpha", getattr(args, "attn_dist_alpha", 0.0))
    actual_attn_dist_mode = getattr(model, "attn_dist_mode", getattr(args, "attn_dist_mode", "none"))
    actual_m2m_supervision_mode = getattr(
        model,
        "m2m_supervision_mode",
        getattr(args, "m2m_supervision_mode", "all_tokens"),
    )
    transform = build_transform()
    amp_dtype = torch.float16
    mode_name = args.m2m_inference_mode if args.model_type == "context_many_to_many_transformer" else "default"

    print("[VAL CONFIG]")
    print(f"split_name=val")
    print(f"model_type={args.model_type}")
    print(f"seq_len={seq_len}")
    print(f"m2m_inference_mode={mode_name}")
    print(f"checkpoint={checkpoint_path}")
    print(f"soft_label_type={args.soft_label_type}")
    print(f"tau={args.tau}")
    print(f"fwhm_mm={args.fwhm_mm}")
    print(f"sampling_interval={args.sampling_interval}")
    print(f"val_patients={len(val_ids)}")
    print("[INFO] Test set is reserved for final reporting only and is not loaded by this evaluator.")
    prediction_rule = get_prediction_rule(args.soft_label_type)
    print(f"[PREDICTION RULE] {args.soft_label_type} -> {prediction_rule}")

    output_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = output_dir / "probability_curves" / mode_name
    plot_dir = output_dir / "plots" / mode_name

    rows = []
    printed_first_patient = False
    for pid in tqdm(val_ids, desc="Validation volume inference"):
        records, gt_mid_ax = get_axial_patient_records(img_root, pid)
        if len(records) == 0:
            rows.append(missing_row(pid, args, seq_len, checkpoint_path, mode_name, prediction_rule))
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
        )
        slnos = [int(rec["slno"]) for rec in records]
        pred_mid_ax = int(slnos[pred_idx])
        prob_at_pred = float(probs[pred_idx])
        gt_idx = next((idx for idx, ax in enumerate(slnos) if gt_mid_ax is not None and int(ax) == int(gt_mid_ax)), None)
        prob_at_gt = "" if gt_idx is None else float(probs[gt_idx])

        if gt_mid_ax is None:
            error_signed_raw = ""
            error_raw = ""
            error_sampled = ""
        else:
            error_signed_raw = int(pred_mid_ax - int(gt_mid_ax))
            error_raw = int(abs(error_signed_raw))
            error_sampled = float(error_raw) / float(args.sampling_interval)

        spacing_mm = read_slice_spacing_mm(img_root, pid)
        error_abs_mm = ""
        if spacing_mm is not None and error_raw != "":
            error_abs_mm = float(error_raw) * spacing_mm

        row = {
            "patient_id": pid,
            "gt_mid_ax": "" if gt_mid_ax is None else int(gt_mid_ax),
            "pred_mid_ax": pred_mid_ax,
            "error_signed_raw": error_signed_raw,
            "error_raw": error_raw,
            "error_sampled": error_sampled,
            "prob_at_pred": prob_at_pred,
            "prob_at_gt": prob_at_gt,
            "prediction_rule": debug.get("prediction_rule", prediction_rule),
            "m2m_inference_mode": mode_name,
            "model_type": args.model_type,
            "seq_len": seq_len,
            "soft_label_type": args.soft_label_type,
            "tau": args.tau,
            "fwhm_mm": "" if args.fwhm_mm is None else args.fwhm_mm,
            "checkpoint": str(checkpoint_path),
            **top_two_peaks(slnos, probs),
            "slice_spacing_mm": "" if spacing_mm is None else spacing_mm,
            "error_abs_mm": error_abs_mm,
            "split_name": "val",
        }
        rows.append(row)

        if not printed_first_patient:
            print("[FIRST VAL PATIENT SANITY]")
            print(f"patient_id={pid}")
            print(f"n_slices={len(slnos)}")
            print(f"first_5_ax_numbers={slnos[:5]}")
            print(f"last_5_ax_numbers={slnos[-5:]}")
            print(f"prob_curve_shape={tuple(probs.shape)}")
            print(f"pred_mid_ax={pred_mid_ax}")
            print(f"gt_mid_ax={gt_mid_ax}")
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
                gt_mid_ax=gt_mid_ax,
                pred_mid_ax=pred_mid_ax,
            )
        if args.save_plots:
            save_curve_plot(
                plot_dir / f"{pid}_{mode_name}_prob_curve.png",
                patient_id=pid,
                slnos=slnos,
                probs=probs,
                gt_mid_ax=gt_mid_ax,
                pred_mid_ax=pred_mid_ax,
                error_raw=error_raw,
                error_sampled=error_sampled,
                inference_mode=mode_name,
            )

    fieldnames = [
        "patient_id", "gt_mid_ax", "pred_mid_ax", "error_signed_raw", "error_raw",
        "error_sampled", "prob_at_pred", "prob_at_gt", "prediction_rule", "m2m_inference_mode",
        "model_type", "seq_len", "soft_label_type", "tau", "fwhm_mm", "checkpoint",
        "top1_ax", "top1_prob", "top2_ax", "top2_prob", "top1_top2_margin",
        "slice_spacing_mm", "error_abs_mm", "split_name",
    ]
    write_csv(output_dir / "patient_volume_results.csv", rows, fieldnames)

    metrics = compute_summary_metrics(rows)
    metrics.update({
        "model_type": args.model_type,
        "seq_len": seq_len,
        "soft_label_type": args.soft_label_type,
        "tau": args.tau,
        "fwhm_mm": args.fwhm_mm,
        "checkpoint": str(checkpoint_path),
        "split_name": "val",
        "prediction_rule": prediction_rule,
        "m2m_inference_mode": mode_name,
        "m2m_supervision_mode": actual_m2m_supervision_mode if args.model_type == "context_many_to_many_transformer" else "not_applicable",
        "attn_dist_alpha": actual_attn_dist_alpha if args.model_type == "context_many_to_many_transformer" else 0.0,
        "attn_dist_mode": actual_attn_dist_mode if args.model_type == "context_many_to_many_transformer" else "none",
        "sampling_interval": args.sampling_interval,
    })
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_csv(
        output_dir / "summary_metrics.csv",
        [{"metric": key, "value": value} for key, value in metrics.items()],
        ["metric", "value"],
    )
    maybe_write_excel(output_dir / "patient_volume_results.xlsx", rows, metrics)

    print(f"[DONE] validation results: {output_dir / 'patient_volume_results.csv'}")
    print(f"[DONE] validation summary: {output_dir / 'summary_metrics.json'}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

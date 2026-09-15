#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shjee/projects/ct_spine}"
IMG_ROOT="${IMG_ROOT:-${PROJECT_ROOT}/axial_slices}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/laplace_val_sweep}"
VAL_ROOT="${VAL_ROOT:-${PROJECT_ROOT}/val_results}"
LOG_DIR="${LOG_DIR:-logs/laplace_val_sweep}"
M2M_SUPERVISION_MODE="${M2M_SUPERVISION_MODE:-all_tokens}"

if [[ "${M2M_SUPERVISION_MODE}" != "all_tokens" && "${M2M_SUPERVISION_MODE}" != "center_token" ]]; then
  echo "[ERROR] M2M_SUPERVISION_MODE must be all_tokens or center_token, got: ${M2M_SUPERVISION_MODE}" >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}" "${VAL_ROOT}" "${LOG_DIR}"

FWHM_VALUES=(15 20 25 30 35 40 45 50 55 60 65 70 75)
M2M_SEQ_LENS=(7 9 11)

tau_from_fwhm() {
  local fwhm="$1"
  "${PYTHON_BIN}" -c "import math; print(f'{float(${fwhm})/(2*math.log(2)):.6f}')"
}

tau_tag_from_fwhm() {
  local fwhm="$1"
  "${PYTHON_BIN}" -c "import math; print(f'{float(${fwhm})/(2*math.log(2)):.3f}')"
}

checkpoint_name() {
  local model_type="$1"
  local fwhm="$2"
  local tau="$3"
  local seq_len="$4"
  local supervision_mode="$5"
  "${PYTHON_BIN}" -c "model='${model_type}'; f=float('${fwhm}'); tau=float('${tau}'); t=int('${seq_len}'); sup='${supervision_mode}'; suffix='' if model != 'context_many_to_many_transformer' else ('_supcenter' if sup == 'center_token' else '_supall'); print(f'best_{model}_soft_laplace_fwhm{f:g}_tau{tau:.6g}_T{t}_alpha0{suffix}.pth')"
}

run_one_config() {
  local model_type="$1"
  local seq_len="$2"
  local model_tag="$3"
  local fwhm="$4"

  local tau
  local tau_tag
  tau="$(tau_from_fwhm "${fwhm}")"
  tau_tag="$(tau_tag_from_fwhm "${fwhm}")"

  local supervision_tag=""
  if [[ "${model_type}" == "context_many_to_many_transformer" ]]; then
    if [[ "${M2M_SUPERVISION_MODE}" == "center_token" ]]; then
      supervision_tag="_supcenter"
    else
      supervision_tag="_supall"
    fi
  fi

  local out_dir="${VAL_ROOT}/laplace_fwhm${fwhm}_b${tau_tag}_T${seq_len}_${model_tag}${supervision_tag}"
  local summary_json="${out_dir}/summary_metrics.json"
  local ckpt="${CHECKPOINT_DIR}/$(checkpoint_name "${model_type}" "${fwhm}" "${tau}" "${seq_len}" "${M2M_SUPERVISION_MODE}")"
  local train_log="${LOG_DIR}/train_laplace_fwhm${fwhm}_b${tau_tag}_T${seq_len}_${model_tag}${supervision_tag}.log"
  local val_log="${LOG_DIR}/val_laplace_fwhm${fwhm}_b${tau_tag}_T${seq_len}_${model_tag}${supervision_tag}.log"

  if [[ -f "${summary_json}" ]]; then
    echo "[SKIP] summary exists: ${summary_json}"
    return 0
  fi

  echo "[CONFIG] model=${model_type} T=${seq_len} fwhm=${fwhm} tau=${tau} supervision=${M2M_SUPERVISION_MODE} output=${out_dir}"

  if [[ -f "${ckpt}" ]]; then
    echo "[SKIP TRAIN] checkpoint exists: ${ckpt}"
  else
    echo "[TRAIN] log=${train_log}"
    "${PYTHON_BIN}" train_ax_08_val_only.py \
      --project_root "${PROJECT_ROOT}" \
      --img_root "${IMG_ROOT}" \
      --checkpoint_dir "${CHECKPOINT_DIR}" \
      --model_type "${model_type}" \
      --target_mode soft \
      --soft_label_type laplace \
      --tau "${tau}" \
      --fwhm_mm "${fwhm}" \
      --seq_len "${seq_len}" \
      --padding_mode replicate \
      --soft_weight_alpha 0 \
      --d_model 256 \
      --num_transformer_layers 1 \
      --nhead 4 \
      --dim_feedforward 512 \
      --transformer_dropout 0.1 \
      --m2m_supervision_mode "${M2M_SUPERVISION_MODE}" \
      --batch_size 32 \
      --num_workers 4 \
      --epochs 30 \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      > "${train_log}" 2>&1
  fi

  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] Expected checkpoint not found after training: ${ckpt}" >&2
    exit 1
  fi

  echo "[VAL] log=${val_log}"
  "${PYTHON_BIN}" val_ax_08.py \
    --project_root "${PROJECT_ROOT}" \
    --img_root "${IMG_ROOT}" \
    --output_dir "${out_dir}" \
    --checkpoint_8 "${ckpt}" \
    --model_type "${model_type}" \
    --soft_label_type laplace \
    --tau "${tau}" \
    --fwhm_mm "${fwhm}" \
    --seq_len "${seq_len}" \
    --d_model 256 \
    --num_transformer_layers 1 \
    --nhead 4 \
    --dim_feedforward 512 \
    --transformer_dropout 0.1 \
    --batch_size 64 \
    --sampling_interval 6 \
    --m2m_inference_mode center \
    --m2m_supervision_mode "${M2M_SUPERVISION_MODE}" \
    > "${val_log}" 2>&1

  echo "[DONE] ${summary_json}"
}

for fwhm in "${FWHM_VALUES[@]}"; do
  run_one_config "one_to_one" "1" "one_to_one" "${fwhm}"
done

for fwhm in "${FWHM_VALUES[@]}"; do
  for seq_len in "${M2M_SEQ_LENS[@]}"; do
    run_one_config "context_many_to_many_transformer" "${seq_len}" "context_m2m" "${fwhm}"
  done
done

echo "[ALL DONE] Validation sweep complete. Test set was not evaluated."

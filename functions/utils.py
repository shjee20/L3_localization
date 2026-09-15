
'''requirements'''

from pathlib import Path 
import random 
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import random, os, numpy as np
from pathlib import Path 
import random 
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
try:
    import mlflow
    import mlflow.pytorch
except ModuleNotFoundError:
    mlflow = None
import torch.nn as nn
import torch.optim as optim
import json
import csv
import math as m


'''Dataset design'''

class SagittalCTDataset(Dataset):
    def __init__(self, img_root, lbl_root, patient_ids, transform=None):

        self.img_root = Path(img_root)
        self.lbl_root = Path(lbl_root)
        self.transform = transform
        self.samples = []   # (img_path, txt_path, patient_id, slice_idx)

        for pid in patient_ids:
            p_img_dir = self.img_root / pid
            p_lbl_dir = self.lbl_root / pid


            for img_path in sorted(p_img_dir.glob("sag*.png")):
                txt_path = p_lbl_dir / f"{img_path.stem}.txt"

                if not txt_path.exists():
                    print(f'{img_path.stem} is not exist')

                    continue

                slno = int(img_path.stem.replace("sag", ""))
                self.samples.append((img_path, txt_path, pid, slno))
                

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path, pid, slno = self.samples[idx]

        '''image'''

        img = Image.open(img_path).convert("L")  # 1채널 grayscale
        if self.transform is not None:
            img = self.transform(img)            # (1,H,W)
        else:
            arr = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)

        '''label 21, 22, 23'''

        y = np.loadtxt(txt_path, dtype=np.float32)    # shape (3,)
        if y.ndim == 0:
            y = np.array([y], dtype=np.float32)
        y = torch.from_numpy(y)                       # (3,)

        meta = {
            "patient_id": pid,
            "slice_idx": slno,
            "img_path": str(img_path)
        }

        return img, y, meta

def collate_ct(batch):
    xs, ys, metas = [], [], []
    for x, y, meta in batch:
        xs.append(x)
        ys.append(y)
        metas.append(meta)
    xs = torch.stack(xs, dim=0)  # (B,1,H,W)
    ys = torch.stack(ys, dim=0)  # (B,3)
    return xs, ys, metas


def set_seed(seed: int = 42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch_ct(model, loader, optimizer, criterion, device, amp_dtype, scaler=None):
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, ys, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)   # (B,1,H,W)
        ys   = ys.to(device, non_blocking=True)     # (B,3)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype,
                            enabled=torch.cuda.is_available()):
            preds = model(imgs)                     # (B,3)
            loss = criterion(preds, ys)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=total_loss / max(1, n))

    return total_loss / max(1, n)


def evaluate_ct(model, loader, criterion, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    sse = 0.0   # sum of squared error
    sae = 0.0   # sum of absolute error
    n_samples = 0

    with torch.no_grad():
        for imgs, ys, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            ys   = ys.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=amp_dtype,
                                enabled=torch.cuda.is_available()):
                preds = model(imgs)     # (B,3)
                loss  = criterion(preds, ys)

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            err = preds - ys
            sse += (err * err).sum().item()
            sae += err.abs().sum().item()

    avg_loss = total_loss / max(1, n_samples)
    mse  = sse / max(1, n_samples * 3)   # 3개 좌표 평균
    rmse = mse ** 0.5
    mae  = sae / max(1, n_samples * 3)

    return avg_loss, mse, rmse, mae


def mid_L3_laplace(z, z_mid, tau=7.21):
    '''
    Laplace (Exponential, L1) soft label for mid-L3 localization.

    Definition:
        w(z) = exp(-|z - z_mid| / tau)

    Parameter setting:
        tau is defined in SLICE units.

        tau is chosen such that:
            w(z_mid ± R) = 0.5

        where:
            R = STEP - 1 = 5 slices  (±5 slices around mid-L3, total 11 slices)

        Derivation:
            exp(-R / tau) = 0.5
            => tau = R / ln(2) ≈ 5 / 0.693 ≈ 7.21 slices

    Interpretation:
        - Symmetric distance-based soft label
        - Encodes localization uncertainty around mid-L3
        - Suitable for axial mid-L3 localization where slice-level ambiguity exists
    '''
    return round(m.exp(-abs(z - z_mid) / tau), 8)


def mid_L3_gaussian(z, z_mid, sigma=4.25):
    '''
    Gaussian (L2) soft label for mid-L3 localization.

    Definition:
        w(z) = exp(-(z - z_mid)^2 / (2 * sigma^2))

    Parameter setting:
        sigma is defined in SLICE units.

        sigma is chosen such that:
            w(z_mid ± R) = 0.5

        where:
            R = STEP - 1 = 5 slices

        Derivation:
            exp(-R^2 / (2 * sigma^2)) = 0.5
            => sigma = R / sqrt(2 * ln(2)) ≈ 5 / 1.177 ≈ 4.25 slices

    Interpretation:
        - Symmetric distance-based soft label
        - Stronger emphasis on the exact mid-L3 slice
        - Commonly used in landmark localization literature
    '''
    d = z - z_mid
    return round(m.exp(-(d * d) / (2 * sigma * sigma)), 8)


def mid_L3_sigmoid(z, z_mid, tau=7.21):
    '''
    Sigmoid (logistic) directional soft label for mid-L3 localization.

    Definition:
        w(z) = 1 / (1 + exp(-(z - z_mid) / tau))

    Parameter setting:
        tau is defined in SLICE units.

        tau controls the slope of the transition around mid-L3.
        The mid-L3 slice is defined as the reference point where:
            w(z_mid) = 0.5

        tau is chosen to match the scale of the surrounding context
        (e.g., ±R = ±(STEP - 1) slices) for fair comparison with
        distance-based soft labels.

        Here:
            R = STEP - 1 = 5 slices
            tau ≈ R / ln(2) ≈ 7.21 slices

    '''
    return round(1.0 / (1.0 + m.exp(-(z - z_mid) / tau)), 8)


def is_patient_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("pat")

def read_labels_json(label_path: Path) -> dict:
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_patient(patient_dir: Path, strict: bool = True) -> dict | None:
    """
    환자 폴더 구조/파일 일치 검증.
    strict=True면 한 개라도 문제 있으면 None 반환(스킵).
    """
    img_dir = patient_dir / "img"
    label_path = patient_dir / "labels.json"

    problems = []

    if not img_dir.exists():
        problems.append("missing img/ directory")
    if not label_path.exists():
        problems.append("missing labels.json")

    if problems:
        if strict:
            print(f"[SKIP] {patient_dir.name}: {', '.join(problems)}")
            return None
        else:
            print(f"[WARN] {patient_dir.name}: {', '.join(problems)}")

    # labels.json 읽기 + png_name 실제 파일 존재 확인
    labels = read_labels_json(label_path)
    png_names = labels.get("png_name", [])
    if not isinstance(png_names, list) or len(png_names) == 0:
        problems.append("labels.json has empty/invalid png_name list")

    missing_png = []
    if img_dir.exists() and isinstance(png_names, list):
        for nm in png_names:
            if not (img_dir / nm).exists():
                missing_png.append(nm)

    if missing_png:
        problems.append(f"{len(missing_png)} png_name entries not found in img/")

    if problems and strict:
        print(f"[SKIP] {patient_dir.name}: {', '.join(problems)}")
        # 원하면 missing list도 출력
        # for nm in missing_png[:5]: print("  missing:", nm)
        return None

    return {
        "patient_id": patient_dir.name,      # ex) pat0004
        "img_dir": str(img_dir),
        "label_path": str(label_path),
        "n_images": len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0,
        "n_listed": len(png_names) if isinstance(png_names, list) else 0,
        "n_missing_listed": len(missing_png),
    }


def split_patients(patient_ids: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    """
    네가 쓰던 방식 그대로: 전체를 섞고 6:1:1(=6/8, 1/8, 나머지)로 split
    """
    ids = patient_ids[:]
    random.seed(seed)
    random.shuffle(ids)

    n_total = len(ids)
    unit = n_total // 10
    n_train = 6 * unit
    n_val = 2 * unit
    n_test = n_total - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids

import re

_AX_RE = re.compile(r"_ax(\d{4})_")

def parse_axial_png_name(png_name: str) -> dict:
    """
    예) pat0001_ax0360_L3_mid.png
        pat0001_ax0000_NL3.png

    return:
      - slno: int
      - is_mid: bool
      - hard_lbl: int (L3면 1, 아니면 0)
    """
    m = _AX_RE.search(png_name)
    if m is None:
        raise ValueError(f"Cannot parse slno from filename: {png_name}")
    slno = int(m.group(1))

    is_mid = ("_L3_mid" in png_name)

    # hard label: "_NL3"면 0, 그 외 "_L3" 포함이면 1
    # (주의: "_L3_mid"도 "_L3"에 포함되므로 1)
    if "_NL3" in png_name:
        hard_lbl = 0
    elif "_L3" in png_name:
        hard_lbl = 1
    else:
        hard_lbl = 0  # 예외 케이스 대비

    return {"slno": slno, "is_mid": is_mid, "hard_lbl": hard_lbl}



def compute_soft_label(
    slno: int,
    slno_mid: int | None,
    soft_type: str = "laplace",  # "laplace" | "gaussian" | "sigmoid"
    tau: float = 7.21,
) -> float:
    if slno_mid is None:
        return 0.0
    if soft_type == "laplace":
        return float(mid_L3_laplace(slno, slno_mid, tau=tau))
    elif soft_type == "gaussian":
        return float(mid_L3_gaussian(slno, slno_mid, sigma=tau))
    elif soft_type == "sigmoid":
        return float(mid_L3_sigmoid(slno, slno_mid, tau=tau))
    else:
        raise ValueError(f"Unknown soft_type: {soft_type}")


class AxialMidL3Dataset(Dataset):
    """
    target_mode:
      - "hard": y = hard label (0/1)           shape (1,)
      - "soft": y = soft label (0~1)           shape (1,)
    """
    def __init__(
        self,
        img_root: Path,
        patient_ids: list[str],
        transform=None,
        target_mode: str = "soft",        
        soft_type: str = "laplace",
        tau: float = 7.21,
        use_only_listed: bool = True,
    ):
        assert target_mode in ["hard", "soft"], "target_mode must be 'hard' or 'soft'"

        self.img_root = Path(img_root)
        self.patient_ids = patient_ids
        self.transform = transform

        self.target_mode = target_mode
        self.soft_type = soft_type
        self.tau = tau
        self.use_only_listed = use_only_listed

        self.samples = []  

        for pid in self.patient_ids:
            pat_dir = self.img_root / pid
            img_dir = pat_dir / "img"
            label_path = pat_dir / "labels.json"
            if not img_dir.exists() or not label_path.exists():
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                label_json = json.load(f)

            png_names = label_json.get("png_name", [])
            if not isinstance(png_names, list) or len(png_names) == 0:
                continue

            slno_mid = None
            for nm in png_names:
                if "_L3_mid" in nm:
                    slno_mid = parse_axial_png_name(nm)["slno"]
                    break

            iter_names = png_names if use_only_listed else [p.name for p in img_dir.glob("*.png")]

            for nm in iter_names:
                img_path = img_dir / nm
                if not img_path.exists():
                    continue
                info = parse_axial_png_name(nm)
                self.samples.append((img_path, info["slno"], slno_mid, info["hard_lbl"], pid))

        if len(self.samples) == 0:
            raise RuntimeError("No samples collected. Check img_root / labels.json / filenames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, slno, slno_mid, hard, pid = self.samples[idx]

        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)  
        else:
            arr = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).unsqueeze(0)

        if self.target_mode == "hard":
            y = torch.tensor([hard], dtype=torch.float32)  # (1,)
        else:
            soft = compute_soft_label(
                slno=slno,
                slno_mid=slno_mid,
                soft_type=self.soft_type,
                tau=self.tau,
            )
            y = torch.tensor([soft], dtype=torch.float32)  # (1,)

        meta = {"patient_id": pid, "png_name": img_path.name, "slno": slno, "slno_mid": slno_mid}
        return img, y, meta


def _validate_seq_len(seq_len: int) -> None:
    if seq_len < 1 or seq_len % 2 == 0:
        raise ValueError(f"seq_len must be a positive odd integer, got {seq_len}")


def get_axial_patient_records(
    img_root: Path,
    patient_id: str,
    use_only_listed: bool = True,
) -> tuple[list[dict], int | None]:
    pat_dir = Path(img_root) / patient_id
    img_dir = pat_dir / "img"
    label_path = pat_dir / "labels.json"
    if not img_dir.exists() or not label_path.exists():
        return [], None

    with open(label_path, "r", encoding="utf-8") as f:
        label_json = json.load(f)

    png_names = label_json.get("png_name", [])
    if not isinstance(png_names, list) or len(png_names) == 0:
        return [], None

    slno_mid = None
    for nm in png_names:
        if "_L3_mid" in nm:
            slno_mid = parse_axial_png_name(nm)["slno"]
            break

    iter_names = png_names if use_only_listed else [p.name for p in img_dir.glob("*.png")]
    records = []
    for nm in iter_names:
        img_path = img_dir / nm
        if not img_path.exists():
            continue
        info = parse_axial_png_name(nm)
        records.append({
            "img_path": img_path,
            "png_name": nm,
            "slno": info["slno"],
            "hard_lbl": info["hard_lbl"],
        })

    records = sorted(records, key=lambda r: r["slno"])
    return records, slno_mid


def load_axial_image_tensor(img_path: Path, transform=None) -> torch.Tensor:
    img = Image.open(img_path).convert("L")
    if transform is not None:
        return transform(img)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def make_context_window(
    records: list[dict],
    center_idx: int,
    seq_len: int,
    transform=None,
    padding_mode: str = "replicate",
) -> tuple[torch.Tensor, list[int], list[str]]:
    _validate_seq_len(seq_len)
    if padding_mode != "replicate":
        raise ValueError(f"Only replicate padding is supported, got {padding_mode}")
    if len(records) == 0:
        raise ValueError("Cannot build a context window from an empty record list")

    k = seq_len // 2
    xs, slnos, names = [], [], []
    last_idx = len(records) - 1
    for offset in range(-k, k + 1):
        src_idx = min(max(center_idx + offset, 0), last_idx)
        rec = records[src_idx]
        xs.append(load_axial_image_tensor(rec["img_path"], transform=transform))
        slnos.append(rec["slno"])
        names.append(rec["png_name"])

    return torch.stack(xs, dim=0), slnos, names


def make_context_target_sequence(
    records: list[dict],
    center_idx: int,
    seq_len: int,
    slno_mid: int | None,
    target_mode: str = "soft",
    soft_type: str = "laplace",
    tau: float = 7.21,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    _validate_seq_len(seq_len)
    if padding_mode != "replicate":
        raise ValueError(f"Only replicate padding is supported, got {padding_mode}")
    if target_mode not in ["hard", "soft"]:
        raise ValueError(f"target_mode must be 'hard' or 'soft', got {target_mode}")
    if len(records) == 0:
        raise ValueError("Cannot build a target sequence from an empty record list")

    k = seq_len // 2
    last_idx = len(records) - 1
    targets = []
    for offset in range(-k, k + 1):
        src_idx = min(max(center_idx + offset, 0), last_idx)
        rec = records[src_idx]
        if target_mode == "hard":
            targets.append(float(rec["hard_lbl"]))
        else:
            targets.append(compute_soft_label(
                slno=rec["slno"],
                slno_mid=slno_mid,
                soft_type=soft_type,
                tau=tau,
            ))
    return torch.tensor(targets, dtype=torch.float32)


class AxialContextWindowDataset(Dataset):
    """
    Context-aware many-to-one dataset.

    x: (T, 1, H, W), centered at slice z with replicate padding at boundaries.
    y: (1,), the hard or soft label for the center slice only.
    """

    def __init__(
        self,
        img_root: Path,
        patient_ids: list[str],
        transform=None,
        target_mode: str = "soft",
        soft_type: str = "laplace",
        tau: float = 7.21,
        seq_len: int = 5,
        padding_mode: str = "replicate",
        use_only_listed: bool = True,
    ):
        assert target_mode in ["hard", "soft"], "target_mode must be 'hard' or 'soft'"
        _validate_seq_len(seq_len)
        if padding_mode != "replicate":
            raise ValueError(f"Only replicate padding is supported, got {padding_mode}")

        self.img_root = Path(img_root)
        self.patient_ids = patient_ids
        self.transform = transform
        self.target_mode = target_mode
        self.soft_type = soft_type
        self.tau = tau
        self.seq_len = seq_len
        self.padding_mode = padding_mode
        self.use_only_listed = use_only_listed

        self.patient_records = {}
        self.samples = []

        for pid in self.patient_ids:
            records, slno_mid = get_axial_patient_records(
                self.img_root,
                pid,
                use_only_listed=use_only_listed,
            )
            if len(records) == 0:
                continue
            self.patient_records[pid] = {
                "records": records,
                "slno_mid": slno_mid,
            }
            for center_idx in range(len(records)):
                self.samples.append((pid, center_idx))

        if len(self.samples) == 0:
            raise RuntimeError("No context-window samples collected. Check img_root / labels.json / filenames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, center_idx = self.samples[idx]
        pdata = self.patient_records[pid]
        records = pdata["records"]
        slno_mid = pdata["slno_mid"]
        center_rec = records[center_idx]

        x, window_slnos, window_png_names = make_context_window(
            records=records,
            center_idx=center_idx,
            seq_len=self.seq_len,
            transform=self.transform,
            padding_mode=self.padding_mode,
        )

        if self.target_mode == "hard":
            y_value = center_rec["hard_lbl"]
        else:
            y_value = compute_soft_label(
                slno=center_rec["slno"],
                slno_mid=slno_mid,
                soft_type=self.soft_type,
                tau=self.tau,
            )

        y = torch.tensor([y_value], dtype=torch.float32)
        meta = {
            "patient_id": pid,
            "png_name": center_rec["png_name"],
            "slno": center_rec["slno"],
            "slno_mid": slno_mid,
            "seq_len": self.seq_len,
            "window_slnos": window_slnos,
            "window_png_names": window_png_names,
        }
        return x, y, meta


class AxialContextSequenceDataset(Dataset):
    """
    Position-aware many-to-many dataset.

    x: (T, 1, H, W), centered at slice z with replicate padding at boundaries.
    y: (T,), hard or soft label sequence for the same padded local window.
    """

    def __init__(
        self,
        img_root: Path,
        patient_ids: list[str],
        transform=None,
        target_mode: str = "soft",
        soft_type: str = "laplace",
        tau: float = 7.21,
        seq_len: int = 5,
        padding_mode: str = "replicate",
        use_only_listed: bool = True,
    ):
        assert target_mode in ["hard", "soft"], "target_mode must be 'hard' or 'soft'"
        _validate_seq_len(seq_len)
        if padding_mode != "replicate":
            raise ValueError(f"Only replicate padding is supported, got {padding_mode}")

        self.img_root = Path(img_root)
        self.patient_ids = patient_ids
        self.transform = transform
        self.target_mode = target_mode
        self.soft_type = soft_type
        self.tau = tau
        self.seq_len = seq_len
        self.padding_mode = padding_mode
        self.use_only_listed = use_only_listed

        self.patient_records = {}
        self.samples = []

        for pid in self.patient_ids:
            records, slno_mid = get_axial_patient_records(
                self.img_root,
                pid,
                use_only_listed=use_only_listed,
            )
            if len(records) == 0:
                continue
            self.patient_records[pid] = {
                "records": records,
                "slno_mid": slno_mid,
            }
            for center_idx in range(len(records)):
                self.samples.append((pid, center_idx))

        if len(self.samples) == 0:
            raise RuntimeError("No context-sequence samples collected. Check img_root / labels.json / filenames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, center_idx = self.samples[idx]
        pdata = self.patient_records[pid]
        records = pdata["records"]
        slno_mid = pdata["slno_mid"]
        center_rec = records[center_idx]

        x, window_slnos, window_png_names = make_context_window(
            records=records,
            center_idx=center_idx,
            seq_len=self.seq_len,
            transform=self.transform,
            padding_mode=self.padding_mode,
        )
        y = make_context_target_sequence(
            records=records,
            center_idx=center_idx,
            seq_len=self.seq_len,
            slno_mid=slno_mid,
            target_mode=self.target_mode,
            soft_type=self.soft_type,
            tau=self.tau,
            padding_mode=self.padding_mode,
        )

        meta = {
            "patient_id": pid,
            "png_name": center_rec["png_name"],
            "slno": center_rec["slno"],
            "slno_mid": slno_mid,
            "seq_len": self.seq_len,
            "window_slnos": window_slnos,
            "window_png_names": window_png_names,
        }
        return x, y, meta


class TargetWeightedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss with optional target weighting:
        weight = 1 + alpha * target
    alpha=0 is equivalent to unweighted BCE.
    """

    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.base_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        loss = self.base_loss(logits, targets)
        if self.alpha == 0.0:
            return loss.mean()
        weights = 1.0 + self.alpha * targets
        return (loss * weights).mean()


@torch.no_grad()
def infer_patient_probability_curve(
    model,
    img_root: Path,
    patient_id: str,
    transform,
    device,
    amp_dtype,
    model_type: str = "context_many_to_one",
    seq_len: int = 5,
    padding_mode: str = "replicate",
    batch_size: int = 64,
):
    records, slno_mid = get_axial_patient_records(img_root, patient_id)
    if len(records) == 0:
        raise RuntimeError(f"No axial records found for patient_id={patient_id}")

    model.eval()
    probs = []
    slnos = [r["slno"] for r in records]

    for start in range(0, len(records), batch_size):
        batch_tensors = []
        end = min(start + batch_size, len(records))
        for center_idx in range(start, end):
            if model_type in ["context_many_to_one", "context_many_to_many_transformer"]:
                x, _, _ = make_context_window(
                    records=records,
                    center_idx=center_idx,
                    seq_len=seq_len,
                    transform=transform,
                    padding_mode=padding_mode,
                )
            elif model_type == "one_to_one":
                x = load_axial_image_tensor(records[center_idx]["img_path"], transform=transform)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            batch_tensors.append(x)

        xs = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=torch.cuda.is_available()):
            logits = model(xs)
            if model_type == "context_many_to_many_transformer":
                logits = logits[:, seq_len // 2]
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.extend(batch_probs.tolist())

    return {
        "patient_id": patient_id,
        "gt_mid_slice": slno_mid,
        "slnos": slnos,
        "probabilities": np.array(probs, dtype=np.float32),
    }


def _write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_safe_float(value):
    value = float(value)
    if np.isfinite(value):
        return value
    return None


@torch.no_grad()
def evaluate_volume_localization(
    model,
    img_root: Path,
    patient_ids: list[str],
    transform,
    device,
    amp_dtype,
    model_type: str = "context_many_to_one",
    seq_len: int = 5,
    padding_mode: str = "replicate",
    batch_size: int = 64,
    save_dir: Path | None = None,
    split_name: str = "test",
    save_probability_curves: bool = False,
):
    rows = []
    pred_slices = []
    gt_slices = []
    abs_errors = []

    curve_dir = None
    if save_dir is not None and save_probability_curves:
        curve_dir = Path(save_dir) / f"{split_name}_probability_curves"
        curve_dir.mkdir(parents=True, exist_ok=True)

    for pid in patient_ids:
        curve = infer_patient_probability_curve(
            model=model,
            img_root=img_root,
            patient_id=pid,
            transform=transform,
            device=device,
            amp_dtype=amp_dtype,
            model_type=model_type,
            seq_len=seq_len,
            padding_mode=padding_mode,
            batch_size=batch_size,
        )
        probs = curve["probabilities"]
        slnos = curve["slnos"]
        gt_mid = curve["gt_mid_slice"]
        pred_idx = int(np.argmax(probs))
        pred_mid = int(slnos[pred_idx])

        signed_error = None if gt_mid is None else pred_mid - int(gt_mid)
        abs_error = None if signed_error is None else abs(signed_error)

        curve_path = ""
        if curve_dir is not None:
            curve_path = str(curve_dir / f"{pid}_probability_curve.csv")
            curve_rows = [
                {"patient_id": pid, "slno": int(slno), "probability": float(prob)}
                for slno, prob in zip(slnos, probs.tolist())
            ]
            _write_csv_rows(
                Path(curve_path),
                curve_rows,
                ["patient_id", "slno", "probability"],
            )

        row = {
            "patient_id": pid,
            "gt_mid_slice": "" if gt_mid is None else int(gt_mid),
            "pred_mid_slice": pred_mid,
            "signed_error": "" if signed_error is None else int(signed_error),
            "abs_error": "" if abs_error is None else int(abs_error),
            "pred_probability_at_pred_z": float(probs[pred_idx]),
            "probability_curve_path": curve_path,
        }
        rows.append(row)

        if gt_mid is not None:
            pred_slices.append(pred_mid)
            gt_slices.append(int(gt_mid))
            abs_errors.append(int(abs_error))

    errors = np.array(abs_errors, dtype=np.float32)
    pred_arr = np.array(pred_slices, dtype=np.float32)
    gt_arr = np.array(gt_slices, dtype=np.float32)

    if len(errors) > 0:
        pearson = np.nan
        if len(errors) > 1 and np.std(pred_arr) > 0 and np.std(gt_arr) > 0:
            pearson = float(np.corrcoef(pred_arr, gt_arr)[0, 1])

        metrics = {
            "n_patients": int(len(errors)),
            "mean_abs_error": _json_safe_float(np.mean(errors)),
            "std_abs_error": _json_safe_float(np.std(errors)),
            "median_abs_error": _json_safe_float(np.median(errors)),
            "exact_match_rate": _json_safe_float(np.mean(errors == 0)),
            "le_1_slice_accuracy": _json_safe_float(np.mean(errors <= 1)),
            "le_2_slice_accuracy": _json_safe_float(np.mean(errors <= 2)),
            "outlier_rate_ge_3": _json_safe_float(np.mean(errors >= 3)),
            "outlier_rate_ge_5": _json_safe_float(np.mean(errors >= 5)),
            "pearson_pred_gt_slice": _json_safe_float(pearson),
        }
    else:
        metrics = {
            "n_patients": 0,
            "mean_abs_error": None,
            "std_abs_error": None,
            "median_abs_error": None,
            "exact_match_rate": None,
            "le_1_slice_accuracy": None,
            "le_2_slice_accuracy": None,
            "outlier_rate_ge_3": None,
            "outlier_rate_ge_5": None,
            "pearson_pred_gt_slice": None,
        }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        result_path = save_dir / f"{split_name}_volume_results.csv"
        _write_csv_rows(
            result_path,
            rows,
            [
                "patient_id",
                "gt_mid_slice",
                "pred_mid_slice",
                "signed_error",
                "abs_error",
                "pred_probability_at_pred_z",
                "probability_curve_path",
            ],
        )

        metrics_json_path = save_dir / f"{split_name}_volume_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        metric_rows = [{"metric": k, "value": v} for k, v in metrics.items()]
        _write_csv_rows(save_dir / f"{split_name}_volume_metrics.csv", metric_rows, ["metric", "value"])

    return metrics, rows
    

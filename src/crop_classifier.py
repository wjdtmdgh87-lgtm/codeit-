"""
src/crop_classifier.py
Stage 2: EfficientNet-B3 기반 pill crop 분류기 (torchvision 사용)

학습: python main.py --mode train_stage2
추론: load_stage2_model() + apply_stage2() — wbf_ensemble.py / model.py에서 호출
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

from config import MODELS_DIR, RESULTS_DIR, CROPS_DIR, STAGE2, STAGE1_BYPASS_THR, STAGE2_SKIP_CLASSES

STAGE2_WEIGHTS = MODELS_DIR / "stage2_best.pt"


class FocalLoss(nn.Module):
    """
    Focal Loss — 다수 클래스(쉬운 샘플)의 기여도를 낮추고 소수 클래스에 집중.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    class_weight와 label_smoothing을 함께 지원합니다.
    """
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma          = gamma
        self.weight         = weight         # class weights (Tensor or None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE with label smoothing + class weights (실제 손실값)
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        # p_t: 정답 클래스에 할당된 확률 (focal 가중치 계산용, smoothing 없이)
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(num_classes: int) -> nn.Module:
    """EfficientNet-B3 + num_classes 헤드"""
    from torchvision import models
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _get_transform(train: bool):
    from torchvision import transforms
    sz = STAGE2["imgsz"]
    if train:
        return transforms.Compose([
            transforms.Resize((sz + 40, sz + 40)),
            transforms.RandomCrop(sz),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _save_results(history: list, best_acc: float, save_dir: Path):
    """학습 결과를 results/stage2/ 에 저장합니다 (CSV + 학습 곡선 이미지)."""
    import csv
    import platform
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # OS별 한글 폰트 설정
    _korean_candidates = {
        "Windows": ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"],
        "Darwin":  ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
                    "/Library/Fonts/AppleGothic.ttf"],
        "Linux":   ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf"],
    }
    for path in _korean_candidates.get(platform.system(), []):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            break

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = save_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "phase", "loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)

    # ── 학습 곡선 ─────────────────────────────────────────────────────────
    epochs   = [r["epoch"]   for r in history]
    losses   = [r["loss"]    for r in history]
    val_accs = [r["val_acc"] for r in history]
    phases   = [r["phase"]   for r in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Phase 구분선
    phase2_start = next((r["epoch"] for r in history if r["phase"] == 2), None)
    for ax in (ax1, ax2):
        if phase2_start:
            ax.axvline(phase2_start - 0.5, color="gray", linestyle="--",
                       linewidth=1, label="Phase 2 시작")

    # Loss
    ax1.plot(epochs, losses, color="steelblue", linewidth=1.5, label="train loss")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Stage 2 학습 곡선  (best val_acc={best_acc:.4f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Val Acc
    best_epoch = history[val_accs.index(max(val_accs))]["epoch"]
    ax2.plot(epochs, val_accs, color="darkorange", linewidth=1.5, label="val acc")
    ax2.axhline(best_acc, color="red", linestyle=":", linewidth=1,
                label=f"best={best_acc:.4f} (ep{best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = save_dir / "results.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()

    print(f"  결과 저장: {csv_path.name}, {plot_path.name} → {save_dir}")


def train_stage2():
    """Stage 2 EfficientNet-B3 분류기 학습"""
    if not CROPS_DIR.exists():
        print("[Stage2] [오류] data/crops/ 없음. 먼저 --mode crop_data 를 실행하세요.")
        return

    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import torch.optim as optim
    import time
    from tqdm import tqdm

    device   = _get_device()
    save_dir = RESULTS_DIR / "stage2"
    save_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Stage2] 학습 시작 — device: {device}  결과 저장: {save_dir}")

    train_ds = ImageFolder(str(CROPS_DIR / "train"), transform=_get_transform(train=True))
    val_ds   = ImageFolder(str(CROPS_DIR / "val"),   transform=_get_transform(train=False))

    # ImageFolder는 폴더명을 알파벳순 정렬 ("10" < "2") → 실제 pill class idx로 역매핑
    idx_to_class = {v: int(k) for k, v in train_ds.class_to_idx.items()}
    num_classes  = len(idx_to_class)

    # ── 클래스 불균형 처리 ────────────────────────────────────────────────
    targets      = torch.tensor(train_ds.targets)
    class_counts = torch.zeros(num_classes)
    for t in targets:
        class_counts[t] += 1

    # 클래스별 역빈도 가중치 (희귀 클래스일수록 높은 가중치)
    class_weights  = 1.0 / class_counts.clamp(min=1)
    class_weights  = class_weights * num_classes / class_weights.sum()  # 정규화

    # WeightedRandomSampler: 배치마다 클래스 균등 샘플링
    sample_weights = class_weights[targets]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    cnt_min = int(class_counts.min())
    cnt_avg = float(class_counts.mean())
    cnt_max = int(class_counts.max())
    print(f"[Stage2] 클래스별 학습 샘플 수  min={cnt_min} / avg={cnt_avg:.1f} / max={cnt_max}")

    import os
    # main.py의 __main__ 가드 하위에서 호출되므로 Windows spawn 데드락 없음
    # Windows spawn 특성상 workers 수가 많을수록 초기 프로세스 생성 비용이 큼 → 4로 제한
    n_workers = min(4, os.cpu_count() or 2)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    cfg      = STAGE2
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch"],
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_dl   = DataLoader(
        val_ds,
        batch_size=cfg["batch"],
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Phase 1 epochs: classifier head만 학습 (backbone 동결)
    # Phase 2 epochs: 전체 unfreeze 후 낮은 lr로 fine-tuning
    PHASE1_EPOCHS = 10
    PHASE2_LR     = cfg["lr0"] * 0.1   # 1e-4
    PATIENCE      = 20                  # unfreeze 이후 진동을 고려해 여유 있게 설정

    print(f"[Stage2] 클래스: {num_classes}개  "
          f"학습: {len(train_ds)}장 ({len(train_dl)}배치)  "
          f"검증: {len(val_ds)}장 ({len(val_dl)}배치)")
    print(f"[Stage2] epochs={cfg['epochs']}  batch={cfg['batch']}  "
          f"phase1={PHASE1_EPOCHS}ep(head only, lr={cfg['lr0']})  "
          f"phase2={(cfg['epochs']-PHASE1_EPOCHS)}ep(full, lr={PHASE2_LR})\n")

    model     = _build_model(num_classes).to(device)
    criterion = FocalLoss(gamma=2.0, weight=class_weights.to(device), label_smoothing=0.1)
    use_amp   = torch.cuda.is_available()
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _fmt(s):
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _run_epoch(epoch, total_epochs, phase_label, verbose_cls=False):
        """학습 1 epoch 실행 후 (avg_loss, val_acc) 반환"""
        model.train()
        train_loss = 0.0
        train_bar  = tqdm(
            train_dl,
            desc=f"Epoch {epoch:3d}/{total_epochs} [{phase_label}|train]",
            leave=False, ncols=95,
        )
        for imgs, lbls in train_bar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        correct = total = 0
        cls_correct = torch.zeros(num_classes)
        cls_total   = torch.zeros(num_classes)
        val_bar = tqdm(val_dl,
                       desc=f"Epoch {epoch:3d}/{total_epochs} [{phase_label}| val ]",
                       leave=False, ncols=95)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for imgs, lbls in val_bar:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds    = model(imgs).argmax(1)
                correct += (preds == lbls).sum().item()
                total   += lbls.size(0)
                for c in range(num_classes):
                    mask = lbls == c
                    cls_correct[c] += (preds[mask] == c).sum().item()
                    cls_total[c]   += mask.sum().item()
                val_bar.set_postfix(acc=f"{correct/total:.4f}")

        if verbose_cls:
            # 정확도 낮은 하위 10개 클래스 출력
            cls_acc = cls_correct / cls_total.clamp(min=1)
            worst   = cls_acc.argsort()[:10]
            print("  [per-class 하위 10]")
            for w in worst:
                pill_cls = idx_to_class[int(w)]
                print(f"    cls {pill_cls:3d}: acc={cls_acc[w]:.2f} "
                      f"({int(cls_correct[w])}/{int(cls_total[w])})")

        return train_loss / len(train_dl), correct / total if total > 0 else 0.0

    best_acc    = 0.0
    no_improve  = 0
    epoch_times = []
    total_start = time.time()
    history     = []   # [{epoch, phase, loss, val_acc}]

    # ── Phase 1: backbone 동결, head만 학습 ──────────────────────────────
    PHASE1_PATIENCE = 5
    print("[ Phase 1 ] backbone 동결 — classifier head만 학습")
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr0"], weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        avg_loss, acc = _run_epoch(epoch, cfg["epochs"], "P1")
        scheduler.step()

        epoch_sec = time.time() - t0
        epoch_times.append(epoch_sec)
        avg_sec   = sum(epoch_times) / len(epoch_times)
        remaining = avg_sec * (cfg["epochs"] - epoch)
        elapsed   = time.time() - total_start

        history.append({"epoch": epoch, "phase": 1, "loss": avg_loss, "val_acc": acc})
        best_mark = " BEST" if acc > best_acc else ""
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  loss={avg_loss:.4f}  "
              f"val_acc={acc:.4f}{best_mark}  "
              f"epoch={_fmt(epoch_sec)}  elapsed={_fmt(elapsed)}  ETA={_fmt(remaining)}")

        if acc > best_acc:
            best_acc   = acc
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "idx_to_class": idx_to_class},
                       STAGE2_WEIGHTS)
            print(f"  → best 저장 (acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= PHASE1_PATIENCE:
                print(f"  Phase 1 Early stopping (patience={PHASE1_PATIENCE})")
                break

    # ── Phase 2: 후반 블록만 unfreeze (features[5:] + classifier) ────────
    # EfficientNet-B3 features: [0]=stem, [1~7]=MBConv blocks, [8]=head conv
    # 초반 블록(저수준 edge/texture)은 ImageNet 가중치 유지, 후반만 fine-tuning
    print(f"\n[ Phase 2 ] features[5:] + classifier unfreeze — lr={PHASE2_LR}")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[5:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  학습 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR, weight_decay=1e-2,
    )
    phase2_epochs = cfg["epochs"] - PHASE1_EPOCHS
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)
    no_improve    = 0   # patience 카운터 리셋

    for epoch in range(PHASE1_EPOCHS + 1, cfg["epochs"] + 1):
        t0 = time.time()
        is_last = (epoch == cfg["epochs"])
        avg_loss, acc = _run_epoch(epoch, cfg["epochs"], "P2", verbose_cls=is_last)
        scheduler.step()

        epoch_sec = time.time() - t0
        epoch_times.append(epoch_sec)
        avg_sec   = sum(epoch_times) / len(epoch_times)
        remaining = avg_sec * (cfg["epochs"] - epoch)
        elapsed   = time.time() - total_start

        history.append({"epoch": epoch, "phase": 2, "loss": avg_loss, "val_acc": acc})
        best_mark = " BEST" if acc > best_acc else ""
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  loss={avg_loss:.4f}  "
              f"val_acc={acc:.4f}{best_mark}  "
              f"epoch={_fmt(epoch_sec)}  elapsed={_fmt(elapsed)}  ETA={_fmt(remaining)}")

        if acc > best_acc:
            best_acc   = acc
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "idx_to_class": idx_to_class},
                       STAGE2_WEIGHTS)
            print(f"  → best 저장 (acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})")
                break

    print(f"\n[Stage2] 학습 완료. best val_acc={best_acc:.4f}")
    print(f"  저장 위치: {STAGE2_WEIGHTS}")

    _save_results(history, best_acc, save_dir)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_stage2_model(weights: str = None) -> Tuple[nn.Module, dict]:
    """
    Stage 2 모델 로드.
    Returns: (model, idx_to_class)
      idx_to_class: {ImageFolder 내부 idx → 실제 pill class idx}
    """
    w = Path(weights) if weights else STAGE2_WEIGHTS
    if not w.exists():
        raise FileNotFoundError(
            f"[Stage2] 가중치 없음: {w}\n"
            "  먼저 python main.py --mode train_stage2 를 실행하세요."
        )

    ckpt         = torch.load(str(w), map_location="cpu")
    idx_to_class = ckpt["idx_to_class"]
    model        = _build_model(len(idx_to_class))
    model.load_state_dict(ckpt["model_state"])

    device = _get_device()
    model  = model.to(device).eval()
    print(f"[Stage2] 모델 로드: {w.name}  device={device}")
    return model, idx_to_class


def _extract_crop(img: np.ndarray, box: list, pad: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    x1n, y1n, x2n, y2n = box
    bw = (x2n - x1n) * w
    bh = (y2n - y1n) * h
    x1 = max(0, int(x1n * w - bw * pad))
    y1 = max(0, int(y1n * h - bh * pad))
    x2 = min(w, int(x2n * w + bw * pad))
    y2 = min(h, int(y2n * h + bh * pad))
    return img[y1:y2, x1:x2]


def apply_stage2(
    img: np.ndarray,
    result: dict,
    stage2_model: nn.Module,
    idx_to_class: dict,
    bypass_thr: float = STAGE1_BYPASS_THR,
    img_name: str = "",
    class_names: dict = None,
) -> dict:
    """
    Stage 2 분류기를 적용해 저신뢰도 박스의 class/score를 보정합니다.

    - score >= bypass_thr : Stage 2 생략
    - score <  bypass_thr : crop 전체를 배치로 묶어 EfficientNet-B3 단일 forward
    """
    from PIL import Image

    boxes  = list(result["boxes"])
    scores = list(result["scores"])
    labels = list(result["labels"])

    device    = next(stage2_model.parameters()).device
    transform = _get_transform(train=False)
    use_amp   = device.type == "cuda"

    def _name(cls_idx):
        if class_names:
            return class_names.get(cls_idx, f"cls_{cls_idx}")
        return f"cls_{cls_idx}"

    # 저신뢰도 박스 인덱스와 crop 텐서를 한꺼번에 수집
    target_indices = []
    tensors        = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if label in STAGE2_SKIP_CLASSES:
            continue
        if score >= bypass_thr:
            continue
        crop_bgr = _extract_crop(img, box)
        if crop_bgr.size == 0:
            continue
        try:
            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            tensors.append(transform(crop_pil))
            target_indices.append(i)
        except Exception as e:
            print(f"  [Stage2 오류] crop 추출: {e}")

    if not tensors:
        return {"boxes": boxes, "scores": scores, "labels": labels}

    # 배치 forward — crop 수만큼 한 번에 처리
    batch = torch.stack(tensors).to(device)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            probs_batch = torch.softmax(stage2_model(batch), dim=1)
    except Exception as e:
        print(f"  [Stage2 오류] batch forward: {e}")
        return {"boxes": boxes, "scores": scores, "labels": labels}

    prefix = f"({img_name}) " if img_name else ""
    for j, i in enumerate(target_indices):
        probs   = probs_batch[j]
        s2_idx  = int(probs.argmax())
        s2_conf = float(probs[s2_idx])
        s2_class = idx_to_class[s2_idx]
        score    = scores[i]
        label    = labels[i]

        if s2_conf >= score:
            if s2_class != label:
                print(f"  [Stage2 보정] {prefix}{_name(label)} → {_name(s2_class)}  "
                      f"conf {score:.2f} → {s2_conf:.2f}  (EfficientNet 채택)")
            else:
                print(f"  [Stage2 확인] {prefix}{_name(label)} 일치  "
                      f"conf {score:.2f} → {s2_conf:.2f}  (EfficientNet 채택)")
            labels[i] = s2_class
            scores[i] = s2_conf
        else:
            print(f"  [Stage2 유지] {prefix}{_name(label)}  "
                  f"YOLO={score:.2f} > EfficientNet={s2_conf:.2f}  (YOLO 유지)")

    return {"boxes": boxes, "scores": scores, "labels": labels}

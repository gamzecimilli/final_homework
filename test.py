"""
Model test scripti.
Önce prepare_cache.py çalıştır, sonra bunu kullan.
"""

import os
import re
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============ AYARLAR ============
EXPERIMENT_ROOT = "out_chest_v1.1"
CACHE_DIR = "cache_v1.1"
OUT_DIR = "chexpert"

BATCH_SIZE = {256: 256, 512: 64}
CLASS_NAMES = ["No Finding", "Pneumonia"]


def get_device():
    """GPU varsa kullan"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("CPU kullanılacak")
    return torch.device("cpu")


def load_cache(size):
    """Cache'den veri yükle"""
    images = np.load(os.path.join(CACHE_DIR, f"images_{size}.npy"))
    labels = np.load(os.path.join(CACHE_DIR, f"labels_{size}.npy"))
    return images, labels


def find_experiments(root):
    """Deney klasörlerini bul"""
    experiments = []
    
    for model in os.listdir(root):
        model_dir = os.path.join(root, model)
        if not os.path.isdir(model_dir):
            continue
        
        for exp in os.listdir(model_dir):
            exp_dir = os.path.join(model_dir, exp)
            if os.path.isdir(exp_dir) and "sz" in exp and "aug_" in exp:
                experiments.append({
                    "model": model,
                    "exp": exp,
                    "dir": exp_dir,
                    "size": int(re.search(r"sz(\d+)", exp).group(1))
                })
    
    return experiments


def find_best_checkpoint(exp_dir):
    """En iyi checkpoint'i bul (en yüksek epoch)"""
    pts = [f for f in os.listdir(exp_dir) if f.endswith(".pt")]
    if not pts:
        return None, -1
    
    best_pt, best_epoch = None, -1
    for pt in pts:
        m = re.search(r"ep(\d+)", pt)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_pt = pt
    
    if best_pt:
        return os.path.join(exp_dir, best_pt), best_epoch
    return None, -1


def compute_metrics(probs, preds, targets):
    """Metrikleri hesapla"""
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    try:
        auc = roc_auc_score(targets, probs)
    except:
        auc = float("nan")
    
    return {
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "sensitivity": recall_score(targets, preds, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "auc": auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cm": cm
    }


def save_confusion_matrix(cm, path, title, metrics):
    """Confusion matrix kaydet"""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=9)
    
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color=color)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    
    txt = f"Bal: {metrics['balanced_accuracy']:.3f} | Sens: {metrics['sensitivity']:.3f} | Spec: {metrics['specificity']:.3f} | AUC: {metrics['auc']:.3f}"
    fig.text(0.5, 0.01, txt, ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


@torch.inference_mode()
def test_model(model, loader, device):
    """Model testi"""
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)
        
        all_probs.append(probs[:, 1].cpu().numpy())
        all_preds.append(probs.argmax(1).cpu().numpy())
        all_targets.append(y.numpy())
    
    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_targets)
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cm_dir = os.path.join(OUT_DIR, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    
    device = get_device()
    torch.backends.cudnn.benchmark = True
    
    # Checkpoint (kaldığı yerden devam)
    ckpt_path = os.path.join(OUT_DIR, "checkpoint.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        completed = set(ckpt["completed"])
        results = ckpt["results"]
        print(f"Devam ediliyor: {len(completed)} tamamlanmış")
    else:
        completed = set()
        results = []
    
    # Deneyleri bul
    experiments = find_experiments(EXPERIMENT_ROOT)
    print(f"Toplam {len(experiments)} deney bulundu")
    
    # Size'a göre grupla
    by_size = {}
    for exp in experiments:
        by_size.setdefault(exp["size"], []).append(exp)
    
    # Her size için test et
    for size in sorted(by_size.keys()):
        exps = by_size[size]
        print(f"\n{'='*50}")
        print(f"Size: {size}px | {len(exps)} model")
        print(f"{'='*50}")
        
        # Cache yükle
        images, labels = load_cache(size)
        print(f"Cache yüklendi: {len(labels)} resim")
        
        # DataLoader
        tensor_x = torch.from_numpy(images)
        tensor_y = torch.from_numpy(labels)
        dataset = TensorDataset(tensor_x, tensor_y)
        bs = BATCH_SIZE.get(size, 64)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, 
                           num_workers=0, pin_memory=True)
        
        # Her model için test
        for exp in tqdm(exps, desc=f"sz{size}"):
            exp_id = f"{exp['model']}__{exp['exp']}"
            
            if exp_id in completed:
                continue
            
            pt_path, epoch = find_best_checkpoint(exp["dir"])
            if not pt_path:
                tqdm.write(f"Checkpoint yok: {exp['dir']}")
                continue
            
            # Model yükle
            try:
                model = torch.jit.load(pt_path, map_location="cpu").to(device)
            except Exception as e:
                tqdm.write(f"Yükleme hatası: {e}")
                continue
            
            # Test
            t0 = time.time()
            probs, preds, targets = test_model(model, loader, device)
            elapsed = time.time() - t0
            
            # Metrikler
            metrics = compute_metrics(probs, preds, targets)
            
            # CM kaydet
            cm_path = os.path.join(cm_dir, f"{exp_id}_ep{epoch:02d}.png")
            save_confusion_matrix(
                metrics["cm"], cm_path,
                f"{exp['model']} | {exp['exp']}", metrics
            )
            
            # Sonuç ekle
            results.append({
                "model": exp["model"],
                "experiment": exp["exp"],
                "size": size,
                "epoch": epoch,
                "n": len(labels),
                "time_sec": round(elapsed, 2),
                "img_per_sec": round(len(labels) / elapsed, 1),
                **{k: v for k, v in metrics.items() if k != "cm"}
            })
            
            # Checkpoint güncelle
            completed.add(exp_id)
            with open(ckpt_path, 'w') as f:
                json.dump({"completed": list(completed), "results": results}, f)
            
            # Temizlik
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
            tqdm.write(f"{exp['model']}/{exp['exp']} | Bal: {metrics['balanced_accuracy']:.3f} | AUC: {metrics['auc']:.3f} | {len(labels)/elapsed:.0f} img/s")
        
        # Bellek temizle
        del images, labels, tensor_x, tensor_y, dataset, loader
        gc.collect()
    
    # Sonuçları kaydet
    df = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
    df.to_excel(os.path.join(OUT_DIR, "results.xlsx"), index=False)
    
    # Checkpoint sil
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    
    print(f"\n{'='*50}")
    print("Tamamlandı!")
    print(f"Sonuçlar: {OUT_DIR}/results.xlsx")
    print(f"{'='*50}")
    
    print("\nTop 5:")
    cols = ["model", "experiment", "balanced_accuracy", "sensitivity", "specificity", "auc"]
    print(df[cols].head().to_string(index=False))


if __name__ == "__main__":
    main()

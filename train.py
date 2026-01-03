# ============================================================
# Chest X-Ray (NORMAL vs PNEUMONIA) - Geliştirilmiş Çoklu Deney Pipeline v2
# ============================================================
# 
# YENİ ÖZELLİKLER (v2):
# ----------------------
# 1. Akıllı Model Kaydetme:
#    - Model adı + balanced accuracy + epoch numarası ile isimlendirme
#    - Her model kendi klasöründe: output_dir/model_name/scenario_xxx/
#    - Otomatik isimlendirme (grid boyutu değişse bile)
#
# 2. 3 Seviyeli Augmentasyon:
#    - none: Augmentasyon yok (sadece resize + crop)
#    - medium: Orta seviye (flip + rotation + hafif renk)
#    - strong: Zengin augmentasyon (elastic, CLAHE simülasyonu, vs.)
#
# 3. Modern PyTorch API:
#    - torch.amp (yeni mixed precision API)
#    - Gradient clipping
#    - CosineAnnealingWarmRestarts scheduler
#
# 4. Detaylı Loglama:
#    - Her epoch: train_loss, val_balanced_acc
#    - En iyi model için: test balanced_acc, F1, sensitivity, specificity
#    - Ayrı Excel dosyası: kaydedilen modelin tüm parametreleri
#
# 5. Otomatik Klasör Yapısı:
#    - output_dir/
#      ├── efficientnet_b1/
#      │   ├── sz512_aug_strong/
#      │   │   ├── efficientnet_b1_ep05_balacc0.9234_BEST.pth
#      │   │   ├── training_log.xlsx
#      │   │   ├── confusion_matrix.png
#      │   │   └── gradcam/
#      │   └── sz224_aug_none/
#      ├── mobilenet_v3_large/
#      └── experiments_summary.xlsx
#
# ============================================================

# %% =====================================================
# IMPORTS - Gerekli Kütüphaneler
# ========================================================
import os
import time
import math
import copy
import random
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Literal
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    precision_score, recall_score, confusion_matrix, roc_auc_score
)

# Uyarıları bastır (temiz çıktı için)
warnings.filterwarnings('ignore', category=UserWarning)

# Grad-CAM (opsiyonel - yüklü değilse otomatik devre dışı)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("⚠️ Grad-CAM paketi yüklü değil. Görselleştirme devre dışı.")
    print("   Yüklemek için: pip install grad-cam")


# %% =====================================================
# CONFIGURATION - Tüm Ayarlar Tek Yerde
# ========================================================
# 
# Bu bölümde tüm deney parametrelerini değiştirebilirsiniz.
# Grid boyutları değişse bile isimlendirme otomatik yapılır.
# ========================================================

CONFIG = {
    # ----- VERİ AYARLARI -----
    # dataset_root: NORMAL ve PNEUMONIA klasörlerini içeren ana dizin
    # Örnek yapı:
    #   /path/to/chest_xray/
    #   ├── NORMAL/
    #   │   ├── img001.jpg
    #   │   └── ...
    #   └── PNEUMONIA/
    #       ├── img001.jpg
    #       └── ...
    "dataset_root": r"chest_full",  #
    "class_names": ["NORMAL", "PNEUMONIA"],       # Klasör adları (sıra önemli: 0=NORMAL, 1=PNEUMONIA)

    # ----- VERİ BÖLME AYARLARI -----
    # Stratified split: Her sınıftan orantılı örnek alınır
    "split": {
        "train_ratio": 0.75,  # Eğitim verisi oranı
        "val_ratio":   0.15,  # Doğrulama verisi oranı  
        "test_ratio":  0.10,  # Test verisi oranı
        "seed": 42,           # Tekrarlanabilirlik için sabit seed
    },

    # ----- DENEY GRID'İ -----
    # Tüm kombinasyonlar otomatik denenir
    # Örnek: 3 boyut × 4 model × 3 aug = 36 deney
    "experiments": {
        # Görüntü boyutları (büyük boyut = daha fazla detay, daha yavaş)
        "input_sizes": [512, 256],
        
        # Test edilecek modeller
        "models": [
            "efficientnet_b1",      # Dengeli performans/hız
            "mobilenet_v3_large",   # Hızlı, mobil uyumlu
            "densenet121",
        ],
        
        # Augmentasyon seviyeleri (3 seviye)
        # "none": Sadece resize + center crop
        # "medium": Flip + rotation + hafif brightness
        # "strong": Yukarıdakiler + blur + contrast + affine
        "augmentation_levels": ["none", "medium", "strong"],
    },

    # ----- EĞİTİM AYARLARI -----
    "train": {
        "batch_size": 32,        # GPU belleğine göre ayarla (512px için 8-16)
        "num_epochs": 25,        # Maksimum epoch (early stopping var)
        "patience": 7,           # Bu kadar epoch iyileşme olmazsa dur
        "learning_rate": 1e-3,   # Başlangıç öğrenme oranı
        "weight_decay": 1e-4,    # L2 regularization
        "num_workers": 4,        # DataLoader paralel işçi sayısı
        "label_smoothing": 0.1,  # Label smoothing (0.0-0.2 arası)
    },

    # ----- LEARNING RATE SCHEDULER -----
    # CosineAnnealingWarmRestarts: Periyodik olarak LR'ı sıfırlar
    "scheduler": {
        "type": "cosine_warm_restarts",  # "cosine_warm_restarts" veya "cosine_annealing"
        "T_0": 10,                        # İlk restart periyodu (epoch)
        "T_mult": 2,                      # Her restart'ta periyod çarpanı
        "eta_min": 1e-5,                  # Minimum öğrenme oranı
    },

    # ----- MIXED PRECISION (AMP) -----
    # GPU'da eğitimi 1.5-2x hızlandırır, bellek tasarrufu sağlar
    "amp": {
        "enabled": True,  # CUDA yoksa otomatik devre dışı
    },

    # ----- SINIF DENGESİZLİĞİ -----
    # WeightedRandomSampler: Az olan sınıftan daha sık örnekleme
    "imbalance": {
        "use_weighted_sampler": True,
    },

    # ----- ÇIKTI AYARLARI -----
    "output": {
        "save_dir": "out_chest_v1.1",          # Ana çıktı klasörü
        "save_torchscript": True,               # TorchScript formatında kaydet (.pt)
        "save_confusion_matrix": True,          # Confusion matrix PNG
        "save_training_curves": True,           # Loss/accuracy grafikleri
        "export_model_params_excel": True,      # Her model için parametre Excel'i
    },

    # ----- GRAD-CAM AYARLARI -----
    "gradcam": {
        "enabled": True,          # Grad-CAM görselleştirmesi
        "num_samples": 8,         # Test setinden kaç örnek
    },
}

# ImageNet normalizasyon değerleri (pretrained modeller için)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# %% =====================================================
# YARDIMCI FONKSİYONLAR
# ========================================================

def seed_everything(seed: int) -> None:
    """
    Tüm random seed'leri sabitler.
    Bu sayede aynı seed ile aynı sonuçlar elde edilir (reproducibility).
    
    Args:
        seed: Sabitlenecek seed değeri
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministik mod (biraz yavaşlatır ama sonuçlar tutarlı)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Kullanılabilir en iyi cihazı döndürür.
    CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: Kullanılacak cihaz
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU bulundu: {torch.cuda.get_device_name(0)}")
        print(f"  Bellek: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Apple Silicon GPU (MPS) kullanılıyor")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU bulunamadı, CPU kullanılıyor (yavaş olacak)")
    return device


def format_time(seconds: float) -> str:
    """Saniyeyi okunabilir formata çevirir (1h 23m 45s)"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def create_experiment_name(model_name: str, input_size: int, aug_level: str) -> str:
    """
    Deney için benzersiz ve okunabilir isim oluşturur.
    
    Örnek: "efficientnet_b1_sz224_aug_medium"
    
    Args:
        model_name: Model adı
        input_size: Görüntü boyutu
        aug_level: Augmentasyon seviyesi
    
    Returns:
        str: Deney adı
    """
    return f"{model_name}_sz{input_size}_aug_{aug_level}"


def create_model_filename(model_name: str, epoch: int, bal_acc: float, 
                          input_size: int, aug_level: str, suffix: str = "") -> str:
    """
    Kaydedilecek model için detaylı dosya adı oluşturur.
    
    Örnek: "efficientnet_b1_sz224_aug_medium_ep05_balacc0.9234_BEST.pth"
    
    Args:
        model_name: Model adı
        epoch: Epoch numarası
        bal_acc: Balanced accuracy değeri
        input_size: Görüntü boyutu
        aug_level: Augmentasyon seviyesi
        suffix: Ek bilgi (BEST, LAST, vs.)
    
    Returns:
        str: Dosya adı
    """
    base = f"{model_name}_sz{input_size}_aug_{aug_level}"
    metrics = f"ep{epoch:02d}_balacc{bal_acc:.4f}"
    
    if suffix:
        return f"{base}_{metrics}_{suffix}.pth"
    return f"{base}_{metrics}.pth"


# %% =====================================================
# VERİ YÜKLEYİCİ SINIFI
# ========================================================

def list_images(dataset_root: str, class_names: List[str]) -> Tuple[List[str], List[int]]:
    """
    Klasörlerden görüntü dosyalarını listeler.
    
    Beklenen yapı:
        dataset_root/
        ├── NORMAL/
        │   ├── img1.jpg
        │   └── ...
        └── PNEUMONIA/
            ├── img1.jpg
            └── ...
    
    Args:
        dataset_root: Veri kök dizini
        class_names: Sınıf klasör adları listesi
    
    Returns:
        Tuple[List[str], List[int]]: (dosya_yolları, etiketler)
    """
    paths, labels = [], []
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_root, class_name)
        
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"❌ Sınıf klasörü bulunamadı: {class_dir}\n"
                f"   Lütfen dataset_root ayarını kontrol edin."
            )
        
        # Klasördeki tüm görüntüleri bul
        count = 0
        for filename in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                paths.append(os.path.join(class_dir, filename))
                labels.append(class_idx)
                count += 1
        
        print(f"  {class_name}: {count} görüntü bulundu")
    
    if len(paths) == 0:
        raise RuntimeError(
            "❌ Hiç görüntü bulunamadı!\n"
            "   Klasör yapısını ve dosya uzantılarını kontrol edin."
        )
    
    return paths, labels


class ChestXRayDataset(Dataset):
    """
    Chest X-Ray görüntüleri için PyTorch Dataset sınıfı.
    
    Özellikler:
    - Görüntüleri RGB'ye çevirir (grayscale olsa bile)
    - Transform uygulanabilir
    - Lazy loading (bellek dostu)
    """
    
    def __init__(self, paths: List[str], labels: List[int], transform=None):
        """
        Args:
            paths: Görüntü dosya yolları
            labels: Sınıf etiketleri (0 veya 1)
            transform: Uygulanacak dönüşümler
        """
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[idx]
        label = self.labels[idx]
        
        # Görüntüyü yükle ve RGB'ye çevir
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# %% =====================================================
# STRATIFIED SPLIT - Dengeli Veri Bölme
# ========================================================

def stratified_split(
    paths: List[str], 
    labels: List[int],
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float,
    seed: int
) -> Tuple[Tuple[List[str], List[int]], ...]:
    """
    Veriyi stratified (katmanlı) olarak train/val/test'e böler.
    
    Stratified: Her bölümde sınıf oranları korunur.
    Örnek: %70 NORMAL, %30 PNEUMONIA -> Her split'te de aynı oran
    
    Args:
        paths: Tüm görüntü yolları
        labels: Tüm etiketler
        train_ratio: Eğitim oranı (örn: 0.70)
        val_ratio: Doğrulama oranı (örn: 0.15)
        test_ratio: Test oranı (örn: 0.15)
        seed: Random seed
    
    Returns:
        Tuple: ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels))
    """
    # Oran kontrolü
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, f"Split oranları toplamı 1.0 olmalı, şu an: {total}"
    
    X = np.array(paths)
    y = np.array(labels)
    
    # İlk bölme: train vs (val + test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=(val_ratio + test_ratio), 
        random_state=seed
    )
    train_idx, temp_idx = next(sss1.split(X, y))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    
    # İkinci bölme: val vs test
    test_size_ratio = test_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=test_size_ratio, 
        random_state=seed
    )
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]
    
    return (
        (X_train.tolist(), y_train.tolist()),
        (X_val.tolist(), y_val.tolist()),
        (X_test.tolist(), y_test.tolist())
    )


# %% =====================================================
# AUGMENTASYON TRANSFORMS - 3 Seviye
# ========================================================

class RandomGaussianBlur:
    """Rastgele Gaussian blur uygular (X-ray için hafif)"""
    def __init__(self, p: float = 0.3, radius_range: Tuple[float, float] = (0.5, 1.5)):
        self.p = p
        self.radius_range = radius_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomBrightnessContrast:
    """Rastgele parlaklık ve kontrast ayarı"""
    def __init__(self, brightness_range: Tuple[float, float] = (0.9, 1.1),
                 contrast_range: Tuple[float, float] = (0.9, 1.1), p: float = 0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            # Parlaklık
            brightness_factor = random.uniform(*self.brightness_range)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)
            
            # Kontrast
            contrast_factor = random.uniform(*self.contrast_range)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        return img



def build_transforms(
    input_size: int, 
    aug_level: Literal["none", "medium", "strong"]
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """
    X-Ray görüntüleri için optimize edilmiş 3 seviyeli augmentasyon.
    
    ÖNEMLİ X-RAY NOTLARI:
    ---------------------
    - HorizontalFlip DÜŞÜK tutulmalı: Kalp sol tarafta, flip anatomik hata yaratır
    - VerticalFlip KULLANILMAMALI: Anatomik olarak anlamsız
    - ColorJitter KULLANILMAMALI: X-ray gri tonlamalı, renk değişimi anlamsız
    - Rotasyon SINIRLI tutulmalı: Gerçek çekimlerde ±10° üstü nadir
    - Agresif crop KULLANILMAMALI: Akciğer kenarları kesilmemeli
    
    Seviyeler:
    ---------
    none (yok):
        - Sadece resize ve center crop
        - Baseline / karşılaştırma için
        - Veri zaten çok büyükse yeterli olabilir
    
    medium (orta):
        - Hafif geometrik dönüşümler
        - Hafif parlaklık/kontrast
        - Genel kullanım için önerilen
    
    strong (zengin):
        - Daha fazla varyasyon ama hala konservatif
        - Gaussian blur (düşük kalite simülasyonu)
        - Veri azsa veya overfitting varsa kullan
    
    Args:
        input_size: Hedef görüntü boyutu (kare, örn: 224, 512)
        aug_level: Augmentasyon seviyesi ("none", "medium", "strong")
    
    Returns:
        Tuple: (train_transform, eval_transform, visualization_transform)
    """
    # Resize boyutu: Crop için biraz büyük tut (%10 fazla)
    resize_size = int(input_size * 1.1)
    
    # =========================================================
    # AUGMENTASYON YOK (Baseline)
    # =========================================================
    # Kullanım: Karşılaştırma için baseline, büyük veri setleri
    if aug_level == "none":
        train_tf = transforms.Compose([
            # Boyutlandırma
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            
            # Tensor'a çevir ve normalize et
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    # =========================================================
    # ORTA SEVİYE AUGMENTASYON (Önerilen)
    # =========================================================
    # Kullanım: Çoğu X-ray projesi için ideal başlangıç
    elif aug_level == "medium":
        train_tf = transforms.Compose([
            # Boyutlandırma
            transforms.Resize(resize_size),
            
            # Hafif random crop: %95-100 oranında (kenarlardan az kes)
            transforms.RandomResizedCrop(input_size, scale=(0.95, 1.0)),
            
            # Yatay flip: DÜŞÜK olasılık (kalp sol tarafta!)
            # p=0.1 → %10 şansla flip (veya tamamen kaldırabilirsin)
            transforms.RandomHorizontalFlip(p=0.1),
            
            # Hafif rotasyon: ±7° (gerçekçi çekim açısı varyasyonu)
            transforms.RandomRotation(degrees=7),
            
            # Hafif parlaklık/kontrast: Farklı cihaz ayarlarını simüle eder
            RandomBrightnessContrast(
                brightness_range=(0.95, 1.05),  # ±%5
                contrast_range=(0.95, 1.05),    # ±%5
                p=0.3  # %30 olasılık
            ),
            
            # Tensor'a çevir ve normalize et
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    # =========================================================
    # ZENGİN AUGMENTASYON (X-Ray için Optimize Edilmiş)
    # =========================================================
    # Kullanım: Veri azsa, overfitting varsa
    # NOT: Genel "strong" augmentasyondan DAHA YUMUŞAK!
    #      X-ray için agresif augmentasyon zararlı olabilir.
    elif aug_level == "strong":
        train_tf = transforms.Compose([
            # Boyutlandırma
            transforms.Resize(resize_size),
            
            # Random crop: %90-100 (biraz daha agresif ama hala konservatif)
            transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
            
            # Yatay flip: Hala düşük olasılık
            transforms.RandomHorizontalFlip(p=0.1),
            
            # Rotasyon: ±10° (maksimum güvenli değer)
            transforms.RandomRotation(degrees=10),
            
            # Affine dönüşümler: Hafif translate, scale, shear
            transforms.RandomAffine(
                degrees=0,               # Rotasyon yukarıda zaten var
                translate=(0.03, 0.03),  # %3 kaydırma (çok az)
                scale=(0.97, 1.03),      # %3 ölçekleme (çok az)
                shear=2                  # 2° shear (çok az)
            ),
            
            # Gaussian blur: Düşük kaliteli görüntü simülasyonu
            RandomGaussianBlur(p=0.15, radius_range=(0.5, 1.0)),
            
            # Parlaklık/Kontrast: Farklı cihaz/ayar simülasyonu
            RandomBrightnessContrast(
                brightness_range=(0.9, 1.1),   # ±%10
                contrast_range=(0.9, 1.1),     # ±%10
                p=0.4  # %40 olasılık
            ),
            
            # Tensor'a çevir ve normalize et
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    else:
        raise ValueError(
            f"Geçersiz aug_level: '{aug_level}'\n"
            f"Geçerli değerler: 'none', 'medium', 'strong'"
        )
    
    # =========================================================
    # DEĞERLENDİRME TRANSFORM (Val/Test için)
    # =========================================================
    # Augmentasyon YOK - Her çalıştırmada aynı sonuç için
    eval_tf = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    # =========================================================
    # GÖRSELLEŞTİRME TRANSFORM (Grad-CAM için)
    # =========================================================
    # Normalize YOK - İnsan gözüyle görüntülemek için
    vis_tf = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        # Normalize yok! Görsel çıktı için [0,1] aralığında kalmalı
    ])
    
    return train_tf, eval_tf, vis_tf


# %% =====================================================
# WEIGHTED SAMPLER - Sınıf Dengesizliği Çözümü
# ========================================================

def make_weighted_sampler(
    labels: List[int], 
    num_classes: int
) -> Tuple[WeightedRandomSampler, List[int], List[float]]:
    """
    Dengesiz sınıflar için ağırlıklı örnekleyici oluşturur.
    
    Çalışma mantığı:
    - Az olan sınıftan daha sık örnekleme yapılır
    - Ağırlık = 1 / sınıf_örnek_sayısı
    - Her batch'te sınıf dağılımı daha dengeli olur
    
    Args:
        labels: Eğitim etiketleri
        num_classes: Sınıf sayısı
    
    Returns:
        Tuple: (sampler, sınıf_sayıları, sınıf_ağırlıkları)
    """
    labels_arr = np.array(labels)
    
    # Her sınıftan kaç örnek var?
    class_counts = np.bincount(labels_arr, minlength=num_classes)
    
    # Sınıf ağırlıkları (ters orantılı)
    class_weights = 1.0 / np.maximum(class_counts, 1)  # 0'a bölme koruması
    
    # Her örnek için ağırlık
    sample_weights = class_weights[labels_arr]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    
    # WeightedRandomSampler: Ağırlıklara göre örnekleme
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Aynı örnek birden fazla seçilebilir
    )
    
    return sampler, class_counts.tolist(), class_weights.tolist()


# %% =====================================================
# MODEL BUILDER - Pretrained Model Oluşturma
# ========================================================



def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower().strip()

    if model_name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == "vgg16_bn":
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16_bn(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("...")

    return model



def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Model parametre sayısını hesaplar.
    
    Returns:
        Tuple: (toplam_parametre, eğitilebilir_parametre)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# %% =====================================================
# METRİK HESAPLAMA
# ========================================================

def compute_binary_metrics(
    probs: np.ndarray, 
    preds: np.ndarray, 
    targets: np.ndarray
) -> Dict:
    """
    İkili sınıflandırma metrikleri hesaplar.
    
    Hesaplanan metrikler:
    - Accuracy: Doğru tahmin oranı
    - Balanced Accuracy: Sınıf dengesizliğine dayanıklı accuracy
    - F1 Score: Precision ve Recall harmonik ortalaması
    - Precision: Pozitif tahminlerin doğruluğu
    - Recall (Sensitivity): Gerçek pozitifleri yakalama oranı
    - Specificity: Gerçek negatifleri yakalama oranı
    - ROC-AUC: ROC eğrisi altındaki alan
    
    Args:
        probs: Softmax olasılıkları (N, 2)
        preds: Tahmin edilen sınıflar (N,)
        targets: Gerçek etiketler (N,)
    
    Returns:
        Dict: Tüm metrikler + confusion matrix
    """
    # Temel metrikler
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='binary')
    precision = precision_score(targets, preds, average='binary', zero_division=0)
    recall = recall_score(targets, preds, average='binary', zero_division=0)  # Sensitivity
    
    # Confusion matrix'ten specificity hesapla
    cm = confusion_matrix(targets, preds)
    # cm[0,0] = TN, cm[0,1] = FP, cm[1,0] = FN, cm[1,1] = TP
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # ROC-AUC (pozitif sınıf olasılığı ile)
    try:
        roc_auc = roc_auc_score(targets, probs[:, 1])
    except Exception:
        roc_auc = float('nan')
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,           # = Sensitivity
        'sensitivity': recall,      # Aynı şey, açık isim
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


@torch.no_grad()
def predict_with_probs(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device,
    use_amp: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Model ile tahmin yapar ve olasılıkları döndürür.
    
    Args:
        model: Eğitilmiş model
        loader: DataLoader
        device: Cihaz
        use_amp: Mixed precision kullan
    
    Returns:
        Tuple: (olasılıklar, tahminler, gerçek_etiketler)
    """
    model.eval()
    
    all_probs = []
    all_preds = []
    all_targets = []
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Mixed precision forward
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
        
        preds = probs.argmax(dim=1)
        
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())
    
    # Birleştir
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    return all_probs, all_preds, all_targets


# %% =====================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# ========================================================

def save_confusion_matrix_plot(
    cm: np.ndarray, 
    class_names: List[str], 
    save_path: str, 
    title: str = "Confusion Matrix"
) -> None:
    """
    Confusion matrix'i PNG olarak kaydeder.
    
    Args:
        cm: Confusion matrix (2x2)
        class_names: Sınıf isimleri
        save_path: Kayıt yolu
        title: Grafik başlığı
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=12)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Hücre değerlerini yaz
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14)
    
    plt.ylabel('Gerçek Etiket', fontsize=11)
    plt.xlabel('Tahmin', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_training_curves(
    history: List[Dict], 
    save_path: str, 
    title: str = "Training Curves"
) -> None:
    """
    Eğitim eğrilerini (loss, accuracy) PNG olarak kaydeder.
    
    Args:
        history: Epoch bazlı metrik listesi
        save_path: Kayıt yolu
        title: Grafik başlığı
    """
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h.get('val_loss', 0) for h in history]
    train_bal_acc = [h['train_balanced_acc'] for h in history]
    val_bal_acc = [h['val_balanced_acc'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=12)
    
    # Loss grafiği
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Loss Eğrisi')
    
    # Accuracy grafiği
    axes[1].plot(epochs, train_bal_acc, 'b-', label='Train Bal Acc', linewidth=2)
    axes[1].plot(epochs, val_bal_acc, 'r-', label='Val Bal Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Balanced Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Balanced Accuracy Eğrisi')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_model_params_excel(
    params: Dict, 
    save_path: str
) -> None:
    """
    Kaydedilen modelin tüm parametrelerini Excel'e yazar.
    
    Args:
        params: Model ve eğitim parametreleri
        save_path: Excel dosya yolu
    """
    # Dict'i DataFrame'e çevir (tek satır)
    df = pd.DataFrame([params])
    
    # Sütun sıralaması
    priority_cols = [
        'model_name', 'input_size', 'augmentation_level',
        'best_epoch', 'best_val_balanced_acc',
        'test_balanced_accuracy', 'test_f1_score', 
        'test_sensitivity', 'test_specificity',
        'test_accuracy', 'test_roc_auc'
    ]
    
    # Öncelikli sütunları öne al
    cols = [c for c in priority_cols if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]
    
    df.to_excel(save_path, index=False)


# %% =====================================================
# GRAD-CAM GÖRSELLEŞTİRME
# ========================================================

def get_gradcam_target_layer(model_name: str, model: nn.Module):
    """
    Her model mimarisi için Grad-CAM hedef katmanını döndürür.
    
    Hedef katman genellikle son convolutional katmandır.
    """
    model_name = model_name.lower()
    
    if model_name == "efficientnet_b1":
        return [model.features[-1]]
    elif model_name == "mobilenet_v3_large":
        return [model.features[-1]]
    elif model_name == "densenet121":
        return [model.features.denseblock4]
    elif model_name == "vgg16_bn":
        return [model.features[-1]]
    else:
        return None


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Normalize edilmiş tensor'ı [0,1] RGB görüntüye çevirir.
    
    Args:
        tensor: 3xHxW normalized tensor
    
    Returns:
        np.ndarray: HxWx3 float32 array [0,1]
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    img = tensor.clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    img = img * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    
    return img.astype(np.float32)


def run_gradcam_analysis(
    model: nn.Module,
    model_name: str,
    dataset: Dataset,
    class_names: List[str],
    device: torch.device,
    save_dir: str,
    num_samples: int = 8
) -> None:
    """
    Grad-CAM analizi yapar ve sonuçları kaydeder.
    
    Args:
        model: Eğitilmiş model
        model_name: Model adı
        dataset: Test dataset
        class_names: Sınıf isimleri
        device: Cihaz
        save_dir: Kayıt dizini
        num_samples: Örnek sayısı
    """
    if not GRADCAM_AVAILABLE:
        print("  ⚠️ Grad-CAM paketi yok, atlanıyor...")
        return
    
    target_layers = get_gradcam_target_layer(model_name, model)
    if target_layers is None:
        print(f"  ⚠️ {model_name} için Grad-CAM hedef katman bulunamadı")
        return
    
    # Model'i Grad-CAM için hazırla
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    
    # Grad-CAM objesi
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Rastgele örnekler seç
    n_samples = min(num_samples, len(dataset))
    indices = np.random.default_rng(42).choice(len(dataset), size=n_samples, replace=False)
    
    # Kayıt klasörü
    gradcam_dir = os.path.join(save_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # Grid figürü
    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
    
    for i, idx in enumerate(indices):
        img_tensor, true_label = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        
        # Tahmin
        with torch.no_grad():
            logits = model(img_input)
            pred_label = logits.argmax(dim=1).item()
        
        # Grad-CAM
        grayscale_cam = cam(input_tensor=img_input)[0]
        
        # Görselleştirme
        rgb_img = denormalize_tensor(img_tensor)
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Grid'e ekle
        ax_orig = axes[0, i] if n_samples > 1 else axes[0]
        ax_cam = axes[1, i] if n_samples > 1 else axes[1]
        
        ax_orig.imshow(rgb_img)
        ax_orig.set_title(f"T:{class_names[true_label]}\nP:{class_names[pred_label]}", fontsize=9)
        ax_orig.axis('off')
        
        ax_cam.imshow(overlay)
        ax_cam.axis('off')
        
        # Tek görüntüyü kaydet
        single_path = os.path.join(gradcam_dir, f"sample_{i:02d}_T{true_label}_P{pred_label}.png")
        plt.imsave(single_path, overlay)
    
    # Grid'i kaydet
    fig.suptitle('Grad-CAM Görselleştirmesi (Üst: Orijinal, Alt: CAM)', fontsize=12)
    fig.tight_layout()
    grid_path = os.path.join(gradcam_dir, 'gradcam_grid.png')
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Grad-CAM kaydedildi: {gradcam_dir}")


# %% =====================================================
# ANA EĞİTİM FONKSİYONU
# ========================================================

def train_single_experiment(
    exp_name: str,
    model_name: str,
    input_size: int,
    aug_level: str,
    train_data: Tuple[List[str], List[int]],
    val_data: Tuple[List[str], List[int]],
    test_data: Tuple[List[str], List[int]],
    class_names: List[str],
    device: torch.device,
    output_base_dir: str
) -> Tuple[pd.DataFrame, Dict]:
    """
    Tek bir deney (model + boyut + augmentasyon kombinasyonu) için eğitim yapar.
    
    Bu fonksiyon:
    1. Model oluşturur
    2. DataLoader'ları hazırlar
    3. Eğitim döngüsünü çalıştırır
    4. En iyi modeli kaydeder (isimde: model_ep05_balacc0.9234_BEST.pth)
    5. Test metriklerini hesaplar
    6. Tüm sonuçları Excel'e yazar
    
    Args:
        exp_name: Deney adı
        model_name: Model adı
        input_size: Görüntü boyutu
        aug_level: Augmentasyon seviyesi
        train_data: (paths, labels) tuple
        val_data: (paths, labels) tuple
        test_data: (paths, labels) tuple
        class_names: Sınıf isimleri
        device: Cihaz
        output_base_dir: Çıktı ana dizini
    
    Returns:
        Tuple[pd.DataFrame, Dict]: (epoch_history, experiment_summary)
    """
    print(f"\n{'='*60}")
    print(f"DENEY: {exp_name}")
    print(f"{'='*60}")
    
    # ===== DENEY KLASÖRÜ OLUŞTUR =====
    # Yapı: output_dir/model_name/sz224_aug_medium/
    exp_dir = os.path.join(
        output_base_dir,
        model_name,
        f"sz{input_size}_aug_{aug_level}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📁 Çıktı klasörü: {exp_dir}")
    
    # ===== VERİLERİ HAZIRLA =====
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    num_classes = len(class_names)
    
    # Transforms
    train_tf, eval_tf, vis_tf = build_transforms(input_size, aug_level)
    
    # Dataset'ler
    train_dataset = ChestXRayDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = ChestXRayDataset(val_paths, val_labels, transform=eval_tf)
    test_dataset = ChestXRayDataset(test_paths, test_labels, transform=eval_tf)
    
    # Weighted sampler (eğitim için)
    sampler = None
    class_counts = None
    class_weights = None
    
    if CONFIG["imbalance"]["use_weighted_sampler"]:
        sampler, class_counts, class_weights = make_weighted_sampler(train_labels, num_classes)
        print(f"📊 Sınıf dağılımı: {dict(zip(class_names, class_counts))}")
        print(f"📊 Sınıf ağırlıkları: {dict(zip(class_names, [f'{w:.4f}' for w in class_weights]))}")
    
    # DataLoader'lar
    batch_size = CONFIG["train"]["batch_size"]
    num_workers = CONFIG["train"]["num_workers"]
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # ===== MODEL OLUŞTUR =====
    model = build_model(model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"🔧 Model: {model_name}")
    print(f"   Toplam parametre: {total_params:,}")
    print(f"   Eğitilebilir: {trainable_params:,}")
    
    # ===== LOSS FONKSİYONU =====
    label_smoothing = CONFIG["train"]["label_smoothing"]
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # ===== OPTİMİZER =====
    lr = CONFIG["train"]["learning_rate"]
    weight_decay = CONFIG["train"]["weight_decay"]
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # ===== LEARNING RATE SCHEDULER =====
    # CosineAnnealingWarmRestarts: Periyodik olarak LR'ı restart eder
    scheduler_cfg = CONFIG["scheduler"]
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_cfg["T_0"],
        T_mult=scheduler_cfg["T_mult"],
        eta_min=scheduler_cfg["eta_min"]
    )
    print(f"📈 LR Scheduler: CosineAnnealingWarmRestarts (T_0={scheduler_cfg['T_0']}, T_mult={scheduler_cfg['T_mult']})")
    
    # ===== MIXED PRECISION (AMP) =====
    use_amp = CONFIG["amp"]["enabled"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("⚡ Mixed Precision (AMP) aktif")
    
    # ===== EĞİTİM DEĞİŞKENLERİ =====
    num_epochs = CONFIG["train"]["num_epochs"]
    patience = CONFIG["train"]["patience"]
    
    best_val_bal_acc = -1.0
    best_epoch = -1
    best_model_state = None
    epochs_no_improve = 0
    
    history = []  # Her epoch için metrikler
    
    print(f"\n🚀 Eğitim başlıyor ({num_epochs} epoch, patience={patience})...")
    print("-" * 80)
    
    # ===== EĞİTİM DÖNGÜSÜ =====
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # ----- TRAIN PHASE -----
        model.train()
        train_loss_sum = 0.0
        train_preds_list = []
        train_targets_list = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass (mixed precision)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            
            # Metrikler için topla
            train_loss_sum += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_preds_list.append(preds.cpu())
            train_targets_list.append(labels.cpu())
        
        # LR Scheduler step (epoch bazlı)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train metrikleri
        train_loss = train_loss_sum / len(train_dataset)
        train_preds = torch.cat(train_preds_list).numpy()
        train_targets = torch.cat(train_targets_list).numpy()
        train_acc = accuracy_score(train_targets, train_preds)
        train_bal_acc = balanced_accuracy_score(train_targets, train_preds)
        
        # ----- VALIDATION PHASE -----
        model.eval()
        val_loss_sum = 0.0
        val_preds_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)
                
                val_loss_sum += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_preds_list.append(preds.cpu())
                val_targets_list.append(labels.cpu())
        
        # Val metrikleri
        val_loss = val_loss_sum / len(val_dataset)
        val_preds = torch.cat(val_preds_list).numpy()
        val_targets = torch.cat(val_targets_list).numpy()
        val_acc = accuracy_score(val_targets, val_preds)
        val_bal_acc = balanced_accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='binary')
        
        epoch_time = time.time() - epoch_start
        
        # History'ye ekle
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_balanced_acc': train_bal_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_balanced_acc': val_bal_acc,
            'val_f1': val_f1,
            'learning_rate': current_lr,
            'epoch_time_sec': epoch_time
        })
        
        # Ekrana yazdır
        print(f"Epoch {epoch:02d}/{num_epochs} | {format_time(epoch_time)} | LR: {current_lr:.2e}")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc*100:.1f}%, bal_acc={train_bal_acc*100:.1f}%")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc*100:.1f}%, bal_acc={val_bal_acc*100:.1f}%, f1={val_f1:.3f}")
        
        # ----- EN İYİ MODEL KONTROLÜ -----
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            
            # En iyi modeli kaydet (isimde epoch ve balanced accuracy var)
            best_filename = create_model_filename(
                model_name, epoch, val_bal_acc, input_size, aug_level, "BEST"
            )
            best_path = os.path.join(exp_dir, best_filename)
            
            # State dict kaydet
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_balanced_acc': val_bal_acc,
                'model_name': model_name,
                'input_size': input_size,
                'augmentation_level': aug_level,
                'config': CONFIG,
            }, best_path)
            
            print(f"  ✓ YENİ EN İYİ! Kaydedildi: {best_filename}")
            
            # TorchScript olarak da kaydet
            if CONFIG["output"]["save_torchscript"]:
                ts_filename = best_filename.replace('.pth', '_torchscript.pt')
                ts_path = os.path.join(exp_dir, ts_filename)
                
                model_cpu = copy.deepcopy(model).cpu().eval()
                example_input = torch.randn(1, 3, input_size, input_size)
                traced = torch.jit.trace(model_cpu, example_input)
                traced.save(ts_path)
        else:
            epochs_no_improve += 1
            print(f"  → İyileşme yok ({epochs_no_improve}/{patience})")
        
        # Early stopping kontrolü
        if epochs_no_improve >= patience:
            print(f"\n⚠️ Early stopping! {patience} epoch boyunca iyileşme olmadı.")
            break
        
        print()
    
    print("-" * 80)
    
    # ===== EN İYİ MODELİ YÜKLE VE TEST ET =====
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\n📊 TEST DEĞERLENDİRMESİ (Epoch {best_epoch} modeli)")
    print("-" * 40)
    
    probs, preds, targets = predict_with_probs(model, test_loader, device, use_amp)
    test_metrics = compute_binary_metrics(probs, preds, targets)
    
    print(f"  Accuracy:         {test_metrics['accuracy']*100:.2f}%")
    print(f"  Balanced Acc:     {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  F1 Score:         {test_metrics['f1_score']:.4f}")
    print(f"  Sensitivity:      {test_metrics['sensitivity']*100:.2f}%")
    print(f"  Specificity:      {test_metrics['specificity']*100:.2f}%")
    print(f"  ROC-AUC:          {test_metrics['roc_auc']:.4f}")
    
    # ===== CONFUSION MATRIX KAYDET =====
    if CONFIG["output"]["save_confusion_matrix"]:
        cm_path = os.path.join(exp_dir, "confusion_matrix.png")
        save_confusion_matrix_plot(
            test_metrics['confusion_matrix'],
            class_names,
            cm_path,
            title=f"{exp_name}\nTest Confusion Matrix"
        )
        print(f"  ✓ Confusion matrix kaydedildi")
    
    # ===== EĞİTİM EĞRİLERİ KAYDET =====
    if CONFIG["output"]["save_training_curves"]:
        curves_path = os.path.join(exp_dir, "training_curves.png")
        save_training_curves(history, curves_path, title=f"{exp_name} - Training Curves")
        print(f"  ✓ Training curves kaydedildi")
    
    # ===== EPOCH LOG EXCEL KAYDET =====
    df_history = pd.DataFrame(history)
    history_excel_path = os.path.join(exp_dir, "training_log.xlsx")
    df_history.to_excel(history_excel_path, index=False)
    print(f"  ✓ Training log kaydedildi")
    
    # ===== MODEL PARAMETRELERİ EXCEL KAYDET =====
    if CONFIG["output"]["export_model_params_excel"]:
        model_params = {
            'experiment_name': exp_name,
            'model_name': model_name,
            'input_size': input_size,
            'augmentation_level': aug_level,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'batch_size': batch_size,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'label_smoothing': label_smoothing,
            'scheduler_type': scheduler_cfg["type"],
            'scheduler_T_0': scheduler_cfg["T_0"],
            'scheduler_T_mult': scheduler_cfg["T_mult"],
            'use_weighted_sampler': CONFIG["imbalance"]["use_weighted_sampler"],
            'use_amp': use_amp,
            'best_epoch': best_epoch,
            'best_val_balanced_acc': best_val_bal_acc,
            'test_accuracy': test_metrics['accuracy'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_f1_score': test_metrics['f1_score'],
            'test_sensitivity': test_metrics['sensitivity'],
            'test_specificity': test_metrics['specificity'],
            'test_roc_auc': test_metrics['roc_auc'],
            'test_tn': test_metrics['tn'],
            'test_fp': test_metrics['fp'],
            'test_fn': test_metrics['fn'],
            'test_tp': test_metrics['tp'],
            'train_class_counts': str(dict(zip(class_names, class_counts))) if class_counts else "N/A",
            'saved_model_path': os.path.join(exp_dir, create_model_filename(
                model_name, best_epoch, best_val_bal_acc, input_size, aug_level, "BEST"
            ))
        }
        
        params_excel_path = os.path.join(exp_dir, "model_parameters.xlsx")
        save_model_params_excel(model_params, params_excel_path)
        print(f"  ✓ Model parameters kaydedildi")
    
    # ===== GRAD-CAM =====
    if CONFIG["gradcam"]["enabled"]:
        # Görselleştirme için dataset (normalize yok)
        test_vis_dataset = ChestXRayDataset(test_paths, test_labels, transform=eval_tf)
        run_gradcam_analysis(
            model, model_name, test_vis_dataset, class_names,
            device, exp_dir, CONFIG["gradcam"]["num_samples"]
        )
    
    # ===== ÖZET DÖNDÜR =====
    summary = {
        'experiment_name': exp_name,
        'model_name': model_name,
        'input_size': input_size,
        'augmentation_level': aug_level,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_epoch': best_epoch,
        'best_val_balanced_acc': best_val_bal_acc,
        'test_accuracy': test_metrics['accuracy'],
        'test_balanced_accuracy': test_metrics['balanced_accuracy'],
        'test_f1_score': test_metrics['f1_score'],
        'test_sensitivity': test_metrics['sensitivity'],
        'test_specificity': test_metrics['specificity'],
        'test_roc_auc': test_metrics['roc_auc'],
        'experiment_dir': exp_dir
    }
    
    return df_history, summary


# %% =====================================================
# ANA ÇALIŞTIRMA FONKSİYONU
# ========================================================

def main():
    """
    Ana çalıştırma fonksiyonu.
    
    Adımlar:
    1. Seed'leri sabitler (reproducibility)
    2. Veriyi yükler ve böler
    3. Tüm deney kombinasyonlarını çalıştırır
    4. Sonuçları özet Excel'e yazar
    """
    print("=" * 60)
    print("CHEST X-RAY SINIFLANDIRMA - ÇOKLU DENEY PIPELINE v2")
    print("=" * 60)
    print(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ===== SEED VE CİHAZ =====
    seed_everything(CONFIG["split"]["seed"])
    device = get_device()
    print()
    
    # ===== VERİYİ YÜKLE =====
    print("📂 Veri yükleniyor...")
    paths, labels = list_images(CONFIG["dataset_root"], CONFIG["class_names"])
    print(f"   Toplam görüntü: {len(paths)}")
    print(f"   Sınıf dağılımı: {dict(zip(CONFIG['class_names'], np.bincount(labels)))}")
    print()
    
    # ===== VERİYİ BÖL =====
    print("✂️ Veri bölünüyor (stratified split)...")
    split_cfg = CONFIG["split"]
    train_data, val_data, test_data = stratified_split(
        paths, labels,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=split_cfg["seed"]
    )
    
    print(f"   Train: {len(train_data[0])} örnek")
    print(f"   Val:   {len(val_data[0])} örnek")
    print(f"   Test:  {len(test_data[0])} örnek")
    print()
    
    # ===== DENEY GRID'İ =====
    exp_cfg = CONFIG["experiments"]
    input_sizes = exp_cfg["input_sizes"]
    model_names = exp_cfg["models"]
    aug_levels = exp_cfg["augmentation_levels"]
    
    total_experiments = len(input_sizes) * len(model_names) * len(aug_levels)
    print(f"🧪 Toplam {total_experiments} deney çalıştırılacak:")
    print(f"   Boyutlar: {input_sizes}")
    print(f"   Modeller: {model_names}")
    print(f"   Augmentasyon: {aug_levels}")
    print()
    
    # ===== ÇIKTI DİZİNİ =====
    output_dir = CONFIG["output"]["save_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Çıktı dizini: {output_dir}")
    print()
    
    # ===== TÜM DENEYLERİ ÇALIŞTIR =====
    all_summaries = []
    all_histories = {}
    experiment_idx = 0
    
    total_start_time = time.time()
    
    for input_size in input_sizes:
        for model_name in model_names:
            for aug_level in aug_levels:
                experiment_idx += 1
                exp_name = create_experiment_name(model_name, input_size, aug_level)
                
                print(f"\n{'#'*60}")
                print(f"DENEY {experiment_idx}/{total_experiments}: {exp_name}")
                print(f"{'#'*60}")
                
                try:
                    df_history, summary = train_single_experiment(
                        exp_name=exp_name,
                        model_name=model_name,
                        input_size=input_size,
                        aug_level=aug_level,
                        train_data=train_data,
                        val_data=val_data,
                        test_data=test_data,
                        class_names=CONFIG["class_names"],
                        device=device,
                        output_base_dir=output_dir
                    )
                    
                    all_summaries.append(summary)
                    all_histories[exp_name] = df_history
                    
                except Exception as e:
                    print(f"❌ HATA: {exp_name}")
                    print(f"   {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    total_time = time.time() - total_start_time
    
    # ===== ÖZET EXCEL OLUŞTUR =====
    print("\n" + "=" * 60)
    print("📊 SONUÇ ÖZETİ")
    print("=" * 60)
    
    if len(all_summaries) > 0:
        df_summary = pd.DataFrame(all_summaries)
        
        # En iyi sonuçlara göre sırala
        df_summary = df_summary.sort_values(
            by=['test_balanced_accuracy', 'test_f1_score'],
            ascending=False
        ).reset_index(drop=True)
        
        # Top 5'i göster
        print("\n🏆 EN İYİ 5 DENEY (Test Balanced Accuracy'e göre):")
        print("-" * 80)
        display_cols = [
            'experiment_name', 'best_epoch', 'best_val_balanced_acc',
            'test_balanced_accuracy', 'test_f1_score', 'test_sensitivity', 'test_specificity'
        ]
        print(df_summary.head(5)[display_cols].to_string(index=False))
        
        # Özet Excel kaydet
        summary_excel_path = os.path.join(output_dir, "experiments_summary.xlsx")
        
        with pd.ExcelWriter(summary_excel_path, engine='openpyxl') as writer:
            # Ana özet sayfası
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Her deney için ayrı sayfa (epoch logları)
            for exp_name, df_hist in all_histories.items():
                # Excel sheet isim limiti: 31 karakter
                sheet_name = exp_name[:31]
                df_hist.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\n✅ Özet Excel kaydedildi: {summary_excel_path}")
    
    print(f"\n⏱️ Toplam süre: {format_time(total_time)}")
    print(f"📅 Bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Tüm deneyler tamamlandı!")


# %% =====================================================
# ÇALIŞTIR
# ========================================================

if __name__ == "__main__":
    main()

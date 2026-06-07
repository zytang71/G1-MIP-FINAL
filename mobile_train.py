import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ALL_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_CLASSES = len(ALL_DISEASES)

TRAIN_CSV = "Data/train_list.csv"
VALID_CSV = "Data/valid_list.csv"
TRAIN_DIR = "split_dataset/train"
VALID_DIR = "split_dataset/valid"

IMG_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 5e-5
MAX_POS_WEIGHT = 10.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MONITOR_DISEASES = [disease for disease in ALL_DISEASES if disease != "Hernia"]


# ==========================================
# 1. 影像前處理 CLAHE
# ==========================================
class ApplyCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)


# ==========================================
# 2. Early Stopping
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7, path="best_model_multilabel.pth", mode="max"):
        self.patience = patience
        self.path = path
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def _is_improved(self, current_value):
        if self.best_value is None:
            return True
        if self.mode == "min":
            return current_value < self.best_value
        return current_value > self.best_value

    def __call__(self, current_value, model):
        if self._is_improved(current_value):
            self.best_value = current_value
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"  -> Early Stopping 計數: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f"  儲存目前最佳模型到 {self.path}")


# ==========================================
# 3. 多標籤資料集
# ==========================================
class ChestXrayMultiLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        findings = str(self.data.iloc[idx]["Finding Labels"]).split("|")
        label_list = [1.0 if disease in findings else 0.0 for disease in ALL_DISEASES]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_list, dtype=torch.float32)


def compute_pos_weights(csv_file):
    df = pd.read_csv(csv_file)
    findings = df["Finding Labels"].fillna("")
    weights = []

    print("\n[根據訓練集自動計算各類別 pos_weight]")
    print("  公式: pos_weight = negative_count / positive_count")
    print(f"  為避免極少數類別權重過大，這裡設定上限為 {MAX_POS_WEIGHT:.1f}")

    for disease in ALL_DISEASES:
        positive_count = findings.str.split("|").apply(lambda labels: disease in labels).sum()
        negative_count = len(df) - positive_count
        raw_weight = negative_count / max(positive_count, 1)
        clipped_weight = min(raw_weight, MAX_POS_WEIGHT)
        weights.append(clipped_weight)
        print(
            f"  - {disease}: positive={positive_count}, negative={negative_count}, "
            f"raw={raw_weight:.4f}, used={clipped_weight:.4f}"
        )

    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def compute_macro_metrics(preds, labels, disease_names):
    precisions = []
    recalls = []
    f1_scores = []

    for disease in disease_names:
        idx = ALL_DISEASES.index(disease)
        tp = ((preds[:, idx] == 1) & (labels[:, idx] == 1)).sum()
        fp = ((preds[:, idx] == 1) & (labels[:, idx] == 0)).sum()
        fn = ((preds[:, idx] == 0) & (labels[:, idx] == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
    }


# ==========================================
# 4. 訓練流程
# ==========================================
def train():
    print(f"開始進行 MobileNetV2 多標籤訓練 | 裝置: {DEVICE}")

    train_trans = transforms.Compose(
        [
            ApplyCLAHE(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_trans = transforms.Compose(
        [
            ApplyCLAHE(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(
        ChestXrayMultiLabelDataset(TRAIN_CSV, TRAIN_DIR, train_trans),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ChestXrayMultiLabelDataset(VALID_CSV, VALID_DIR, val_trans),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[0] = nn.Dropout(p=0.5, inplace=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.to(DEVICE)

    pos_weight = compute_pos_weights(TRAIN_CSV)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=10, path="best_model_multilabel.pth", mode="max")

    for epoch in range(MAX_EPOCHS):
        start_time = time.time()

        # --- 訓練階段 ---
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss = criterion(logits, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, lbls)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        # --- 驗證階段 ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_lbls = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        logits = model(imgs)
                        loss = criterion(logits, lbls)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, lbls)

                val_loss += loss.item()
                preds = (logits > 0).float()
                all_preds.append(preds.cpu().numpy())
                all_lbls.append(lbls.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_lbls = np.vstack(all_lbls)

        full_metrics = compute_macro_metrics(all_preds, all_lbls, ALL_DISEASES)
        core_metrics = compute_macro_metrics(all_preds, all_lbls, MONITOR_DISEASES)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(
            f"  全 14 類 Macro -> P: {full_metrics['precision']:.2%} | "
            f"R: {full_metrics['recall']:.2%} | F1: {full_metrics['f1']:.2%}"
        )
        print(
            f"  排除 Hernia -> P: {core_metrics['precision']:.2%} | "
            f"R: {core_metrics['recall']:.2%} | F1: {core_metrics['f1']:.2%}"
        )

        scheduler.step(core_metrics["f1"])
        early_stopping(core_metrics["f1"], model)

        if early_stopping.early_stop:
            print(f">>> 排除 Hernia 的驗證 Macro-F1 連續 {early_stopping.patience} 個 epoch 沒有進步，提前停止訓練。")
            break

        print(f"  耗時: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    train()

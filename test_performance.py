import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models, transforms

# ==========================================
# 1. 測試資料與模型設定
# ==========================================
TEST_DIR = "split_dataset/test"
TEST_CSV = "Data/test_list.csv"
MODEL_PATH = "best_model_multilabel.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
MONITOR_DISEASES = [disease for disease in ALL_DISEASES if disease != "Hernia"]
NUM_CLASSES = len(ALL_DISEASES)


# ==========================================
# 2. 前處理與資料集
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


class ChestXrayTestMultiLabelDataset(torch.utils.data.Dataset):
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


def compute_per_class_metrics(all_labels, all_preds):
    metrics = {}

    for i, disease in enumerate(ALL_DISEASES):
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i], labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[disease] = {
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "f1": f1,
        }

    return metrics


def summarize_macro(metrics, disease_names):
    return {
        "precision": float(np.mean([metrics[d]["precision"] for d in disease_names])),
        "recall": float(np.mean([metrics[d]["recall"] for d in disease_names])),
        "specificity": float(np.mean([metrics[d]["specificity"] for d in disease_names])),
        "f1": float(np.mean([metrics[d]["f1"] for d in disease_names])),
    }


# ==========================================
# 3. 測試流程
# ==========================================
def evaluate_test_set():
    # 與訓練一致，保留 CLAHE / Resize / Normalize，只拿掉隨機增強
    test_transforms = transforms.Compose(
        [
            ApplyCLAHE(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_loader = DataLoader(
        ChestXrayTestMultiLabelDataset(TEST_CSV, TEST_DIR, test_transforms),
        batch_size=32,
        shuffle=False,
    )

    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []

    print(f"開始測試 MobileNetV2 ({len(test_loader.dataset)} 張影像)，正在進行多標籤評估...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    metrics = compute_per_class_metrics(all_labels, all_preds)
    full_macro = summarize_macro(metrics, ALL_DISEASES)
    core_macro = summarize_macro(metrics, MONITOR_DISEASES)

    print("\n" + "=" * 60)
    print("多標籤疾病分類 - MobileNet 測試結果")
    print("=" * 60)
    print(f"{'Disease':<20} | {'Recall':<8} | {'Precision':<9} | {'Specificity':<11} | {'F1-Score':<8}")
    print("-" * 65)

    for disease in ALL_DISEASES:
        result = metrics[disease]
        print(
            f"{disease:<20} | "
            f"{result['recall']:>7.2%} | "
            f"{result['precision']:>8.2%} | "
            f"{result['specificity']:>10.2%} | "
            f"{result['f1']:>7.2%}"
        )

    print("-" * 65)
    print(
        f"{'全 14 類 Macro':<20} | "
        f"{full_macro['recall']:>7.2%} | "
        f"{full_macro['precision']:>8.2%} | "
        f"{full_macro['specificity']:>10.2%} | "
        f"{full_macro['f1']:>7.2%}"
    )
    print(
        f"{'排除 Hernia':<20} | "
        f"{core_macro['recall']:>7.2%} | "
        f"{core_macro['precision']:>8.2%} | "
        f"{core_macro['specificity']:>10.2%} | "
        f"{core_macro['f1']:>7.2%}"
    )
    print("=" * 60)


if __name__ == "__main__":
    evaluate_test_set()

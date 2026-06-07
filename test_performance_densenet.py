import os

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
MODEL_PATH = "best_densenet_multilabel.pth"
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
NUM_CLASSES = len(ALL_DISEASES)


# ==========================================
# 2. 測試資料集
# ==========================================
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


# ==========================================
# 3. 測試流程
# ==========================================
def evaluate_test_set():
    test_transforms = transforms.Compose(
        [
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

    model = models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    print(f"開始測試 DenseNet-121 ({len(test_loader.dataset)} 張影像)，正在進行多標籤評估...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)

    print("\n" + "=" * 60)
    print("多標籤疾病分類 - DenseNet 測試結果")
    print("=" * 60)
    print(f"{'Disease':<20} | {'Recall':<8} | {'Precision':<9} | {'Specificity':<11} | {'F1-Score':<8}")
    print("-" * 65)

    recalls = []
    precisions = []
    specificities = []
    f1_scores = []

    for i, disease in enumerate(ALL_DISEASES):
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i], labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        recalls.append(recall)
        precisions.append(precision)
        specificities.append(specificity)
        f1_scores.append(f1)

        print(f"{disease:<20} | {recall:>7.2%} | {precision:>8.2%} | {specificity:>10.2%} | {f1:>7.2%}")

    print("-" * 65)
    print(
        f"{'Macro Average':<20} | "
        f"{np.mean(recalls):>7.2%} | "
        f"{np.mean(precisions):>8.2%} | "
        f"{np.mean(specificities):>10.2%} | "
        f"{np.mean(f1_scores):>7.2%}"
    )
    print("=" * 60)


if __name__ == "__main__":
    evaluate_test_set()

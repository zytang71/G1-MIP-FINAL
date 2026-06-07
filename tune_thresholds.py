import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
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
VALID_DIR = "split_dataset/valid"
VALID_CSV = "Data/valid_list.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)


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


class ChestXrayValidMultiLabelDataset(Dataset):
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


def build_model(backbone):
    if backbone == "mobilenet":
        model = models.mobilenet_v2()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model

    if backbone == "densenet":
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")


def compute_prf(labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return precision, recall, f1, specificity


def summarize_macro(probabilities, labels, thresholds):
    precisions = []
    recalls = []
    f1_scores = []

    for idx in range(NUM_CLASSES):
        preds = (probabilities[:, idx] >= thresholds[idx]).astype(np.float32)
        precision, recall, f1, _ = compute_prf(labels[:, idx], preds)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1_scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="Tune per-class thresholds on validation set.")
    parser.add_argument("--backbone", choices=["mobilenet", "densenet"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    val_transforms = transforms.Compose(
        [
            ApplyCLAHE(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_loader = DataLoader(
        ChestXrayValidMultiLabelDataset(VALID_CSV, VALID_DIR, val_transforms),
        batch_size=32,
        shuffle=False,
    )

    model = build_model(args.backbone)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_probs = []

    print(f"開始在 valid set 上調整 {args.backbone} 的各類別 threshold...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    default_thresholds = [0.5] * NUM_CLASSES
    default_macro = summarize_macro(all_probs, all_labels, default_thresholds)

    tuned_thresholds = []
    per_class_results = {}

    print("\n[各類別最佳 threshold]")
    for idx, disease in enumerate(ALL_DISEASES):
        best_threshold = 0.5
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0
        best_specificity = 0.0

        for threshold in THRESHOLD_GRID:
            preds = (all_probs[:, idx] >= threshold).astype(np.float32)
            precision, recall, f1, specificity = compute_prf(all_labels[:, idx], preds)
            if f1 > best_f1:
                best_threshold = float(threshold)
                best_f1 = float(f1)
                best_precision = float(precision)
                best_recall = float(recall)
                best_specificity = float(specificity)

        tuned_thresholds.append(best_threshold)
        per_class_results[disease] = {
            "threshold": best_threshold,
            "precision": best_precision,
            "recall": best_recall,
            "specificity": best_specificity,
            "f1": best_f1,
        }
        print(
            f"  - {disease}: threshold={best_threshold:.2f}, "
            f"P={best_precision:.2%}, R={best_recall:.2%}, F1={best_f1:.2%}"
        )

    tuned_macro = summarize_macro(all_probs, all_labels, tuned_thresholds)

    payload = {
        "backbone": args.backbone,
        "checkpoint": args.checkpoint,
        "thresholds": {disease: threshold for disease, threshold in zip(ALL_DISEASES, tuned_thresholds)},
        "default_macro": default_macro,
        "tuned_macro": tuned_macro,
        "per_class": per_class_results,
    }

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print("\n[Macro 指標比較]")
    print(
        f"  預設 0.5 -> P: {default_macro['precision']:.2%} | "
        f"R: {default_macro['recall']:.2%} | F1: {default_macro['f1']:.2%}"
    )
    print(
        f"  調整後  -> P: {tuned_macro['precision']:.2%} | "
        f"R: {tuned_macro['recall']:.2%} | F1: {tuned_macro['f1']:.2%}"
    )
    print(f"\n已輸出 threshold 設定到 {args.output}")


if __name__ == "__main__":
    main()

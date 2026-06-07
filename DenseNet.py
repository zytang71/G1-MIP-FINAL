import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report







##                  要先整理資料夾，然後接著把改好的模型拿去試試看








# ==========================================
# 1. 參數設定
# ==========================================
TRAIN_DIR = '../split_dataset/train'
VALID_DIR = '../split_dataset/valid'
TEST_DIR = '../split_dataset/test'

TRAIN_CSV = '../Data/train_list.csv'
VALID_CSV = '../Data/valid_list.csv'
TEST_CSV = '../Data/test_list.csv'

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
THRESHOLD = 0.5

ALL_DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
                'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
                'Pleural_Thickening', 'Hernia']
RED_INDICES = [2, 4, 6, 7, 8, 9]

# Early Stopping 設定
PATIENCE = 7
MIN_DELTA = 1e-4

MODEL_PATH = 'densenet121_baseline_best_val_loss.pth'
RESULT_CSV = 'densenet121_baseline_test_results.csv'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. Dataset
# ==========================================
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])

        image = Image.open(img_name).convert('RGB')
        label_str = str(self.data.iloc[idx]['target'])
        label_list = [float(x) for x in label_str.split(',')]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_list, dtype=torch.float32)


# ==========================================
# 3. Transform
# ==========================================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. 建立 DenseNet-121
# ==========================================
def build_model():
    model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1
    )

    num_ftrs = model.classifier.in_features

    # 多標籤分類，輸出 14 個 logits (不加 Sigmoid)
    model.classifier = nn.Linear(num_ftrs, 14)

    return model.to(DEVICE)


# ==========================================
# 5. 模型複雜度
# ==========================================
def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_model_size_mb(model):
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024 / 1024


# ==========================================
# 6. 指標計算
# ==========================================
def calculate_metrics_from_counts(tp, fn, fp, tn):
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1": f1
    }


# ==========================================
# 7. 驗證 / 測試共用函式
# ==========================================
def evaluate_model(model, data_loader, criterion=None, threshold=0.5, measure_time=False):
    model.eval()

    total_loss = 0.0
    all_severe_labels = []
    all_severe_preds = []

    total_inference_time = 0.0

    # GPU warm-up，避免第一次推論包含 CUDA 初始化時間
    if measure_time and DEVICE.type == "cuda":
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        torch.cuda.synchronize()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if measure_time and DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            logits = model(images)  # Shape: (Batch, 14)

            if measure_time and DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            if measure_time:
                total_inference_time += end_time - start_time

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()

            probs = torch.sigmoid(logits)  # Shape: (Batch, 14)

            # --- 最終嚴重性判定邏輯 ---
            # 只要有任何一個紅燈疾病的機率 >= threshold，就判斷為嚴重 (1)
            batch_severe_preds = (probs[:, RED_INDICES].max(dim=1).values >= threshold).float()
            # 只要真實標籤中有任何一個紅燈疾病是 1，就是嚴重 (1)
            batch_severe_labels = labels[:, RED_INDICES].max(dim=1).values

            all_severe_labels.extend(batch_severe_labels.cpu().numpy())
            all_severe_preds.extend(batch_severe_preds.cpu().numpy())

    cm = confusion_matrix(all_severe_labels, all_severe_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = calculate_metrics_from_counts(tp, fn, fp, tn)

    metrics["tp"] = tp
    metrics["fn"] = fn
    metrics["fp"] = fp
    metrics["tn"] = tn

    if criterion is not None:
        metrics["loss"] = total_loss / len(data_loader)
    else:
        metrics["loss"] = None

    if measure_time:
        num_images = len(data_loader.dataset)
        avg_time_per_image = total_inference_time / num_images
        throughput = num_images / total_inference_time

        metrics["total_inference_time_sec"] = total_inference_time
        metrics["avg_time_per_image_ms"] = avg_time_per_image * 1000
        metrics["throughput_images_per_sec"] = throughput
    else:
        metrics["total_inference_time_sec"] = None
        metrics["avg_time_per_image_ms"] = None
        metrics["throughput_images_per_sec"] = None

    metrics["all_labels"] = all_severe_labels
    metrics["all_preds"] = all_severe_preds

    return metrics


# ==========================================
# 8. 訓練 DenseNet baseline
# ==========================================
def train():
    train_dataset = ChestXrayDataset(TRAIN_CSV, TRAIN_DIR, train_transforms)
    valid_dataset = ChestXrayDataset(VALID_CSV, VALID_DIR, eval_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    model = build_model()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params, trainable_params = count_model_parameters(model)
    model_size_mb = get_model_size_mb(model)

    print("=" * 80)
    print("DenseNet-121 Baseline Training")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Early Stopping Patience: {PATIENCE}")
    print(f"Min Delta: {MIN_DELTA}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Estimated Model Size: {model_size_mb:.2f} MB")
    print("=" * 80)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -------------------------
        # Validation
        # -------------------------
        valid_metrics = evaluate_model(
            model=model,
            data_loader=valid_loader,
            criterion=criterion,
            threshold=THRESHOLD,
            measure_time=False
        )

        epoch_time = time.time() - epoch_start

        print("=" * 80)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Time: {epoch_time:.1f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {valid_metrics['loss']:.4f}")
        print(
            f"Val Accuracy: {valid_metrics['accuracy']:.4f} | "
            f"Val Recall: {valid_metrics['recall']:.4f} | "
            f"Val Precision: {valid_metrics['precision']:.4f} | "
            f"Val Specificity: {valid_metrics['specificity']:.4f} | "
            f"Val F1: {valid_metrics['f1']:.4f}"
        )
        print(
            f"TP: {valid_metrics['tp']} | "
            f"FN: {valid_metrics['fn']} | "
            f"FP: {valid_metrics['fp']} | "
            f"TN: {valid_metrics['tn']}"
        )

        # -------------------------
        # Save best model by validation loss
        # -------------------------
        if valid_metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = valid_metrics["loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "threshold": THRESHOLD,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "val_accuracy": valid_metrics["accuracy"],
                "val_recall": valid_metrics["recall"],
                "val_precision": valid_metrics["precision"],
                "val_specificity": valid_metrics["specificity"],
                "val_f1": valid_metrics["f1"],
                "val_tp": valid_metrics["tp"],
                "val_fn": valid_metrics["fn"],
                "val_fp": valid_metrics["fp"],
                "val_tn": valid_metrics["tn"],
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_size_mb": model_size_mb
            }, MODEL_PATH)

            print(f">> Val loss 下降，DenseNet baseline 權重已儲存：{MODEL_PATH}")

        else:
            epochs_without_improvement += 1

            print(
                f">> Val loss 未明顯改善，"
                f"Early Stopping 計數：{epochs_without_improvement}/{PATIENCE}"
            )

            if epochs_without_improvement >= PATIENCE:
                print(">> 觸發 Early Stopping，停止訓練")
                break

    print("=" * 80)
    print("DenseNet baseline 訓練完成")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Model Path: {MODEL_PATH}")
    print("=" * 80)


# ==========================================
# 9. 測試 DenseNet baseline
# ==========================================
def test():
    test_dataset = ChestXrayDataset(TEST_CSV, TEST_DIR, eval_transforms)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    model = build_model()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        threshold = checkpoint.get("threshold", THRESHOLD)
        best_epoch = checkpoint.get("best_epoch", None)
        best_val_loss = checkpoint.get("best_val_loss", None)
    else:
        model.load_state_dict(checkpoint)
        threshold = THRESHOLD
        best_epoch = None
        best_val_loss = None

    model.to(DEVICE)
    model.eval()

    total_params, trainable_params = count_model_parameters(model)
    model_size_mb = get_model_size_mb(model)

    print("=" * 80)
    print("DenseNet-121 Baseline Test")
    print("=" * 80)
    print(f"Test Images: {len(test_loader.dataset)}")
    print(f"Device: {DEVICE}")
    print(f"Threshold: {threshold}")
    print(f"Loaded Model: {MODEL_PATH}")

    if best_epoch is not None:
        print(f"Best Epoch from Validation: {best_epoch}")

    if best_val_loss is not None:
        print(f"Best Validation Loss: {best_val_loss:.4f}")

    print("=" * 80)

    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=None,
        threshold=threshold,
        measure_time=True
    )

    print("\n" + "=" * 80)
    print("【DenseNet-121 Baseline - Test Set Report】")
    print("=" * 80)

    print("Confusion Matrix:")
    print(f"TP，紅燈抓對: {test_metrics['tp']}")
    print(f"FN，重症漏抓: {test_metrics['fn']}  <-- 越低越安全")
    print(f"FP，綠燈誤報: {test_metrics['fp']}")
    print(f"TN，綠燈抓對: {test_metrics['tn']}")

    print("-" * 80)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Sensitivity / Recall: {test_metrics['recall']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"F1-score: {test_metrics['f1']:.4f}")

    print("-" * 80)
    print("Efficiency Metrics:")
    print(f"Total Inference Time: {test_metrics['total_inference_time_sec']:.4f} 秒")
    print(f"Average Time per Image: {test_metrics['avg_time_per_image_ms']:.4f} ms/image")
    print(f"Throughput: {test_metrics['throughput_images_per_sec']:.2f} images/sec")
    print(f"Batch Size: {BATCH_SIZE}")

    print("-" * 80)
    print("Model Complexity:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Estimated Model Size: {model_size_mb:.2f} MB")

    print("-" * 80)
    print("Classification Report:")
    print(classification_report(
        test_metrics["all_labels"],
        test_metrics["all_preds"],
        target_names=["Green / Negative", "Red / Positive"],
        digits=4,
        zero_division=0
    ))

    print("=" * 80)

    result_df = pd.DataFrame([{
        "model": "DenseNet-121 Baseline",
        "threshold": threshold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_images": len(test_loader.dataset),
        "tp": test_metrics["tp"],
        "fn": test_metrics["fn"],
        "fp": test_metrics["fp"],
        "tn": test_metrics["tn"],
        "accuracy": test_metrics["accuracy"],
        "recall": test_metrics["recall"],
        "precision": test_metrics["precision"],
        "specificity": test_metrics["specificity"],
        "f1": test_metrics["f1"],
        "total_inference_time_sec": test_metrics["total_inference_time_sec"],
        "avg_time_per_image_ms": test_metrics["avg_time_per_image_ms"],
        "throughput_images_per_sec": test_metrics["throughput_images_per_sec"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE)
    }])

    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    print(f"測試結果已儲存為：{RESULT_CSV}")


# ==========================================
# 10. 主程式
# ==========================================
if __name__ == "__main__":
    # train()
    test()
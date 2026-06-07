import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os
from PIL import Image

# ==========================================
# 1. 參數與路徑
# ==========================================
TEST_DIR = 'split_dataset/test'
TEST_CSV = 'Data/test_list.csv'
MODEL_PATH = 'best_model_multilabel.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
    'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(ALL_DISEASES)

# ==========================================
# 2. 測試集讀取器
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
        image = Image.open(img_name).convert('RGB')
        
        # 多標籤處理
        findings = str(self.data.iloc[idx]['Finding Labels']).split('|')
        label_list = [1.0 if disease in findings else 0.0 for disease in ALL_DISEASES]
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_list, dtype=torch.float32)

# ==========================================
# 3. 測試執行
# ==========================================
def evaluate_test_set():
    # 影像轉換 (與驗證集一致，這裡不套用訓練時的隨機增強)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # 訓練若是 448 則此處需改 448
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_loader = DataLoader(ChestXrayTestMultiLabelDataset(TEST_CSV, TEST_DIR, test_transforms), 
                              batch_size=32, shuffle=False)

    # 載入模型結構與權重
    from torchvision import models
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    print(f"正在對測試集 ({len(test_loader.dataset)} 張影像) 進行最終多標籤評估...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            
            # 使用 Sigmoid 將 Logits 轉為機率
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)

    print("\n" + "="*60)
    print("【多標籤疾病分類系統 - 測試集各類別評估報告】")
    print("="*60)
    print(f"{'Disease':<20} | {'Recall':<8} | {'Precision':<9} | {'Specificity':<11} | {'F1-Score':<8}")
    print("-" * 65)

    recalls, precisions, specificities, f1_scores = [], [], [], []

    for i, disease in enumerate(ALL_DISEASES):
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i], labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # 如果測試集中某類別全為0或全為1，會回傳 1x1
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
    print(f"{'Macro Average':<20} | {np.mean(recalls):>7.2%} | {np.mean(precisions):>8.2%} | {np.mean(specificities):>10.2%} | {np.mean(f1_scores):>7.2%}")
    print("="*60)

if __name__ == "__main__":
    evaluate_test_set()
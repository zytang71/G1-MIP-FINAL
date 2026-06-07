import os
# 解決 OpenMP 重複初始化問題
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import time
import cv2
import numpy as np

# ==========================================
# 1. 自定義 CLAHE 預處理類別 (醫療影像增強)
# ==========================================
class ApplyCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        # 轉換至 LAB 空間進行亮度增強，避免色彩失真
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)

# ==========================================
# 2. 參數設定 (配合 RTX 5080 算力)
# ==========================================
TRAIN_DIR = 'split_dataset/train'
VALID_DIR = 'split_dataset/valid'
TRAIN_CSV = 'Data/train_list.csv'  
VALID_CSV = 'Data/valid_list.csv'

IMG_SIZE = 448        # 提升至 448x448 以獲取更多細節
BATCH_SIZE = 32      # 5080 顯存充足，甚至可以試試 64
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. 數據增強與預處理
# ==========================================
train_transforms = transforms.Compose([
    ApplyCLAHE(),                       # 第一步：對比度增強
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),       # 模擬拍攝角度傾斜
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 隨機平移
    transforms.ColorJitter(brightness=0.2, contrast=0.2),    # 亮度隨機化
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    ApplyCLAHE(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. Dataset 類別
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
        label = self.data.iloc[idx]['target'] # 讀取 0/1 二元標籤
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 5. 模型初始化 (MobileNetV2)
# ==========================================
def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    # 輸出層改為單個神經元 + Sigmoid (二元分類)
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model.to(DEVICE)

# ==========================================
# 6. 訓練主流程
# ==========================================
def train():
    print(f"環境檢查：正在使用 {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    train_loader = DataLoader(ChestXrayDataset(TRAIN_CSV, TRAIN_DIR, train_transforms), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(ChestXrayDataset(VALID_CSV, VALID_DIR, val_transforms), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model()
    criterion = nn.BCELoss()
    # 加入 Weight Decay 防止模型在細節上過擬合
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # 學習率排程：若 Recall 3 輪不進步則調降學習率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    best_recall = 0.0

    print("-" * 30)
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 驗證階段
        model.eval()
        val_tp, val_fn, val_fp, val_tn = 0, 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images).squeeze()
                preds = (outputs > 0.5).float() # 預設分流門檻 0.5
                
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()

        # 指標計算
        recall = val_tp / (val_tp + val_fn + 1e-8)
        precision = val_tp / (val_tp + val_fp + 1e-8)
        avg_loss = running_loss / len(train_loader)
        
        # 步進學習率
        scheduler.step(recall)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Recall: {recall:.4%}")
        print(f"  TP: {val_tp}, FN: {val_fn} (漏抓), FP: {val_fp}, TN: {val_tn}")

        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), 'best_recall_model_v2.pth')
            print(f"  >> 敏感度突破！模型已存儲。時間: {time.time()-start_time:.1f}s")

    print("-" * 30)
    print(f"訓練結束！最佳 Recall: {best_recall:.4%}")

if __name__ == "__main__":
    train()
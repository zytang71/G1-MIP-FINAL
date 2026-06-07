import os
# 解決 OpenMP 與重複初始化問題
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

ALL_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
    'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(ALL_DISEASES)

# ==========================================
# 1. 影像預處理：CLAHE
# ==========================================
class ApplyCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)

# ==========================================
# 2. 早停機制 (Early Stopping)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7, path='best_model_multilabel.pth'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            print(f"  -> 早停計數器: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f"  ⭐ 驗證損失下降，最佳權重已儲存至: {self.path}")

# ==========================================
# 3. 數據集與參數設定
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
        image = Image.open(img_name).convert('RGB')
        
        # 讀取 Finding Labels，將其轉為 14 維 Multi-hot Encoding
        findings = str(self.data.iloc[idx]['Finding Labels']).split('|')
        label_list = [1.0 if disease in findings else 0.0 for disease in ALL_DISEASES]
        
        if self.transform: 
            image = self.transform(image)
            
        return image, torch.tensor(label_list, dtype=torch.float32)

IMG_SIZE = 224 # 如果顯存夠可以調高至 448
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 4. 訓練流程
# ==========================================
def train():
    print(f"🚀 啟動多標籤優化訓練流程 | 裝置: {DEVICE}")

    # 資料轉換與載入
    train_trans = transforms.Compose([
        ApplyCLAHE(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_trans = transforms.Compose([
        ApplyCLAHE(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(ChestXrayMultiLabelDataset('Data/train_list.csv', 'split_dataset/train', train_trans), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ChestXrayMultiLabelDataset('Data/valid_list.csv', 'split_dataset/valid', val_trans), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 建立模型：使用 MobileNetV2，輸出 14 個 logits (配合 BCEWithLogitsLoss)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # 增加 Dropout 比例以減緩 Overfitting
    model.classifier[0] = nn.Dropout(p=0.5, inplace=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES) 
    model.to(DEVICE)

    # 損失函數與優化器
    # 針對極度不平衡的資料集，給予正樣本 10 倍的懲罰權重
    pos_weight = torch.ones([NUM_CLASSES]).to(DEVICE) * 10.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    # 提高 weight_decay (L2正則化) 來減輕 Overfitting (從 1e-5 提高到 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    # 放寬 early stopping，避免太早停下來
    early_stopping = EarlyStopping(patience=15, path='best_model_multilabel.pth')

    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        
        # --- 訓練階段 ---
        model.train()
        train_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
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
        v_loss = 0
        all_preds = []
        all_lbls = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(imgs)
                        loss = criterion(logits, lbls)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, lbls)
                    
                v_loss += loss.item()
                
                # 收集預測與真實標籤以計算 Macro-Recall
                preds = (logits > 0).float() # Logits > 0 等同於 Probability > 0.5
                all_preds.append(preds.cpu().numpy())
                all_lbls.append(lbls.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_lbls = np.vstack(all_lbls)
        
        # 簡單計算平均 Recall (Macro-Recall)
        recalls = []
        for i in range(NUM_CLASSES):
            tp = ((all_preds[:, i] == 1) & (all_lbls[:, i] == 1)).sum()
            fn = ((all_preds[:, i] == 0) & (all_lbls[:, i] == 1)).sum()
            if (tp + fn) > 0:
                recalls.append(tp / (tp + fn))
        macro_recall = np.mean(recalls) if recalls else 0.0

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = v_loss / len(val_loader)

        print(f"Epoch [{epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Macro-Recall: {macro_recall:.2%}")

        # 步進學習率與早停判斷
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print(f">>> 驗證集損失連續 {early_stopping.patience} 輪未下降，啟動早停。")
            break
            
        print(f"  耗時: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    train()
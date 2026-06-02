import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import time

# ==========================================
# 1. 參數設定
# ==========================================
TRAIN_DIR = 'split_dataset/train'
VALID_DIR = 'split_dataset/valid'
TRAIN_CSV = 'train_list/train_list.csv'
VALID_CSV = 'train_list/valid_list.csv'

BATCH_SIZE = 32      # 顯存若不足 (Out of Memory) 請調低至 16
EPOCHS = 15          # 醫療影像通常需要較多輪次來收斂
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 醫療影像專用 Dataset
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
        label = self.data.iloc[idx]['target']
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 3. 數據增強 (Data Augmentation)
# ==========================================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), # 模擬拍攝角度偏差
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. 模型初始化 (MobileNetV2)
# ==========================================
def build_model():
    # 使用預訓練權重
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # 修改最後的分類器
    # MobileNetV2 的 classifier 是一個 Sequential，最後一層在 [1]
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid() # 二元分類輸出 0~1 的機率
    )
    return model.to(DEVICE)

# ==========================================
# 5. 訓練與驗證邏輯
# ==========================================
def train():
    train_loader = DataLoader(ChestXrayDataset(TRAIN_CSV, TRAIN_DIR, train_transforms), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ChestXrayDataset(VALID_CSV, VALID_DIR, val_transforms), 
                            batch_size=BATCH_SIZE, shuffle=False)

    model = build_model()
    criterion = nn.BCELoss() # 二元交叉熵損失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"開始訓練於裝置: {DEVICE}")
    best_recall = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- 訓練階段 ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 驗證階段 ---
        model.eval()
        val_tp, val_fn, val_fp, val_tn = 0, 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images).squeeze()
                preds = (outputs > 0.5).float() # 預設門檻 0.5
                
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()

        # 計算指標
        recall = val_tp / (val_tp + val_fn + 1e-8)
        precision = val_tp / (val_tp + val_fp + 1e-8)
        
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | "
              f"Recall (敏感度): {recall:.4f} | Precision: {precision:.4f} | Time: {duration:.1f}s")

        # 儲存 Recall 表現最好的模型
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), 'best_recall_model.pth')
            print("  >> 敏感度提升，模型已儲存！")

if __name__ == "__main__":
    train()
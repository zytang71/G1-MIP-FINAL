import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
from PIL import Image

# ==========================================
# 1. 參數與路徑
# ==========================================
TEST_DIR = 'split_dataset/test'
TEST_CSV = 'Data/test_list.csv'
MODEL_PATH = 'best_model_standard.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 測試集讀取器
# ==========================================
class ChestXrayTestDataset(torch.utils.data.Dataset):
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
# 3. 測試執行
# ==========================================
def evaluate_test_set():
    # 影像轉換 (與驗證集一致)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_loader = DataLoader(ChestXrayTestDataset(TEST_CSV, TEST_DIR, test_transforms), 
                              batch_size=32, shuffle=False)

    # 載入模型結構與權重
    from torchvision import models
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Sequential(nn.Linear(model.classifier[1].in_features, 1), nn.Sigmoid())
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []

    print(f"正在對測試集 ({len(test_loader.dataset)} 張影像) 進行最終評估...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images).squeeze()
            
            # 分流系統決策閾值 (Threshold)
            # 你可以調整這個數字，例如 0.3 代表更敏感（寧可誤報，不可漏抓）
            preds = (outputs > 0.5).float() 
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # --- 輸出報表 ---
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)

    print("\n" + "="*50)
    print("【紅綠燈分流系統 - 測試集評估報告】")
    print("="*50)
    print(f"真正例 (TP, 紅燈抓對): {tp}")
    print(f"偽陰性 (FN, 重症漏抓): {fn}  <-- 這個越低越安全！")
    print(f"偽陽性 (FP, 綠燈誤報): {fp}")
    print(f"真陰性 (TN, 綠燈抓對): {tn}")
    print("-" * 30)
    print(f"● 敏感度 Sensitivity (Recall): {recall:.2%}")
    print(f"● 特異度 Specificity: {specificity:.2%}")
    print(f"● 精確率 Precision: {precision:.2%}")
    print("="*50)

if __name__ == "__main__":
    evaluate_test_set()
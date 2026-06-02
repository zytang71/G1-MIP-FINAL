import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix
import pandas as pd
from PIL import Image
import cv2
import numpy as np

# ==========================================
# 1. 參數與路徑 (請確認與訓練一致)
# ==========================================
TEST_DIR = 'split_dataset/test'
TEST_CSV = 'train_list/test_list.csv'
MODEL_PATH = 'best_model_standard.pth' # 請確認檔名是否為 best_model_final.pth
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 448 # 必須與訓練時一致

# ==========================================
# 2. 醫療影像增強 CLAHE (必須與訓練一致)
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

class ChestXrayTestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx]['target']
        if self.transform: image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 3. 測試執行
# ==========================================
def evaluate_test_set():
    # 影像轉換同步：加入 CLAHE 並提升至 448
    test_transforms = transforms.Compose([
        ApplyCLAHE(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_loader = DataLoader(ChestXrayTestDataset(TEST_CSV, TEST_DIR, test_transforms), 
                              batch_size=16, shuffle=False, num_workers=4)

    # 載入模型：必須完全符合訓練時的結構 (單層 Linear)
    model = models.mobilenet_v2()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1) # 修改為訓練時的單層 Linear
    
    # 載入權重
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []

    print(f"正在對測試集 ({len(test_loader.dataset)} 張影像) 進行最終評估...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            # 推論
            logits = model(images).squeeze()
            
            # 因為模型輸出是 Logits (未經過 Sigmoid)，所以判定門檻 0.5 對應到 0
            # 若 logits > 0 代表機率 > 0.5
            preds = (logits > 0).float() 
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # --- 輸出報表 ---
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print("\n" + "="*50)
    print("【紅綠燈分流系統 - 測試集最終報告 (V2)】")
    print("="*50)
    print(f"真正例 (TP, 紅燈抓對): {tp}")
    print(f"偽陰性 (FN, 重症漏抓): {fn}  <-- 安全指標關鍵")
    print(f"偽陽性 (FP, 綠燈誤報): {fp}")
    print(f"真陰性 (TN, 綠燈抓對): {tn}")
    print("-" * 30)
    print(f"● 敏感度 Sensitivity (Recall): {recall:.2%}")
    print(f"● 特異度 Specificity: {specificity:.2%}")
    print(f"● 精確率 Precision: {precision:.2%}")
    print("="*50)

if __name__ == "__main__":
    evaluate_test_set()
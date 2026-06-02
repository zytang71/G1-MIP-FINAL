import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 參數與環境設定
# ==========================================
# 請修改此處為你想測試的圖片路徑 (建議從測試集的 TP 清單中挑選)
INPUT_IMAGE_PATH = 'split_dataset/train/00018921_017.png' 
MODEL_WEIGHTS_PATH = 'best_recall_model.pth'
OUTPUT_DIR = 'gradcam_results' # 輸出結果的資料夾

# 決策門檻 (與測試時一致)
DECISION_THRESHOLD = 0.5 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 載入模型結構與權重
# ==========================================
def load_model():
    print(f"正在載入模型權重: {MODEL_WEIGHTS_PATH} 到 {DEVICE}...")
    # 重建 MobileNetV2 結構
    model = models.mobilenet_v2()
    # 修改最後一層分類器為二元分類
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    # 載入權重
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # 設為評估模式
    return model

# ==========================================
# 3. Grad-CAM 核心類別
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None
        
        # 註冊 Hook 來捕捉前向傳播的特徵圖與反向傳播的梯度
        self.target_layer.register_forward_hook(self.save_feature_map)
        # 針對新版 PyTorch 使用 register_full_backward_hook
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_feature_map(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor):
        self.model.zero_grad()
        
        # 1. 前向傳播得到預測機率
        output = self.model(input_tensor)
        
        # 2. 針對「紅燈 (類別1)」進行反向傳播計算梯度
        # 因為是二元分類且輸出經過 Sigmoid，直接對輸出求梯度即可
        output.backward()

        # 3. 計算神經元重要性權重 (GAP over gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 4. 執行特徵圖的加權求和
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze().cpu().numpy()
        
        # 5. 通過 ReLU 唯保留對分類有正向貢獻的區域
        cam = np.maximum(cam, 0)
        
        # 6. 正規化到 0~1 之間
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, output.item()

# ==========================================
# 4. 主程式：執行視覺化
# ==========================================
def main():
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"錯誤：找不到輸入圖片 {INPUT_IMAGE_PATH}")
        return

    # A. 載入模型與準備 Grad-CAM 物件
    model = load_model()
    # 針對 MobileNetV2，選擇 features 的最後一層卷積層 [18]
    target_layer = model.features[18]
    gradcam = GradCAM(model, target_layer)

    # B. 影像預處理
    # 用 PIL 讀取供模型使用
    img_pil = Image.open(INPUT_IMAGE_PATH).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

    # C. 生成熱力圖與機率
    print("正在執行推論與生成熱力圖...")
    heatmap_raw, prob = gradcam(input_tensor)

    # D. 判定紅綠燈與處理圖像
    # 用 OpenCV 讀取原始尺寸供顯示與疊加
    img_bgr = cv2.imread(INPUT_IMAGE_PATH)
    height, width, _ = img_bgr.shape

    # 調整熱力圖大小至原始尺寸
    heatmap_resized = cv2.resize(heatmap_raw, (width, height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # 套用色彩映射 (Jet 模式，紅色代表關注重點)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 將熱力圖疊加在原始圖片上 (設定透明度)
    # 此處僅在判定為「紅燈」時完整疊加，以利臨床診斷重點呈現
    alpha = 0.4
    if prob > DECISION_THRESHOLD:
        status = "RED (Priority)"
        title_color = 'red'
        overlay_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    else:
        status = "GREEN (Routine)"
        title_color = 'green'
        # 綠燈時選擇不疊加熱力圖，保持影像清晰，符合臨床分流邏輯
        overlay_img = img_bgr

    # E. 顯示結果與儲存
    # 將 BGR 轉為 RGB 供 matplotlib 顯示
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original X-ray\nFilename: {os.path.basename(INPUT_IMAGE_PATH)}")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"AI Decision: {status}\nProb for Red: {prob:.4f}", color=title_color, fontweight='bold')
    plt.imshow(overlay_rgb)
    plt.axis('off')

    # 儲存結果
    save_path = os.path.join(OUTPUT_DIR, f"gradcam_{os.path.basename(INPUT_IMAGE_PATH)}")
    plt.savefig(save_path)
    print(f"結果已儲存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
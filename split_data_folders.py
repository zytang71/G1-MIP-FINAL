import pandas as pd
import shutil
import os

# ==========================================
# 1. 路徑設定
# ==========================================
# 原始圖片所在的大資料夾
SOURCE_DIR = 'all_images/' 

# 準備產出的目標根目錄
DEST_ROOT = 'split_dataset/'

# 對應的三個 CSV 檔案名稱
CSV_FILES = {
    'train': 'train_list/train_list.csv',
    'valid': 'train_list/valid_list.csv',
    'test': 'train_list/test_list.csv'
}

# ==========================================
# 2. 執行分類邏輯
# ==========================================
def split_images_by_csv():
    # 檢查原始來源是否存在
    if not os.path.exists(SOURCE_DIR):
        print(f"錯誤：找不到來源資料夾 '{SOURCE_DIR}'，請確認路徑。")
        return

    print("開始執行影像分類作業...")

    for split_name, csv_path in CSV_FILES.items():
        # 建立子資料夾，例如 split_dataset/train/
        target_dir = os.path.join(DEST_ROOT, split_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 讀取 CSV 名單
        if not os.path.exists(csv_path):
            print(f"警告：找不到檔案 {csv_path}，跳過此子集。")
            continue
            
        df = pd.read_csv(csv_path)
        # 假設 CSV 第一欄是 Image Index (檔名)
        image_list = df['Image Index'].tolist()
        
        print(f"正在處理 [{split_name}] 子集：預計處理 {len(image_list)} 張圖片...")
        
        success = 0
        fail = 0
        
        for img_name in image_list:
            src_file = os.path.join(SOURCE_DIR, img_name)
            dst_file = os.path.join(target_dir, img_name)
            
            try:
                # 執行複製 (保留元數據)
                shutil.copy2(src_file, dst_file)
                success += 1
            except FileNotFoundError:
                fail += 1
            except Exception as e:
                print(f"處理 {img_name} 時發生未知錯誤: {e}")
                fail += 1
        
        print(f"  -> [{split_name}] 完成。成功: {success}, 失敗: {fail}")

    print("\n" + "="*50)
    print(f"任務完成！分類後的影像位於: {os.path.abspath(DEST_ROOT)}")
    print("="*50)

if __name__ == "__main__":
    split_images_by_csv()
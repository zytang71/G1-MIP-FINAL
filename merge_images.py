import os
import shutil

# ==========================================
# 1. 路徑設定
# ==========================================
# 存放 image_001 ~ image_012 的根目錄 (通常是目前的目錄)
BASE_DIR = './' 

# 目標資料夾
TARGET_DIR = 'all_images'

# ==========================================
# 2. 執行移動邏輯
# ==========================================
def merge_nih_images():
    # 建立目標資料夾
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"已建立目標資料夾: {TARGET_DIR}")

    total_moved = 0

    # 迴圈處理 001 到 012
    for i in range(1, 13):
        # 格式化資料夾名稱，例如 image_001, image_002...
        folder_name = f"images_{i:03d}"  # 這裡假設資料夾名為 images_001
        
        # 建立完整的來源路徑: images_001/images/
        source_path = os.path.join(BASE_DIR, folder_name, 'images')

        if os.path.exists(source_path):
            print(f"正在處理 {folder_name}...")
            files = os.listdir(source_path)
            
            moved_in_this_folder = 0
            for f in files:
                # 只搬移圖片檔 (NIH 主要是 .png)
                if f.lower().endswith('.png'):
                    src_file = os.path.join(source_path, f)
                    dst_file = os.path.join(TARGET_DIR, f)
                    
                    # 執行移動 (Move 比 Copy 快非常多)
                    try:
                        shutil.move(src_file, dst_file)
                        moved_in_this_folder += 1
                    except Exception as e:
                        print(f"搬移 {f} 時發生錯誤: {e}")
            
            total_moved += moved_in_this_folder
            print(f"  -> 已搬移 {moved_in_this_folder} 張圖片。")
        else:
            print(f"跳過: 找不到路徑 {source_path}")

    print("\n" + "="*50)
    print(f"所有任務完成！")
    print(f"總共搬移了 {total_moved} 張圖片到 {TARGET_DIR} 底下。")
    print("="*50)

if __name__ == "__main__":
    merge_nih_images()
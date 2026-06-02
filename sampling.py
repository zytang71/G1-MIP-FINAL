import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# 1. 參數設定 (可根據需求調整)
# ==========================================
CSV_PATH = 'Data_Entry_2017.csv'  
TARGET_TOTAL_SAMPLES = 6250
TRAIN_RATIO = 0.8                 
VALID_RATIO = 0.1                 
TEST_RATIO = 0.1                  

# 紅燈與綠燈疾病定義
RED_LIGHT_DISEASES = ['Pneumothorax', 'Edema', 'Pneumonia', 'Consolidation', 'Effusion', 'Mass']
# 綠燈中除了 "No Finding" 以外的疾病 (僅供統計參考)
GREEN_OTHER_DISEASES = ['Atelectasis', 'Infiltration', 'Emphysema', 'Fibrosis', 
                        'Pleural_thickening', 'Nodule', 'Hernia', 'Cardiomegaly']

# ==========================================
# 2. 輔助函數
# ==========================================
def classify_target(finding_labels):
    individual_labels = finding_labels.split('|')
    if any(label in RED_LIGHT_DISEASES for label in individual_labels):
        return 1
    return 0

def get_detailed_stats(df, title):
    """計算並顯示標籤細節"""
    print(f"\n>>> [{title}] 詳細統計：")
    print(f"  總影像數: {len(df)}")
    print(f"  總病患數: {df['Patient ID'].nunique()}")
    
    # 計算各個紅燈病症出現的次數 (因為是多標籤，總和可能超過總圖數)
    print("  紅燈組成細節:")
    for disease in RED_LIGHT_DISEASES:
        count = df['Finding Labels'].str.contains(disease).sum()
        print(f"    - {disease}: {count}")
    
    red_count = (df['target'] == 1).sum()
    green_count = (df['target'] == 0).sum()
    print(f"  分類統計: 紅燈 = {red_count} ({red_count/len(df):.1%}), 綠燈 = {green_count} ({green_count/len(df):.1%})")

# ==========================================
# 3. 主流程
# ==========================================
def main():
    print("="*50)
    print("系統啟動：NIH Chest X-ray 資料分流處理")
    print("="*50)

    # 讀取資料
    df = pd.read_csv(CSV_PATH)
    df['target'] = df['Finding Labels'].apply(classify_target)
    
    # 顯示原始資料全局資訊
    get_detailed_stats(df, "原始資料集")

    # --- 步驟 A: 病患級別切分 ---
    print("\n[正在執行病患 ID 隔離切分...]")
    gss_temp = GroupShuffleSplit(n_splits=1, train_size=TRAIN_RATIO, random_state=42)
    train_idx, temp_idx = next(gss_temp.split(df, groups=df['Patient ID']))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    gss_val_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val_test.split(temp_df, groups=temp_df['Patient ID']))
    
    valid_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    # --- 檢查 Data Leakage (資料洩漏) ---
    train_patients = set(train_df['Patient ID'])
    test_patients = set(test_df['Patient ID'])
    overlap = train_patients.intersection(test_patients)
    print(f"\n[安全性檢查] 訓練集與測試集重疊病患數: {len(overlap)}")
    if len(overlap) == 0:
        print("  >> OK: 通過病患隔離檢查，無資料洩漏。")

    # --- 步驟 B: 平衡抽樣 ---
    def balance_sample_verbose(data_frame, target_size, name):
        half_size = target_size // 2
        reds = data_frame[data_frame['target'] == 1]
        greens = data_frame[data_frame['target'] == 0]
        
        n_red = min(len(reds), half_size)
        n_green = min(len(greens), n_red) # 強制 1:1
        
        red_sampled = reds.sample(n=n_red, random_state=42)
        green_sampled = greens.sample(n=n_green, random_state=42)
        
        result = pd.concat([red_sampled, green_sampled]).sample(frac=1, random_state=42)
        print(f"  {name} 抽樣完成: 紅燈 {n_red} 張 + 綠燈 {n_green} 張 = 共 {len(result)} 張")
        return result

    train_size = int(TARGET_TOTAL_SAMPLES * TRAIN_RATIO)
    valid_size = int(TARGET_TOTAL_SAMPLES * VALID_RATIO)
    test_size = int(TARGET_TOTAL_SAMPLES * TEST_RATIO)

    print("\n[正在執行平衡抽樣...]")
    train_final = balance_sample_verbose(train_df, train_size, "訓練集 (Train)")
    valid_final = balance_sample_verbose(valid_df, valid_size, "驗證集 (Valid)")
    test_final = balance_sample_verbose(test_df, test_size, "測試集 (Test)")

    # --- 輸出結果 ---
    train_final.to_csv('train_list/train_list.csv', index=False)
    valid_final.to_csv('train_list/valid_list.csv', index=False)
    test_final.to_csv('train_list/test_list.csv', index=False)

    print("\n" + "="*50)
    print("任務成功結束！")
    print(f"1. 訓練清單已儲存至: train_list.csv")
    print(f"2. 驗證清單已儲存至: valid_list.csv")
    print(f"3. 測試清單已儲存至: test_list.csv")
    print("="*50)

if __name__ == "__main__":
    main()
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

# 所有 14 種疾病清單
ALL_DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
                'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
                'Pleural_Thickening', 'Hernia']

# 紅燈疾病定義 (目前僅用於抽樣平衡)
RED_LIGHT_DISEASES = ['Pneumothorax', 'Edema', 'Pneumonia', 'Consolidation', 'Effusion', 'Mass']

# ==========================================
# 2. 輔助函數
# ==========================================
def classify_severity(finding_labels):
    individual_labels = finding_labels.split('|')
    if any(label in RED_LIGHT_DISEASES for label in individual_labels):
        return 1
    return 0

def create_multi_hot_encoding(finding_labels):
    """將文字標籤轉換為 14 維 Multi-hot Encoding，以逗號分隔"""
    labels = finding_labels.lower().split('|')
    encoded = []
    for disease in ALL_DISEASES:
        # 特別注意有些標籤大小寫或底線問題，轉小寫後比較最安全
        if disease.lower() in labels:
            encoded.append(1)
        else:
            encoded.append(0)
    return ','.join(map(str, encoded))

def get_detailed_stats(df, title):
    """計算並顯示標籤細節"""
    print(f"\n>>> [{title}] 詳細統計：")
    print(f"  總影像數: {len(df)}")
    print(f"  總病患數: {df['Patient ID'].nunique()}")
    
    red_count = (df['severity_label'] == 1).sum()
    green_count = (df['severity_label'] == 0).sum()
    print(f"  嚴重性統計: 紅燈 = {red_count} ({red_count/len(df):.1%}), 綠燈 = {green_count} ({green_count/len(df):.1%})")

# ==========================================
# 3. 主流程
# ==========================================
def main():
    print("="*50)
    print("系統啟動：NIH Chest X-ray 資料分流處理 (保持 1:1 平衡與舊抽樣數，但輸出 14 維標籤)")
    print("="*50)

    # 讀取資料
    df = pd.read_csv(CSV_PATH)
    # severity_label 用來做後續的平衡抽樣
    df['severity_label'] = df['Finding Labels'].apply(classify_severity)
    # target 則是模型真正要訓練的 14 維度目標
    df['target'] = df['Finding Labels'].apply(create_multi_hot_encoding)
    
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
        # 改用 severity_label 做平衡
        reds = data_frame[data_frame['severity_label'] == 1]
        greens = data_frame[data_frame['severity_label'] == 0]
        
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
    train_final.to_csv('Data/train_list.csv', index=False)
    valid_final.to_csv('Data/valid_list.csv', index=False)
    test_final.to_csv('Data/test_list.csv', index=False)

    print("\n" + "="*50)
    print("任務成功結束！")
    print(f"1. 訓練清單已儲存至: train_list.csv")
    print(f"2. 驗證清單已儲存至: valid_list.csv")
    print(f"3. 測試清單已儲存至: test_list.csv")
    print("="*50)

if __name__ == "__main__":
    main()
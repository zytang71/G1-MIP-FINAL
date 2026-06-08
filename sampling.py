import numpy as np
import pandas as pd

# ==========================================
# 1. 基本設定
# ==========================================
CSV_PATH = "Data_Entry_2017.csv"
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
EVAL_SPLIT_SAMPLES = 1250

# Train 直接保留完整 split，不再壓縮成小子集
USE_FULL_TRAIN_SPLIT = True

# Valid / Test 保留較高的陽性覆蓋，方便評估各類別
EVAL_NEGATIVE_RATIO = 0.20

SPLIT_SEARCH_TRIALS = 300
BASE_RANDOM_STATE = 42

ALL_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


# ==========================================
# 2. 標籤處理
# ==========================================
def parse_labels(finding_labels):
    if pd.isna(finding_labels):
        return set()
    labels = [label.strip() for label in str(finding_labels).split("|") if label.strip()]
    if labels == ["No Finding"]:
        return set()
    return set(labels)


def add_label_columns(df):
    label_sets = df["Finding Labels"].apply(parse_labels)

    for disease in ALL_DISEASES:
        df[disease] = label_sets.apply(lambda labels: int(disease in labels))

    df["label_count"] = df[ALL_DISEASES].sum(axis=1)
    df["target"] = df[ALL_DISEASES].astype(int).astype(str).agg(",".join, axis=1)
    return df


# ==========================================
# 3. 病人層級切分搜尋
# ==========================================
def compute_split_score(train_df, valid_df, test_df):
    train_counts = train_df[ALL_DISEASES].sum()
    valid_counts = valid_df[ALL_DISEASES].sum()
    test_counts = test_df[ALL_DISEASES].sum()

    zeros_train = int((train_counts == 0).sum())
    zeros_valid = int((valid_counts == 0).sum())
    zeros_test = int((test_counts == 0).sum())

    min_valid_test = min(valid_counts.min(), test_counts.min())
    balance_penalty = valid_counts.std() + test_counts.std()

    score = (
        -(10000 * zeros_valid + 10000 * zeros_test + 1000 * zeros_train)
        + 20 * min_valid_test
        - 0.05 * balance_penalty
    )
    return score


def search_best_group_split(df):
    best_score = None
    best_splits = None

    for trial in range(SPLIT_SEARCH_TRIALS):
        random_state = BASE_RANDOM_STATE + trial
        rng = np.random.default_rng(random_state)

        patient_ids = df["Patient ID"].drop_duplicates().to_numpy()
        shuffled_patients = rng.permutation(patient_ids)

        train_end = int(len(shuffled_patients) * TRAIN_RATIO)
        valid_end = train_end + int(len(shuffled_patients) * VALID_RATIO)

        train_patients = set(shuffled_patients[:train_end])
        valid_patients = set(shuffled_patients[train_end:valid_end])
        test_patients = set(shuffled_patients[valid_end:])

        train_df = df[df["Patient ID"].isin(train_patients)].copy()
        valid_df = df[df["Patient ID"].isin(valid_patients)].copy()
        test_df = df[df["Patient ID"].isin(test_patients)].copy()

        score = compute_split_score(train_df, valid_df, test_df)
        if best_score is None or score > best_score:
            best_score = score
            best_splits = (train_df, valid_df, test_df, random_state)

    return best_splits


# ==========================================
# 4. 抽樣工具
# ==========================================
def summarize_sampled_dataset(df, split_name, target_size):
    positive_counts = df[ALL_DISEASES].sum()
    negative_count = int((df["label_count"] == 0).sum())

    print(f"\n[{split_name}] 抽樣結果")
    print(f"  目標筆數: {target_size}")
    print(f"  實際筆數: {len(df)}")
    print(f"  全陰性筆數: {negative_count}")
    print("  各類別正樣本數:")
    for disease in ALL_DISEASES:
        print(f"    - {disease}: {int(positive_counts[disease])}")


def choose_seed_indices(label_matrix, min_positive_per_disease):
    selected = set()

    for disease_idx in np.argsort(np.maximum(label_matrix.sum(axis=0), 1)):
        available_indices = np.where(label_matrix[:, disease_idx] == 1)[0]
        if len(available_indices) == 0:
            continue

        target_count = min(min_positive_per_disease, len(available_indices))
        current_count = 0

        for idx in available_indices:
            if idx in selected:
                if label_matrix[idx, disease_idx] == 1:
                    current_count += 1
                continue

            selected.add(int(idx))
            current_count += 1

            if current_count >= target_count:
                break

    return selected


def prepare_full_train_set(data_frame, split_name, random_state):
    combined = data_frame.sample(frac=1, random_state=random_state).reset_index(drop=True)
    summarize_sampled_dataset(combined, split_name, len(combined))
    return combined


def sample_eval_set(data_frame, target_size, split_name, random_state):
    shuffled = data_frame.sample(frac=1, random_state=random_state).reset_index(drop=True)
    positive_df = shuffled[shuffled["label_count"] > 0].reset_index(drop=True)
    negative_df = shuffled[shuffled["label_count"] == 0].reset_index(drop=True)

    target_negatives = min(len(negative_df), int(round(target_size * EVAL_NEGATIVE_RATIO)))
    target_positives = min(len(positive_df), target_size - target_negatives)

    label_matrix = positive_df[ALL_DISEASES].to_numpy(dtype=np.int32)
    class_availability = np.maximum(label_matrix.sum(axis=0), 1)
    rarity_weights = 1.0 / class_availability

    selected = choose_seed_indices(label_matrix, min_positive_per_disease=1)
    class_counts = (
        label_matrix[list(selected)].sum(axis=0).astype(np.int32)
        if selected
        else np.zeros(len(ALL_DISEASES), dtype=np.int32)
    )

    while len(selected) < target_positives:
        remaining_indices = [idx for idx in range(len(positive_df)) if idx not in selected]
        if not remaining_indices:
            break

        remaining_matrix = label_matrix[remaining_indices]
        scores = (remaining_matrix * (rarity_weights / (class_counts + 1.0))).sum(axis=1)
        best_offset = int(np.argmax(scores))
        best_index = remaining_indices[best_offset]

        selected.add(best_index)
        class_counts += label_matrix[best_index]

    selected_positive_df = positive_df.iloc[sorted(selected)].copy()
    selected_negative_df = negative_df.iloc[:target_negatives].copy()

    combined = pd.concat([selected_positive_df, selected_negative_df], ignore_index=True)

    if len(combined) < target_size:
        used_images = set(combined["Image Index"].tolist())
        fallback_pool = shuffled[~shuffled["Image Index"].isin(used_images)]
        fill_count = min(target_size - len(combined), len(fallback_pool))
        if fill_count > 0:
            combined = pd.concat([combined, fallback_pool.iloc[:fill_count].copy()], ignore_index=True)

    combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    summarize_sampled_dataset(combined, split_name, target_size)
    return combined


# ==========================================
# 5. 資料統計
# ==========================================
def print_dataset_stats(df, title):
    print(f"\n>>> [{title}]")
    print(f"  總筆數: {len(df)}")
    print(f"  病人數: {df['Patient ID'].nunique()}")
    print(f"  全陰性筆數: {(df['label_count'] == 0).sum()}")
    print("  各類別在此集合的正樣本數:")
    for disease in ALL_DISEASES:
        print(f"    - {disease}: {int(df[disease].sum())}")


# ==========================================
# 6. 主程式
# ==========================================
def main():
    print("=" * 60)
    print("從 NIH Chest X-ray 重新抽樣 14 類多標籤資料")
    print("=" * 60)
    print("Train: 保留完整 split")
    print("Valid/Test: 保留每類覆蓋，方便評估")

    df = pd.read_csv(CSV_PATH)
    df = add_label_columns(df)

    print_dataset_stats(df, "原始 NIH 全資料")

    print("\n[開始搜尋病人層級 train/valid/test 切分...]")
    train_df, valid_df, test_df, best_seed = search_best_group_split(df)
    print(f"  最佳 random_state = {best_seed}")

    train_patients = set(train_df["Patient ID"])
    valid_patients = set(valid_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])

    print(f"  Train/Valid 重疊病人數: {len(train_patients.intersection(valid_patients))}")
    print(f"  Train/Test 重疊病人數: {len(train_patients.intersection(test_patients))}")
    print(f"  Valid/Test 重疊病人數: {len(valid_patients.intersection(test_patients))}")

    print_dataset_stats(train_df, "切分後原始 Train")
    print_dataset_stats(valid_df, "切分後原始 Valid")
    print_dataset_stats(test_df, "切分後原始 Test")

    valid_size = EVAL_SPLIT_SAMPLES
    test_size = EVAL_SPLIT_SAMPLES

    print("\n[開始依新策略抽樣...]")
    if USE_FULL_TRAIN_SPLIT:
        train_final = prepare_full_train_set(train_df, "Train", best_seed)
    else:
        train_final = train_df.sample(frac=1, random_state=best_seed).reset_index(drop=True)
    valid_final = sample_eval_set(valid_df, valid_size, "Valid", best_seed + 1)
    test_final = sample_eval_set(test_df, test_size, "Test", best_seed + 2)

    train_final.to_csv("Data/train_list.csv", index=False)
    valid_final.to_csv("Data/valid_list.csv", index=False)
    test_final.to_csv("Data/test_list.csv", index=False)

    print("\n" + "=" * 60)
    print("抽樣完成，已輸出以下檔案")
    print("  1. Data/train_list.csv")
    print("  2. Data/valid_list.csv")
    print("  3. Data/test_list.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()

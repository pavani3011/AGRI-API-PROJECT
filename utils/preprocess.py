import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib 
from config import (
    CROP_CSV, FEATURE_COLS, TARGET_COL, 
    RANDOM_STATE, TEST_SIZE, LABEL_ENC_PATH
)

def load_and_split():

    df = pd.read_csv(CROP_CSV)

    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f" csv missing columns: {missing}")
    
    X = df[FEATURE_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])

    X_train, X_test , y_train, y_test  = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    joblib.dump(le, LABEL_ENC_PATH)
    print(f" label encoder saved in {LABEL_ENC_PATH}")
    print(f" Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"crops: {list(le.classes_)}")

    return X_train, X_test, y_train, y_test, le

if __name__ == "__main__":
    load_and_split()

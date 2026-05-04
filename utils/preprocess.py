import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib 
from config import (
    CROP_CSV, FEATURE_COLS, TARGET_COL, 
    RANDOM_STATE, TEST_SIZE, LABEL_ENC_PATH
)
from torchvision import transforms

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

# #training - makes model see variety
# train_transforms = transforms.Compose([
#     transforms.Resize((128,128)),
#     transforms.RandomHorizontalFlip(), #augumentation
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],
#                          [0.229,0.224,0.225])
# ])

# #validation - fair evaluation - no augmentation
# val_transform = transforms.Compose([
#     transforms.Resize((128,128)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],
#                          [0.229,0.224,0.225])
# ])

if __name__ == "__main__":
    load_and_split()

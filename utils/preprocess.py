import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib 
from config import (
    CROP_CSV, FEATURE_COLS, TARGET_COL, 
    RANDOM_STATE, TEST_SIZE, LABEL_ENC_PATH
)
from torchvision import transforms, datasets
import os
from pathlib import Path
from torch.utils.data import DataLoader 
import shutil
import torch

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

#training - makes model see variety

IMG_SIZE =128
BATCH_SIZE = 32
NUM_WORKERS = 2
DATA_DIR = Path("data/plant_village")

IMAGENET_MEAN = [0.485,0.456, 0.406]
IMAGENET_STD= [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(), #augumentation
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast = 0.2,
                           saturation = 0.2, hue = 0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
])

#validation - fair evaluation - no augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
])

# dataset + dataloader
def get_dataloaders(data_dir = DATA_DIR, batch_size = BATCH_SIZE):
    train_dataset = datasets.ImageFolder(
        root = data_dir/"train",
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        root = data_dir/"val",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size = batch_size,
        shuffle = True,
        num_workers = NUM_WORKERS,
        pin_memory = False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = NUM_WORKERS,
        pin_memory = True
    )

    return train_loader, val_loader, train_dataset.classes

#split raw data into train/val 
def split_dataset(src_dir, dist_dir = DATA_DIR, val_size = 0.2, seed = 42):

    src = Path(src_dir)
    dst = Path(dist_dir)

    for class_dir in src.iterdir():
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*.jpg"))+ list(class_dir.glob("*.JPG"))
        train_imgs , val_imgs = train_test_split(
            images, test_size=val_size, random_state=seed
        )

        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            out = dst/split/class_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, out/img.name)
    print(f"split complete -> { dst}")



if __name__ == "__main__":
    load_and_split()

    train_loader, val_loader, classes = get_dataloaders()
    print(f"train batches : {len(train_loader)}")
    print(f"val batches : {len(val_loader)}")
    print(f"classes: {len(classes)}")
    print(f"sample classes: {classes[:3]}")

    imgs, lbls = next(iter(train_loader))
    print(f"batch shape: {imgs.shape}")
    print(f"labels sample: {lbls[:5]}")
    

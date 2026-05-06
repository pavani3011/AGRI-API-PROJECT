from pathlib import Path
import torch 

# ---Paths---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR/"data"
MODEL_DIR = BASE_DIR/"models"
OUTPUT_DIR = BASE_DIR/"outputs"

# ---create folders if they does not exist---
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---Data files---
CROP_CSV = DATA_DIR/ "Crop_recommendation.csv"
LABEL_ENC_PATH = MODEL_DIR/"label_encoder.pkl"
CROP_MODEL_PATH = MODEL_DIR/ "crop_model.pkl"

FEATURE_COLS = ["N","P","K","temperature","humidity","ph","rainfall"]
TARGET_COL = "label"

# ---Training config---
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---Hyperparameter grid---
PARAM_GRID ={
    "model__n_estimators": [100,200],
    "model__max_depth": [10,None],
    "model__min_samples_split": [2,5],
    "smote__k_neighbors": [3,5],
}

# ---CNN config---
CNN_CONFIG = {
    #data
    "data_dir": DATA_DIR/"plant_village",
    "img_size": 128,
    "num_classes": 38,

    #training
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,

    #paths
    "model_save_path": MODEL_DIR/"disease_model.pth",
    "device":(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
}

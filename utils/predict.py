import pandas as pd
import joblib
from config import CROP_MODEL_PATH, LABEL_ENC_PATH, FEATURE_COLS, CNN_CONFIG
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from utils.train import build_model, device


def load_artifacts():
    # load model and le from disk and call once at startup
    model = joblib.load(CROP_MODEL_PATH)
    le = joblib.load(LABEL_ENC_PATH)
    return model, le

def predict_crop(input_dict: dict, model = None, le= None):
    # input_dict : eg: {"N":90, "P": 42, "K": 43, so on.....}

    if model is None or le is None:
        model,le = load_artifacts()

    #convert dict into single row dataframe----preserves column order
    X = pd.DataFrame([input_dict])[FEATURE_COLS]

    #smote step is skipped at predict time auto....
    pred_idx = model.predict(X)[0]
    crop_name = le.inverse_transform([pred_idx])[0]

    return crop_name
    
def predict_proba(input_dict,model=None,le=None):
    if model is None or le is None:
        model,le = load_artifacts()
    
    X=pd.DataFrame([input_dict])[FEATURE_COLS]
    probas= model.predict_proba(X)[0]

    return{
        crop: round(float(prob),4)
        for crop, prob in zip(le.classes_,probas)
    }



# ── Constants ────────────────────────────────────────────────────
IMG_SIZE      = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Inference transform (same as val, no augmentation) ───────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Load model once ──────────────────────────────────────────────
def load_model(model_path="models/disease_model.pth"):
    checkpoint = torch.load(model_path, map_location=device)
    classes    = checkpoint["classes"]
    model      = build_model(num_classes=len(classes), use_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


# ── Predict a single image ───────────────────────────────────────
def predict(image_path, model, classes, top_k=3):
    """
    Args:
        image_path : path to .jpg / .png leaf image
        model      : loaded CNN model
        classes    : list of class names
        top_k      : return top-k predictions

    Returns:
        list of dicts [{"class": str, "confidence": float}, ...]
    """
    img    = Image.open(image_path).convert("RGB")
    tensor = infer_transform(img).unsqueeze(0).to(device)   # [1, 3, 128, 128]

    with torch.no_grad():
        output = model(tensor)                               # [1, num_classes]
        probs  = F.softmax(output, dim=1)[0]                # [num_classes]

    top_probs, top_indices = torch.topk(probs, k=top_k)

    results = []
    for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
        results.append({
            "class":      classes[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return results


# ── Predict from raw bytes (for FastAPI endpoint) ────────────────
def predict_from_bytes(image_bytes, model, classes, top_k=3):
    """Used by the FastAPI /disease-detect endpoint."""
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = infer_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=top_k)

    return [
        {
            "class":      classes[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        }
        for prob, idx in zip(top_probs.cpu(), top_indices.cpu())
    ]


# ── Entry point ──────────────────────────────────────────────────
      

if __name__ == "__main__":

    sample= {
        "N":90, "P":42, "K":43,
        "temperature":20.8, "humidity": 82.0,
        "ph":6.5,"rainfall": 202.9
    }

    print("predicted crop:", predict_crop(sample))
    print("probabilities:", predict_proba(sample))

    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_leaf.jpg"

    model, classes = load_model("models/disease_model.pth")
    results        = predict(image_path, model, classes, top_k=3)

    print(f"\nImage: {image_path}")
    print("─" * 40)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['class']:<35} {r['confidence']:>6.2f}%")
    




    



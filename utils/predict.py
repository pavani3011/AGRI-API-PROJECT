import pandas as pd
import joblib
from config import CROP_MODEL_PATH, LABEL_ENC_PATH, FEATURE_COLS, CNN_CONFIG
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


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

# def load_cnn_model():
#     cfg = CNN_CONFIG
#     model = models.resnet18(pretrained= False)
#     model.fc = nn.Linear(512,cfg["num_classes"])
#     model.load_state_dict(torch.load(cfg['model_save_path'],
#                                      map_location = cfg["device"]))
#     model.eval()
#     return model

# def predict_disease(image_path, model, classes):
#     img = Image.open(image_path)
#     t = transforms.Compose([
#         transforms.Resize((128,128)),
#         transforms.ToTensor()
#     ])
#     tensor = t(img).unsqueeze(0)
#     with torch.no_grad():
#         out = model(tensor)
#     idx = out.argmax().item()
#     conf = torch.softmax(out,1).max().item()
#     return classes[idx],round(conf * 100,1)
    

if __name__ == "__main__":

    sample= {
        "N":90, "P":42, "K":43,
        "temperature":20.8, "humidity": 82.0,
        "ph":6.5,"rainfall": 202.9
    }

    print("predicted crop:", predict_crop(sample))
    print("probabilities:", predict_proba(sample))

    # load_cnn_model()
    # predict_disease()
    




    



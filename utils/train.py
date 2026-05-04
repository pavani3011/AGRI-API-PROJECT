from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
from config import (
    CROP_MODEL_PATH, PARAM_GRID,
    RANDOM_STATE, CV_FOLDS
)
from utils.preprocess import load_and_split
from config import CNN_CONFIG
import torch, torch.nn as nn
from torchvision import models

def build_pipeline():
    #return an untrained imbalanced pipeline
    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model",RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        )),
    ])

def train(X_train, y_train):
    # run gridsearch over param_grid , refit best model, save to disk , return fitted estimator.
    pipe = build_pipeline()

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    #find best parameters and best score
    gs = GridSearchCV( 
        estimator=pipe,
        param_grid=PARAM_GRID,
        cv= cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print("Starting GridSearch CV...")
    gs.fit(X_train, y_train) 

    best_model = gs.best_estimator_

    joblib.dump(best_model, CROP_MODEL_PATH)
    print(f"Best params: {gs.best_params_}")
    print(f"Best cv F1: {gs.best_score_:.4f}")

    return best_model,{
        "best_params": gs.best_params_,
        "best_cv_score":round(gs.best_score_,4)
    }

# #Cnn training loop
# def get_model(inpt_size,num_classes):
#     model= nn.Sequential(
#         nn.Linear(inpt_size,64),
#         nn.ReLU(),
#         nn.Linear(64,32),
#         nn.ReLU(),
#         nn.Linear(32,num_classes)
#     )
#     return model

# def train_cnn(X_train, X_test , y_train, y_test):
#     cfg = CNN_CONFIG
#     input_dim = X_train.shape[1]
#     model = get_model(input_dim,cfg["num_classes"]).to(cfg["device"])

#     X_train_t = torch.tensor(X_train.values, dtype= torch.float32).to(cfg["device"])
#     y_train_t = torch.tensor(y_train, dtype= torch.long).to(cfg["device"])
#     X_test_t = torch.tensor(X_test.values, dtype= torch.float32).to(cfg["device"])
#     y_test_t = torch.tensor(y_test, dtype= torch.long).to(cfg["device"])

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr = cfg["learning_rate"])

#     for epoch in range(1,51):
#         model.train()
#         pred = model(X_train_t)
#         loss= criterion(pred,y_train_t)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if epoch % 10 == 0:
#             model.eval()
#             with torch.no_grad():
#                 acc = (model(X_test_t).argmax(1)== y_test_t).float().mean()
#             print(f"epoch {epoch:3d} || loss : {loss:.4f} || acc : {acc:.2%}")

#     torch.save(model.state_dict(), cfg['model_save_path'])
#     print("Saved to", cfg["model_save_path"])

if __name__ == "__main__":
    X_train, X_test , y_train, y_test, le = load_and_split()

    train(X_train, y_train)

    # train_cnn(X_train, X_test , y_train, y_test)

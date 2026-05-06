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
from utils.preprocess import load_and_split , get_dataloaders
from config import CNN_CONFIG
import torch, torch.nn as nn
from torchvision import models
from pathlib import Path

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
device = CNN_CONFIG["device"]

def build_model(num_classes, use_pretrained=True):
    model = models.resnet18(weights = "IMAGENET1K_V1" if use_pretrained else None)

    if use_pretrained:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256,num_classes)
    )
    return model.to(device)

def build_custom_cnn(num_classes):
    model = nn.Sequential(
        #block 1
        nn.Conv2d(3,32, kernel_size=3, padding= 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        #block 2
        nn.Conv2d(32,64, kernel_size=3, padding= 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        #block 3
        nn.Conv2d(64,128, kernel_size=3, padding= 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # classifier head
        nn.Flatten(),
        nn.Linear(128*16*16, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
        
    )
    return model.to(device)

#training loop
def train_cnn(model, train_loader, val_loader,classes, epochs=20, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p : p.requires_grad, model.parameters()), lr = lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):

        # train phase
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss+= loss.item()

        avg_loss = running_loss/len(train_loader)

        #val phase 
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs).argmax(dim = 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct/total
        scheduler.step()

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        print(f" epoch {epoch:3d}/{epochs} | "
              f"loss: {avg_loss:.4f} | "
              f"val acc: {val_acc:.2%}")
        
        #save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(CNN_CONFIG["model_save_path"])
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
            }, save_path)
            print(f" saved best model (val_acc ={val_acc:.2%})")
    print(f"\nTraining complete. best val acc : {best_val_acc:.2%}")
    return history

# unfreeze all layers for fine-tuning
def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

#model point      
if __name__ == "__main__":
    X_train, X_test , y_train, y_test, le = load_and_split()
    
    train(X_train, y_train)

    train_loader , val_loader , classes = get_dataloaders()
    num_classes = len(classes)
    print(f"Training on {device} | Classes: {num_classes}")

    #Phase 1: train only the head (5 epochs)
    model = build_model(num_classes, use_pretrained=True)
    history = train_cnn(model, train_loader, val_loader,classes, epochs=5, lr= 0.001)

    #Phase 2: unfreeze and fine-tune all layers (15 more epochs)
    model = unfreeze_model(model)
    history2 = train_cnn(model,train_loader, val_loader,classes, epochs=15, lr = 0.0001)


    

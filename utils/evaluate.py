import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix,
    top_k_accuracy_score
)
import joblib
from config import OUTPUT_DIR, LABEL_ENC_PATH , CROP_MODEL_PATH, CNN_CONFIG
from utils.preprocess import load_and_split, get_dataloaders
import torch 
from pathlib import Path
from utils.train import build_model


def evaluate(model, X_test, y_test):
    le = joblib.load(LABEL_ENC_PATH)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    F1 = f1_score(y_test,y_pred, average="macro")
    recall = recall_score(y_test,y_pred, average="macro")

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 score: {F1:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\n Per class report:")
    print(classification_report(y_test,y_pred,target_names=le.classes_))


    cm = confusion_matrix(y_test,y_pred)
    fig,ax = plt.subplots(figsize=(14,11))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax
    )

    ax.set_title("confusion matrix - crop prediction")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = OUTPUT_DIR/"confusion_matrix.png"
    plt.savefig(out_path, dpi= 150)
    plt.close(fig)
    
    return {
        "accuracy": round(acc,4),
        "F1": round(F1,4),
        "Recall": round(recall,4),
        
    }


device = CNN_CONFIG["device"]

# load saved model
def load_model(model_path=None):
    if model_path is None:
        model_path = CNN_CONFIG["model_save_path"]
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    cnn_model = build_model(num_classes=len(classes), use_pretrained=False)
    cnn_model.load_state_dict(checkpoint["model_state_dict"])
    cnn_model.eval()
    print(f"loaded model | val acc at save: { checkpoint['val_acc']:.2%}")
    return cnn_model, classes

# get all predictions 
def get_predictions(cnn_model, loader):
    all_preds= []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = cnn_model(imgs)
            probs = torch.softmax(outputs, dim= 1)
            preds = outputs.argmax(dim =1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return(
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )

def evaluate_cnn(cnn_model, val_loader, classes, save_dir = "outputs"):
    save_dir= Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    preds, labels, probs = get_predictions(cnn_model, val_loader)

    acc = (preds == labels).mean()
    print(f"\n overall accuracy: {acc:.2%}")

    top_k = min(5, len(classes))
    top5 = top_k_accuracy_score(labels, probs, k=top_k)
    print(f"top-{top_k} accuracy: {top5:.2%}")

    print("\nClassification report:")
    print(classification_report(labels,preds, target_names=classes))

    plot_confusion_matrix(preds, labels, classes, save_dir)
    plot_per_class_accuracy(preds, labels, classes,save_dir)

    return acc, top5

# confusion matrix plot 
def plot_confusion_matrix(preds,labels, classes, save_dir):
    cm = confusion_matrix(labels, preds)

    top_n = min(15, len(classes))
    per_class_err = (cm.sum(axis=1)- np.diag(cm))
    top_idx = np.argsort(per_class_err)[-top_n:]
    cm_sub = cm[np.ix_(top_idx, top_idx)]
    sub_cls = [classes[i] for i in top_idx]

    fig,ax = plt.subplots(figsize= (14,12))
    sns.heatmap(
        cm_sub, annot=True, fmt="d", cmap="Greens",
        xticklabels=sub_cls, yticklabels=sub_cls, ax=ax
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize =12)
    ax.set_title("Confusion matrix (top confused classes)", fontsize =14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize =8)
    plt.tight_layout()
    out= save_dir/"crop_confusion_matrix.png"
    plt.savefig(out,dpi =150)
    plt.show()
    print(f"saved -> {out}")

# per-class accuracy bar chart
def plot_per_class_accuracy(preds, labels, classes, save_dir):
    per_cls_acc= []
    for i in range(len(classes)):
        mask = labels == i
        if mask.sum()> 0:
            per_cls_acc.append((preds[mask] == labels[mask]).mean())
        else:
            per_cls_acc.append(0.0)
    
    sorted_idx= np.argsort(per_cls_acc)
    fig,ax = plt.subplots(figsize= (10,14))
    colors = ["#D85A30" if v < 0.8 else "#1D9E75" for v in
              [per_cls_acc[i] for i in sorted_idx]]
    
    ax.barh(
        [classes[i] for i in sorted_idx],
        [per_cls_acc[i] for i in sorted_idx],
        color = colors
    )

    ax.axvline(0.8, color="gray", linestyle="--", linewidth=1, label="80% threshold")
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-class accuracy")
    ax.legend()
    plt.tight_layout()
    out = save_dir/"per_class_accuracy.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"saved -> {out}")

#plot training curves
def plot_training_curves(history, save_dir="outputs"):
    save_dir= Path(save_dir)
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))

    ax1.plot(history["train_loss"], color = "#D85A30")
    ax1.set_title("training loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    ax2.plot([v * 100 for v in history["val_acc"]], color="#1D9E75")
    ax2.set_title("validation accuracy (%)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy %")

    plt.tight_layout()
    out = save_dir/"trining_curves.png"
    plt.savefig(out, dpi = 150)
    plt.show()
    print(f"saved -> {out}")

#entry point
if __name__ == "__main__":
    X_train, X_test , y_train, y_test, le = load_and_split()
    model = joblib.load(CROP_MODEL_PATH)

    evaluate(model, X_test,y_test)

    _,val_loader, _ = get_dataloaders()
    cnn_model, classes = load_model(CNN_CONFIG["model_save_path"])
    save_dir = CNN_CONFIG.get("output_dir", "outputs")
    acc, top5 = evaluate_cnn(cnn_model, val_loader, classes, save_dir= save_dir)
    
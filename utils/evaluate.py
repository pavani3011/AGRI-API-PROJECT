import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix
)
import joblib
from config import OUTPUT_DIR, LABEL_ENC_PATH , CROP_MODEL_PATH
from utils.preprocess import load_and_split
import torch 


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



# def evaluate_cnn(model, loader, classes, device):
#     model.eval()
#     all_preds, all_labels = [],[]
#     with torch.no_grad():
#         for X, y in loader:
#             preds = model(X.to(device)).argmax(1).cpu()
#             all_preds.extend(preds); all_labels.extend(y)
#     print(classification_report(all_labels, all_preds, target_names= classes))

#     cm = confusion_matrix(all_labels, all_preds)
#     sns.heatmap(cm,xticklabels=classes, yticklabels=classes)
#     plt.title("disease detection confusion matrix")
#     plt.savefig("../outputs/confusion_matrix_torch.png")
#     plt.show()


if __name__ == "__main__":
    X_train, X_test , y_train, y_test, le = load_and_split()
    model = joblib.load(CROP_MODEL_PATH)

    evaluate(model, X_test,y_test)
    
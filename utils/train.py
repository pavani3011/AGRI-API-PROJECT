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

if __name__ == "__main__":
    X_train, X_test , y_train, y_test, le = load_and_split()

    train(X_train, y_train)

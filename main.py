''' 
CROP PREDICTION PIPELINE
RUN THIS TO LOAD -> TRAIN -> EVALUATE -> PREDICT -> SAVE ARTIFACTS
'''

import time
from utils.preprocess import load_and_split
from utils.train import train
from utils.evaluate import evaluate
from utils.predict import predict_crop, load_artifacts


def run_pipeline():
    print("="*50)
    print(" AGRI AI - CROP PREDICTION PIEPLINE")
    print("="*50)

    #load and split data
    print("\n[1/4] loading and splitting data....")
    X_train, X_test, y_train, y_test, le = load_and_split()

    #train with gridsearchCV
    print("\n[2/4] training model....")
    t0 = time.time()
    model,train_results = train(X_train, y_train)
    print(f" training took {round(time.time()-t0, 1)}s")

    #evaluate on test set
    print("\n[3/4] evaluating on held-out test set....")
    metrics = evaluate(model, X_test, y_test)

    #smoke test the predict module
    print("\n[4/4] smoke test - single prediction....")
    sample= {
        "N":90, "P":42, "K":43,
        "temperature":20.8, "humidity": 82.0,
        "ph":6.5,"rainfall": 202.9
    }

    #predict crop
    loaded_model, loaded_le = load_artifacts()
    crop = predict_crop(sample,model=loaded_model, le=loaded_le)
    print(f"input: {sample}")
    print(f"predicted crop: {crop}")

    #summary
    print("\n"+ "="*50)
    print("Pipeline complete")
    print(f" accuracy: {metrics['accuracy']*100:.2f}")
    print(f" marco f1: {metrics['F1']:.4f}")
    print(f" best params: {train_results['best_params']}")
    print("="*50)


if __name__ == "__main__":
    run_pipeline()



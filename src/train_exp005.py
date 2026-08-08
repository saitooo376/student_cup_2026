import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path

from src.data import load_train, load_test, convert_category
from src.cv_features import create_cv_features
from src.models.model_exp005 import train_cv
from src.config import load_config
from src.io import save_metrics
from src.paths import OUTPUT_DIR


def main():

    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    exp_name = config["experiment"]["name"]

    save_dir = OUTPUT_DIR / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)


    # load data
    train = load_train()
    test = load_test()

    target = config["data"]["target"]
    id_col = config["data"]["id"]
    drop_cols = config["feature"]["drop_columns"] + [target] + [id_col]
    cat_features = config["feature"]["categorical_features"]

    train, test = convert_category(train, test, cat_features)

    X = train.drop(columns=drop_cols)
    y = train[target]

    test = test.drop(columns=config["feature"]["drop_columns"] )
    _, full_train_stats = create_cv_features(X)
    test, _ = create_cv_features(test, stats_dict=full_train_stats)

    # train
    model, oof, importance, metrics = train_cv(
        X,
        y,
        cat_features,
        config
    )

    # [変更] CV内で特徴量が増えるため
    feature_columns = model.models[0].feature_name_

    for fold, fold_model in enumerate(model.models[1:], start=1):
        if fold_model.feature_name_ != feature_columns:
            raise ValueError(f"fold {fold} で特徴量列が一致していません。")


    # threshold探索
    thresholds = np.arange(0.01, 0.5, 0.01)

    best_f1 = 0
    best_threshold = 0

    for t in thresholds:
        pred = (oof >= t).astype(int)

        score = f1_score(y, pred)

        if score > best_f1:
            best_f1 = score
            best_threshold = t

    metrics.update({
    "oof_f1": float(best_f1),
    "best_threshold": float(best_threshold)})


    # save model
    joblib.dump(model, save_dir / "model.pkl")

    
    # save features
    feature_path = save_dir / "features.txt"
    with open(feature_path, "w", encoding="utf-8") as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    
    joblib.dump(feature_columns, save_dir / "feature_columns.pkl")

    # save oof
    oof_df = pd.DataFrame({"企業ID": train[id_col], "target": y, "prediction": oof})
    oof_df.to_csv(save_dir / "oof.csv", index=False)

    
    # save importance
    importance.to_csv(save_dir / "feature_importance.csv", index=False)

 
    # save metrics
    save_metrics(metrics, save_dir / "metrics.json")

 
    # inference
    pred = model.predict_proba(test.drop(columns=id_col))[:, 1]
    pred_label = (pred >= best_threshold).astype(int)
    submission = pd.DataFrame({
        id_col: test[id_col],
        "target": pred_label
    })

    submission.to_csv(save_dir / "submission.csv", index=False, header=False)

    print("=" * 40)
    print(f"AUC      : {metrics['cv_auc']:.5f}")
    print(f"OOF F1   : {metrics['oof_f1']:.5f}")
    print(f"Threshold: {metrics['best_threshold']:.2f}")
    print(f"Saved to : {save_dir}")
    print("=" * 40)


if __name__ == "__main__":
    main()
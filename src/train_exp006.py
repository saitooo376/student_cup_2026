import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path

from src.data import load_train, load_test, convert_category
from src.cv_features import create_cv_features
from src.models.model_exp006 import train_cv
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

    test_ids = test[id_col].copy()
    test_X = test.drop(columns=config["feature"]["drop_columns"] )
    test_X = test_X.drop(columns=id_col)

    _, full_train_stats = create_cv_features(X)
    test_X, _ = create_cv_features(test_X, stats_dict=full_train_stats)

    # train
    model, oof, importance, metrics = train_cv(
        X,
        y,
        cat_features,
        config
    )


    feature_columns = (
        model.models[0]["model"]
        .feature_name_
    )

    # 全モデルで特徴量が一致しているか確認
    for i, model_info in enumerate(
        model.models
    ):

        current_features = (
            model_info["model"]
            .feature_name_
        )

        if current_features != feature_columns:

            raise ValueError(
                f"model {i} "
                f"(seed={model_info['seed']}, "
                f"fold={model_info['fold']}) "
                f"で特徴量列が一致していません。"
            )

    # testも学習時と同じ列順にする
    test_X = test_X[feature_columns]


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
    pred = model.predict_proba(test_X)[:, 1]
    pred_label = (pred >= best_threshold).astype(int)
    submission = pd.DataFrame({
        id_col: test_ids,
        "target": pred_label
    })

    submission.to_csv(save_dir / "submission.csv", index=False, header=False)

    print("=" * 40)
    print(f"AUC      : {metrics['cv_auc']:.5f}")
    print(f"OOF F1   : {metrics['oof_f1']:.5f}")
    print(f"Threshold: {metrics['best_threshold']:.2f}")
    print(f"Seeds    : {metrics['seeds']}")
    print(f"Models   : {metrics['n_models']}")
    print(f"Saved to : {save_dir}")
    print("=" * 40)


if __name__ == "__main__":
    main()
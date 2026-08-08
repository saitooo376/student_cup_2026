from seaborn import categorical
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.cv_features import create_cv_features


class EnsembleModel:
    """
    複数フォールドのLightGBMモデルをまとめるアンサンブル用クラス
    train.py の model.predict_proba(test)[:, 1] に対応
    """
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        # 各フォールドのモデルで予測を行い、その平均（確率）を算出
        preds = np.mean([model.predict_proba(X) for model in self.models], axis=0)
        return preds


def train_cv(X, y, cat_features, config):
    """
    LightGBMによるCross Validation学習関数

    Args:
        X (pd.DataFrame): 訓練用特徴量データ
        y (pd.Series): 訓練用ターゲットデータ
        config (dict): 設定ファイルの内容

    Returns:
        model (EnsembleModel): テストデータ予測用のアンサンブルモデル
        oof (pd.DataFrame): OOF（OutOfFold）の予測確率結果
        importance (pd.DataFrame): 各特徴量の重要度（Mean・Std）
        metrics (dict): CVスコア等の評価指標
    """
    # Configからの設定読み込み（指定がない場合はデフォルト値を使用）
    model_params = config.get("model", {}).get("params", {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "random_state": 42,
        "verbose": -1,
    })
    
    n_splits = config.get("train", {}).get("n_splits", 5)
    random_state = config.get("train", {}).get("random_state", 42)
    early_stopping_rounds = config.get("train", {}).get("early_stopping_rounds", 50)
    
    # 評価指標の設定（デフォルトは AUC）
    
    oof = np.zeros(len(X))
    fold_scores = []
    models = []
    feature_importances_dict = {}

    # K-Foldの設定（分類問題を想定して StratifiedKFold を使用）
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"Starting Training with {n_splits}-Fold CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        #==================変更================
        # CV内特徴量　の追加
        X_train, stats_dict = create_cv_features(X_train)
        X_val, _ = create_cv_features(X_val, stats_dict=stats_dict)
        #=====================================

        # fit 直前で確認
        for col in X_train.columns:
            if X_train[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                print(f"【原因列】 {col} に配列データが入っています！")

        # モデル作成
        model = lgb.LGBMClassifier(**model_params)

        # Early Stopping と Callbacks の設定
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0)
        ]

        # 学習
        model.fit(
            X_train, y_train, categorical_feature=cat_features,
            eval_X=X_val, eval_y=y_val,
            callbacks=callbacks
        )

        # 検証データの予測確率取得 (1クラス目の確率)
        val_preds = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_preds

        # スコア出力
        fold_auc = roc_auc_score(y_val, val_preds)

        # モデルの安定性確認用
        fold_f1 = f1_score(y_val, (val_preds >= 0.5).astype(int))

        #　foldごと各指標の保存
        fold_scores.append({"fold": fold + 1, 
                            "auc": fold_auc, 
                            "f1": fold_f1, 
                            "best_iteration": model.best_iteration_
                            })

        print(f"Fold {fold+1} "
                f"AUC:{fold_auc:.5f} "
                f"F1:{fold_f1:.5f}")
                                            

        # モデルの保存と特徴量重要度の記録
        models.append(model)
        feature_importances_dict[f"fold_{fold + 1}"] = model.feature_importances_
        feature_names = X_train.columns
    # CVスコア計算
    cv_score = roc_auc_score(y, oof)

    # 特徴量重要度の整理 (各フォールドの平均と標準偏差)
    feature_importances = pd.DataFrame(feature_importances_dict, index=feature_names)

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": feature_importances.mean(axis=1).values,
        "importance_std": feature_importances.std(axis=1).values
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)



    # メトリクスのまとめ
    metrics = { "cv_auc": float(cv_score),

                "fold_scores": fold_scores,

                "auc_mean": np.mean([x["auc"] for x in fold_scores]),

                "auc_std": np.std([x["auc"] for x in fold_scores]),

                "f1_mean": np.mean([x["f1"] for x in fold_scores]),

                "f1_std": np.std([x["f1"] for x in fold_scores])
                }

    return EnsembleModel(models), oof, importance, metrics
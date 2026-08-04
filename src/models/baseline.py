from seaborn import categorical
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


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
    
    oof_preds = np.zeros(len(X))
    models = []
    feature_importances = pd.DataFrame(index=X.columns)

    # K-Foldの設定（分類問題を想定して StratifiedKFold を使用）
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"Starting Training with {n_splits}-Fold CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

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
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )

        # 検証データの予測確率取得 (1クラス目の確率)
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        # スコア出力
        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold + 1} - AUC: {fold_auc:.5f}")

        # モデルの保存と特徴量重要度の記録
        models.append(model)
        feature_importances[f"fold_{fold + 1}"] = model.feature_importances_

    # CVスコア計算
    cv_score = roc_auc_score(y, oof_preds)

    # 特徴量重要度の整理 (各フォールドの平均と標準偏差)
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": feature_importances.mean(axis=1).values,
        "importance_std": feature_importances.std(axis=1).values
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    # OOFデータの整理
    oof = pd.DataFrame({
        "oof_pred": oof_preds
    })

    # メトリクスのまとめ
    metrics = {
        "cv_score": float(cv_score)
    }

    return EnsembleModel(models), oof, importance, metrics
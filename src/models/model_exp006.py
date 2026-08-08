import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.cv_features import create_cv_features


class EnsembleModel:
    """
    複数seed * 複数foldのLightGBMモデルをまとめるアンサンブルモデル。

    modelsには以下のような辞書を格納する。

    [
        {
            "seed": 42,
            "fold": 1,
            "model": <LGBMClassifier>
        },
        ...
    ]

    predict_proba()では全モデルの予測確率を平均する。
    """

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        """
        全seed * foldモデルの予測確率を平均する。
        """

        preds = np.mean(
            [
                item["model"].predict_proba(X)
                for item in self.models
            ],
            axis=0
        )

        return preds


def train_cv(X, y, cat_features, config):
    """
    複数seed * K-Fold CVでLightGBMを学習する。

    例:
        seeds = [42, 52, 62, 72, 82]
        n_splits = 5

        → 5 * 5 = 25モデル

    Returns
    -------
    ensemble_model : EnsembleModel
        全25モデルを平均するアンサンブルモデル

    oof : np.ndarray
        5 seed分のOOF予測を平均したもの

    importance : pd.DataFrame
        25モデルのfeature importance

    metrics : dict
        CV評価指標
    """

    # =========================================================
    # 1. Config
    # =========================================================

    model_params = config.get("model", {}).get(
        "params",
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "verbose": -1,
        }
    )

    n_splits = config.get(
        "train", {}
    ).get(
        "n_splits", 5
    )

    seeds = config.get(
        "train", {}
    ).get(
        "seeds",
        [42, 52, 62, 72, 82]
    )

    early_stopping_rounds = config.get(
        "train", {}
    ).get(
        "early_stopping_rounds", 50
    )

    # =========================================================
    # 2. 保存用変数
    # =========================================================

    # 最終的なseed-average OOF
    oof = np.zeros(len(X))

    # 全モデル
    models = []

    # 全モデルのfeature importance
    feature_importances_list = []

    # foldごとのスコア
    fold_scores = []

    # seedごとのOOF AUC
    seed_scores = []

    # =========================================================
    # 3. 学習開始
    # =========================================================

    total_models = len(seeds) * n_splits

    print(
        f"Starting Training: "
        f"{len(seeds)} seeds * "
        f"{n_splits} folds "
        f"= {total_models} models"
    )

    # =========================================================
    # 4. Seed loop
    # =========================================================

    for seed_idx, seed in enumerate(seeds):

        # このseed専用OOF
        seed_oof = np.zeros(len(X))

        # seedごとにCV splitを変える
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )

        # =====================================================
        # 5. Fold loop
        # =====================================================

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X, y),
            start=1
        ):

            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx]

            X_val = X.iloc[val_idx].copy()
            y_val = y.iloc[val_idx]

            # =================================================
            # 6. CV内特徴量生成
            # =================================================

            X_train, stats_dict = create_cv_features(
                X_train
            )

            X_val, _ = create_cv_features(
                X_val,
                stats_dict=stats_dict
            )

            # =================================================
            # 7. 配列データチェック
            # =================================================

            for col in X_train.columns:

                if X_train[col].apply(
                    lambda x: isinstance(
                        x,
                        (list, np.ndarray)
                    )
                ).any():

                    raise ValueError(
                        f"【原因列】{col} "
                        f"に配列データが入っています。"
                    )

            # =================================================
            # 8. 特徴量列の整合性確認
            # =================================================

            if list(X_train.columns) != list(X_val.columns):

                raise ValueError(
                    f"Seed {seed}, Fold {fold} で"
                    f"train / validationの特徴量列が一致していません。"
                )

            feature_names = list(X_train.columns)

            # =================================================
            # 9. LightGBM
            # =================================================

            current_params = model_params.copy()

            # seedをLightGBMにも渡す
            current_params["random_state"] = seed

            model = lgb.LGBMClassifier(
                **current_params
            )

            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=False
                ),
                lgb.log_evaluation(
                    period=0
                )
            ]

            # =================================================
            # 10. 学習
            # =================================================

            model.fit(
                X_train,
                y_train,
                eval_X=X_val, 
                eval_y=y_val,
                callbacks=callbacks
            )

            # =================================================
            # 11. OOF予測
            # =================================================

            val_preds = model.predict_proba(
                X_val
            )[:, 1]

            # seed単位OOF
            seed_oof[val_idx] = val_preds

            # =================================================
            # 12. fold評価
            # =================================================

            fold_auc = roc_auc_score(
                y_val,
                val_preds
            )

            fold_f1 = f1_score(
                y_val,
                (val_preds >= 0.5).astype(int)
            )

            fold_scores.append(
                {
                    "seed": int(seed),
                    "fold": int(fold),
                    "auc": float(fold_auc),
                    "f1": float(fold_f1),
                    "best_iteration": int(
                        model.best_iteration_
                    )
                }
            )

            print(
                f"Seed {seed} "
                f"Fold {fold} "
                f"AUC: {fold_auc:.5f} "
                f"F1: {fold_f1:.5f} "
                f"BestIter: {model.best_iteration_}"
            )

            # =================================================
            # 13. モデル保存
            # =================================================

            models.append(
                {
                    "seed": int(seed),
                    "fold": int(fold),
                    "model": model
                }
            )

            # =================================================
            # 14. Feature Importance
            # =================================================

            feature_importances_list.append(
                model.feature_importances_
            )

        # =====================================================
        # 15. Seed単位のOOF評価
        # =====================================================

        seed_auc = roc_auc_score(
            y,
            seed_oof
        )

        seed_f1 = f1_score(
            y,
            (seed_oof >= 0.5).astype(int)
        )

        seed_scores.append(
            {
                "seed": int(seed),
                "auc": float(seed_auc),
                "f1": float(seed_f1)
            }
        )

        print(
            f"\nSeed {seed} "
            f"OOF AUC: {seed_auc:.5f} "
            f"F1: {seed_f1:.5f}"
        )

        # =====================================================
        # 16. 最終OOFに加算
        # =====================================================

        oof += seed_oof / len(seeds)

    # =========================================================
    # 17. 最終seed-average OOF
    # =========================================================

    cv_auc = roc_auc_score(
        y,
        oof
    )

    cv_f1 = f1_score(
        y,
        (oof >= 0.5).astype(int)
    )

    # =========================================================
    # 18. Feature Importance
    # =========================================================

    feature_importances = pd.DataFrame(
        feature_importances_list,
        columns=feature_names
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,

            "importance_mean":
                feature_importances.mean(
                    axis=0
                ).values,

            "importance_std":
                feature_importances.std(
                    axis=0
                ).values
        }
    )

    importance = (
        importance
        .sort_values(
            by="importance_mean",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # 19. Metrics
    # =========================================================

    metrics = {
        "cv_auc": float(cv_auc),

        "cv_f1_at_0.5": float(cv_f1),

        "n_seeds": int(len(seeds)),

        "seeds": [int(s) for s in seeds],

        "n_splits": int(n_splits),

        "n_models": int(len(models)),

        "fold_scores": fold_scores,

        "seed_scores": seed_scores,

        "auc_mean": float(
            np.mean(
                [
                    x["auc"]
                    for x in fold_scores
                ]
            )
        ),

        "auc_std": float(
            np.std(
                [
                    x["auc"]
                    for x in fold_scores
                ]
            )
        ),

        "f1_mean": float(
            np.mean(
                [
                    x["f1"]
                    for x in fold_scores
                ]
            )
        ),

        "f1_std": float(
            np.std(
                [
                    x["f1"]
                    for x in fold_scores
                ]
            )
        )
    }

    # =========================================================
    # 20. Ensemble
    # =========================================================

    ensemble_model = EnsembleModel(
        models
    )

    return (
        ensemble_model,
        oof,
        importance,
        metrics
    )
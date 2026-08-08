import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

def create_group_relative_features(
    df: pd.DataFrame, 
    group_col: str = '業界グループ', 
    target_cols: list = None,
    stats_dict: dict = None
):
    if target_cols is None:
        target_cols = [
            'r_sw支出_対_売上比',
            'r_有利子負債営業CF倍率',
            '従業員数',
            'r_ROA_経常利益'
        ]

    df_out = df.copy()
    eps = 1e-6

    # カテゴリ型対策
    group_series = df_out[group_col].astype(str)

    # 1. 統計情報の集計 (Train適用時)
    if stats_dict is None:
        stats_dict = {}
        for col in target_cols:
            means_series = df_out.groupby(group_col, observed=True)[col].mean()
            stats_dict[col] = {
                'means': means_series.dropna().to_dict(),
                'dists': df_out.groupby(group_col, observed=True)[col].apply(lambda x: x.dropna().values).to_dict(),
                'global_mean': float(df_out[col].mean()),
                'global_dist': df_out[col].dropna().values
            }

    # 2. 特徴量の作成
    for col in target_cols:
        col_stats = stats_dict[col]
        group_means = col_stats['means']
        group_dists = col_stats['dists']
        global_mean = col_stats['global_mean']
        global_dist = col_stats['global_dist']

        # A. 業界グループ平均差
        means = group_series.map(group_means).fillna(global_mean).astype(float)
        df_out[f'{col}_業界グループ平均差'] = df_out[col].astype(float) - means

        # B. 業界グループ平均比
        df_out[f'{col}_業界グループ平均比'] = df_out[col].astype(float) / (means + eps)

        # C. 業界グループパーセンタイル
        def calc_percentile(row):
            val = row[col]
            grp = str(row[group_col])
            if pd.isna(val):
                return np.nan
            
            dist = group_dists.get(grp, global_dist)
            if len(dist) == 0:
                return np.nan
            
            # percentileofscore の結果を確実に float（数値）に変換
            pct_val = percentileofscore(dist, val, kind='weak') / 100.0
            return float(pct_val)

        # apply の結果を float 型で明示的に保持
        df_out[f'{col}_業界グループpct'] = df_out.apply(calc_percentile, axis=1).astype(float)

    return df_out, stats_dict

# cv_features.py
def create_cv_features(df, stats_dict=None):
    return create_group_relative_features(df, stats_dict=stats_dict)
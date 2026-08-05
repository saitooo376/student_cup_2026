import pandas as pd
import numpy as np

def add_base_fetures(df):
    import numpy as np
import pandas as pd


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    description.csvの定義に基づき、一般的な財務比率・指標（特徴量）を追加する関数.

    Parameters
    ----------
    df : pd.DataFrame
        財務・企業情報が格納された DataFrame

    Returns
    -------
    pd.DataFrame
        財務比率特徴量が追加された DataFrame
    """
    df = df.copy()

    # ゼロ割りを防止するヘルパー関数 (分母の0をNaNに置換)
    def safe_div(num, den):
        den_clean = den.replace(0, np.nan)
        return num / den_clean

    # 中間指標・集計値の作成
    # 有利子負債（短期借入金 + 長期借入金）
    if "短期借入金" in df.columns and "長期借入金" in df.columns:
        df["有利子負債"] = df["短期借入金"].fillna(0) + df["長期借入金"].fillna(0)

    # フリーキャッシュフロー（営業CF + 投資CF）
    if "営業CF" in df.columns and "投資CF" in df.columns:
        df["FCF"] = df["営業CF"].fillna(0) + df["投資CF"].fillna(0)

    # 拠点数合計
    location_cols = [
        c for c in ["事業所数", "工場数", "店舗数"] if c in df.columns
    ]
    if location_cols:
        df["拠点数合計"] = df[location_cols].sum(axis=1)

    # 収益性指標 (Profitability Ratios)
    if "売上" in df.columns:
        if "営業利益" in df.columns:
            df["売上高営業利益率"] = safe_div(df["営業利益"], df["売上"])
        if "経常利益" in df.columns:
            df["売上高経常利益率"] = safe_div(df["経常利益"], df["売上"])
        if "当期純利益" in df.columns:
            df["売上高純利益率"] = safe_div(df["当期純利益"], df["売上"])

    if "総資産" in df.columns:
        if "営業利益" in df.columns:
            df["ROA_営業利益"] = safe_div(df["営業利益"], df["総資産"])
        if "経常利益" in df.columns:
            df["ROA_経常利益"] = safe_div(df["経常利益"], df["総資産"])
        if "当期純利益" in df.columns:
            df["ROA_当期純利益"] = safe_div(df["当期純利益"], df["総資産"])

    if "自己資本" in df.columns and "当期純利益" in df.columns:
        df["ROE"] = safe_div(df["当期純利益"], df["自己資本"])


    # 安全性・財務健全性指標 (Safety & Capital Structure Ratios)
    if "総資産" in df.columns:
        if "自己資本" in df.columns:
            df["自己資本比率"] = safe_div(df["自己資本"], df["総資産"])
        if "負債" in df.columns:
            df["負債比率_対総資産"] = safe_div(df["負債"], df["総資産"])
        if "流動資産" in df.columns:
            df["流動資産比率"] = safe_div(df["流動資産"], df["総資産"])
        if "固定資産" in df.columns:
            df["固定資産比率"] = safe_div(df["固定資産"], df["総資産"])
        if "有利子負債" in df.columns:
            df["有利子負債依存度"] = safe_div(df["有利子負債"], df["総資産"])

    if "自己資本" in df.columns:
        if "負債" in df.columns:
            # D/Eレシオ
            df["デット・エクイティ・レシオ"] = safe_div(
                df["負債"], df["自己資本"]
            )
        if "有利子負債" in df.columns:
            df["有利子負債倍率"] = safe_div(df["有利子負債"], df["自己資本"])


    # 効率性指標 (Efficiency Ratios)
    if "売上" in df.columns and "総資産" in df.columns:
        df["総資産回転率"] = safe_div(df["売上"], df["総資産"])


    # 生産性・1人当たり指標 (Productivity Ratios)
        if "売上" in df.columns:
            df["一人当たり売上"] = safe_div(df["売上"], df["従業員数"])
        if "営業利益" in df.columns:
            df["一人当たり営業利益"] = safe_div(
                df["営業利益"], df["従業員数"]
            )
        if "経常利益" in df.columns:
            df["一人当たり経常利益"] = safe_div(
                df["経常利益"], df["従業員数"]
            )
        if "総資産" in df.columns:
            df["一人当たり総資産"] = safe_div(df["総資産"], df["従業員数"])


    # キャッシュフロー指標 (Cash Flow Ratios)
    if "営業CF" in df.columns:
        if "売上" in df.columns:
            df["営業CFマージン"] = safe_div(df["営業CF"], df["売上"])
        if "有利子負債" in df.columns:
            # 有利子負債返済能力（営業CFに対する有利子負債の倍率）
            df["有利子負債営業CF倍率"] = safe_div(
                df["有利子負債"], df["営業CF"]
            )

    return df


import numpy as np
import pandas as pd


def add_softwhere_features(df: pd.DataFrame) -> pd.DataFrame:
    """総資産、売上、従業員数に対する無形固定資産変動の対数特徴量を追加する関数.

    ※ 不要な中間対数カラム（log_総資産など）は削除し、最終結果のみを保持します。

    Parameters
    ----------
    df : pd.DataFrame
        財務・企業情報が格納された DataFrame

    Returns
    -------
    pd.DataFrame
        最終特徴量のみが追加された DataFrame
    """
    df = df.copy()

    # ゼロ割りを防止するヘルパー関数
    def safe_div(num, den):
        den_clean = den.replace(0, np.nan)
        return num / den_clean

    # 無形固定資産変動のカラム名を判定
    intangible_col = "無形固定資産変動(ソフトウェア関連)"
      
    # 一時的に使用する作業用カラム名のリスト
    temp_cols = []

    # 1. 一時的な対数変数の作成
    if intangible_col is not None:
        abs_change = -df[intangible_col]
        df["_temp_log_無形正数"] = np.log1p(abs_change.clip(lower=0))
        temp_cols.append("_temp_log_無形正数")

    if "総資産" in df.columns:
        df["_temp_log_総資産"] = np.log1p(df["総資産"].clip(lower=0))
        temp_cols.append("_temp_log_総資産")

    if "売上" in df.columns:
        df["_temp_log_売上"] = np.log1p(df["売上"].clip(lower=0))
        temp_cols.append("_temp_log_売上")

    if "従業員数" in df.columns:
        df["_temp_log_従業員数"] = np.log1p(df["従業員数"].clip(lower=0))
        temp_cols.append("_temp_log_従業員数")

    # 2. 最終特徴量の作成
    if "_temp_log_無形正数" in df.columns:
        if "_temp_log_総資産" in df.columns:
            df["sw支出_対_総資産比"] = safe_div(
                df["_temp_log_無形正数"], df["_temp_log_総資産"]
            )

        if "_temp_log_売上" in df.columns:
            df["sw支出_対_売上比"] = safe_div(
                df["_temp_log_無形正数"], df["_temp_log_売上"]
            )

        if "_temp_log_従業員数" in df.columns:
            df["一人当たりsw支出"] = safe_div(
                df["_temp_log_無形正数"], df["_temp_log_従業員数"]
            )

    # 3. 中間作成した作業用対数カラムの破棄
    df = df.drop(columns=temp_cols, errors="ignore")

    return df


#　上と一番効いていた特徴量にかける
# def add_softwhere_survey_features(df: pd.DataFrame) -> pd.DataFrame:
#     """無形固定資産変動の対数特徴量に対し、アンケート項目（1, 2, 7, 8）を掛け合わせた特徴量を追加する関数.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         財務・企業情報、対数特徴量、およびアンケート結果が格納された DataFrame

#     Returns
#     -------
#     pd.DataFrame
#         アンケート項目との掛け算特徴量が追加された DataFrame
#     """
#     df = df.copy()

#     # 対象とする無形固定資産変動系の対数特徴量カラムを特定
#     target_log_cols = [
#         col
#         for col in [
#             "log_無形固定資産変動_対_総資産比",
#             "log_無形固定資産変動_対_売上比",
#             "log_一人当たり無形固定資産変動",
#         ]
#         if col in df.columns
#     ]

#     # 掛け算対象となる対数特徴量が存在しない場合は警告を出してそのまま返す
#     if not target_log_cols:
#         print(
#             "Warning: 対象となる対数特徴量が見つかりませんでした。"
#             "先に add_softwhere_features() を実行してください。"
#         )
#         return df

#     # 1. (6 - アンケート２) * アンケート７ * アンケート８ の重み計算
#     q_cols_A = ["アンケート２", "アンケート７", "アンケート８"]
#     if all(col in df.columns for col in q_cols_A):
#         weight_A = (
#             (6 - df["アンケート２"])
#             * df["アンケート７"]
#             * df["アンケート８"]
#         )

#         for col in target_log_cols:
#             df[f"{col}_x_アンケート2_7_8"] = df[col] * weight_A

#     # 2. アンケート１ の重み計算
#     if "アンケート１" in df.columns:
#         weight_B = df["アンケート１"]

#         for col in target_log_cols:
#             df[f"{col}_x_アンケート1"] = df[col] * weight_B

#     return df


def create_features(df):
    df = add_financial_features(df)
    df = add_softwhere_features(df)
    return df

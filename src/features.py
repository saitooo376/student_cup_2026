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





def create_features(df):
    df = add_financial_features(df)
    return df

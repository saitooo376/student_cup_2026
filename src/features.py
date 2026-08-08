import pandas as pd
import numpy as np


import numpy as np
import pandas as pd


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """description.csvの定義に基づき、一般的な財務比率・指標（特徴量）を追加する関数.

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

    # 安全な割り算を行うヘルパー関数 (分母が 0 以下の場合は NaN に置換)
    def safe_div(num, den):
        den_clean = den.where(den > 0, np.nan)
        return num / den_clean

    # 1. 中間指標・集計値の作成（※金額や個数の合計のため比率プレフィックス r_ は付与しない）
    # 有利子負債（短期借入金 + 長期借入金）
    if "短期借入金" in df.columns and "長期借入金" in df.columns:
        df["有利子負債"] = df["短期借入金"].fillna(0) + df[
            "長期借入金"
        ].fillna(0)

    # フリーキャッシュフロー（営業CF + 投資CF）
    if "営業CF" in df.columns and "投資CF" in df.columns:
        df["FCF"] = df["営業CF"].fillna(0) + df["投資CF"].fillna(0)

    # 拠点数合計
    location_cols = [
        c for c in ["事業所数", "工場数", "店舗数"] if c in df.columns
    ]
    if location_cols:
        df["拠点数合計"] = df[location_cols].sum(axis=1)

    # 2. 収益性指標 (Profitability Ratios)
    if "売上" in df.columns:
        if "営業利益" in df.columns:
            df["r_売上高営業利益率"] = safe_div(df["営業利益"], df["売上"])
        if "経常利益" in df.columns:
            df["r_売上高経常利益率"] = safe_div(df["経常利益"], df["売上"])
        if "当期純利益" in df.columns:
            df["r_売上高純利益率"] = safe_div(df["当期純利益"], df["売上"])

    if "総資産" in df.columns:
        if "営業利益" in df.columns:
            df["r_ROA_営業利益"] = safe_div(df["営業利益"], df["総資産"])
        if "経常利益" in df.columns:
            df["r_ROA_経常利益"] = safe_div(df["経常利益"], df["総資産"])
        if "当期純利益" in df.columns:
            df["r_ROA_当期純利益"] = safe_div(df["当期純利益"], df["総資産"])

    if "自己資本" in df.columns and "当期純利益" in df.columns:
        df["r_ROE"] = safe_div(df["当期純利益"], df["自己資本"])

    # 3. 安全性・財務健全性指標 (Safety & Capital Structure Ratios)
    if "総資産" in df.columns:
        if "自己資本" in df.columns:
            df["r_自己資本比率"] = safe_div(df["自己資本"], df["総資産"])
        if "負債" in df.columns:
            df["r_負債比率_対総資産"] = safe_div(
                df["負債"], df["総資産"]
            )
        if "流動資産" in df.columns:
            df["r_流動資産比率"] = safe_div(df["流動資産"], df["総資産"])
        if "固定資産" in df.columns:
            df["r_固定資産比率"] = safe_div(df["固定資産"], df["総資産"])
        if "有利子負債" in df.columns:
            df["r_有利子負債依存度"] = safe_div(
                df["有利子負債"], df["総資産"]
            )

    if "自己資本" in df.columns:
        if "負債" in df.columns:
            df["r_デット・エクイティ・レシオ"] = safe_div(
                df["負債"], df["自己資本"]
            )
        if "有利子負債" in df.columns:
            df["r_有利子負債倍率"] = safe_div(
                df["有利子負債"], df["自己資本"]
            )

    # 4. 効率性指標 (Efficiency Ratios)
    if "売上" in df.columns and "総資産" in df.columns:
        df["r_総資産回転率"] = safe_div(df["売上"], df["総資産"])

    # 5. 生産性・1人当たり指標 (Productivity Ratios)
    if "従業員数" in df.columns:
        if "売上" in df.columns:
            df["r_一人当たり売上"] = safe_div(
                df["売上"], df["従業員数"]
            )
        if "営業利益" in df.columns:
            df["r_一人当たり営業利益"] = safe_div(
                df["営業利益"], df["従業員数"]
            )
        if "経常利益" in df.columns:
            df["r_一人当たり経常利益"] = safe_div(
                df["経常利益"], df["従業員数"]
            )
        if "総資産" in df.columns:
            df["r_一人当たり総資産"] = safe_div(
                df["総資産"], df["従業員数"]
            )

    # 6. キャッシュフロー指標 (Cash Flow Ratios)
    if "営業CF" in df.columns:
        if "売上" in df.columns:
            df["r_営業CFマージン"] = safe_div(df["営業CF"], df["売上"])
        if "有利子負債" in df.columns:
            df["r_有利子負債営業CF倍率"] = safe_div(
                df["有利子負債"], df["営業CF"]
            )

    return df


def add_software_features(df: pd.DataFrame) -> pd.DataFrame:
    """総資産、売上、従業員数に対する無形固定資産変動（ソフトウェア）の比率特徴量を追加する関数.

    Parameters
    ----------
    df : pd.DataFrame
        財務・企業情報が格納された DataFrame

    Returns
    -------
    pd.DataFrame
        比率特徴量（r_sw支出_対_総資産比 等）が追加された DataFrame
    """
    df = df.copy()

    # 安全な割り算を行うヘルパー関数 (分母が 0 以下の場合は NaN に置換)
    def safe_div(num, den):
        den_clean = den.where(den > 0, np.nan)
        return num / den_clean

    intangible_col = "無形固定資産変動(ソフトウェア関連)"

    if intangible_col in df.columns:
        # 会計上の変動額から正の支出額に変換（マイナス値をプラス反転・負の数値は0にクリップ）
        sw_expenditure = (-df[intangible_col]).clip(lower=0)

        # 1. sw支出 対 総資産比
        if "総資産" in df.columns:
            df["r_sw支出_対_総資産比"] = safe_div(
                sw_expenditure, df["総資産"]
            )

        # 2. sw支出 対 売上比
        if "売上" in df.columns:
            df["r_sw支出_対_売上比"] = safe_div(
                sw_expenditure, df["売上"]
            )

        # 3. 一人当たりsw支出
        if "従業員数" in df.columns:
            df["r_一人当たりsw支出"] = safe_div(
                sw_expenditure, df["従業員数"]
            )

    return df



def add_location_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    事業所数・工場数・店舗数・従業員数から拠点集中度や1拠点あたり人数の比率特徴量(r_*)を生成する関数

    Parameters
    ----------
    df : pd.DataFrame
        '従業員数', '事業所数', '工場数', '店舗数' を含むデータフレーム

    Returns
    -------
    pd.DataFrame
        新しい特徴量を追加したデータフレーム
    """
    df = df.copy()

    # 各カラムの取得（欠損値は0として扱う）
    emp = df["従業員数"].fillna(0)
    office = df["事業所数"].fillna(0)
    factory = df["工場数"].fillna(0)
    store = df["店舗数"].fillna(0)

    # 全拠点数（分母）
    total_locations = office + factory + store

    # 1. r_1拠点あたり従業員数
    df["r_1拠点あたり従業員数"] = np.where(
        total_locations > 0, emp / total_locations, np.nan
    )

    # 2. r_オフィス集中度
    df["r_オフィス集中度"] = np.where(
        total_locations > 0, office / total_locations, np.nan
    )

    return df


def transform_numeric_features(
    df: pd.DataFrame,
    skew_threshold: float = 1.5,
    kurt_threshold: float = 5.0,
    iqr_mult_threshold: float = 5.0,
    clip_quantile_lower: float = 0.01,
    clip_quantile_upper: float = 0.99,
) -> pd.DataFrame:
    """全ての数値変数に対して歪度・スパイク判定を行い、

    条件に応じた変換（Clip / log1p / arcsinh）で上書きして返す関数。

    Parameters
    ----------
    df : pd.DataFrame
        対象データフレーム（特徴量追加済みのもの）
    skew_threshold : float (default=1.5)
        歪度の絶対値の閾値（|Skew| > 1.5 で要変換）
    kurt_threshold : float (default=5.0)
        尖度の閾値（Kurt > 5.0 で要変換）
    iqr_mult_threshold : float (default=5.0)
        IQRに対する外れ値倍率の閾値（> 5.0 で要変換）
    clip_quantile_lower : float (default=0.01)
        r_系特徴量のクリッピング下限（1%点）
    clip_quantile_upper : float (default=0.99)
        r_系特徴量のクリッピング上限（99%点）

    Returns
    -------
    df_transformed : pd.DataFrame
        数値変数が変換後の値で上書きされた DataFrame
    summary_df : pd.DataFrame
        各特徴量の統計量と適用された変換処理のログ
    """
    df_transformed = df.copy()

    # 数値型のカラムのみ自動抽出
    numeric_cols = df_transformed.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    stats_list = []

    for col in numeric_cols:
        s = df_transformed[col].dropna()

        # サンプル不足や定数値は変換スキップ
        if len(s) < 3 or s.nunique() <= 1:
            continue

        skew = s.skew()
        kurt = s.kurt()
        has_negative = (s < 0).any()

        # IQRに対する外れ値倍率の計算
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            upper_iqr_mult = (s.max() - q3) / iqr
            lower_iqr_mult = (q1 - s.min()) / iqr
            max_iqr_mult = max(upper_iqr_mult, lower_iqr_mult)
        else:
            max_iqr_mult = 0.0

        # 変換要否の判定
        is_high_skew = abs(skew) > skew_threshold
        is_high_kurt = kurt > kurt_threshold
        is_extreme_outlier = max_iqr_mult > iqr_mult_threshold

        needs_transform = is_high_skew or is_high_kurt or is_extreme_outlier

        applied_action = "なし (変換不要)"

        # ----------------------------------------------------
        # 変換処理の実行（上書き）
        # ----------------------------------------------------
        if needs_transform:
            # 条件1: r_で始まる特徴量 -> クリッピング
            if col.startswith("r_"):
                q_low = df_transformed[col].quantile(clip_quantile_lower)
                q_high = df_transformed[col].quantile(clip_quantile_upper)
                df_transformed[col] = df_transformed[col].clip(
                    lower=q_low, upper=q_high
                )
                applied_action = f"クリッピング ({clip_quantile_lower*100:.0f}%-{clip_quantile_upper*100:.0f}%)"

            # 条件2: 負の値を含む特徴量 -> arcsinh 変換
            elif has_negative:
                df_transformed[col] = np.arcsinh(df_transformed[col])
                applied_action = "arcsinh 変換"

            # 条件3: 全て0以上の値の特徴量 -> log1p 変換
            else:
                df_transformed[col] = np.log1p(df_transformed[col])
                applied_action = "log1p 変換"

        # ログ記録
        stats_list.append({
            "特徴量名": col,
            "判定": "要変換" if needs_transform else "正常",
            "適用処理": applied_action,
            "変換前_歪度(Skew)": round(skew, 2),
            "変換前_尖度(Kurt)": round(kurt, 2),
            "変換前_最大IQR倍率": round(max_iqr_mult, 1),
            "負の値あり": has_negative,
        })

    summary_df = pd.DataFrame(stats_list)
    #summary_df.to_csv('../data/summary_df.csv')

    return df_transformed



def add_gyoukai_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    「業界」列を元に、新しいカテゴリ分類である「業界グループ」列を追加する関数
    """
    # 元の業界から新しい業界グループへのマッピング辞書
    mapping = {
        # 製造業
        '機械': '製造業',
        '製造': '製造業',
        '電気製品': '製造業',
        '食品': '製造業',
        '化学': '製造業',
        '生活用品': '製造業',

        # 自動車
        '自動車・乗り物': '自動車・乗り物',
        
        # 通信
        'IT': 'IT・ゲーム',
        'ゲーム': 'IT・ゲーム',

        # 通信
        '通信': '通信',
        '通信機器': '通信',
        
        # # 建設・不動産
        '建設・工事': '建設・不動産',
        '不動産': '建設・不動産',
        
        # 小売・外食
        '小売': '小売・外食',
        '外食': '小売・外食',

        # アパレル・美容・エンタメ
        'アパレル・美容': 'アパレル・美容・エンタメ',
        'エンタメ': 'アパレル・美容・エンタメ',
        
        # 商社
        '商社': '商社',

        # 運輸・物流
        '運輸・物流': '運輸・物流',
        
        # 金融
        '金融': '金融',
        
        # コンサル・教育
        'コンサルティング': 'コンサル・教育',
        '教育': 'コンサル・教育',
        
        # 医療・福祉
        '医療・福祉': '医療・福祉',

        # 人材
        '人材': '人材',
        
        # エネルギー
        'エネルギー': 'エネルギー',
        
        # # その他サービス
        'その他サービス': 'その他サービス',
        'その他': 'その他サービス',
        '専門サービス': 'その他サービス',
        '広告':'その他サービス',
        'マスコミ':'その他サービス',
        '機械関連サービス':'その他サービス'
    }
    
    
    # マッピング（未定義の値がある場合は元の値を維持）
    df['業界グループ'] = df['業界'].map(mapping).fillna(df['業界'])
    
    return df


def create_features(df):
    df = add_financial_features(df)
    df = add_software_features(df)
    df = add_location_ratio_features(df)
    df = transform_numeric_features(df)
    df = add_gyoukai_group(df)
    return df

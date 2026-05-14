"""
デモ用サンプルデータ生成モジュール。
Kaggleデータがなくてもアプリの動作確認ができるよう、
実際のJRAレースに近いサンプルデータを生成する。
"""
import pandas as pd
import numpy as np
from datetime import date, timedelta
import random

random.seed(42)
np.random.seed(42)

SIRES = ["ディープインパクト", "キングカメハメハ", "ハーツクライ", "ロードカナロア",
         "エピファネイア", "モーリス", "ルーラーシップ", "ダイワメジャー",
         "スクリーンヒーロー", "オルフェーヴル"]
JOCKEYS = ["C.ルメール", "川田将雅", "横山武史", "戸崎圭太", "松山弘平",
           "岩田康誠", "福永祐一", "浜中俊", "武豊", "坂井瑠星",
           "和田竜二", "池添謙一", "北村友一", "吉田隼人", "団野大成"]
VENUES = ["東京", "中山", "阪神", "京都", "中京", "新潟", "福島", "小倉"]
SURFACES = ["芝", "ダート"]
CONDITIONS = ["良", "良", "良", "稍重", "重", "不良"]
DISTANCES = [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500, 3000]
CLASSES = ["G1", "G2", "G3", "オープン", "3勝", "2勝", "1勝", "未勝利"]
CLASS_WEIGHTS = [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.14]

HORSE_NAMES = [
    "テイエムオペラオー", "スペシャルウィーク", "グラスワンダー", "エルコンドルパサー",
    "ディープインパクト", "キタサンブラック", "アーモンドアイ", "ジェンティルドンナ",
    "ゴールドシップ", "ウオッカ", "ダイワスカーレット", "ブエナビスタ",
    "オルフェーヴル", "フェノーメノ", "ジャスタウェイ", "エピファネイア",
    "ラブリーデイ", "ショウナンパンドラ", "ドゥラメンテ", "リアルスティール",
    "モーリス", "ルージュバック", "マリアライト", "ラブリーデイ",
    "サトノクラウン", "ミッキーロケット", "スワーヴリチャード", "レイデオロ",
    "アルアイン", "ペルシアンナイト", "カデナ", "クリンチャー",
    "サウンズオブアース", "トーセンバジル", "プロディガルサン", "ヴィブロス",
    "シャケトラ", "スティッフェリオ", "キセキ", "アエロリット",
]

def generate_demo_race_results(n_rows: int = 5000) -> pd.DataFrame:
    """
    Kaggleデータの構造を模したデモ用レース結果DataFrameを生成する。
    """
    rows = []
    start_date = date(2019, 1, 1)
    end_date = date(2023, 12, 31)
    date_range = (end_date - start_date).days

    for _ in range(n_rows):
        race_date = start_date + timedelta(days=random.randint(0, date_range))
        surface = random.choice(SURFACES)
        distance = random.choice(DISTANCES)
        venue = random.choice(VENUES)
        condition = random.choice(CONDITIONS)
        race_class = random.choices(CLASSES, weights=CLASS_WEIGHTS)[0]
        n_horses = random.randint(8, 18)
        popularity = random.randint(1, n_horses)
        # 人気に基づく着順生成（人気馬ほど好走しやすい）
        win_prob = max(0.02, (n_horses - popularity + 1) / n_horses * 0.25)
        rank = 1 if random.random() < win_prob else random.randint(2, n_horses)
        gate = random.randint(1, 8)
        # オッズは人気から逆算（人気ほど低オッズ）
        base_odds = popularity * 3.5 + random.gauss(0, popularity * 1.5)
        odds = max(1.1, round(base_odds, 1))

        rows.append({
            "date": race_date,
            "venue": venue,
            "race_name": f"{venue}{'重賞' if 'G' in race_class else ''}レース",
            "race_class": race_class,
            "surface": surface,
            "distance": distance,
            "track_condition": condition,
            "horse_name": random.choice(HORSE_NAMES),
            "sire": random.choice(SIRES),
            "jockey": random.choice(JOCKEYS),
            "gate": gate,
            "horse_no": random.randint(1, n_horses),
            "popularity": popularity,
            "odds": odds,
            "rank": rank,
            "horse_weight": random.randint(440, 540),
            "weight_carried": random.choice([54, 55, 56, 57, 58]),
            "last_3f": round(random.gauss(35.5, 1.2), 1),
            "corner_order": f"{random.randint(1, n_horses)}-{random.randint(1, n_horses)}-{random.randint(1, n_horses)}-{random.randint(1, n_horses)}",
        })

    df = pd.DataFrame(rows)
    # 派生列の追加
    from data_loader import categorize_distance
    df["distance_cat"] = df["distance"].apply(categorize_distance)
    df["win_flag"] = (df["rank"] == 1).astype(int)
    df["place_flag"] = (df["rank"] <= 3).astype(int)
    df["implied_prob"] = 1.0 / df["odds"]
    return df


def get_demo_race_entries(
    surface: str = "芝",
    distance: int = 2000,
    venue: str = "東京",
    condition: str = "良",
    n_horses: int = 16,
) -> list[dict]:
    """
    デモ用の出走馬リストを生成する（手動入力フォームの初期値として使用）。
    """
    entries = []
    popularities = list(range(1, n_horses + 1))
    random.shuffle(popularities)

    for i in range(n_horses):
        pop = popularities[i]
        base_odds = pop * 3.2 + abs(random.gauss(0, pop * 1.2))
        odds = max(1.1, round(base_odds, 1))
        gate = ((i) // 2) + 1

        entries.append({
            "horse_no": str(i + 1),
            "horse_name": random.choice(HORSE_NAMES),
            "jockey": random.choice(JOCKEYS),
            "gate": gate,
            "odds": odds,
            "popularity": pop,
            "sire": random.choice(SIRES),
            "horse_weight": random.randint(450, 520),
            "surface": surface,
            "distance": distance,
            "venue": venue,
            "track_condition": condition,
        })
    return sorted(entries, key=lambda x: x["popularity"])

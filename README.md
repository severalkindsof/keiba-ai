# 競馬予想AI

JRA 中央競馬専用の期待値（EV）ベース予想サポートツール。
LightGBM + XGBoost + CatBoost の Stacking モデル + Benter Odds Blending + Harville 多券種期待値 + Conformal Prediction を統合。

詳細な変更履歴は [`CHANGELOG.md`](CHANGELOG.md) を参照。

---

## クイックスタート

### 1. 必要パッケージのインストール

```powershell
pip install streamlit pandas numpy plotly
pip install lightgbm xgboost catboost optuna scikit-learn joblib
pip install requests beautifulsoup4 lxml
pip install anthropic                         # Claude AI サマリ（任意）
pip install pyarrow                           # parquet 用
```

### 2. データ準備

TFJV (Target Front JV-Link) から CSV をエクスポートして `C:/TFJV/TXT/` 配下に配置。

```powershell
cd "C:\Users\somaf\Desktop\keiba-ai"
python convert_tfjv.py    # TFJV CSV → tfjv_all.parquet
```

### 3. モデル学習（初回・週次）

```powershell
python update_all.py      # 全パイプライン一括（推定 15〜30分）
```

または個別実行：

```powershell
python train_lgbm.py            # LightGBM 学習 + Isotonic 校正 + Benter重みフィット + Conformal 再フィット
python train_stacking.py        # XGBoost + CatBoost + メタモデル
python tune_lgbm.py --trials 50 # Optuna ハイパーパラメータ最適化（任意・夜間）
python horse_elo.py             # 馬 Elo レーティング更新
python trainer_jockey_matrix.py # 厩舎×騎手マトリクス
python speed_index.py           # タイム指数ベースライン
python track_variant.py 2024-01-01  # トラックバイアス時系列
python course_bias.py           # コースバイアス恒常推定
python market_prob.py           # 市場確率テーブル
```

### 4. アプリ起動

```powershell
python -m streamlit run app.py
```

スマホアクセス用に LAN URL も表示されます。

---

## 主要機能

### モデル
- **LightGBM Ranker**：44 特徴量、AUC 0.83 / Brier 0.059 / Walk-Forward 評価
- **3-way Stacking**：LightGBM + XGBoost + CatBoost + メタモデルで安定化（AUC 0.83）
- **Benter Odds Blending**：自モデルと市場確率（人気→経験勝率）を log-linear 統合
- **Isotonic + Beta Calibration**：確率値の校正
- **Conformal Prediction**：レース単位の予測信頼区間 → 見送り判定

### ファクター
- 馬 Elo レーティング（Margin-of-Victory 拡張）
- PCI / RPCI（ペースチェンジ指数）
- 自作タイム指数（条件別基準）
- 厩舎×騎手ペア統計（21,712 ペア）
- 厩舎 30日 form（上り調子検知）
- コースバイアス恒常（外枠/内枠有利）
- 直近トラックバイアス（日次速報）
- ペース × PCI マッチ判定

### ベッティング
- Harville (1973) 多券種確率モデル（馬連・馬単・ワイド・三連複・三連単）
- 多券種オッズ自動取得（netkeiba API）
- 軸+押さえ自動構成（馬連・三連複）
- 複数券種 variance 分散（単勝・複勝・ワイド）
- 分数 Kelly（1/4〜フル）+ Multi-Bet Portfolio Kelly
- 動的 EV 閾値（人気帯別 0.05〜0.35）
- 連敗時の自動ストップロス

### UI
- Anthropic 風アイボリー基調 + Noto Serif JP + テラコッタアクセント
- モバイル対応（@media max-width: 768px）
- Claude AI による上位3頭の短評（任意・要 API キー）

---

## 監査ツール

```powershell
python tools/audit.py                       # 全項目
python tools/audit.py --section mismatch    # confluence 期待 vs 実供給キーの不整合
python tools/audit.py --section ttl          # キャッシュ TTL 分布（60 / 900 / 3600 階層）
python tools/audit.py --section runtime      # 実走時 eval_df.columns ベースの真の整合性
```

---

## 設定ファイル

### `.streamlit/secrets.toml`

任意：netkeiba 認証 Cookie / Anthropic API キー / Discord Webhook 等。

```toml
[netkeiba]
cookie = "TlRBMU5...="     # netkeiba ログイン Cookie

ANTHROPIC_API_KEY = "sk-ant-..."           # Claude AI サマリ用
DISCORD_WEBHOOK_URL = "https://..."        # 通知用
```

### `data/lgbm_best_params.json`
Optuna で最適化されたハイパーパラメータ。`tune_lgbm.py` 実行で生成、`train_lgbm.py` が次回実行時に自動読込。

### `data/benter_weights.json`
Benter Odds Blending の α / β 重み。`train_lgbm.py` 末尾で自動フィット。

### `data/conformal_q.json`
Conformal Prediction の q_alpha。

### `data/stacking_weights.json`
3-way Stacking のメタモデル重み（w_lgbm / w_xgb / w_cb）。

---

## ディレクトリ構成

```
keiba-ai/
├── app.py                  # メイン Streamlit アプリ
├── train_lgbm.py           # LightGBM 学習パイプライン
├── train_stacking.py       # XGBoost + CatBoost + Stacking メタモデル
├── tune_lgbm.py            # Optuna HPO
├── update_all.py           # 全パイプライン一括実行
├── tools/audit.py          # 監査ツール
├── ev_calculator.py        # EV 計算 + LightGBM 推論
├── stacking_predictor.py   # XGBoost + CatBoost 推論
├── confluence.py           # 総合信頼スコア (confidence_score)
├── bet_builder.py          # 馬券構成・Kelly配分
├── harville.py             # 多券種確率モデル
├── conformal.py            # Conformal Prediction
├── portfolio_kelly.py      # ポートフォリオ Kelly
├── market_prob.py          # 市場確率テーブル
├── horse_elo.py            # 馬 Elo レーティング
├── pci_calculator.py       # PCI / RPCI
├── speed_index.py          # タイム指数
├── track_variant.py        # トラックバイアス（日次）
├── course_bias.py          # コースバイアス（恒常）
├── trainer_jockey_matrix.py# 厩舎×騎手統計
├── pace_analyzer.py        # ペース予測
├── ensemble.py             # アンサンブル合成
├── favorite_longshot.py    # Snowberg-Wolfers 補正
├── ai_summary.py           # Claude AI 短評
├── ui_theme.py             # UI テーマ・色定数
├── utils.py                # 共通ヘルパー (safe_int / safe_float / to_seconds)
├── scraper.py              # netkeiba スクレイピング
├── data_loader.py          # 過去データロード
├── data/                   # 統計テーブル・モデル
├── assets/style.css        # カスタム CSS
└── .streamlit/config.toml  # Streamlit テーマ
```

---

## トラブルシューティング

### `confluence.WEIGHTS 合計が 1.0 から外れています` warning
`confluence.py` の WEIGHTS 辞書を変更した時。合計が 1.0 になるよう調整してください。

### 「未確定オッズ」表示
発走直前にオッズ取得できなかった馬。「🔄 オッズ・出走馬を再取得」で更新。

### 「データ不足」表示
horse_latest_features に過去5戦未満の馬。LightGBM 予測スキップ、ルックアップフォールバック使用。

### 監査で `mismatch` 発生
新しいファクターを追加した時、`_bonus_cols` リストに追加忘れの可能性。`python tools/audit.py --section mismatch` で確認。

### GitHub Actions の失敗通知
`.github/workflows/prefetch_weekend.yml` は TFJV 移行で不要になり DEPRECATED。schedule 無効化済み（2026-06-09）。

---

## 参考文献

- Bill Benter (1994) "Computer Based Horse Race Handicapping and Wagering Systems"
- Ziemba & Hausch "Dr. Z's Beat the Racetrack"
- Harville (1973) "Assigning probabilities to the outcomes of multi-entry competitions"
- Snowberg & Wolfers (2010) "Explaining the Favorite-Longshot Bias"
- Kull et al. (2017) "Beta calibration"
- Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
- Kovalchik (2020) "Extension of the Elo rating system to margin of victory"

学術的背景の詳細は `CHANGELOG.md` の最終セクション参照。

# 競馬AI 残タスク

最終更新: 2026-06-09

これまでに第1〜第4波で多くのファクター・モデルを実装済み。
本ファイルは「意図的に見送ったもの／優先度を下げたもの」を一覧化したもの。
着手する場合は上から順を推奨。

---

## 🟠 P3：着手価値が高い残タスク

### SUPER-7: 時系列バリデーション体系化

- **現状**: `train_lgbm.py` は学習/検証/テストの時系列分割が緩い（または無作為）。`_run_backtest` 関数も `df.sample()` を使っており、リーケージのリスクあり。
- **やること**:
  - `train_lgbm.py` に `train_test_split_time(df, train_until, valid_until, test_until)` を導入
  - 例: 学習 2010-2024 / 検証 2025前半 / テスト 2025後半
  - LightGBM の `early_stopping_rounds=50` を valid set 基準で適用
  - 検証後にモデル再保存 + `fit_calibration.py` も自動で再実行する一連スクリプトに統合
- **作業量**: 1日
- **期待効果**: バックテスト数値の信頼性向上、過学習の検出
- **関連ファイル**: `train_lgbm.py`, `fit_calibration.py`, `ev_calculator.py`, `app.py:84-162`（`_run_backtest`）

### NEW-5: トラックバイアス（馬場差）の時系列記録

- **現状**: `realtime_bias.py` は当日の単発入力のみ。日付別の蓄積・推移グラフ無し。
- **やること**:
  - 新規 `track_variant.py` 作成 — その日の全レースの「平均ラスト3F vs 過去同条件平均」差を自動計算
  - `data/track_variants.json`（日付×会場×馬場 → バイアスベクトル）を蓄積
  - 過去30日のバイアス推移を可視化（折線グラフ）
- **作業量**: 1〜2日
- **関連ファイル**: `realtime_bias.py`（既存ロジック）, 新規 `track_variant.py`

### オッズ自動取得（多券種）

- **現状**: 単勝・複勝のみ自動取得。三連複・馬連・馬単・三連単は手動入力。
- **やること**:
  - `scraper.py` に `fetch_multi_odds(race_id, ticket_types)` を追加
  - netkeiba API: `type=4`（馬連）, `type=6`（馬単）, `type=7`（三連複）, `type=8`（三連単）
  - キャッシュ TTL は単勝と同じ 900 秒
  - Harville タブで自動的に「公平オッズ vs 実オッズ」の差分（+EV組合せ）を一覧化
- **作業量**: 半日〜1日
- **関連ファイル**: `scraper.py:230` `_enrich_odds`, `app.py` 馬券構成タブの Harville セクション

---

## 🟡 P4：余力があれば

### NEW-2: PCI / RPCI（ペースチェンジ指数）

- **公式**: `PCI = (前半部3F換算 / 上り3F) × 100 - 50`
- **データ源**: 既に `tfjv_all.parquet` に `last_3f` と `finish_time` あり → 既存データで計算可
- **応用**: 「前走 PCI が 50 から大きく離れたが着順は良かった」=ペース崩れの中で力を出した=巻返し候補
- **作業量**: 1日
- **新規ファイル**: `pci_calculator.py`

### NEW-6: アンサンブル（LightGBM + LR + 市場ベースライン）

- **やること**: 既存 LightGBM に加え、ロジスティック回帰（外れ値に強い）と「人気順そのまま」（市場のベースライン）を加重平均
- **重み**: バックテストで最適化
- **作業量**: 2〜3日
- **関連ファイル**: `ev_calculator.py` に `ensemble_predict()` 追加

### NEW-7: タイム指数の自作

- **背景**: Mshimia氏「LightGBM特徴量重要度トップ」
- **公式**: `タイム指数 = (基準タイム - 走破タイム) × 距離係数 + 馬場差補正`
- **既存**: `horse_profiler.calc_time_rank` あり（馬個別）。これとは別の「絶対指数」を作る
- **作業量**: 2〜3日

### NEW-8: 厩舎×乗替マトリクス

- **やること**: `(trainer, jockey)` ペアの過去成績マトリクスを構築 → 初コンビ・好相性コンビを判定
- **新規ファイル**: `trainer_jockey_matrix.py`
- **作業量**: 1日

### NEW-9: 多券種ケリー最適化

- **現状**: `bet_builder.py` は単勝中心。三連複・三連単・ワイドのケリー配分は弱い
- **やること**: スコア上位3頭×相手の組み合わせで、券種ごとに期待値×ケリーで自動配分（Harville の確率を流用）
- **作業量**: 2〜3日

### EFF-2: 複数レース一括分析

- **既存**: `scan_weekend_races` あり
- **やること**: TFJV CSV 群（複数ファイル）を一度に投げると全部分析する UI
- **作業量**: 半日

### EFF-3: Claude 自然言語サマリ

- **既存**: `claude_chat.py` あり
- **やること**: 分析完了後、上位3頭について「なぜこの評価か」を3行で自動生成 → ユーザーは数字を読まずに済む
- **作業量**: 半日

### EFF-4: モバイル縦長カードビュー

- **問題**: スマホで表が崩れる
- **やること**: `hide_index` だけでなく「縦長カード」モードを追加（馬1頭=1カード）
- **作業量**: 半日

### EFF-5: 過去同条件レース自動リコール

- **やること**: 「東京芝2400m良 18頭」のような条件キーで過去5年の同条件レース10件を自動引用 → 平均勝ち時計・PCI・1〜3着馬の傾向を表示
- **作業量**: 1日

### SUPER-8: ΔR² 可視化（ファクター寄与度監視）

- **背景**: Benter「ΔR² 0.01 未満のファクターはノイズ」
- **やること**: バックテストタブに「市場オッズ単独 R²」「自モデル R²」「ブレンド後 R²」を並べる
- **作業量**: 半日

---

## 🔴 既知の小バグ・未処理 ISSUE

### ISSUE-2: `fetch_today_races` 失敗時の `fetch_mode` 上書きが効かない
- **場所**: `app.py:886`
- **現状**: TFJV/直接 race_id モードへ移行することで実害は減ったが、根本未解決
- **作業量**: 2時間

### ISSUE-3: `fetch_race_entries` キャッシュ TTL 900 秒
- **場所**: `scraper.py:154`
- **現状**: 直前オッズ・馬体重発表を取り損ねるリスク
- **やること**: 「🔄 オッズ再取得」ボタン UI 追加 or TTL を 300秒に短縮
- **作業量**: 1時間

### ISSUE-4 + LATENT-10: `odds=10.0 / popularity=9` ハードコードフォールバック
- **場所**: `scraper.py:267-268`, `scraper.py:303`
- **やること**: `None` 許容にして表示・分析側で「未確定」扱い
- **作業量**: 半日

### ISSUE-7: netkeiba Cookie ハードコード
- **場所**: `scraper.py:28` `_nk_cookie = "TlRBMU5USTFOdz09"`
- **やること**: `st.secrets` 経由のみにしてハードコード削除
- **作業量**: 30分

### LATENT-4: `fetch_race_entries` で `horse_no` 空のまま行が混入
- **場所**: `scraper.py:170, 214`
- **やること**: `horse_no` 空も除外
- **作業量**: 15分

### LATENT-9: scraper の `time.sleep(REQUEST_INTERVAL=3)` が呼ぶたび3秒
- **場所**: `scraper.py:25, 47`
- **やること**: API呼び出しごとに別調整、または動的バックオフ
- **作業量**: 半日
- **効果**: スキャン時の致命的な遅さを改善（1レース最低9秒→3秒程度に）

### LATENT-14: state 3変数併存（TFJV/direct/preselected）
- **やること**: `_analysis_target = {"source": ..., "race_id": ...}` への単一state集約
- **作業量**: 半日

### LATENT-18: `_get` の encoding が `apparent_encoding`
- **やること**: `resp.encoding = "EUC-JP"` 固定にして高速化
- **作業量**: 15分

### LATENT-19: 単勝→複勝オッズ取得が直列
- **場所**: `scraper.py:244-252`
- **やること**: `asyncio` / `concurrent.futures` で並列化
- **作業量**: 半日

---

## 📋 まとめ：着手優先順

| 優先 | 項目 | 作業量 | 種別 |
|------|------|--------|------|
| 🟠 P3 | オッズ自動取得（多券種） | 半日〜1日 | 機能追加 |
| 🟠 P3 | LATENT-9 + LATENT-19 並列化 | 1日 | 体感速度↑ |
| 🟠 P3 | ISSUE-7 Cookie 撤廃 | 30分 | セキュリティ |
| 🟠 P3 | SUPER-7 時系列バリデーション | 1日 | モデル品質 |
| 🟠 P3 | NEW-5 トラックバイアス時系列 | 1-2日 | 馬場補正 |
| 🟡 P4 | NEW-2 PCI / RPCI | 1日 | 巻返し発見 |
| 🟡 P4 | EFF-3 Claude 自然言語サマリ | 半日 | UX |
| 🟡 P4 | NEW-9 多券種ケリー最適化 | 2-3日 | 馬券構成 |
| 🟡 P4 | （その他） | 各種 | 各種 |

---

## 🚀 完了済み（参考）

| 波 | 内容 | 主な成果物 |
|----|------|-----------|
| 第1波 | TFJV CSV 連携 + 致命的バグ修正（datetime/venue/horse_weight）+ 直接モード堅牢化 | `extract_race_id_from_tfjv_csv`, `fetch_race_meta` 拡張 |
| 第2波 | Benter Odds Blending + Isotonic 校正 + 分数ケリー | `market_prob.py`, `ev_calculator.blend_with_market`, `fit_calibration.py` |
| 第3波 | Harville モデル（5券種の確率と公平オッズ） | `harville.py`, 馬券構成タブ拡張 |
| 第4波 | 累乗フィルター + 予測確度バナー + データ質除外 + お任せモード + Favorite-Longshot + 馬 Elo | `bet_builder.apply_power_filter/apply_data_quality_filter`, `favorite_longshot.py`, `horse_elo.py` |

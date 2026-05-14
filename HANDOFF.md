# 競馬予想AI 引き継ぎドキュメント

## プロジェクト概要

**目的**: JRA中央競馬専用の予想分析サポートWebアプリ  
**ユーザー**: 2年間で70〜80万円負けた穴馬狙いのユーザー（非エンジニア）  
**起動**: `streamlit run app.py`  
**場所**: `C:\Users\somaf\Desktop\keiba-ai\`

---

## ユーザーの特性（AIへの注意事項）

- 穴馬の選定は合っているが、**人気馬の相手選びで外す**のが最大の弱点
- **叩き1走目は買わない**（rotation_weight.py でペナルティ -0.03〜-0.05）
- 1番人気は嫌う傾向あり（過剰人気を嫌う）
- 芝↔ダート変わりはむしろ積極的に買う（season_climate.py でボーナス）
- 3連複（穴馬軸×人気馬×流し）と馬連がメイン、3連単は2〜3点
- **ハンデ戦の土曜平場**で衝動買いしやすい → ブレーキ機能あり
- 週予算 5,000円/レース、10,000円/週末

---

## ファイル構成（28ファイル）

### エントリポイント
| ファイル | 役割 |
|---------|------|
| `app.py` | Streamlit メインアプリ（1869行）。全タブのUI・パイプライン統合 |

### データ取得
| ファイル | 役割 |
|---------|------|
| `data_loader.py` | Kaggle JRA CSV読み込み・列名正規化・統計テーブル構築 |
| `scraper.py` | netkeiba スクレイピング（出走表・オッズ・レース結果） |
| `training_scraper.py` | 調教タイム取得・併せ馬パートナー分析・土曜勝ち馬照合 |
| `demo_data.py` | Kaggleデータなし時のデモ用サンプルデータ生成 |

### スコアリング・分析エンジン
| ファイル | 役割 |
|---------|------|
| `ev_calculator.py` | EV計算・勝率推定・knowledge_base を統合した馬別スコア |
| `confluence.py` | 14ファクターを統合した0〜100点の総合信頼スコア |
| `horse_profiler.py` | 馬別詳細プロファイル（1366行）。距離/馬場/会場適性・上がり3F・時計ランク・有力馬撃破・近走詳細分析・右左回り・初条件・厩舎近況・頭数変化・ペース適合 |
| `pace_analyzer.py` | ペース予測・展開恩恵スコア |
| `draw_bias.py` | 枠順バイアス（静的テーブル＋動的バイアス） |
| `realtime_bias.py` | 当日の馬場バイアス分析（ユーザー入力テキスト or ルール） |
| `rotation_weight.py` | ローテ・馬体重・クラス変動・叩き台検出・障害叩き→平地 |
| `jockey_change.py` | 騎手乗り替わり分析・乗り替わり理由推定 |
| `season_climate.py` | 季節適性・馬場状態適性・芝↔ダート転向分析 |
| `race_level.py` | 前走レースレベル・ラップ適性 |
| `position_correction.py` | 前走位置取り補正 |
| `weight_handicap.py` | 斤量馬体重比・頭数×本命信頼度・ハンデ戦斤量トレンド |
| `nicks.py` | ニックス（父×母父の相性）スコア |
| `longshot_evaluator.py` | 穴馬の「ロマン買い vs 構造的根拠あり」7段階判定 |

### 意思決定
| ファイル | 役割 |
|---------|------|
| `bet_builder.py` | 馬券構成の自動提案（3連複・馬連・3連単 予算内最適化） |
| `bankroll_manager.py` | 週末複数レースへの資金配分（修正ケリー基準） |
| `race_selector.py` | 週末スキャン・狙い目レース絞り込み |

### ナレッジベース
| ファイル | 役割 |
|---------|------|
| `knowledge_base.json` | ユーザーの競馬メモ（騎手/血統/レース特性/荒れ条件/短期外国人騎手/格言） |
| `knowledge_base.py` | JSONローダー・各種クエリ関数。app内のメモ編集UIからも更新可能 |

### 通知・外部連携
| ファイル | 役割 |
|---------|------|
| `discord_notify.py` | Discord Webhook通知（オッズ急落・分析完了・買い目確定・週次レポート） |
| `odds_monitor.py` | リアルタイムオッズ監視（20%急落/30%急騰検出） |
| `claude_chat.py` | Claude API チャット（レース分析データ＋KBコンテキストをシステムプロンプトに注入） |

### 記録・振り返り
| ファイル | 役割 |
|---------|------|
| `race_diary.py` | SQLite日記（予想記録・結果取得・Claude振り返り分析・穴馬履歴サマリー） |
| `session_store.py` | 前日分析の保存・翌日読み込み |

---

## アプリのタブ構成（11タブ）

| タブ名 | 主な機能 |
|--------|---------|
| 🏠 ホーム | 週末スキャン結果・資金配分・65点以上馬一覧 |
| 🔭 週末スキャン | 土日の全レース自動分析・推奨レース優先順位 |
| 📋 レース詳細分析 | 出走馬全頭のEV・総合スコア・注目馬アラート一覧 |
| 🐎 馬プロファイル | 1頭詳細（適性・近走詳細分析・トレンドグラフ全部） |
| 💰 馬券構成 | 自動買い目提案・衝動買いブレーキ確信ボタン |
| 🤖 AI相談 | Claude APIチャット（KBメモ自動注入） |
| 📡 オッズ監視 | リアルタイムオッズ変動・Discord自動通知 |
| 📓 振り返り日記 | 予想記録・結果取得・Claude振り返り・穴馬履歴 |
| 📊 バックテスト | EVプラス馬のみ買い続けた場合のシミュレーション |
| 🏆 騎手・血統 | 穴馬に強い騎手ランキング・血統別適性 |
| 📖 メモ編集 | knowledge_base.json のUI編集・バックアップ |

---

## スコアリングパイプライン（app.py 分析ボタン押下後の処理順）

```
入力（出走馬リスト）
  ↓
1. ペース・展開予測              pace_analyzer.py
2. 騎手乗り替わり分析            jockey_change.py
3. ローテ・体重・クラス・障害叩き  rotation_weight.py
4. 枠順バイアス                  draw_bias.py
5. 前走位置取り補正              position_correction.py
6. 斤量馬体重比                  weight_handicap.py
7. ニックス                      nicks.py
8. 季節・馬場状態・芝↔ダート     season_climate.py
9. 前走レースレベル・ラップ適性   race_level.py
10. リアルタイムバイアス          realtime_bias.py
11. 上がり3F・会場距離・時計ランク・体重トレンド・
    ハンデ斤量・右左回り・初条件・厩舎近況・
    頭数変化・ペース適合・短期外国人騎手     horse_profiler.py / weight_handicap.py / knowledge_base.py
  ↓
EV計算（knowledge_base ボーナス込み）       ev_calculator.py
  ↓
調教タイム・併せ馬・土曜勝ち馬照合           training_scraper.py
  ↓
有力馬撃破スコア・近走詳細・VM格言          horse_profiler.py / knowledge_base.py
  ↓
Confluence スコア（0〜100点）              confluence.py
  ↓
穴馬根拠判定                               longshot_evaluator.py
  ↓
表示 / Discord通知（65点以上の馬がいれば）
```

---

## Confluence スコアの重み（14ファクター）

| ファクター | 重み | 加算されるもの |
|-----------|-----|--------------|
| EV期待値 | 22% | ev（KB/右左/初条件/障害/短期外国人 ボーナスがadj_win_rateに反映済み） |
| ペース展開 | 10% | pace_benefit |
| 枠順 | 7% | draw_bonus |
| 騎手 | 10% | jockey_change_bonus |
| ローテーション | 8% | rotation_bonus + tatakidai_bonus |
| クラス変動 | 7% | class_bonus |
| 馬体重変化 | 3% | weight_bonus |
| 前走位置取り | 8% | position_correction_bonus |
| 斤量比 | 5% | weight_ratio_bonus |
| ニックス | 5% | nicks_bonus |
| 季節・馬場適性 | 5% | season_bonus + condition_apt_bonus + surface_change_bonus + weight_trend_bonus + handicap_trend_bonus + turn_dir_bonus + hurdle_to_flat_bonus + short_term_foreign_bonus + proverb_bonus + first_time_bonus + stable_bonus + field_size_bonus + pace_fit_bonus |
| 前走レベル | 5% | race_level_bonus + beat_bonus + resume_bonus_total |
| ラップ適性 | 5% | lap_bonus + last3f_bonus + time_rank_bonus |
| 当日バイアス | 10% | realtime_bias_bonus |

**追加ボーナス（加重平均後に直接加算）:**
- 調教スコア: ±5点
- 土曜勝ち馬と併せ勝ち: +8点、引き分け/負け: +5点
- プラスファクター4個以上: +3点、6個以上: +6点、8個以上: +10点

---

## knowledge_base.json の構造

```json
{
  "special_signals":         // G1ブリンカー、G1連対経験、荒れ馬場外国馬
  "jockey_patterns":         // 25件超の騎手×条件パターン（買い/消し）
  "jockey_change_patterns":  // 乗り替わりシグナル
  "trainer_jockey_combos":   // 調教師×騎手の組み合わせ
  "sire_patterns":           // 30件超の血統×コース条件パターン
  "race_specific_patterns":  // NHKマイル・高松宮記念・VM等のチェックリスト
  "upset_race_conditions":   // 荒れやすいレース・コース条件
  "avoid_conditions":        // 夏ローカル最終週芝短距離未勝利など
  "short_term_foreign_jockeys": // モレイラ・ビュイック・ムーア等9名リスト
  "race_proverbs":           // VM格言・宝塚格言等
}
```

**編集方法**: アプリ内「📖 メモ編集」タブから追加・削除・バックアップが可能。  
`reload_kb()` を呼べばキャッシュが自動更新される。

---

## secrets.toml の設定項目

場所: `C:\Users\somaf\Desktop\keiba-ai\.streamlit\secrets.toml`

```toml
ANTHROPIC_API_KEY    = "sk-ant-..."   # Claude API（チャット・振り返り分析）
DISCORD_WEBHOOK_URL  = "https://discord.com/api/webhooks/..."  # Discord通知（任意）
```

---

## データファイル

| ファイル/フォルダ | 内容 |
|----------------|------|
| `data/*.csv` | Kaggle JRA Dataset（1986〜2021）。なければデモモードで動作確認可 |
| `race_diary.db` | SQLite。予想記録・買い目・結果・factor_log が蓄積される |
| `odds_history.json` | オッズ監視の履歴スナップショット |
| `knowledge_base.json` | ユーザーの競馬メモ（直接編集もUI編集も可） |
| `sessions/` | 前日分析の保存ファイル（session_store.py が管理） |

---

## 既知の制約・注意事項

### スクレイピングについて
- `scraper.py` はnetkeiba個人利用目的のみ。リクエスト間隔は3秒以上
- `training_scraper.py` の `fetch_training_with_partner()` は調教ページのHTML構造変更に弱い
- `fetch_saturday_winners()` は土曜のレース結果ページから勝ち馬を取得するが、ページ構造変更時は要修正

### データについて
- Kaggleデータは2021年まで。2022年以降の馬は「過去データなし」として扱われる
- `demo_data.py` の生成データはランダムなので実際の競馬とは無関係
- `horse_weight`（馬体重）列はKaggleデータに含まれていない場合がある

### Windows固有
- ターミナルでの日本語表示が文字化けするが、アプリ本体（ブラウザ）は問題なし
- cp932エンコードエラーは表示のみの問題

### pandas 3.x 対応済み
- `style.applymap` → `style.map` に変更済み
- `NaN place_rate` のバグ修正済み（`np.isnan` で明示チェック）

---

## 未実装・今後の候補

| 機能 | 優先度 | メモ |
|------|--------|------|
| 大幅除外歴フラグ（手動チェックボックス） | 中 | スクレイピングでは取得困難 |
| netkeibaレースコメント（不利記載）の自動取得 | 中 | 個別レース結果ページにある場合がある |
| 出走取消・競走除外の自動検出 | 低 | 出馬表確定後のデータで対応 |
| PDF出力（週次レポート） | 低 | fpdf2またはreportlabで実装可能 |
| Streamlit Community Cloudデプロイ | 高 | GitHubにプッシュするだけ。Kaggleデータは別途Google Drive連携が必要 |
| knowledge_base.json の自動更新（振り返り結果から学習） | 高 | Claude APIでログから自動エントリ生成 |

---

## よく使う質問パターン（Claude Code向け）

**「〇〇を実装してほしい」系**
- 実装先は基本的に既存モジュールに追加し、`app.py` のパイプラインに統合する
- 新ボーナスは `confluence.py` の `season_b` か `race_level_b` か `lap_b` のいずれかに加算
- UIアラートは `tab_race` の注目馬アラートセクションに追加

**「エラーが出た」系**
- `style.applymap` → `style.map`（pandas 3.x）
- `NaN` の真偽値判定は `pd.isna()` または `np.isnan()` を使う
- `eval_df.get("column")` ではなく `"column" in eval_df.columns` で列存在確認

**「knowledge_base を更新したい」系**
- アプリ内「📖 メモ編集」タブから追加可能
- または直接 `knowledge_base.json` を編集後 `reload_kb()` を呼ぶ
- 構造は `jockey_patterns` / `sire_patterns` / `race_specific_patterns` etc.

---

## 起動コマンド

```bash
cd C:\Users\somaf\Desktop\keiba-ai
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開く。  
スマホ（同じWiFi）からは `http://[PCのIPアドレス]:8501` でアクセス可能。

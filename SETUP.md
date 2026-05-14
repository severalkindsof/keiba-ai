# 競馬予想AI セットアップ手順

## 必要なもの
- PC（Windows/Mac どちらでも可）
- インターネット接続
- Kaggleアカウント（無料）

---

## STEP 1：Pythonをインストールする

1. https://www.python.org/downloads/ にアクセス
2. 「Download Python 3.11.x」をクリックしてダウンロード
3. インストーラを実行。**「Add Python to PATH」にチェックを入れてから** Install Now をクリック

確認方法：コマンドプロンプト（Windowsキー → 「cmd」と入力 → Enter）を開いて以下を入力：
```
python --version
```
`Python 3.11.x` と表示されればOK。

---

## STEP 2：競馬AIをセットアップする

コマンドプロンプトで以下を順番に実行：

```
cd C:\Users\somaf\Desktop\keiba-ai
pip install -r requirements.txt
```

※インストールに2〜5分かかります。エラーが出たら Claude Code に貼り付けてください。

---

## STEP 3：Kaggleデータをダウンロードする

1. https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset にアクセス
2. Kaggleにログイン（Googleアカウントで登録可）
3. 「Download」ボタンをクリックして ZIPファイルをダウンロード
4. ZIPを解凍して、中のCSVファイルをすべて `keiba-ai/data/` フォルダに入れる

---

## STEP 4：アプリを起動する

コマンドプロンプトで：

```
cd C:\Users\somaf\Desktop\keiba-ai
streamlit run app.py
```

ブラウザが自動で開きます（http://localhost:8501）。
スマホからは同じWifi内で `http://PCのIPアドレス:8501` にアクセス。

---

## STEP 5（任意）：スマホ専用URLで公開する

1. https://github.com で無料アカウントを作成
2. Claude Code に「Streamlit Cloudにデプロイして」と頼む
3. 発行されたURLをスマホのホーム画面に追加

---

## よくあるエラー

| エラーメッセージ | 対処法 |
|---|---|
| `python is not recognized` | PythonをPATHに追加して再インストール |
| `ModuleNotFoundError` | `pip install -r requirements.txt` を再実行 |
| データが表示されない | `data/` フォルダにCSVが入っているか確認 |

# Data Importer (`importer.py`)

## 概要 (Overview)

`importer.py` は、JPX（日本取引所グループ）の先物生Tickデータや、MT5（MetaTrader 5）から出力された外部環境Tickデータを読み込み、システム共通の**秒足（デフォルト1秒足）Parquetフォーマット**に変換するための統合データパイプラインスクリプトです。

本スクリプトにより生成されたデータは、後続の特徴量エンジニアリングや、Transformerモデルの学習・推論における基盤データとして使用されます。

## 主な機能 (Features)

本スクリプトは、対象となるデータソースに合わせて2つのモード（`nk225`, `mt5`）を提供しています。

### 1. NK225モード (日経225ミニ先物)

* **生データパース**: JPXフォーマットのTSVファイルを読み込み、日時結合（JST naive）や型変換を安全に実行します。
* **期近限月フィルタリング**: データ内に複数限月が混在している場合、その日の期近限月を自動判定してフィルタリングします。
* **売買方向の推定 (Tick Test)**: 歩み値の価格変動（Tick）から、その約定が「買い（Up Tick）」か「売り（Down Tick）」かを推定します。
* **動的リサンプリング**: 指定された秒足（`config.BAR_SECONDS`）にリサンプリングし、以下の微細構造（Microstructure）ベースの指標を集計します。
* 価格情報（Open, High, Low, Close, Price Std）
* 買い・売り別の出来高（Buy/Sell Volume）および約定回数
* 買い・売り別の平均約定サイズ（Avg Buy/Sell Size）
* 大口検知用の最大約定サイズ（Max Trade Size）および出来高標準偏差（Volume Size Std）
* 純買い圧力（Delta Volume）


* **複数ファイルの一括処理**: リストファイル（パスが列挙されたテキスト）を渡すことで、複数日のTSVを一括処理して結合可能です。

### 2. MT5モード (外部環境データ)

USD/JPY、S&P500、XAU/USD、BTC/USDなどのMT5出力データに対応します。

* **遅延評価（LazyFrame）**: 大容量のTickデータをPolarsのLazyFrameで効率的にストリーミング処理します。
* **タイムゾーン変換**: MT5のサーバータイム（通常 US/Eastern 基準の -7h等）から、日本標準時（JST / Asia/Tokyo）へ正確にタイムゾーンを変換します。
* **Askベース秒足作成**: 秒ごとのAsk価格をベースに、OHLC、Tickボリューム、スプレッド平均を集計し、ルックアヘッドバイアス（カンニング）防止のためタイムスタンプをオフセット（`available_ts`）して保存します。

## 必要要件 (Requirements)

* Python 3.9+
* `polars`
* プロジェクトルートの `config.py` (`cfg.BAR_SECONDS` の参照等に利用)

## 使い方 (Usage)

コマンドラインから、処理モードと入出力パスを指定して実行します。

```bash
python importer.py [-h] {nk225,mt5} [--input INPUT | --list LIST] --output OUTPUT

```

### 引数オプション

* `mode`: `nk225` または `mt5` （必須）
* `-i`, `--input`: 単一の入力TSVファイルへのパス
* `-l`, `--list`: 処理対象のTSVファイルパスが1行ずつ記載されたテキストファイルへのパス（`nk225`モード専用）
* `-o`, `--output`: 変換後の保存先Parquetファイルパス（必須）

### 実行例 (Examples)

**例1: 日経225ミニの単一TickファイルをParquetに変換**

```bash
python importer.py nk225 -i inputs/NK225/future_tick_19_202310.tsv -o data/nk225_202310.parquet

```

**例2: リストファイルを使って日経225ミニの複数ファイルを一括処理**

```bash
python importer.py nk225 -l inputs/NK225_file_list.txt -o data/nk225_merged.parquet

```

**例3: MT5のUSDJPYのTickデータをParquetに変換**

```bash
python importer.py mt5 -i inputs/MT5/USDJPY.tsv -o data/usdjpy.parquet

```

## アーキテクチャの制約と注意点

* **メモリ使用量**: `nk225` モードにおいて巨大なリストファイルを一括処理する場合、最終結合時にメモリ（RAM）を大量に消費する可能性があります。メモリ不足が発生する場合は、月ごとなどに分割して実行することを推奨します。
* **タイムゾーン**: `mt5` モード内のタイムゾーンロジックは、利用しているMT5ブローカーの冬時間/夏時間（DST）仕様に依存します。現在のコードは `US/Eastern` 基準の -7h 仕様（一般的なブローカー仕様）を前提としています。
* **出力先**: 出力先のディレクトリが存在しない場合はエラーになる可能性があるため、事前に保存先ディレクトリ（例: `data/`）を作成しておいてください。
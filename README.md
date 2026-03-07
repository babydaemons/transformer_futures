# Temporal Fusion Transformer for Nikkei 225 Day-Trading (NK225-TFT)

本プロジェクトは、日経225先物ミニ（NK225M）を対象とした **1分足デイトレードシステム** の実装です。
**改良型 Temporal Fusion Transformer (TFT)** アーキテクチャを採用し、歩み値から生成された市場微細構造（Microstructure）指標と、外部資産（USDJPY, S&P500, 等）の先行遅行効果、および時間帯特性を統合。
スキャルピング（数秒〜数分）ではなく、**数分〜最大4時間** のホライゾンで、往復コスト（15円）を確実に上回るボラティリティの拡大局面（Edge）を特定します。

## 1. システムアーキテクチャ


本戦略は、単純な価格予測ではなく、トレードの意思決定プロセスを模倣した **2段階分類 (Two-Stage Classification) ＋ 動的利確/損切り予測** と **収益性直接最適化** に基づいています。

### 1.1 Model: Modified TFT (`model.py`)

標準的なTFTに対し、デイトレード特有のノイズ耐性とパターン認識能力を強化するための改良を加えています。

* **Bidirectional LSTM Encoder:**
  * 時系列の文脈抽出において、過去から未来、未来から過去（シーケンス内）の双方向から情報を集約。チャートパターンの認識精度を向上させています。

* **Variable Selection Network (VSN):**
  * 「板圧力が効いている局面」「マクロ連動が強い局面」「時間帯特有の動き」など、その瞬間に有効な特徴量を動的に選択・重み付けします。

* **Vectorized Projection:**
  * `nn.Conv1d` を用いた高速な特徴量埋め込み処理により、学習・推論速度を向上。

* **Triple Output Heads:**
  1. **Trade Head (Binary: Neutral vs Action):** 値幅（ボラティリティ）がコスト（手数料＋スリッページ）を超えて拡大するかを判定。
  2. **Direction Head (Binary: Long vs Short):** 価格変動の方向性を判定。
  3. **SL/TP Head (Regression):** その局面における最適な動的損切り幅（SL）と利食い幅（TP）のATR乗数を予測。

### 1.2 Training: Hybrid Loss Landscape (`train.py`, `core/losses.py`)

不均衡データ（Neutralが圧倒的多数）に対処し、かつ「勝率」よりも「期待値」を重視するための複合損失関数を採用しています。

1. **PnL Optimization Loss (Sharpe-like Momentum):**
   * `-(Expected_Return / Global_Std_Dev)` を最小化する損失関数。EMAによる大域分散を用いて安定化を図ります。
   * SL/TP Headの予測値を元にリターンをクリッピングし、モデルが実現不可能な架空の利益を学習しないよう制限しています。

2. **Focal Loss:**
   * 予測が容易なサンプル（ノイズ）の重みを下げ、予測困難な重要な局面に学習を集中させます。

3. **Regularization Penalties:**
   * **Directional Penalty:** 方向予測の自信度が低い場合にペナルティを与え、確信を持てない局面でのエントリーを抑制。
   * **False Action Penalty:** Neutral相場でAction（エントリー）を予測することに対して強くペナルティを与え、ポジポジ病（Over-trading）を防ぎます。
   * **Logit Margins:** ActionとNeutralのLogit差を強制的に広げ、判定の曖昧さを排除します。

## 2. データエンジニアリング


### 2.1 データソースと処理フロー (`import_data.py`, `data/fetcher.py`)

生データ（Tick/TSV）を読み込み、以下の処理を経て `BAR_SECONDS` (デフォルト1分足) に統合されます。

| データ種別     | ソース   | 処理概要                             | 特記事項                                                                           |
| -------------- | -------- | ------------------------------------ | ---------------------------------------------------------------------------------- |
| **NK225 Mini** | JPX/Tick | 期近限月の結合、売買方向推定         | 歩み値から「買い/売り」を判定し動的リサンプリングを実施。                          |
| **USD/JPY**    | MT5/Tick | タイムゾーン変換、動的リサンプリング | **Lag処理**: 未来情報の漏洩（Look-ahead Bias）を防ぐため、常に過去のデータを結合。 |
| **S&P 500**    | MT5/Tick | タイムゾーン変換、動的リサンプリング | 同上。Globex市場とのLead-Lag効果（相関・乖離）を利用。                             |

### 2.2 特徴量エンジニアリング (`features.py`)

デイトレードに特化した以下の高度な特徴量を生成します。


#### A. Microstructure (需給・板情報)
* **Daily Cumulative Volume Delta (CVD):** その日の朝からの「買い圧力 - 売り圧力」の累積値（正規化済み）。デイトレードにおけるトレンドの強さを示唆。
* **Amihud Illiquidity Proxy:** `|Return| / Volume` で算出される流動性指標。値が跳ねたときは「薄い板を食った」ことを示唆。
* **Tick Speed Ratio (Acceleration):** 直近平均に対する約定頻度の加速率。モメンタムの初動を検知。
* **Buy Pressure / OFI:** 買い出来高と売り出来高の不均衡、およびローソク足形状を加味した圧力。
* **Max Trade Size / Vol Std:** 大口注文の検知と、約定サイズのばらつき（大口/小口の質）。
* **Volume Profile (POC) & Volume Skew:** 日次・週次の最大出来高価格帯(POC)からの乖離や、VWAPより上で取引された出来高の割合。

#### B. Macro Lead-Lag (外部環境)
* **USDJPY Lead Spread / Cum Divergence:** ドル円のリターンと日経平均のリターンの乖離、およびその累積値。
* **USDJPY Bollinger Score:** ドル円のボリンジャーバンド乖離。
* **RS (Relative Strength):** S&P500に対する相対強度。
* **Rolling Beta (beta_sp500_1h):** 対S&P500のローリング・ベータ。相関の強弱を動的に測定。

#### C. Volatility & Technicals
* **Garman-Klass Volatility:** 始値・高値・安値・終値を考慮した高精度ボラティリティ。
* **Efficiency Ratio (Kaufman):** トレンドの「純度（ノイズの少なさ）」を測定（短期15分/長期4時間）。
* **VWAP Deviation:** 短期(1m)、中期(4時間)、および**日次VWAP**からの乖離率。

#### D. Time Structure
* **Minutes From Open / To Close:** 寄り付きからの経過時間と、引けまでの残り時間。
* **Session Flags:** 日中/夜間セッション、市場オープン直後、ランチ休憩、引け前などの時間帯フラグ。
* **High Volatility Window:** 特定の高ボラティリティ時間帯フラグ。
* **Minute of Day / TOD:** 1日の中の細かい時間帯の周期性（サイン・コサイン波）。

## 3. ラベリングと検証戦略


### 3.1 Dynamic Triple Barrier Method (`data/dataset.py`)

ボラティリティ環境とトレードコストに応じて動的に変化する閾値を用いたラベリングを採用しています。

* **閾値:** `max(ATR * Scale, コスト由来の下限値)`
* 取引手数料やスリッページにバッファを加えた「コスト負けしない値幅」をベースラインとし、ATRでボラティリティに追従させます。
* **判定:** 予測ホライゾン（標準60バー）以内に、上側バリアに先に触れればLong、下側ならShort。
* **Auto-Tuning (`train.py`):** Foldごとに「Neutral」と「Trade」の比率が適正（例: Neutral 60%〜85%）になるよう、ATR倍率を自動調整します。

### 3.2 Walk-Forward Validation

時系列の順序を厳守したスライディングウィンドウ方式で検証を行います。

* **Train:** 30日
* **Validation:** 5日
* **Test:** 1日 (OOS)

## 4. トレーディングロジック (`trade.py`, `backtest/fast_sim.py`)


非同期GPU転送を活用した高速推論パイプライン（`data/inference.py`）と、Numba JITを用いた高速シミュレーション（`backtest/fast_sim.py`）により、以下の厳格なルールでバックテストを行います。

### 4.1 エントリーフィルタ

シグナル発生時でも、以下の環境下ではエントリーを見送ります。

1. **Tick Speed Filter:** 市場が閑散としている（Tick Speed Ratio < 1.2 など）場合。
2. **Dead Market Filter:** 直近の値幅（Spread）が極端に小さい（5円以下）場合。
3. **Time to Close:** 引けまで1分を切っている場合。
4. **Conditional Signals:** 方向予測確率が閾値（`min_dir_conf`）に満たない場合。

### 4.2 エントリー & エグジット & ポジションサイジング

* **Entry:**
  * `P(Action) > Threshold_Trade` かつ `P(Direction) > Threshold_Dir`
  * **Next Bar Entry:** シグナル点灯足の「次の足の始値」でエントリー（スリッページ考慮）。

* **Position Sizing:**
  * モデルの確信度（超過確率）とボラティリティ（ATR）に基づき、ベースロット（`base_lots`）から最大ロット（`max_lots`）の間で動的に建玉数を調整します。

* **Exit:**
  * **Time Stop:** 最大4時間（`max_holding_sec`）経過、または引け時刻で強制決済。
  * **Dynamic TP/SL:** モデルが予測した SL/TP のATR乗数を用いた利確・損切り。コスト負けを防ぐためTPにはフロア値（最小値）が適用されます。
  * **Trailing Stop:** 含み益が一定のATR倍率（`trailing_act_mult`）を超えた後、最高値/最安値から一定割合逆行した場合に決済し、利益を確保します（時間減衰ファクター適用済み）。

### 4.3 Threshold Optimization

検証データ（OOS）に対して、以下の指標を複合したスコアを最大化する閾値を自動探索します。

* **Profit Factor (PF):** 損失に対する利益の比率（上限キャップあり）。
* **PnL Consistency:** 平均利益がコストを十分に回収できているか。
* **Trade Count:** 統計的有意性を保つための取引回数制約（多すぎず少なすぎず適正レンジに誘導）。
* **Directional Accuracy:** エントリー方向の正答率。

## 5. ファイル構成

```text
.
├── config.py                 # システム全体の設定 (Feature, Train, Backtest Params)
├── import_data.py            # JPX/MT5 Tickデータの読み込みとリサンプリング
├── train.py                  # 学習ループエントリポイント
├── trade.py                  # バックテストシミュレーションの統合・パラメータ最適化
├── trade_simulator.py        # 動的ロット計算・TP/SLシミュレータインターフェース
├── core/
│   ├── losses.py             # Focal Loss などの基本損失関数
│   ├── pnl_loss.py           # 期待PnLと分散に基づくカスタム損失関数
│   └── trainer.py            # 学習プロセス管理、Early Stopping
├── data/
│   ├── dataset.py            # Walk-Forward 分割、TFTDataset (Triple Barrier)
│   ├── builder.py            # DataLoader構築、ラベル閾値のAuto-Tuning
│   ├── fetcher.py            # データの遅延評価読み込み (Polars)
│   └── inference.py          # 非同期GPU転送を用いた高速推論プロファイラ
├── features/
│   ├── pipeline.py           # 特徴量生成オーケストレーター (FeatureEngineer)
│   ├── microstructure.py     # 需給・板情報特徴量
│   ├── macro.py              # 外部資産連動特徴量
│   ├── technicals.py         # RSI, MACD, ADX, Volatility等
│   └── volume_profile.py     # POC, VWAP乖離, Volume Skew
├── backtest/
│   └── fast_sim.py           # Numba JIT による超高速シグナル生成・バックテスト
├── model/
│   ├── tft.py                # Modified TFT メインアーキテクチャ
│   └── blocks.py             # GRN, GLU, VSN などのビルディングブロック
└── util/
    ├── utils.py              # ロギング, PerfTimer, RankGauss, EMA
    └── trade_log.py          # 詳細なTSVトレードログ出力と決済理由の評価

```

## 6. 今後の改善案（Next Steps）

try08での成績悪化を受け、システムの安定性とパフォーマンスを回復・向上させるために以下のステップを実行します。

1. **バグと情報漏洩（Look-ahead Bias）の徹底調査:**
   * 特徴量計算を分割した際に、`shift()` や `rolling()` の計算漏れ・ズレが発生し、未来の情報が学習に混入していないか（または必要なラグが失われていないか）を検証する。
   * NumbaシミュレータとPythonネイティブコード間で、配列のインデックス参照（Off-by-oneエラー）が起きていないか確認する。

2. **切り戻し（A/Bテスト）による原因特定:**
   * try07とtry08のコードベースで同じSeed値・同じデータ期間を用いて学習と推論を行い、出力ロジット（`probs_action`, `probs_short`）の分布差異を比較する。
   * 悪化の原因が「特徴量の追加」によるものか、「コード分割によるバグ」によるものかを切り分ける。

3. **過学習（Overfitting）の緩和:**
   * 特徴量が多すぎることによる次元の呪いを防ぐため、VSN（Variable Selection Network）の出力ウェイトを分析し、寄与度の低い特徴量を思い切って削除する。
   * Dropout率の再調整、または `weight_decay` の強化を検討する。

4. **動的バリアと損失関数の再調整:**
   * `Focal Loss` の `gamma` 値や、`PnL Optimization



## 7. 変更履歴（Changelog）

### try08
* **コードのリファクタリング:** `features/pipeline.py` などの巨大モジュールをドメインごとに分割（Microstructure, Technicals, Macro, Volume Profile）。
* **機能拡張:** 特徴量やシミュレーションロジックの追加・整理。
* **⚠️ Issue:** リファクタリングおよび機能拡張後、バックテスト（OOS）のパフォーマンス（PnL、PF等）が悪化する現象を確認。

### try07

* **特徴量拡充:** 日次・週次POCからの乖離、Volume Skew（VWAPより上で取引された割合）、細かい時間帯周期性（Sine/Cosine）、Sessionフラグ、および対S&P500ローリング・ベータなどを追加。
* **ホールド時間拡張:** 予測ホライゾンを見直し、最大ホールド時間を4時間に拡張（`max_holding_sec`）。
* **ポジションサイジング:** 確信度とボラティリティに基づく動的ロットサイズ調整（`lots`）を実装。
* **モジュール分割:** `core/`, `data/`, `backtest/` などのサブディレクトリへファイルを分割し保守性を向上。
* **データソース拡張準備:** MT5データソースとして XAU/USD (Gold) および BTC/USD (Bitcoin) の受け入れ体制を整備（`import_data.py`）。
* **バックテストロジック強化:** NumbaによるJITコンパイルシミュレータを導入。トレイリングストップに時間減衰ファクター（Time-Decay）を追加。

### try06

* **特徴量拡充:** 日次・週次POCからの乖離、Volume Skew（VWAPより上で取引された割合）、細かい時間帯周期性（Sine/Cosine）、Sessionフラグなどを追加。
* **SL/TP Head導入:** 動的SL/TP乗数（`m_sl`, `m_tp`）を回帰予測する第3の出力ヘッドを追加。
* **PnL Loss改善:** EMAによる大域分散を用いたSharpe Ratio Lossの安定化。モデル予測SL/TP幅によるリターンクリッピングの実装。
* **バックテストロジック強化:** コストを考慮したラベル最小値補正（`label_min_limit_mode`）、トレイリングストップの実装、およびNumbaによるJITコンパイルシミュレータの引数最適化。

### try01

* 初期プロトタイプ。固定閾値のTriple Barrier、単純なFocal Lossによる学習。

# OpenGCMC

**多孔性材料と柔軟なゲスト分子のための、キャッシュを中心に設計したグランドカノニカル Monte Carlo。**

[简体中文](README.md) · [English](README.en.md) · **日本語**

OpenGCMC は、繰り返し発生する近傍探索、相互作用パラメータの混合、環境の再構築を試行評価から切り離します。厳密なエネルギー評価と柔軟分子の配置バイアス Monte Carlo（CBMC）を組み合わせ、C++ ネイティブサンプラー、汎用 Python インターフェース、オプションの CUDA エネルギーバックエンドを提供します。

### 熱力学と電気化学

- **化学ポテンシャル応答と自由エネルギー。** 粒子数の揺らぎから応答行列を求め、固定バイアス窓を MBAR で統合して粒子数と電極電荷の自由エネルギー地形を計算します。
- **電解質と定電位界面。** 化学量論に従って塩を交換し、電荷制約の下で Gaussian 電極の電荷を求めます。陽に扱う SPC/E 水中で、固定分子トポロジーのプロトン化、酸化還元、プロトン共役電子移動をサンプリングします。
- **較正と電流。** アルケミカル窓から反応状態間の自由エネルギー差を推定し、指定した速度定数や拡散係数から実時間の電流応答、クロノアンペロメトリー、サイクリックボルタンメトリーを計算します。

[電気化学モデルと使用方法](docs/electrochemistry.md)

## 科学計算ケース

OpenGCMC は、帯電 MOF 中の混合吸着、定電位電解質界面、陽な水を含む反応自由エネルギー、拡散電気化学の各計算を一連の流れで実行できます。

- **帯電 IRMOF-1 中の CO2/CH4 競争吸着。** RASPA3 公式例の CIF、サイト電荷、力場を使用します。3 本の独立鎖は 2.647 および 4.257 分子/セルを与え、RASPA3 の 2.615 および 4.246 と比較されます。同時に化学ポテンシャル応答行列と MBAR 自由エネルギー地形を生成します。
- **NaCl/SPC/E 定電位界面。** 2 枚の Gaussian 電極、62 個の陽な水分子、1 対のイオンを一つの計算で扱います。電極電荷の独立解との差は 1.1×10⁻¹⁵ e、分子エネルギーの全再計算との差は 1.95×10⁻⁷ K で、Na-O と Cl-O の配位数も直接出力します。
- **フェリシアン化物/フェロシアン化物の拡散電気化学。** 実験拡散係数を用いてクロノアンペロメトリーとサイクリックボルタンメトリーを計算します。Cottrell 極限との最大電流差は 0.108% で、2 つの走査速度のピーク電流は Randles-Sevcik 予測と一致します。
- **陽な水を含む反応と自由エネルギー較正。** 剛体 SPC/E 水、固定トポロジー反応状態、NCMC、7 窓 MBAR を同じ経路で実行し、反応状態間の自由エネルギー差、窓重複、電荷応答の診断値を出力します。

これらのケースは、実在結晶、実験パラメータ、標準物理極限を、構造準備、サンプリング、自由エネルギー、電流解析の再現可能な数値出力へ接続します。[詳細なケース解析を読む](docs/scientific_cases.md)。

![OpenGCMC の吸着サンプラー構成](docs/assets/opengcmc_architecture.svg)

## 性能

[2 種類の CPU を用いたベンチマーク](benchmarks/cpu/README.md)では、NIST の分布との照合、同一提案則での受理法比較、RASPA3 との精度到達時間の比較を行った。サンプルと再現用スクリプトも収録している。

Cu-BTC 中の中性 Lennard–Jones プローブを用い、各ソフトウェアと同一システム上でペア比較した結果、キャッシュを用いる OpenGCMC ネイティブサンプラーは次のスループット向上を示しました。

| 直接比較 | 向上倍率の中央値 | 3 回のペア測定の範囲 | 計算条件 |
|---|---:|---:|---|
| RASPA3 | **21.60×** | 20.06–23.27× | 10,000 回の予備試行後、40,000 回の本計算試行 |
| gRASPA | **30.17×** | 28.34–30.25× | 空の初期状態から 1,000 回の試行；WSL、CUDA 同期起動、共有 GPU |

![RASPA3 および gRASPA とのスループット比較](docs/assets/opengcmc_performance.svg)

各比較は、それぞれの同一システム上で逐次実行し、別々に表示しています。OpenGCMC は CPU ネイティブサンプリング、gRASPA は GPU サンプリングを使用します。倍率はペアごとのスループット比の中央値です。初期化、準備、最終エネルギー照合は計時区間の外に置き、4,992 サイトの骨格、単一サイトプローブ、カットオフ、ポテンシャルの規約を揃えています。

各測定時間、配置エネルギーの照合、再現コマンドは[ベンチマーク文書](docs/benchmarks.md)と[生データ](benchmarks/results)に掲載しています。

4,416 サイトの Ni_TPT_TATBA 固定骨格内の帯電 p-キシレンを、有限直接 Ewald 和で評価しました。低吸着量での 3 シードの単一 CPU コア測定では、二段階サンプリングの中央値は毎秒約 6,400–6,600 試行でした。64³ 骨格場の読み込みは 0.24 秒、初回の構築と書き込みは 18.37 秒でした。[計時条件と個別データ](docs/benchmarks.md#charged-p-xylene-in-ni_tpt_tatba)。

## 中核アルゴリズム

- **厳密な空間・パラメータキャッシュ。** 周期的な分率座標のセル分割と CSR 候補表により近傍関係を再利用し、混合パラメータ行を事前に結び付けます。各問い合わせでは実距離と厳密なカットオフを評価します。
- **受理状態の近傍更新。** ネイティブサンプラーは移動分子が占める周期セルを局所的に更新します。単一成分と混合物で同じエネルギーカーネルを使います。
- **密度補正付きバイアス CBMC。** 全領域で正の確率を持つ固定 voxel alias 分布が、有利な領域へ根サイトを提案します。順方向の成長と逆方向の再追跡の両方に `1 / [V q(r)]` を含め、目標分布を保ちます。
- **試行の一括評価と骨格データの共有。** 同じ不変環境とパラメータを用いるエネルギー要求をまとめます。独立鎖は骨格キャッシュを共有し、ゲスト状態、乱数生成器、受理判定を個別に保持します。
- **厳密な目標分布への補正。** Python 経路では、グリッドまたは学習代理モデルで提案を選別し、二段階受理規則により厳密な目標エネルギーへ補正できます。

- **固定骨格グリッドと二段階受理。** 骨格の LJ 場と静電場を事前計算し、C++ の CBMC 成長と再追跡に使います。選択された分子配置を目標ポテンシャルで補正します。
- **長距離静電相互作用。** 直接 Ewald は CPU の調和成分の縮約とオプションの FP64 CUDA テンソル演算に対応します。PME では指定メッシュが目標エネルギーを定めます。

[アルゴリズムとソースコードの対応](docs/algorithms.md) · [長距離サンプリング](docs/long_range.md)

## RASPA・gRASPA に対する利点

OpenGCMC は、再利用可能なエネルギー表現、柔軟分子の成長、拡張可能なエネルギーインターフェースを統合します。RASPA3 との直接比較は、上記プローブ計算での本計算スループット向上を示しています。明示的な JSON 入力、Python API、剛体 LJ 入力の変換も提供します。gRASPA に対しては、木構造の柔軟ゲストの逐次成長、再追跡、再成長を備え、記載した短時間実行条件で試行スループットが 30.17× 向上しました。

| ワークフロー | OpenGCMC の機能 |
|---|---|
| 剛体・柔軟ゲスト | 剛体、木構造トポロジー、固定または有界の結合長・結合角、Fourier ねじれ項、CBMC 挿入・削除・再成長 |
| 高速ネイティブサンプリング | 多成分の LJ、DSF、直接 Ewald、PME に対応する C++ ループと固定骨格グリッド |
| 汎用サンプリング | Python 多成分 GCMC、Ewald 静電相互作用、オプションの厳密な目標補正 |
| GPU エネルギー評価 | 常駐する近傍・パラメータ表、再利用可能な作業領域、NumPy/CuPy インターフェース |
| 解析・再利用 | 粒子数統計、ESS、自己相関、チェックポイント、骨格データを共有する独立鎖 |
| RASPA3 入力変換 | 物理量と単位を確認できる剛体 LJ サブセットの明示的変換 |

gRASPA は GPU 並列エネルギー評価と剛体 CBMC を中心とする実装です。機能比較は[公開ソース](https://github.com/snurr-group/gRASPA/tree/e6da4e376f19047df6cb85b9d20685d1bf0777e8)に基づいています。実行構成と入力形式は[詳細比較](docs/algorithms.md#comparison-with-raspa3-and-graspa)を参照してください。

## インストール

Python 3.10 以降と C++17 対応コンパイラが必要です。

```bash
git clone https://github.com/owenxyzhang-seseTJ/OpenGCMC.git
cd OpenGCMC
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python native/build.py --portable --no-openmp
```

OpenMP 対応コンパイラでは `--no-openmp` を省略できます。CUDA エネルギーバックエンドには CUDA 12 と CuPy を使用します。

```bash
python -m pip install -e '.[cuda]'
```

配布名とコマンド名は `opengcmc`、Python のインポート名は `poremc` です。

## 使用方法

```bash
# ネイティブの柔軟分子サンプリング
opengcmc run examples/flexible_cached.json -o output/flexible

# 全領域を覆う根サイトのバイアス提案
opengcmc run examples/flexible_cached_bias.json -o output/biased

# Python サンプラーと CUDA エネルギーバックエンド
opengcmc run examples/flexible_cuda.json -o output/cuda

# グリッド提案と厳密な目標補正
opengcmc run examples/grid_delayed_acceptance.json -o output/corrected

# 帯電した柔軟ゲストの二段階サンプリング
opengcmc run examples/longrange/small_delayed_ewald.json -o output/delayed

# 定電位電解質 GCMC
opengcmc-thermo run examples/thermodynamics/interface_constant_potential.json -o output/interface

# 陽な水を含む反応状態のサンプリングとサイクリックボルタンメトリー
opengcmc-reactive run examples/electrochemistry/hydrated_interface.json -o output/hydrated
opengcmc-reactive transport examples/electrochemistry/diffusion_voltammetry.json -o output/cv
```

出力には統計、入力設定、チェックポイント、最終座標が含まれます。例では明示的なデモ用パラメータを使用しています。対象系に合わせて骨格、ゲスト、力場、電荷、温度、リザーバーを JSON で指定します。

```python
from poremc.fast import FastGCMC
from poremc.engine import GCMC
from poremc.batching import evaluate_requests, run_native_replicas
```

## リポジトリの内容

| パス | 内容 |
|---|---|
| [`poremc/`](poremc) | Python API、CBMC、ポテンシャル、CUDA、統計、入出力 |
| [`native/`](native) | 厳密な C++17 エネルギーカーネルとネイティブサンプリングループ |
| [`examples/`](examples) | 吸着、電解質、定電位界面、反応較正、電流計算の入力例 |
| [`benchmarks/`](benchmarks) | 再現用入力、各測定値、比較ソフトウェアの生出力 |
| [`tools/`](tools) | 直接比較、数値再生、データ処理ツール |
| [`figures/`](figures) | 編集可能な描画ソース、図の元データ、キャプション |
| [`docs/`](docs) | [技術レポート](docs/OpenGCMC_Technical_Report.pdf)：数学的基礎、物理モデル、アルゴリズムと実装（PDF、36 ページ） |
| [`tests/`](tests) | 数値 fixture とアルゴリズムテスト |

## ライセンス・参考資料

[MIT License](LICENSE) で公開しています。アルゴリズムの参考文献は [`docs/references.bib`](docs/references.bib) を参照してください。

[RASPA3](https://github.com/iRASPA/RASPA3) · [gRASPA](https://github.com/snurr-group/gRASPA)

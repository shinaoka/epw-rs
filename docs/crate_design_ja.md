# EPW 下流の構造分析とクレート分割案

作成日：2026-10-15．QE develop ブランチの `EPW/src` について，モジュールの `USE` 文と `CALL` 文を集計し，主ドライバ `use_wannier.f90` の流れを読んだ結果です．

## 1. 現状の構造

### 1.1 主ドライバは「q ループ＋大域変数経由のコールバック」

下流の計算はほぼすべて `use_wannier.f90` の 1 つの q ループの中で起きます．

1. **準備：** Fermi 準位の推定（`fast_fermi`），細かい k 格子の生成（`load_mesh`），Fermi 窓（`fermiwindow`），`epmatwp` を MPI-IO で開く．
2. **q ループ（`DO iqq = iq_restart, totq`）：**
   - フォノン：`dynwan2bloch`（または `dynifc2blochf`）で ω_qν，e_qν．
   - g を半分変換：`ephwan2blochp` で Rₚ → q．
   - k ループ：`hamwan2bloch_batch` で ε_k，ε_k+q．`ephwan2bloch` で Rₑ → k，バンドとモードへの回転．極性なら `rgd_blk_epw` を加算．速度は `vmewan2bloch`．
   - 結果は大域配列 `etf`，`wf`，`epf17` などに置かれる．
   - **物理量のコールバック：** `selfen_elec_q`，`selfen_phon_q`，`write_ephmat`（超伝導用），`scattering_rate_q`（輸送），`indabs_main`（間接吸収）など．引数は `(iqq, iq, totq, first_cycle)` 程度で，入力データはすべて大域変数から読む．
3. **後処理：** 反復 BTE（`iter_merge`），スペクトル関数，DOS，a2F の出力．Eliashberg は別の段（`supercond_driver`）で，書き出したファイルを読んで解く．

この「q ごとに補間 → 各物理量が累積」という形は，Rust では**補間器が q ごとのバッチを返し，物理量はそれを受け取る累積器（accumulator）として実装する**形に素直に写せます．大域変数を経由しないので，各物理量を単独でテストできます．

### 1.2 下流が QE から使っているもの

下流の各ファイルの `USE ..., ONLY :` を集計すると，QE から取っているのは次だけです．前回の調査メモの「MPI ラッパー，格子，定数程度」より少し多く，対称性と四面体法も含みます．

| QE モジュール | 使っているもの | Rust での置き換え |
|---|---|---|
| `cell_base`，`ions_base`，`modes` | 格子，原子位置，質量，モード数 | 境界データの結晶型 |
| `pwcom`，`klist` | `ef`，`nelec`，`nbnd`，`degauss`，`ngauss` | 境界データと入力 |
| `symm_base` | 対称操作，`find_sym`，`set_sym_bl`，時間反転 | 対称性 crate（moyo など）．境界データに対称操作を持たせてもよい |
| `ktetra` | 四面体法 | 自前の小 crate |
| `noncollin_module`，`lsda_mod` | スピンの扱い | 境界データのフラグ |
| `rigid` の `cdiagh2`，`fft_scalar` の `cft_1z` | エルミート対角化，1 次元 FFT | 線形代数と FFT の crate |
| `mp*`，`io_global`，`io_files` | MPI，入出力 | `Comm` trait（当面は 1 プロセス実装のみ）．入出力は I/O crate |
| `constants` | 物理定数 | 定数 crate |

QE 本体の重い内部（波動関数，FFT 格子，擬ポテンシャル，DFPT）は下流からは一切使われていません．

### 1.3 下流内部の結合の問題点

- **大域変数：** 物理量のサブルーチンは `global_var`（約 290 変数）と `input` から直接読みます．例えば `selfen_elec_q` は `etf`，`wf`，`epf17`，`xqf`，`ef` を大域変数から読みます．テストには大域状態を全部組み立てる必要があります．
- **分野をまたぐ依存：** `selfen` が `transport`（`transport_prepare`）を使い，`utilities` が `transport`，`indabs`，`wannier2bloch` を使います．「utilities」が上位の物理モジュールに依存しているため，層が崩れています．
- **共有の雑多なモジュール：** `supercond_common` が `selfen`，`spectral`，`printing`，`low_lvl` からも使われます．超伝導用の共有状態が他分野に漏れています．
- **計算と I/O と MPI の混在：** `io_*` 系（合計約 1.2 万行）は書式と MPI の集約を兼ねます．物理のサブルーチンの中で `mp_sum` と出力が起きます．縮約を累積器の型に持たせ，出力を物理の外に出せば，この部分のかなりが不要になります．

### 1.4 中間ファイルという第 2 の境界

EPW は段の間で中間ファイルを書き，次の段がそれを読みます．これらはそのままクレートの境界とテストの fixture になります．

| 中間データ | 書く側 | 読む側 | 意味 |
|---|---|---|---|
| `crystal.fmt`，`epwdata.fmt`，`wigner.fmt`，`*.epmatwp` | 上流（`epwwrite`） | 下流全体（`epwread`） | Wannier 表現の境界 |
| `<prefix>.a2f`（`fila2f`） | フォノン自己エネルギー | 等方的 Eliashberg | α²F(ω) だけで等方的 Eliashberg が解ける（`epw_super/epw5`） |
| `*.ephmat`（`ephwrite`） | q ループ | 異方的 Eliashberg | Fermi 面上の g² |
| 自己エネルギーの出力 | `selfen` | `spectral`，`cumulant` | Σ(k,ω) |

## 2. 設計方針

1. **境界はデータ型で決める．** Wannier 表現の境界データを I/O から独立した型にし，読み込みは別クレートにします．QE（EPW の `.fmt`），EPW-VASP の HDF5，将来の phonopy などは読み込みアダプタを足すだけで対応します．
2. **大域変数を作らない．** 補間器は q ごとに `EphBatch`（ε_k，ε_k+q，ω_q，g，必要なら速度）を返し，物理量は `trait QAccumulator` を実装して受け取ります．入力パラメータは各クレートの設定構造体として明示的に渡します．
3. **MPI は実装を後回しにし，データ構造には最初から入れる．** 当面は 1 プロセスとスレッド並列（rayon）で動かします．ただし，データの分割，所有範囲，縮約の種類，通信の口は最初から型として持たせます．後から MPI を足すときに物理クレートを書き換えずに済むようにするためです．詳細は 2.1 節．
4. **再利用できる部品は EPW に依存させない．** 補間核，長距離項，BZ 格子，smearing，Eliashberg ソルバーは，EPW を知らない汎用クレートにします．
5. **テストの層を分ける．** 解析解と toy model（L0），EPW から切り出した小さな fixture（L1），中間ファイル境界での比較（L2），全体の回帰（L3）．

### 2.1 分散を前提にしたデータ構造

EPW 自体は 3 種類の分散を使っています．k 点をプール間で分ける（`fkbounds`），q 点をイメージ間で分ける（`image_division`），巨大な g(Rₑ,Rₚ) をプロセス間で分けるかディスクから逐次読む（`etf_mem`，`ephwan2blochp_dist`）です．同じ 3 つの軸を最初から型で表します．

**(1) 全体と局所を区別する索引**

- 格子（`bz-mesh`）は常に全体の索引を持ちます．各プロセスが持つのは `Partition`（全体索引の所有範囲）だけです．
- `EphBatch` などの配列は「局所の配列＋全体索引への対応」を持ちます．物理クレートは全体索引で結果を書き込みます．
- 当面は `Partition` が常に全範囲ですが，コードは最初から局所範囲だけを回すように書きます．

**(2) 縮約の種類を累積器の型に持たせる**

物理量ごとに，部分結果の合わせ方が違います．これを累積器の関連型で宣言させます．

| 種類 | 例 | 合わせ方 |
|---|---|---|
| 和 | α²F(ω)，λ，DOS，移動度の分子 | allreduce（和） |
| k ごと | 電子自己エネルギー Σ(k,ω)，緩和時間 τ_nk | 全体索引での gather（各 k は 1 つのランクだけが持つ） |
| q ごと | フォノン線幅 γ_qν，λ_qν | 同上（q 軸） |

累積器は `accumulate`（局所の部分結果を作る）と，宣言した種類に従う縮約だけを持ちます．縮約そのものは `Comm` が行うので，物理クレートは通信を直接呼びません．スレッド並列も同じ縮約で処理します．

**(3) 巨大データは「源」として抽象化**

g(Rₑ,Rₚ) は数 GB から数百 GB になります．`eph-wannier-data` では g を配列として持たず，`EphMatrixSource` trait（Rₚ のブロック単位で取り出す）にします．

- 当面の実装：全部メモリに載せる，またはファイルからブロック単位で読む（EPW の `etf_mem = 1` 相当）．
- 後の実装：Rₚ の範囲をランクで分けて持ち，q ごとの部分和を allreduce する（`ephwan2blochp_dist` 相当）．
- メモリ配置は Rₚ を最外にします．ブロック単位の読み出しと分散のどちらにも都合がよいためです．

**(4) 通信できる形のデータ**

- 境界データと結果は，連続した数値配列とスカラーだけで構成します（参照カウントやポインタの入れ子を持たない）．broadcast や gather がそのまま書けます．
- 入力ファイルの読み込みは 1 ランクで行い，`Comm::broadcast` で配る前提にします（当面は何もしない）．

**(5) 決定性と分割不変性のテスト**

- 和は全体索引の順に固定した順序で取ります．スレッド数やランク数で結果が変わらないようにするためです．
- `ThreadComm`（スレッドで複数ランクを模擬する実装）を使い，「1 ランクと 4 ランクで結果がビット一致する」ことを L0 と L1 のテストに入れます．MPI を実装する前に，分割の正しさを検証できます．
- 再開（restart）も全体の q 索引で記録します．分割を変えて再開しても同じ結果になります．

## 3. クレート分割案

名前は仮です（crates.io での衝突確認と接頭辞の統一は後で決めます）．

### 3.1 基盤層（EPW を知らない汎用部品）

| クレート | 内容 | EPW の対応部分（行数） | 主なテスト |
|---|---|---|---|
| `phys-units` | 物理定数と単位変換（Ry，Bohr，eV，K） | `ep_constants` 他 | 換算の往復 |
| `eph-comm` | `Comm` trait，分割（`Partition`），縮約の種類．実装は `SelfComm`（1 プロセス）と `ThreadComm`（テスト用の模擬多ランク）．`MpiComm` は後で追加 | `parallelism`，`mp*` | 分割の被覆と重なりなし，縮約の結果が分割数によらない |
| `smearing` | Gauss，Methfessel-Paxton，Fermi-Dirac，Bose-Einstein，適応 smearing，四面体法．Fermi 準位の決定 | `utilities`，`low_lvl` の一部（約 1.5k） | 規格化，和則，既知の DOS |
| `bz-mesh` | 均一格子，k+q の対応表，Fermi 窓，IBZ と既約化（対称性は moyo などに委譲） | `bzgrid`，`kfold`，`kgen`，`symmetry` の一部（約 5k） | 格子点数，重みの和，対称性での不変性 |
| `wannier-interp` | WS 格子と重複度，R → k の Fourier 補間，H(k) と D(q) の対角化，g(Rₑ,Rₚ) → g(k,q) と回転，速度行列要素．GEMM を差し替え可能に（CPU，GPU） | `wannier2bloch`，`wannier2bloch_opt`，`wigner`（約 5.8k） | 1 次元鎖や graphene の tight-binding で解析解，エルミート性，並進不変性，音響和則 |
| `eph-longrange` | 極性補正（Fröhlich 双極子，四重極），2 次元の Coulomb 切断 | `longrange`（約 2k） | 既知の極限（q → 0 の LO-TO 分裂） |

`wannier-interp` と `eph-longrange` は，Wannier 補間をする他のコード（phonopy 系，Wannier90 の後処理，DeepH 系）からも使える単位です．

### 3.2 境界データ層

| クレート | 内容 | 主なテスト |
|---|---|---|
| `eph-wannier-data` | 境界データの型：結晶（格子，原子，質量），WS 格子 3 組，H(R)，力定数，g(Rₑ,Rₚ)，Z*，ε∞，対称操作．I/O なし | 型の整合性チェック（次元，エルミート性） |
| `eph-io-epw` | EPW の `crystal.fmt`，`epwdata.fmt`，`wigner.fmt`，`*.epmatwp` を読む | Pb，MgB₂，Si の fixture の読み込み |
| `eph-io-h5` | EPW-VASP の `elphon_wannier.h5` を読み書き（HDF5 は実行時読み込み） | EPW 形式との往復で一致 |

### 3.3 物理層（`EphBatch` を受け取る累積器）

| クレート | 内容 | EPW の対応部分（行数） | 入力 | 主なテスト |
|---|---|---|---|---|
| `eph-selfen` | 電子自己エネルギー，フォノン自己エネルギーと線幅，λ_qν，α²F，nesting 関数 | `selfen`（約 2.8k） | `EphBatch` | Holstein 模型の解析解，∫α²F/ω = λ/2，Pb fixture |
| `eph-spectral` | スペクトル関数，キュムラント展開 | `spectral`，`cumulant`（約 3.3k） | Σ(k,ω) | 解析的な Σ で既知の A(ω) |
| `eph-transport` | SERTA，反復 BTE，磁場下（Hall） | `transport`，`transport_mag`（約 2.9k） | `EphBatch`，速度 | 定数緩和時間の解析解，Si fixture |
| `eliashberg` | 等方的と異方的 Migdal-Eliashberg（虚軸，実軸，線形化による Tc），μ*．Matsubara 和は sparse-ir-rs | `supercond*`（約 1.2 万） | 等方的は α²F(ω) のみ，異方的は Fermi 面上の g² | Einstein 模型，McMillan と Allen-Dynes との比較，MgB₂ の `.a2f` |
| `eph-polaron` | ポーラロン（自己無撞着，励起子ポーラロン） | `polaron*`，`expolaron`，`exphonon`（約 8k） | `EphBatch` | 後回し |
| `eph-optics` | 間接吸収，量子ドット吸収 | `indabs`，`qdabs`（約 5.4k） | `EphBatch`，速度 | 後回し |
| `eph-tdbe` | 時間依存 Boltzmann，フォノン間散乱 | `tdbe_mod`，`phph`，`screening`（約 5.4k） | `EphBatch`，3 次の力定数 | 後回し |

`eliashberg` の等方的ソルバーは α²F(ω) だけを入力にするので，DFT とも EPW とも無関係に使え，テストもできます．sparse-ir-rs の利点を見せる場所としても最も単純です．

### 3.4 ドライバ層

| クレート | 内容 | EPW の対応部分 |
|---|---|---|
| `eph-driver` | q ループ，k と q の分割，スレッド並列（rayon），再開（restart），累積器の登録と縮約 | `use_wannier`，`global_setups`，`parallelism` |
| `epw-rs`（バイナリ） | EPW の入力（namelist）を読む互換層と出力．Fortran 版と同じ入力で比較するためのもの | `readin`，`input`，`printing`，`io_*` |

入力互換層をバイナリ側に閉じ込めることで，ライブラリ側は EPW の 200 を超える入力変数に縛られず，物理量ごとの設定構造体だけを持ちます．

### 3.5 依存関係

```
eph-comm  phys-units  smearing  bz-mesh  wannier-interp  eph-longrange
     \          |        |          |               /
      +---------+--------+----------+--------------+
                          |
                   eph-wannier-data  <-- eph-io-epw, eph-io-h5
                          |
                      (EphBatch)
          +-------+-------+--------+---------+
          |       |       |        |         |
     eph-selfen  eph-transport  eph-polaron  eph-optics ...
          |
       (α²F, Fermi 面上の g²)
          |
      eliashberg  <-- sparse-ir
                          |
                     eph-driver  -->  epw-rs（バイナリ）
```

物理層のクレートは互いに依存しません．現在の EPW にある `selfen` → `transport`，`utilities` → `indabs` のような横の依存はなくします．共通に必要なもの（Fermi 準位，占有数）は `smearing` に下ろします．

## 4. テストの層

| 層 | 内容 | 実行時間 | 例 |
|---|---|---|---|
| L0 | 解析解，toy model，和則 | ミリ秒 | 1 次元鎖の補間，Holstein 模型の自己エネルギー，Einstein 模型の Eliashberg |
| L1 | Wannier 境界の fixture から物理量まで | 秒 | Pb の λ，α²F（`epw_metal`），Si の移動度（`epw_mob`） |
| L2 | 中間ファイル境界での比較 | 秒 | MgB₂ の `.a2f` から等方的 Eliashberg（`epw_super/epw5`），`.ephmat` から異方的 Eliashberg |
| L3 | EPW と同じ入力で全体を比較 | 分 | `epw_*` の各ケースを Fortran と並べる |

L0 と L1 が AI 開発のファーストフィードバックになります．L1 と L2 の fixture は，QE＋EPW を 1 回だけ流して作り，リポジトリに置きます（小さいものに限る）．

## 5. 段階的な実装順

1. **Pb（最小例）：** `eph-comm`（`SelfComm` と `ThreadComm`），`phys-units`，`smearing`（Gauss のみ），`bz-mesh`（全格子のみ，対称性なし），`wannier-interp`（極性なし），`eph-wannier-data`，`eph-io-epw`，`eph-selfen`（フォノン線幅，λ，α²F），簡易ドライバ．
2. **MgB₂：** `eliashberg`（等方的，sparse-ir-rs）．L2 で `.a2f` から Tc．
3. **Si：** `bz-mesh` に対称性と IBZ，`eph-transport`（SERTA），速度行列要素．
4. **極性物質：** `eph-longrange`．
5. **EPW-VASP：** `eph-io-h5`．
6. **MPI：** `MpiComm`（MPI は実行時読み込み），`EphMatrixSource` の分散実装．物理クレートは変更しない．

## 6. 未確認の点

- ~~`epmatwp` のバイト配置~~：確認済み．記録マーカーのない complex128 の column-major 配列 `(nbnd, nbnd, nrr_k, nmodes, nrr_g)` で，Rₚ が最外です（[pb_reproduction_ja.md](pb_reproduction_ja.md)）．
- ~~EPW の Fermi 準位の再計算規則~~：確認済み．`efermi_read` でなければ細かい k 格子で常に再決定します．QE の `efermig`（Gauss で二分法 → Methfessel-Paxton で Newton 法 → 失敗したら二分法）を使います．
- 対称性の扱いを境界データに含めるか，結晶構造から再計算するか．
- 行数は QE develop ブランチの 2026-10 時点の集計です．

# 下流だけのテストと最小例の計画

作成日：2026-10-15．EPW 開発者に見せる「動く最小例」の計画です．sparse-ir にはこだわらず，物質の計算を見せることを優先します．

## 1. 下流だけテストできる仕組みは既存テストにある

QE の `test-suite/epw_*` は，すでに「上流で 1 回書き出し，下流は読み込むだけ」という構成です．

- **上流：** `epw1.in` が `epwwrite = .true.`，`wannierize = .true.` で Wannier 表現のデータをファイルに書き出します．
- **下流：** `epw2.in` 以降はほぼすべて `epwread = .true.`，`wannierize = .false.` で，そのファイルを読むだけです．例：`epw_super` の epw3 から epw11，`epw_mob` の epw2 から epw6．
- **境界のファイル：**
  - `crystal.fmt`：原子数，モード数，電子数，格子（`at`，`bg`），体積，`alat`，原子位置，質量，原子種，`noncolin`（`io/io.f90` で書き出し）．
  - `epwdata.fmt`：H(R)，力定数，E_F など．
  - `wigner.fmt`：Wigner-Seitz ベクトルと重複度（電子，フォノン，g の 3 組）．
  - `<prefix>.epmatwp`：Wannier 表現の g(Rₑ,Rₚ)．Fortran の direct-access バイナリ（MPI-IO で書き出し）．
  - 必要に応じて `vmedata.fmt`（速度行列要素），`dmedata.fmt`．

Fortran の EPW を 1 回だけ流してこれらを fixture として固めれば，Rust の下流は pw.x も ph.x も QE も無しで，何度でも数秒でテストできます．EPW-VASP の `elphon_wannier.h5` も，ほぼ同じ情報を HDF5 にしたものです．

## 2. 候補の比較

| テスト | 物質 | 下流で必要なもの | 最小例としての評価 |
|---|---|---|---|
| **epw_metal** | **Pb**（1 原子，Wannier 4 本，3 モード） | H，D，g の補間，フォノン線幅から λ，α²F，ω_log | **最適**．金属なので極性補正が要らず，データも最小 |
| epw_base | ダイヤモンド（2 原子，4 本） | 補間，電子自己エネルギー | 簡単だが，見せる結果が地味 |
| epw_super | MgB₂（3 原子，5 本） | 上に加えて Fermi 面上の g²，等方的 Eliashberg（虚軸，`liso`，`limag`） | 第 2 弾向き．Matsubara 和なので sparse-ir の出番 |
| epw_mob | Si（2 原子，16 本） | 上に加えて速度，キャリア密度からの E_F，scissor，IBZ の k リスト（`kpt.txt`），対称化，SERTA | 開発者に一番響くが，部品が多い．第 3 弾向き |

## 3. 最小例（Pb，`epw_metal`）でやること

`epw_metal/epw1.in` は `phonselfen = .true.`，`a2f = .true.`，`nbndsub = 4`，粗い格子 3³，細かい格子 6³，`fsthick = 6` eV，`degaussw = 0.1` eV です．

1. **読み込み：** `crystal.fmt`，`epwdata.fmt`，`wigner.fmt`，`pb.epmatwp`．
2. **Wannier から Bloch への補間：**
   - H(R) から ε_nk．
   - 力定数から ω_qν，e_qν．
   - g(Rₑ,Rₚ) から g_mnν(k,q)．電子とフォノンの固有ベクトルで回転し，フォノン変位で正規化．
3. **物理量：** フォノン線幅 γ_qν（デルタ関数の Gauss smearing と `fsthick` 窓），λ_qν，α²F(ω)，λ，ω_log，Allen-Dynes の Tc．
4. **規模：** Rust で 1,000 行前後の見込み．6³×6³ 格子なら一瞬で終わり，対称性を使わず全格子で計算できます．

## 4. 開発者に見せる形

- **同じ入力での一致：** 同じ fixture から，Fortran の出力（`lambda___`，`gamma___`，`logavg` など）と Rust の値を並べ，α²F を重ね描きします．
- **物理の絵：** 密な格子（EPW の Pb チュートリアル程度の規模）で α²F と Tc を出し，実験値と並べます．Rust 版の速度もここで示せます．
- **ユニットテスト（DFT 不要）：**
  - H(k) のエルミート性，R の並進不変性，音響和則，∫α²F(ω)/ω dω = λ/2．
  - 1 バンドの tight-binding と Holstein の toy model で解析解と比較．
- **`cargo add` で他から使える形**にします．OpenMX 資料の主張の実演にもなります．

## 5. 注意点

- **テスト用設定の数値は物理的に意味がない：**
  - `epw_metal/epw1` の参照出力には `Minimization algorithm failed to find Fermi energy: reverting to bisection` とあり，E_F が粗い格子の 9.88 eV から細かい格子の 11.85 eV に跳んでいます．同じ出力の最初の q 点では λ と γ がすべて 0 と出ています．
  - testcode の許容誤差も，λ によっては絶対値 1.5 と非常に緩いものがあります．
  - したがって「テストに合格した」ではなく，**同じ fixture で Fortran と桁まで一致すること**を主張にします．物理の絵は別に密な格子で作ります．
- **一致には EPW の規約の再現が必要：** 細かい格子での E_F の再計算方法，`degaussw`／`degaussq`，`eps_acoustic` による Γ 点音響モードの除外，WS 重複度の重み，単位（Ry，Bohr）．
- **バイナリの読み方は要確認：** `epmatwp` は `(nbnd, nbnd, nrr_k, nmodes)` を 1 レコードとして nrr_g 方向に並ぶ形式に見えます（`bloch2wannier.f90` の `ephbloch2wanp_mem`）．MPI-IO で書かれているので，バイトの並びは実際のファイルで確かめます．
- **ゲージ：** fixture の H(R) と g は同じ EPW 実行から出るので一貫しています．EPW-VASP 形式に対応するときは付録 C の規約差を確認します．

## 6. 進め方

1. QE+EPW を一度ビルドし，`epw_metal` を流して fixture を作る．
2. Pb：読み込み，補間，λ，α²F，Tc．Fortran と数値一致を確認．
3. MgB₂：等方的 Eliashberg．ここで sparse-ir-rs を入れる．
4. Si：SERTA の移動度．

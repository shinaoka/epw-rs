# EPW の構造と QE との結合：調査メモ

調査日：2026-10-15．対象は Quantum ESPRESSO（QE）の develop ブランチにある `EPW/src` と `test-suite/epw_*` です（GitLab API で読み込み）．

## 要点

- EPW には OpenMX と同じく**ユニットテストがなく**，全系を流して結果を比べる回帰テストだけがあります．
- QE 内部に深く依存しているのは，粗い格子で電子格子結合 g を作る**上流の約 1 万行だけ**です．
- Wannier 補間より後の**下流（全体の約 7 割）は QE との結合が浅く**，Rust の独立した crate 群に書き直すのは現実的です．
- この境界は，EPW と VASP の開発者が EPW-VASP で引いた境界とほぼ一致します．下流が QE なしで成り立つことは，すでに実証されています．

構造の図は [epw_structure.svg](epw_structure.svg) にあります．

## 1. 規模とテスト

- **規模：** `EPW/src` に Fortran ファイルが 78 本，合計約 10.4 万行あります．
- **大域変数：** `global_var.f90` の冒頭コメントは「EPW で使うすべての大域変数を収めるモジュール」です．`SAVE` 付きで，約 290 個の変数があります．OpenMX の `openmx_common.h` と同じ構図です．
- **ユニットテスト：** ありません．EPW のソースツリーに test と名の付くファイルは 1 本もありません．
- **回帰テスト：** QE の `test-suite/` に `epw_*` が 21 件あります．
  - 件名：epw_2D，epw_base，epw_hall，epw_lsda_metal，epw_metal，epw_mob，epw_mob_U，epw_mob_ibte，epw_mob_ibte_sym，epw_mob_polar，epw_pl，epw_plrn，epw_polar，epw_qdpt，epw_soc，epw_super，epw_tdbe，epw_trev，epw_trev_paw，epw_trev_uspp，epw_wfpt．
  - 仕組み：testcode を使い，`jobconfig` と `userconfig.tmp` で量ごとに許容誤差を決めて，参照出力（`benchmark.out.git.*`）と比べます．
  - 各ケースは `pw.x`，`ph.x`，`epw.x` を順に実行します．例えば `epw_base` は 17 段の入力を順に流します．

## 2. QE との結合の深さ

モジュールの `USE` 文を集計した結果です．

| 層 | 主なファイル | QE への依存 |
|---|---|---|
| **上流**：粗い格子での g の計算，Wannier 化 | `ep_coarse`（832 行），`ep_coarse_unfolding`（1368），`pw2wan`（4093），`dvqhub`（1802），`dvqpsi`（1205），`init`（267），`setups`（183）．計約 9,750 行 | 深い．`wvfct`，`gvect`，`fft_*`，`uspp*`，`becmod`，DFPT の内部状態（`phcom`，`lrus`，`eqv`，`qpoint`），DFT+U（`ldau*`），`paw_variables`，wannier90 ライブラリを直接使う |
| **下流**：Wannier 補間と物性計算 | `wannier2bloch`，`selfen`，`spectral`，`transport`，`polaron*`，`supercond_*`，`tdbe` など | 浅い．使うのは MPI ラッパー（`mp`，`mp_global`），`io_global`，`cell_base`，`ions_base`，物理定数程度．加えて対称操作（`symm_base`），四面体法（`ktetra`），`ef` と `nelec`（`pwcom`）．例えば `selfen` が QE から取るのは `klist` の `degauss` と `ngauss` だけ．詳細は [crate_design_ja.md](crate_design_ja.md) |

**自然な境界は「Wannier 表現の g(Rₑ,Rₚ)，H(R)，力定数，格子と原子の情報」**です．上流は QE の中に残し，下流は独立した crate 群にできます．下流が実際にやっているのは，Fourier 補間，対角化，BZ 上の和，Eliashberg 方程式や Boltzmann 方程式の反復です．

前例として，[Phoebe](https://github.com/mir-group/phoebe)（C++，MIT ライセンス）は Wannier 補間による電子格子散乱を EPW とは独立に実装しています．

## 3. sparse-ir

**EPW はすでに sparse-ir を使っています．**

- `utilities/sparse_ir.f90`（約 4,000 行）は，冒頭コメントによると [sparse-ir-fortran](https://github.com/SpM-lab/sparse-ir-fortran)（CC0）を改変して取り込んだ（vendoring した）ものです．
- 基底データは事前計算したファイル `EPW/irobjs/ir_nlambda5_ndigit6.dat` で持っています．

Rust なら，コードの複製も事前計算ファイルも不要で，sparse-ir-rs を crate として依存に加えるだけで済みます．上流の改良もバージョンを上げれば取り込めます．「他で開発された高性能なライブラリを再利用しやすい」という主張の具体例になります．

## 4. ライセンス

- `EPW/License` は **GPLv2** です．ソースのヘッダは「GNU General Public License」とだけ書いていて，バージョンを指定していません．
  - GPLv2 第 9 条には，バージョン指定がなければ任意のバージョンを選べるという規定があります．ただし，EPW の場合にどう解釈されるかは確認が必要です．
  - GPLv2 のみと解釈される場合，Apache-2.0 とは両立しません．Rust で標準的な「MIT OR Apache-2.0」との組み合わせに注意が要ります．
- 著作権表示は多数の名義に分かれています．Noffsinger，Malone，Giustino（2007 から 2009），Poncé，Margine，Verdi，Giustino（2010 から 2016），EPW-Collaboration（2016 以降）です．許諾を得るなら，OpenMX より相手が多くなります．
- 一方，下流の物理は論文（EPW の論文群，Giustino の総説など）に式が詳しく書かれています．**論文の式から独立に実装し，EPW の出力と数値を突き合わせて検証する**道が取りやすいと考えます．出力の比較自体はコードの複製ではありません．Phoebe もこの形で成り立っています．どこまでが「改変」に当たるかの判断には注意が残ります．

## 5. EPW-VASP との関係

出典は 2025 年 11 月のプレプリント [arXiv:2511.21905](https://arxiv.org/abs/2511.21905) です（著者：Radevych，Thorn，Engel，Kolmogorov，Tiwari，Kresse，Giustino，Margine）．EPW-VASP が共有しているのは，**EPW の下流だけ**です．上流は丸ごと VASP に置き換わっています．

- **VASP 側（上流の代わり）：** 電子格子行列要素を，DFPT ではなくスーパーセルの有限変位で計算します．
- **受け渡し：** HDF5 ファイル `elphon_wannier.h5` 1 本です．中身は次の通りです．
  - 格子ベクトルと原子位置（基本セルとスーパーセル）
  - Wannier 中心
  - Wannier 表現の実空間ハミルトニアン H(R)
  - 原子間力定数（`phonon_data.h5` で別に渡すこともできる）
  - Wannier 表現の実空間 g(Rₑ,Rₚ)
  - 極性補正を使う場合は誘電テンソルと Born 有効電荷
- **EPW 側（下流を再利用）：** Wannier から Bloch への変換と Fourier 補間，長距離の極性補正，異方的 Migdal-Eliashberg ソルバー，Boltzmann 輸送ソルバーです．入力フラグ `lvasp = .true.` で HDF5 を読みます．

**示唆**

- 上の「境界」は，EPW と VASP の開発者自身が引いた境界とほぼ一致します．
- 同じプレプリントは，QE，Koopmans，Yambo，phonopy から EPW 用の入力を作る関連ツール（EPWpy，ElePhAny）にも触れています．上流は複数の候補，下流は EPW 一択という構図になりつつあります．
- 下流を Rust の独立した crate にしてこの HDF5 形式を読めるようにすれば，QE，VASP，phonopy のどの上流とも組み合わせられます．

**注意点**

- **未リリース：** QE の develop ブランチの EPW ソースには，まだ `lvasp` も `elphon_wannier` も入っていません．2026 年春の VASP フォーラムでも「リリースは両コードの進捗次第」とされていました．
- **ゲージの一貫性：** Wannier 関数の構築と g の評価で同じ波動関数を使わないと結果が無意味になる，と EPW の開発者は警告しています．境界のデータを作る側の責任です．
- **未確認：** 付録 C（規約の違い：Wannier の定義，ゲージ）の後半はまだ読めていません．

## 6. OpenMX 資料から流用できる部分

[related/openmx_crates_and_license_ja.md](related/openmx_crates_and_license_ja.md) の 1 節（AI によるファーストフィードバック），2 節（大域変数とテスト，速度），4 節（crate，`cargo add`，実行時読み込み，ライブラリ再利用）は，ほぼそのまま使えます．差し替えが必要なのは次の 3 点です．

- 現状の節を EPW の数字に置き換える．
- 部品化の提案を「上流と下流の境界」にする．
- ライセンスの節を GPLv2 版にし，「独立実装」という選択肢を足す．

## 出典

- [EPW/License](https://gitlab.com/QEF/q-e/-/blob/develop/EPW/License)
- [EPW/src](https://gitlab.com/QEF/q-e/-/tree/develop/EPW/src)
- [QE test-suite](https://gitlab.com/QEF/q-e/-/tree/develop/test-suite)
- [sparse-ir-fortran](https://github.com/SpM-lab/sparse-ir-fortran)
- [Phoebe](https://github.com/mir-group/phoebe)
- [EPW paper (arXiv:1005.4418)](https://arxiv.org/abs/1005.4418)
- [EPW-VASP interface (arXiv:2511.21905)](https://arxiv.org/abs/2511.21905)
- [VASP forum: electron-phonon coupling](https://vasp.at/forum/viewtopic.php?p=33209)
- [EPW forum](https://forum.epw-code.org/viewtopic.php?p=3446)

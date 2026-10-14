# epw-rs

EPW（Quantum ESPRESSO の電子格子結合コード）の下流部分，つまり Wannier 補間以降の物性計算を Rust で書き直す構想のリポジトリです．EPW 開発者に提案し，動く最小例を見せることを目標にしています．

現時点ではコードはなく，調査資料と計画だけを置いています．

## 資料

- [docs/epw_analysis_ja.md](docs/epw_analysis_ja.md)：EPW の構造，QE との結合の深さ，テスト，sparse-ir，ライセンス，EPW-VASP との関係の調査．
- [docs/epw_structure.svg](docs/epw_structure.svg)：EPW と QE の結合構造の図．
- [docs/minimal_example_plan_ja.md](docs/minimal_example_plan_ja.md)：下流だけをテストする方法と，既存テストを元にした最小例（Pb の λ，α²F，Tc）の計画．
- [docs/related/openmx_crates_and_license_ja.md](docs/related/openmx_crates_and_license_ja.md)：OpenMX 開発者向けの提案資料（AI 開発，crate 化，ライセンス）．EPW 提案で論点を流用するための参照用コピーです．原本は tensor4all-meta にあります．

## 方針

- EPW の Fortran ソースはこのリポジトリに入れません（GPL）．実装は論文の式から独立に行い，EPW の出力と数値を突き合わせて検証します．
- テストの fixture は，EPW が `epwwrite` で書き出す Wannier 表現のデータ（`crystal.fmt`，`epwdata.fmt`，`wigner.fmt`，`*.epmatwp`）を使います．

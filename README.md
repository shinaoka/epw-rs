# epw-rs

> **Not intended for publication.** This repository is a **technical demonstration for the EPW development team** only.
> It is not a released product, is not published on crates.io, and makes no claim to replace or compete with EPW.
> EPW and Quantum ESPRESSO remain the reference implementations; all numbers here are validated against them.

EPW（Quantum ESPRESSO の電子格子結合コード）の下流部分，つまり Wannier 補間以降の物性計算を Rust で書き直す構想のリポジトリです．EPW 開発者に提案し，動く最小例を見せることを目標にしています．

最小例として，Pb（QE の `test-suite/epw_metal`）のフォノン線幅，λ，α²F，Allen-Dynes の Tc を EPW と数値一致させるところまで実装しています．詳細と検証結果は [docs/pb_reproduction_ja.md](docs/pb_reproduction_ja.md) にあります．

## 動かし方

```
cargo run --release --bin epw-rs -- tests/fixtures/pb_epw_metal/wannier out/
cargo test --workspace
```

`out/` に `linewidths.dat` と `a2f.dat` を書き，E_F，N_F，λ，ω_log，Tc を表示します．`--ranks N` を付けると N ランクをスレッドで模擬します（結果はビット単位で同じです）．Rust 1.96 以上が必要です．

## 資料

- [docs/epw_analysis_ja.md](docs/epw_analysis_ja.md)：EPW の構造，QE との結合の深さ，テスト，sparse-ir，ライセンス，EPW-VASP との関係の調査．
- [docs/epw_structure.svg](docs/epw_structure.svg)：EPW と QE の結合構造の図．
- [docs/crate_design_ja.md](docs/crate_design_ja.md)：下流の構造分析と，再利用しやすくテストしやすいクレート分割案．
- [docs/minimal_example_plan_ja.md](docs/minimal_example_plan_ja.md)：下流だけをテストする方法と，既存テストを元にした最小例（Pb の λ，α²F，Tc）の計画．
- [docs/pb_reproduction_ja.md](docs/pb_reproduction_ja.md)：Pb の再現の作業記録（設計判断，EPW との一致，見つかった点）．
- [tests/fixtures/pb_epw_metal/README.md](tests/fixtures/pb_epw_metal/README.md)：fixture の由来，原著者，再生成の手順．
- [docs/related/openmx_crates_and_license_ja.md](docs/related/openmx_crates_and_license_ja.md)：OpenMX 開発者向けの提案資料（AI 開発，crate 化，ライセンス）．EPW 提案で論点を流用するための参照用コピーです．原本は tensor4all-meta にあります．

## 方針

- EPW の Fortran ソースはこのリポジトリに入れません（GPL）．実装は論文の式から独立に行い，EPW の出力と数値を突き合わせて検証します．
- テストの fixture は，EPW が `epwwrite` で書き出す Wannier 表現のデータ（`crystal.fmt`，`epwdata.fmt`，`wigner.fmt`，`*.epmatwp`）を使います．
- テンソル演算は tensor4all の [tenferro-rs](https://github.com/tensor4all/tenferro-rs)（column-major）を使います．Fourier 和は `tenferro-fft`，縮約は `tenferro-einsum`，対角化は `tenferro-linalg` です．
- 式は論文から実装しています（各クレートの冒頭に出典）．ファイル形式と数値上の規約（Fermi 準位の探索，Fermi 窓，縮退モードの平均など）は，EPW 6.1（QE 7.6）のソースを読んで確認し，それに合わせています．Fortran のコードを翻訳したものではありません．

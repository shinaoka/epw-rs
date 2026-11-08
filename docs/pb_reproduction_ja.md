# Pb の再現：作業記録

作成日：2026-10-15．[minimal_example_plan_ja.md](minimal_example_plan_ja.md) の第 1 段（Pb の λ，α²F，Tc）の実装記録です．

## 結果

QE 7.6（EPW 6.1）の `test-suite/epw_metal` から作った fixture（[tests/fixtures/pb_epw_metal](../tests/fixtures/pb_epw_metal/README.md)）で，Rust 版は EPW が出す値をすべて再現します．EPW 側の参照値は，同じ Wannier ファイルだけを置いたディレクトリで `epwread` した出力です．

| 量 | EPW | Rust | 差 |
|---|---|---|---|
| E_F（細かい k 格子） | 11.755998 eV | 11.7559985 eV | 印字桁内（4.6e-7 eV） |
| E_F の求め方 | Newton 失敗 → 二分法 | 同じ経路 | — |
| N_F（スピンあたり） | 0.518460 /eV | 0.518460 /eV | 印字桁内（2.8e-7 /eV） |
| Fermi 窓のバンド | 2〜4 | 2〜4 | — |
| ω_qν（648 個） | `linewidth.phself` | — | 最大 1.7e-7 meV |
| γ_qν（648 個） | `linewidth.phself` | — | 最大 2.5e-9 meV（最大値 0.085 meV に対し 3e-8） |
| λ_qν（648 個） | `lambda.phself`（5 桁） | — | 相対 5e-5 以内（印字桁内） |
| α²F（500 点 × 10 smearing） | `pb.a2f.01.300.000`（小数 7 桁） | — | 最大 5.0e-8（印字の丸め） |
| λ（q 和） | 0.1608789 | 0.1608789 | 2.8e-8 |
| λ（2∫α²F/ω） | 0.1608847 | 0.1608847 | 2.8e-9 |
| ω_log | 0.0004572 Ry | 0.0004572 Ry | 2.3e-8 Ry |
| Allen-Dynes Tc（μ* = 0.10） | 0.000000003020 K | 0.000000003020 K | 印字桁で一致 |

検証は `crates/eph-driver/tests/pb_epw_metal.rs` で，各比較の残差を表示します．この設定の λ ≈ 0.16 は回帰テスト用の粗い格子の値で，物理的な意味はありません（Pb の実験値は約 1.2）．

λ_tr と α²F_tr（速度行列要素と cos θ が要る）は今回は実装していません．

## 設計判断

- **クレート分割：** [crate_design_ja.md](crate_design_ja.md) §5 の第 1 段に従い，`phys-units`，`eph-comm`，`smearing`，`bz-mesh`，`eph-wannier-data`，`eph-io-epw`，`wannier-interp`，`eph-selfen`，`eph-driver` の 9 クレートにしました．各クレートは Pb に要る部分だけを持ちます．すべて `publish = false` です．
- **tenferro-rs：** crates.io の 0.10.0 を使います．host の `TypedTensor` と，`CpuEngine`（実行用のプールを持つ `CpuEngine`）での直接実行（AD なし）です．配列はすべて column-major です．
  - **R → k の Fourier 和：** WS ベクトル R を細かい格子に畳み込んで（R mod N），3 次元の逆 FFT（`tenferro-fft`）を取ります．格子点 k では e^{2πik·R} が畳み込んだ位置の位相と一致するので，畳み込みは近似ではなく厳密です．直接和との一致を `wannier-interp/tests/fourier.rs` で確認しています（|HU − UE| < 1e-12 Ry）．
  - **Rₚ → q：** q ブロックごとに位相行列を作り，einsum で縮約します．Rₚ は `EphMatrixSource` からブロックで読みます（EPW の `etf_mem = 1` 相当）．ディスクから逐次読む実装（`EpmatwpFile`）と全部メモリに載せる実装の結果は一致します．
  - **Bloch 基底とモード基底への回転：** `U_{k+q}† g̃ U_k` とモード変位 u_ν を，2 項の einsum を 3 回で縮約します．
  - **対角化：** `tenferro-linalg` のバッチ `eigh` です．
- **格子の順序：** EPW の細かい格子の点の順序（最後の成分が最速）は，形 `[n3, n2, n1]` の column-major の添字と同じです．FFT の格子をこの形で持つので，並べ替えは要りません．k+q の対応表は，q 格子が k 格子を割り切ることを前提にしています．
- **分散：** 分散するのは **大域で固定した q ブロック**です（`Config::q_block`）．各ランクは自分の q ブロックだけを補間し，q ごとの結果を大域の順序で gather します．α²F と λ はその後で和を取ります．これにより 1 ランクと 4 ランク（`ThreadComm`）の結果がビット単位で一致し，テストで確認しています．
- **EPW の規約への合わせ方：** 式は論文から書いています（出典は各クレートの冒頭）．数値上の規約は EPW 6.1 のソースを読んで確認し，それに合わせています．Fortran のコードの翻訳ではありません．合わせた規約は次のとおりです．
  - Fermi 準位：Gauss の二分法，Methfessel-Paxton の Newton 法，失敗したら二分法．許容誤差は 1e-10，括りは [min − 10σ, max + 10σ]．
  - Fermi 窓：k と k+q の両方にバンドが `fsthick` 内にある対だけを数え，バンド範囲は格子全体で決めます．
  - フォノン：ω ≤ `eps_acoustic`（0.1 cm⁻¹）のモードは除き，1e-6 Ry 以内の縮退モードで平均します．
  - 占有数は 300 K の Fermi-Dirac，エネルギー保存のデルタは Gauss です．N_F は MP1 のデルタです．
  - α²F：負の λ は 0 とします．

## 見つかった点

1. **EPW の `linewidth.phself` は 2γ を書きます．** 標準出力の `gamma___` は γ です．同じ値の 2 倍がファイルに出ます．比較ではこれに合わせています．
2. **EPW 内で H(k) の Hermite 化が経路によって違います（整合性の指摘．影響は確認できていません）．** E_F と Fermi 窓に使う `hamwan2bloch` は (H + H†)/2 を対角化します．q ループの `hamwan2bloch_batch` は Hermite 化せずに `ZHEEVD('L')`（下三角だけ）です．この fixture では H(R) が H(−R)† と 2.8e-17 Ry で一致し，固有値の差は 0 でした．`HermitianPart::LowerTriangle` で EPW の q ループの規約も選べるようにしました．
3. **tenferro の多項 einsum の縮約順序（tenferro への報告候補）．** 4 項の einsum を同じ入力で 20 回呼ぶと，最後のビットの異なる結果が 4 通り出ました．2 項ずつ明示的に縮約すると毎回同じになります．planner の縮約経路の選び方が呼び出しごとに変わると推定しています．
4. **tenferro のスレッド数．** スレッド数を固定すれば結果は毎回同じです．スレッド数を変えると最後のビットが変わります．そこで `Config::threads_per_rank`（既定 1）でスレッド数を固定し，並列はランクで取ります．

## 残る制約と未確認の点

- EPW との差のうち，絶対値の小さな γ（最大値の 1e-7 以下）は相対 3e-5 程度ずれます．最も大きいものでも絶対値は最大値の 1.8e-12 です．E_F のずれや，バンド 1 と 2 の縮退が原因ではないことは確かめましたが，原因は特定できていません．比較の許容誤差は「最大値の 1e-7 ＋ 相対 1e-6」としています．
- 対応しているのは `use_ws = .false.`（WS の重複度がベクトルあたり 1 つ）の `wigner.fmt` だけです．非共線，LSDA，極性補正，対称性による既約化には未対応です．
- 計算時間（参考）：Rust 版は 0.47 s（1 ランク，release）です．EPW の `epwread` は 1.32 s ですが，電子の自己エネルギーも計算しているので，そのままでは比べられません．

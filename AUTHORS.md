# Authors and copyright

`epw-rs` is an independent Rust re-implementation of the **downstream** of
[EPW](https://epw-code.org), the electron-phonon code distributed as part of
[Quantum ESPRESSO](https://www.quantum-espresso.org). It is not a fork: it shares no source
with EPW, and every number it produces is validated against EPW's own output.

Because the design, the algorithms and the reference data come from EPW, **EPW's authors and
its licence are carried over**:

- **Copyright (C) The EPW Collaboration.** EPW is distributed under the GNU General Public
  License, version 2; `epw-rs` is distributed under the same licence (see `LICENSE`).
- EPW is part of the Quantum ESPRESSO suite: P. Giannozzi et al.,
  *J. Phys.: Condens. Matter* **21**, 395502 (2009), https://www.quantum-espresso.org.

## EPW papers

Please cite these when using EPW's methods, or results reproduced here:

1. F. Giustino, M. L. Cohen, and S. G. Louie, *Phys. Rev. B* **76**, 165108 (2007).
2. S. Poncé, E. R. Margine, C. Verdi and F. Giustino, *Comput. Phys. Commun.* **209**, 116 (2016).
   (EPW v4 – v5.6)
3. H. Lee, S. Poncé, K. Bushick, S. Hajinazar, J. Lafuente-Bartolome, J. Leveillee, C. Lian,
   J.-M. Lihm, F. Macheda, H. Mori, H. Paudyal, W. H. Sio, S. Tiwari, M. Zacharias, X. Zhang,
   N. Bonini, E. Kioupakis, E. R. Margine, and F. Giustino, *npj Comput. Mater.* **9**, 156 (2023).
   (EPW v5.7 and beyond)

## This repository

Written and maintained by Hiroshi Shinaoka, largely through AI-agent development; the port was
also used to exercise [tenferro-rs](https://github.com/tensor4all/tenferro-rs).

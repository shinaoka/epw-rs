#!/bin/bash
# Regenerate the pb_epw_metal fixture from a Quantum ESPRESSO source tree.
#
# Usage: QE_SRC=/path/to/q-e QE_BIN=/path/to/q-e/build/bin ./regenerate.sh WORKDIR
#
# Runs scf -> nscf -> epw1 (wannierize + epwwrite) in WORKDIR/run, then an
# epwread-only pass in WORKDIR/read that sees nothing but the files in wannier/.
# Everything runs on 1 MPI rank so that pb.epmatwp has a single-writer layout.
set -euo pipefail

: "${QE_SRC:?set QE_SRC to the q-e source tree (tag qe-7.6)}"
: "${QE_BIN:?set QE_BIN to the directory containing pw.x and epw.x}"
WORK=${1:?usage: regenerate.sh WORKDIR}
HERE=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=1
MPI="mpirun -np 1"

mkdir -p "$WORK/run" "$WORK/read"
cp -r "$QE_SRC/test-suite/epw_metal/save" "$WORK/run/"
cp "$HERE"/qe_inputs/{scf_epw.in,nscf_epw.in,epw1.in} "$WORK/run/"
cp "$HERE/qe_inputs/epw_read.in" "$WORK/read/"
cp "$QE_SRC/pseudo/pb_s.UPF" "$WORK/run/"
sed -i "s#'../../pseudo/'#'./'#" "$WORK/run/scf_epw.in"

cd "$WORK/run"
$MPI "$QE_BIN/pw.x" -npool 1 -input scf_epw.in > scf_epw.out
$MPI "$QE_BIN/pw.x" -npool 1 -input nscf_epw.in > nscf_epw.out
$MPI "$QE_BIN/epw.x" -npool 1 -input epw1.in > epw1.out

cd "$WORK/read"
cp "$WORK"/run/{crystal.fmt,epwdata.fmt,wigner.fmt,pb.epmatwp,vmedata.fmt,dmedata.fmt,pb.ukk} .
$MPI "$QE_BIN/epw.x" -npool 1 -input epw_read.in > epw_read.out

echo "Done. Compare $WORK/read against $HERE/reference/epw_read and $HERE/wannier."

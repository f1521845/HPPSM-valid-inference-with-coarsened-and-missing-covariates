#!/bin/bash
# Reproduce all results of the paper from the command line (resumable; see README).
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/code"
for s in M1 M2 M3 M4; do python3 s2_study.py --scen $s --reps 1000; python3 s2_study.py --scen $s --reps 1000 --smc; done
for s in n500 n1000 misslow misshigh self05 self10 self20 quad; do python3 s2_study.py --scen $s --reps 200; python3 s2_study.py --scen $s --reps 200 --smc; done
for s in M1 M2 M3 M4; do python3 s2_study.py --scen $s --reps 20 --dense; done
python3 s2_study.py --scen M1 --cindex --reps 20
python3 me_sim.py 250
[ -f p12_summary.pkl ] || python3 p12_build.py "${P12_RAWDIR:-physionet-challenge-2012}" p12_summary.pkl
python3 p12_study.py --task me
python3 p12_study.py --task dense --reps 100
python3 p12_study.py --task coverage --scen A --reps 1000
python3 p12_study.py --task coverage --scen B --reps 1000
python3 p12_study.py --task vertex --reps 1000
python3 p12_study.py --task app

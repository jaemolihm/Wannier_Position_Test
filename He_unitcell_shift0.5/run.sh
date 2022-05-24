#!/bin/bash
set -ex

mpirun -np 4 pw.x -in scf.in > scf.out
mpirun -np 4 pw.x -in nscf.in > nscf.out

wannier90.x -pp He
mpirun -np 4 pw2wannier90.x -in pw2wan.in > pw2wan.out
wannier90.x He

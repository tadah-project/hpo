# Shared example data

Two bcc Ta datasets from VASP 5.4.4 (ISMEAR = 1, SIGMA = 0.20), converted
with `tadah data convert`:

| File | Structures | Source |
|---|---|---|
| `eos_bcc.tadahdb` | 138 two-atom cubic cells (ENCUT 600 eV): 37 with a = 2.60-3.68 A in 0.03 A steps, and a fine scan of 101 with a = 3.3200-3.3400 A in 0.0002 A steps | one single-point OUTCAR per lattice constant, `E_vs_V/bcc_reduced/a*/OUTCAR` and `E_vs_V/bcc_reduced/MINIMA/a*/OUTCAR` |
| `elastic.tadahdb` | 29 two-atom cells (ENCUT 630 eV): structure 1 is the reference, a0 = 3.320087 A; 2-13 uniaxial strains of +-1 % and +-2 % along x, y and z; 14-25 shear strains of +-1 % and +-2 % in the xy, yz and zx planes; 26-29 the reference cell with atom 1 moved by +-0.01 and +-0.02 A along x | one finite-difference OUTCAR (IBRION = 6), `ELASTIC/OUTCAR` |

The labels keep the source paths (`examples/data/...`).

**What they store** (since the re-conversion of 2026-10-01):
- **Energy:** the free energy F (VASP's `free  energy   TOTEN`), whose
  gradient the forces and the stress are, plus 3.69807734 eV per atom.
  That reference is minus the free energy of a spin-polarised free Ta
  atom (ENCUT 630 eV), so the energies are cohesive-energy-like.
- **Stress:** the Cauchy stress in eV/A^3, positive under tension: minus
  VASP's `in kB` tensor times 6.241509074e-4.
- **Forces and cells** as VASP printed them. The rows of the cell block
  are the lattice vectors.

**History.**
- Every earlier version of these files, the 1.6.0 development snapshots
  included, held the energy without entropy and VASP's pressure-signed
  stress.
- They were re-converted on 2026-10-01 from the same OUTCARs with the 1.6.0
  readers.
- Cells, positions and forces are unchanged. The stress is negated, and each
  energy moved by its F - E_wo (-2.9 to +1.5 meV/atom).

To re-convert, with the OUTCARs under `$SRC`:

```bash
mkdir -p examples/data/E_vs_V examples/data/ELASTIC
ln -s "$SRC/E_vs_V/bcc_reduced" examples/data/E_vs_V/bcc_reduced
ln -s "$SRC/ELASTIC/archive_original_P0/OUTCAR" examples/data/ELASTIC/OUTCAR
# the order of the shipped file: its path labels, in order
mapfile -t EOS < <(awk 'NR==1 || prev=="" {print $1} {prev=$0}' eos_bcc.tadahdb)
tadah data convert --dft-file "${EOS[@]}" -o eos_raw.tadahdb
tadah data convert --dft-file examples/data/ELASTIC/OUTCAR -o elastic_raw.tadahdb
tadah data balance -d eos_raw.tadahdb --eshift -3.69807734 -o eos_bcc.tadahdb
tadah data balance -d elastic_raw.tadahdb --eshift -3.69807734 -o elastic.tadahdb
```

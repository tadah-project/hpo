# Shared example data

Two bcc Ta datasets from VASP 5.4.4 (ENCUT 600 eV, ISMEAR = 1, SIGMA = 0.20),
converted with `tadah data convert`:

| File | Structures | Source |
|---|---|---|
| `eos_bcc.tadahdb` | 138 two-atom cubic cells, a = 2.60-3.68 A (with a fine scan 3.3202-3.3246 A) | one single-point OUTCAR per lattice constant, `E_vs_V/bcc_reduced/a*/OUTCAR` and `E_vs_V/bcc_reduced/MINIMA/a*/OUTCAR` |
| `elastic.tadahdb` | 29 two-atom cells: the reference a0 = 3.320087 A and +-1 %, +-2 % uniaxial and shear strains | one finite-difference OUTCAR (IBRION = 6), `ELASTIC/OUTCAR` |

The labels keep the source paths (`examples/data/...`).

**What they store** (since 1.6.0):
- **Energy:** the free energy F (VASP's `free  energy   TOTEN`), whose
  gradient the forces and the stress are, plus 3.69807734 eV per atom.
  That reference is minus the free energy of a spin-polarised free Ta
  atom, so the energies are cohesive-energy-like.
- **Stress:** the Cauchy stress in eV/A^3, positive under tension: minus
  VASP's `in kB` tensor times 6.241509074e-4.
- **Forces and cells** as VASP printed them. The rows of the cell block
  are the lattice vectors.

**History.**
- Before 1.6.0 these files held the energy without entropy and VASP's
  pressure-signed stress.
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

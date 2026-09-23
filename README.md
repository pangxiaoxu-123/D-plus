# D‑plus
> A maximum-likelihood phylogenomic downstream companion for D-statistic: it controls rate-variation-driven false positives and detects and characterizes ghost and non-sister introgression.

## Features
- Correct false positives caused by lineage‑specific substitution‑rate variation
- Distinguish ghost introgression, non‑sister introgression, and their combinations

## Dependencies
1. C compiler (`gcc` or `clang`)
2. [GSL (GNU Scientific Library)](https://www.gnu.org/software/gsl/) **version ≥ 2.8**
3. OpenMP (supported in most modern compilers)

> If you install GSL to a custom path, remember to supply that path during compilation.

## Compilation
One‑line compile command (replace `/path/to/gsl` with your real GSL prefix):
```bash
cc -o D-plus -O3 -std=c99 -I /path/to/gsl/include -L /path/to/gsl/lib D-plus.c -fopenmp -lm -lgsl -lgslcblas
```

## Quick start
### Input files
1. **Control file**
2. **Sequence file**: Phylip‑format multiple sequence alignments.

### Run example
```bash
# Use the example provided in the example folder
./D-plus example/example-run.ctl
```

### Output files
- `*.out`: Human‑readable main output: parsed control settings, site‑pattern statistics, maximum‑likelihood parameter estimates, log‑likelihood values.
- `*.BFGS`: Optimization trace file, records iteration history, objective function values and parameter vectors during BFGS maximum‑likelihood search.

## Control file key parameters
| Parameter | Brief description |
|---|---|
| `jobname` | Prefix for output files |
| `seqfile` | Path to phylip‑format alignment |
| `stree` | Three taxon labels: P1 P2 Outgroup |
| `nloci` | Max number of loci to use |
| `RV` | `1` to enable substitution‑rate‑heterogeneity model; `0` to disable |
| `run` | Execution mode switch |
| `model` | Model specification |
| `repeat` | Number of random initial‑point restarts for ML optimization |
| `nthreads` | OpenMP thread count |
| `npoints` | Number of Gaussian quadrature integration points |

## Citation
If you use D‑plus in your research, please cite our manuscript:
> *D-plus: Beyond the D-statistic — Accurate Detection and Characterization of Introgression Despite Among-Lineage Rate Variation*

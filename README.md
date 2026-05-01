# SymbolicRegressionPackage

Wolfram Mathematica package for brute-force symbolic regression: exhaustive search over expression trees built from user-specified constants, unary functions, and binary operations.


## Core functionality


- `RecognizeConstant[x]` — identify an analytic formula for a numeric value, searching in order of growing Kolmogorov complexity.
- `RecognizeFunction`, `RecognizeSequence` — same idea for univariate data and integer sequences.
- `RandomExpression[]`, `EnumerateExpressions[]` — generate random or exhaustive expression trees from given building blocks.
- `VerifyBaseSet[constants, functions, operations]` — bootstrapping completeness check: given a set of primitives, verify whether elementary functions can be reconstructed from given primitives.

## Quick start

Check out examples from `SymbolicRegressionPackage_Examples.nb`.


## Repository structure

- `SymbolicRegression.m` — main Mathematica package
- `EML_toolkit/` — EML compiler (Python), numerical test harnesses (C, NumPy, PyTorch, mpmath), symbolic verification notebooks, PyTorch tree trainer, figure scripts, and CUDA shortest-expression search tools
- `rust_*/` — Rust reimplementation of the bootstrapping procedure and search tools (~35 s vs ~40 min in Mathematica)


## Local modifications (this fork)

This fork extends Odrzywołek's SymbolicRegressionPackage with infrastructure
for empirical study of EML symbolic regression at scale, particularly
focused on understanding the depth-scaling cliff in fit-success rates
documented in Section 4.3 of the paper.

### `PyTorch_v17_batched/` — GPU-batched trainer (new)

A new trainer derived from `v16_final` that batches multiple (seed, strategy)
slots through a single forward pass, amortizing Python and kernel-launch
overhead. ~14× speedup over sequential CPU training on RTX 5090.

Key additions beyond v16:

- Batched-seed training with PerSeedAdam (per-seed Adam state isolation)
- Per-seed expression export for failure-mode analysis
- Trajectory history saving (`--save-history`)
- CLI hooks for ablation interventions: `--lam-active-bypass`,
  `--force-root-right-bypass-from-iter`, `--force-root-right-bypass-until-iter`

### `v16_final` CUDA support

`tree_prototype_torch_v16_final.py` extended with `--device` flag and
device-aware tensor creation. Default behavior unchanged on CPU-only
machines. Required by `v17_batched`, which imports `v16_final` utilities.

### Cross-implementation test harness (new)

`Test_torch/test_eml_torch.py`, `Test_numpy/test_eml_numpy.py`, and
`Test_mpmath/test_eml_mpmath.py` — three implementations of the same
deeply-nested EML expression (sqrt approximation) for cross-precision
sanity checking. Complemented by upstream's existing `test_eml.c`.

### Compiler portability

`EmL_compiler/Test_C_math_h/make_eml_c.py` and `make_eml_binary_c.py`
extended with clang/gcc/cc fallback chain for Windows machines without
Intel `icx` installed.

## Relationship to upstream

This fork is intended to support empirical research on EML's symbolic
regression properties at depths 2–6. Findings related to failure-mode
analysis, optimization landscape characterization, and schedule
interventions are documented in research artifacts external to this
repo (currently private; may be published separately).

Upstream commits from `VA00/SymbolicRegressionPackage` can be pulled
via the `upstream` remote: `git pull upstream master`.

## Requirements

- Wolfram Mathematica (≥13.0)
- Python ≥3.9 with NumPy, PyTorch, mpmath (for EML toolkit)
- Rust ≥1.70 (for rust tools)
- C compiler (for Eml_verify)
- CUDA toolkit (for EmL_recognizer)


## License

MIT

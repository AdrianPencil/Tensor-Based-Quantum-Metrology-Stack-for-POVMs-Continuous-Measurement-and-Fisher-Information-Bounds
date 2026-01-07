# qmet

A modular quantum metrology stack:

**sensor → measurement → estimation → metrics → workflows**

This repository is intentionally small and test-backed. It gives you a clean baseline for coding and comparing canonical quantum sensing experiments (Ramsey, weak continuous measurement, Fisher/CRB limits, stability metrics) without forcing you to read the whole codebase to understand what’s going on.

---

## What it does

`qmet` provides:

- **Protocol models** (closed-form, estimation-ready)
  - Ramsey signal model + derivative w.r.t. parameter (for Fisher/CRB)
  - Echo envelope model
  - Interferometric fringe model
  - A metrology-facing squeezing container (Wineland parameter)

- **Measurement models**
  - Discrete **POVM** evaluation and sampling
  - **Continuous diffusive measurement** record model
  - **Backaction** update step (stochastic master equation step) for a Hermitian observable
  - Additive **Gaussian readout noise** likelihood helper

- **Estimation stack**
  - Gaussian IID log-likelihood, score, Fisher
  - CRB helpers (scalar + matrix, optional ridge)
  - 1D grid Bayesian posterior update
  - Minimal scalar Kalman filter (random-walk tracking)

- **Noise & stability**
  - Welch PSD estimate
  - Overlapping Allan deviation
  - Random walk / linear drift generators

- **Workflows / reports / viz**
  - One canonical Ramsey frequency pipeline (bounds + optional simulation)
  - Markdown and JSON report writers
  - Simple plotting helper (lazy `matplotlib` import)

- **Flagship experiments**
  - Ramsey under dephasing (end-to-end)
  - Weak continuous measurement of a qubit (trajectory simulation)
  - Classical Fisher vs quantum benchmark (simple QFI reference)

- **Case studies**
  - NV-inspired Ramsey sensitivity study (design insight: which time points contribute most)
  - Readout chain example (sensor → device response → noise → likelihood)

---

## Motivation

Quantum metrology discussions often jump between:
- abstract quantum limits (QFI, Heisenberg scaling),
- practical device readout models (gain, offset, noise),
- and estimation theory (Fisher/CRB, filtering, Bayesian updates).

In practice, you want one thing:

> A consistent, minimal pipeline where *the same parameter* flows from model → measurement → likelihood → Fisher/CRB → metrics, with tests that guard basic invariants.

This project is built around that “observable map” philosophy.

---

## How it’s organized


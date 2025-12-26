
# Momenta.jl

# Momenta.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dazhwu.github.io/Momenta.jl/dev/)
[![Build Status](https://github.com/dazhwu/Momenta.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dazhwu/Momenta.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/dazhwu/Momenta.jl/blob/main/LICENSE)

**Momenta** is a next-generation, high-performance library for **Panel Vector Autoregression (PVAR)** analysis.

It provides a rigorous implementation of the **Generalized Method of Moments (GMM)** estimators for dynamic panel data, strictly adhering to the econometric frameworks established by Holtz-Eakin et al. (1988), Arellano & Bover (1995), and Blundell & Bond (1998).

Functionally equivalent to (and faster than) the R package `panelvar` (Sigmund & Ferstl, 2019) and Stata's `pvar` (Abrigo & Love, 2016), **Momenta** leverages Julia's Just-In-Time (JIT) compilation and multi-threading to handle large-scale datasets and intensive bootstrapping procedures with unmatched efficiency.

> **For Python Users:** Access the full power of Momenta via our seamless wrapper: `pip install pymomenta`.

## Why Momenta?

While existing tools like R's `panelvar` or Stata's `xtabond2` are comprehensive, they often suffer from performance bottlenecks in large  panels or during bootstrap inference. Momenta is built to supersede them:

| Feature | Momenta.jl | R (panelvar) | Stata (pvar) |
| --- | --- | --- | --- |
| **Core Engine** | **Julia (Compiled)** | R (Interpreted) | Stata (ADO) |
| **Parallelism** | **Native Multi-threading** | Limited | Limited |
| **Memory Efficiency** | **Zero-Copy Views** | High Overhead | Moderate |
| **GMM Estimators** | Diff & System GMM | Diff & System GMM | Diff GMM |
| **Transformation** | FD & FOD | FD & FOD | FD & FOD |
| **Python Support** | **Native Wrapper** | No | No |

## Key Features

### 1. Advanced GMM Estimation

* **System GMM**: Full implementation of the Blundell-Bond estimator, utilizing both level and difference equations for higher efficiency in persistent series.
* **Difference GMM**: Standard Arellano-Bond estimator.
* **Windmeijer Correction**: Finite-sample correction for standard errors (Windmeijer, 2005).

### 2. Rigorous Data Transformation

* **Forward Orthogonal Deviations (FOD)**:
* Implements the Arellano & Bover (1995) transformation, preserving sample size in unbalanced panels better than First Differences (FD).
* **Strict Listwise Deletion**: Unlike some implementations that introduce bias by retaining partial observations, Momenta enforces strict validity checks: if  is missing, corresponding instruments and regressors are correctly invalidated to preserve moment conditions.



### 3. Structural Analysis (PVAR)

* **Impulse Response Functions (IRF)**: Orthogonalized IRFs based on Cholesky decomposition.
* **Generalized IRF (GIRF)**: Order-invariant impulse responses (Pesaran & Shin, 1998).
* **Forecast Error Variance Decomposition (FEVD)**: Quantify the contribution of shocks to variable variability.
* **Fast Bootstrapping**: Multi-threaded bootstrapping for accurate confidence intervals in seconds, not minutes.

### 4. Specification Testing

* **Hansen J-Test**: For overidentifying restrictions.
* **Lag Selection**: Andrews-Lu MMSC criteria (MBIC, MAIC, MHQIC).
* **Stability Tests**: Eigenvalue checks for the companion matrix.

## Installation

### Julia

```julia
using Pkg
Pkg.add("Momenta")

```

### Python

```bash
pip install pymomenta

```

## Quick Start

### Julia Workflow

```julia
using Momenta, CSV, DataFrames

# 1. Load Data
df = CSV.read("dahlberg_data.csv", DataFrame)

# 2. Configure Model (Replicating Sigmund & Ferstl, 2019)
m = Momenta.fit(df, 
        ["id", "year"],  
        "n w  ~ lag(n, 1:2) lag(w, 1:2) k", 
        "GMM(n w, 2:4) IV(k)", 
        "fod" 
)


# 3. Structural Analysis
irf = Momenta.irf(m, 8)
bootstrap_result=Momenta.bootstrap(m, 8, 200)
all_plots=Momenta.plot_irf(m, bootstrap_result)

```

### Python Workflow

```python
import pandas as pd
from pymomenta import fit

df = pd.read_csv("dahlberg_data.csv")

# Estimate using the high-performance Julia backend
m = fit(df, 
        ["id", "year"],  
        "n w  ~ lag(n, 1:2) lag(w, 1:2) k", 
        "GMM(n w ,2:4) IV(k)", 
        "fod" 
)


```

## Methodology & References

Momenta is designed to strictly follow the theoretical foundations laid out in:

* **Sigmund, M., & Ferstl, R. (2019).** Panel Vector Autoregression in R with the Package panelvar. *The Quarterly Review of Economics and Finance*. (The methodology implemented here mirrors and extends this work).
* **Arellano, M., & Bover, O. (1995).** Another look at the instrumental variable estimation of error-components models. *Journal of Econometrics*.
* **Blundell, R., & Bond, S. (1998).** Initial conditions and moment restrictions in dynamic panel data models. *Journal of Econometrics*.
* **Roodman, D. (2009).** How to do xtabond2: An introduction to difference and system GMM in Stata. *The Stata Journal*.

## License

This project is licensed under the MIT License.


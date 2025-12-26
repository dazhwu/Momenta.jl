
# Momenta.jl User Guide

**Momenta.jl** is an advanced dynamic panel data estimation package for Julia. It supports single-equation dynamic panel models and Panel Vector Autoregression (PVAR) models. The package provides flexible syntax for defining GMM and IV instruments, supporting Difference GMM, System GMM, and Forward Orthogonal Deviations (FOD) transformation.

## Installation

```julia
using Pkg
Pkg.add("Momenta")
using Momenta

```

## Core Function: `fit`

The main entry point for model estimation is the `fit` function:

```julia
m = fit(df, panel_ids, model_str, instr_str, options_str)

```

### Parameters

* **`df`**: `DataFrame`. The dataset containing your panel data.
* **`panel_ids`**: `Vector{String}` containing exactly two elements: `[Individual ID, Time Variable]`. Example: `["id", "year"]`.
* **`model_str`**: `String`. Defines the regression model (equations).
* **`instr_str`**: `String`. Defines the instruments (GMM and IV).
* **`options_str`**: `String`. Controls estimation methods (e.g., FOD, One-step, etc.).

---

## 1. Model Syntax (`model_str`)

Models are defined using a tilde `~` to separate dependent variables (LHS) and independent variables (RHS). The `lag()` function is supported for specifying lags.

### Single Equation Model

Suitable for standard dynamic panel estimation (e.g., Arellano-Bond).

```julia
# Example: n depends on the 1st lag of n and contemporaneous k
"n ~ lag(n, 1) k"

# Example: n depends on the 1st through 2nd lags of n
"n ~ lag(n, 1:2) k"

```

### Panel Vector Autoregression (PVAR)

When **multiple variables** appear on the left side of the tilde, the package automatically identifies it as a PVAR model. The system estimates these equations simultaneously.

**Note**: PVAR models require symmetry, meaning all dependent variables must have the same number of lags.

```julia
# Bivariate PVAR(1): n and w cause each other, both with 1 lag
"n w ~ lag(n, 1) lag(w, 1)"

# With an exogenous variable k
"n w ~ lag(n, 1) lag(w, 1) k"

```

---

## 2. Instrument Syntax (`instr_str`)

The instrument string supports two definition blocks: `GMM(...)` and `IV(...)`. You can combine them as needed.

### GMM Instruments

Used to define lagged endogenous variables as instruments (Internal Instruments).

* **Syntax**: `GMM(variable_names, lag_range)`
* **Special Symbol**: `.` represents the maximum available time length ().

```julia
# Use lags 2 to 4 of n as instruments
"GMM(n, 2:4)"

# Use lags 2 to max available (Standard Arellano-Bond style)
"GMM(n, 2:.)"

# Define multiple variables simultaneously
"GMM(n w, 2:4)"

```

### IV (Standard) Instruments

Used to define strictly exogenous variables (External Instruments).

```julia
# Treat k as a contemporaneous exogenous instrument
"IV(k)"

# Treat the 1st lag of k as the instrument
"IV(lag(k, 1))"

# Combined syntax
"IV(k lag(w, 1))"

```

---

## 3. Options Configuration (`options_str`)

`options_str` is a space-separated string used to enable different estimation features.

| Option Keyword | Meaning | Description |
| --- | --- | --- |
| **(Empty)** | **System GMM** | Default behavior. Includes both Level and Difference equations, estimated using the two-step method. |
| `nolevel` | **Difference GMM** | Uses Difference equations only (similar to basic Arellano-Bond). |
| `fod` | **FOD Transformation** | Uses Forward Orthogonal Deviations instead of First Differencing. Suitable for unbalanced panels or to preserve observations (Arellano-Bover). |
| `collapse` | **Collapse Instruments** | Limits the instrument count to one per variable per lag distance (instead of per time period) to prevent overfitting ("instrument proliferation"). |
| `onestep` | **One-step Estimation** | Performs only one-step GMM estimation (Default is two-step. Two-step is generally more efficient but requires finite-sample correction for standard errors). |

---

## 4. Common Model Examples

The following examples demonstrate how to combine parameters to implement classic econometric models.

### Scenario A: Standard System GMM (Blundell-Bond)

* **Model**: AR(1)
* **Instruments**: Lags of `n` for the difference equation, differences of `n` for the level equation (handled automatically).
* **Options**: Empty (defaults to Level/System GMM).

```julia
fit(df, ["id", "year"], 
    "n ~ lag(n, 1) k",      # Model
    "GMM(n, 2:.) IV(k)",    # GMM lags 2 to max
    ""                      # Default: Two-step System GMM
)

```

### Scenario B: Difference GMM (Arellano-Bond)

* **Options**: Add `"nolevel"`. This explicitly turns off the Level equations, using only Difference equations.

```julia
fit(df, ["id", "year"], 
    "n ~ lag(n, 1) k", 
    "GMM(n, 2:4) IV(k)", 
    "nolevel"               # Disable Level equations
)

```

### Scenario C: Using FOD Transformation (Arellano-Bover)

* **Options**: Add `"fod"`. This is often superior to first differencing in unbalanced panels. It can be combined with `"nolevel"` or used alone (FOD-System GMM).

```julia
fit(df, ["id", "year"], 
    "n ~ lag(n, 1) k", 
    "GMM(n, 2:4) IV(k)", 
    "fod"                   # Use Forward Orthogonal Deviations
)

```

### Scenario D: Panel Vector Autoregression (PVAR)

* **Model**: Specify multiple variables on the LHS.
* **Instruments**: Typically apply GMM instruments to all endogenous variables.

```julia
fit(df, ["id", "year"], 
    "n w ~ lag(n, 1) lag(w, 1)",   # PVAR(1)
    "GMM(n w, 2:4)",               # GMM instruments for both n and w
    "fod"                          # FOD is recommended for PVAR
)

```

### Scenario E: Collapsing Instruments

* When  is large, the instrument matrix can become very wide. Use `collapse` to reduce the number of instruments.

```julia
fit(df, ["id", "year"], 
    "n ~ lag(n, 1) k", 
    "GMM(n, 2:.) IV(k)", 
    "collapse"
)

```

---

## 5. Post-Estimation

### Impulse Response Functions (IRF) & Bootstrap

For PVAR models, you can use Bootstrap to calculate confidence intervals for Impulse Response Functions.

```julia
# 1. Fit the model
m = fit(df, ["id", "year"], "n w ~ lag(n, 1) lag(w, 1)", "GMM(n w, 2:4)", "fod")

# 2. Run Bootstrap
# Parameters: model, steps ahead, number of draws
res = bootstrap(m, 8, 200)
res = bootstrap(m, 8, 200, "girf") # "girf" is the default method
res = bootstrap(m, 8, 200, "oirf")

# 3. Plotting (Requires `using Plots`)
all_plots=Momenta.plot_irf(m, bootstrap_result)
display(all_plots["n on w"]) # show the impact of n on w
display(all_plots["full"])  # show a plot with all subplots


```

### Regression Summary

```julia
# Print detailed summary
print_summary(m)

# Export results to HTML or LaTeX
export_html(m, "results.html")
export_latex(m, "results.tex")

```
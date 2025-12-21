# module data_struct
# export List_Variables, model_options, regular_variable, gmm_var, model, append_variable, purge_list_varaibles, check_contiguous

const IRF_girf = 1
const IRF_oirf = 2

const TRANSFORMATION_FD = 1
const TRANSFORMATION_FOD = 2

const MMSC_LU_BIC = 1
const MMSC_LU_HQIC = 2
const MMSC_LU_AIC =
    const largest_T::Int64 = typemax(Int16) #typemax(Int64)
const ALL_LAGS::Int64 = typemax(Int16)

abstract type Instrument end      # Parent type for all linear instruments
abstract type AbstractNLConstraint end    # Parent type for all nonlinear constraints

# 2. Struct definitions
struct GMM <: Instrument
    vars::Vector{Symbol}
    lags::UnitRange{Int64}  # <--- Explicit declaration as Int64, consistent with the rest of your program
    collapse::Bool
end

# 3. Constructor
# Here the lag default is also of type Int64
GMM(vars...; lag=2:ALL_LAGS, collapse=false) = GMM([vars...], lag, collapse)

# === 2.2 Define IV instrument ===
struct IV <: Instrument
    vars::Vector{Symbol}
    lag::Int64  # Usually 0 (current period) or 1 (1 lag)
end

IV(vars...; lag=0) = IV([vars...], lag)

# # For use by the computation engine (Internal)
# struct IV_Definition
#     var_index::Int64   # Column index of variable in data matrix (critical optimization)
#     lag::Int64         # Lag order (usually 0 or 1)
# end
# Corresponds to nl(noserial)
struct AhnSchmidt <: AbstractNLConstraint
    lags::Int  # Allows user to fine-tune the number of lags
end
# Default constructor, default lags=1
AhnSchmidt(; lags::Int=1) = AhnSchmidt(lags)

# === 3.2 Homoskedasticity constraint ===
# Corresponds to nl(iid)
struct Homoskedasticity <: AbstractNLConstraint
    # Might not need parameters here, just an on/off switch
end

struct Options
    level::Bool         # Whether to use System GMM (enabled by default)
    steps::Int64        # Whether to use two-step method (enabled by default)
    robust::Bool        # Whether to use Windmeijer correction (enabled by default)
    timedumm::Bool
    # === Data transformation ===
    # User-friendly symbol: :fd (difference) or :fod (forward orthogonal deviation)
    transformation::String
    # === Other settings ===
    collapse::Bool        # Global collapse (fallback for Instrument collapse)
    #constant::Bool         # Whether to include a constant

    # === Advanced settings ===
    # Additional parameters can be added here, e.g. max_iter, etc.
    # compact::Bool = false

end

# function Options(;
#     level=true,
#     steps=2,
#     robust=true,
#     timedummy=false,
#     transformation="fd",
#     collapse=false,
#     constant=true
# )
#     if constant && !level
#         # Including constant in Diff GMM is usually lost due to differencing, or should be treated as instrument
#         # Here you can keep your error logic, or change to warning as needed
#         throw(ArgumentError("Conflict: Constant cannot be used with nolevel (Diff GMM only)."))
#     end

#     if 

#     if steps < 1
#         throw(ArgumentError("Steps must be ast least 1"))
#     end

#     return Options(level, steps, robust, timedummy, transformation, collapse, constant)
# end

mutable struct InternalOptions
    steps::Int64
    level::Bool
    timedumm::Bool
    collapse::Bool
    #mmsc::Int64  # MMSC_LU_BIC = BIC, MMSC_LU_HQIC = HQIC, MMSC_LU_AIC = AIC
    transformation::Int64
end

function InternalOptions(;
    steps=2, level=true, timedumm=false,
    collapse=false, transformation=TRANSFORMATION_FD
    )::InternalOptions

    return InternalOptions(steps, level, timedumm, collapse, transformation)
end

struct regular_var
    name::String
    p_index::Int64   # Index pointing to names/positions (logical layer index)
    lag::Int64
end

struct gmm_var
    name::String
    p_index::Int64   # Index pointing to names/positions
    min_lag::Int64
    max_lag::Int64
    collapse::Bool
end

mutable struct MMSC_LU
    BIC::Float64
    HQIC::Float64
    AIC::Float64
end

mutable struct Hansen_test_info
    test_value::Float64
    df::Int64
    P_value::Float64
    critical_value::Float64
end

mutable struct bootstrap_results
    ahead::Int64
    draws::Int64
    lower::Matrix{Float64}
    upper::Matrix{Float64}
    irf::Matrix{Float64}

end

mutable struct step_result
    residual::Vector{Matrix{Float64}}
    #_residual_t::Matrix{Float64}
    XZ_W::Matrix{Float64}
    W::Matrix{Float64}
    W_inv::Matrix{Float64}
    W_next::Matrix{Float64}
    _M_XZ_W::Matrix{Float64}
    zs::Matrix{Float64}
    qs::Matrix{Float64}
    #ZuuZ::Matrix{Float64}
    uz_list::Vector{Matrix{Float64}}
    M::Matrix{Float64}
    #_zs_list::Matrix{Float64}
    beta::Matrix{Float64}
    vcov::Matrix{Float64}
    std_err::Matrix{Float64}

    function step_result(residual::Vector{Matrix{Float64}}, XZ_W::Matrix{Float64}, W::Matrix{Float64}, W_inv::Matrix{Float64}, W_next::Matrix{Float64}, _M_XZ_W::Matrix{Float64}, zs::Matrix{Float64}, qs::Matrix{Float64}, uz_list::Vector{Matrix{Float64}}, M::Matrix{Float64}, beta::Matrix{Float64})
        tbr = new()
        tbr.residual = residual
        #tbr._residual_t = _residual_t
        tbr.XZ_W = XZ_W
        tbr.W = W
        tbr.W_inv = W_inv
        tbr.W_next = W_next
        tbr._M_XZ_W = _M_XZ_W
        tbr.zs = zs
        tbr.qs = qs
        tbr.uz_list = uz_list
        #tbr.ZuuZ = ZuuZ
        tbr.M = M
        #tbr._zs_list = _zs_list
        tbr.beta = beta

        return tbr
    end

    function step_result(residual::Vector{Matrix{Float64}}, beta::Matrix{Float64})
        tbr = new()
        tbr.residual = residual
        tbr.beta = beta

        return tbr
    end

end

mutable struct basic_info
    N::Int64 
    T::Int64 

    first_diff_index::Int64 
    last_diff_index::Int64 
    first_level_index::Int64 
    last_level_index::Int64 
    max_lag::Int64 

    num_dep::Int64 
    num_dep_lags::Int64 
    num_indep::Int64 
    
    num_xy_width::Int64 

    num_obs::Int64 
    max_obs::Int64 
    min_obs::Int64 
    avg_obs::Float64 

    diff_width::Int64 
    diff_height::Int64 
    level_width::Int64 
    level_height::Int64 
    z_width::Int64   # number of instruments
    z_height::Int64 

    irf_ahead::Int64 

    basic_info() = new()

end

mutable struct regression_result
    beta::Matrix{Float64}
    std_err::Matrix{Float64}
    z_stats::Matrix{Float64}
    p_values::Matrix{Float64}

    function regression_result(beta::Matrix{Float64}, std_err::Matrix{Float64})
        tbr = new()
        tbr.beta = beta
        tbr.std_err = std_err

        tbr.z_stats = beta ./ std_err
        tbr.p_values = 2 * (1 .- cdf(Normal(), abs.(tbr.z_stats)))

        return tbr
    end
end

mutable struct model

    info::basic_info
    options::InternalOptions

    deps::Vector{regular_var}
    indeps::Vector{regular_var}
    Dgmm_vars::Vector{gmm_var}
    Lgmm_vars::Vector{gmm_var}
    IV_vars::Vector{regular_var}
    command_str::String
    hansen::Hansen_test_info
    results::Vector{step_result}
    mmsu_lu::MMSC_LU




    #irf::Matrix{Float64}na_records
    #upper::Matrix{Float64}
    #lower::Matrix{Float64}
    stability::Vector{Float64}

    regression_result::regression_result
    residuals::Vector{Matrix{Float64}}

    cache_Cx::Vector{Matrix{Float64}}
    cache_Cy::Vector{Matrix{Float64}}
    cache_z_list::Vector{Matrix{Float64}}
    cache_xz_list::Vector{Matrix{Float64}}
    cache_zy_list::Vector{Matrix{Float64}}
    cache_zHz_list::Vector{Matrix{Float64}}
    cache_H1::Matrix{Float64}
    cache_na_records::Vector{Int64}

    function model(deps, indeps, Dgmm_vars, Lgmm_vars, IV_vars, command_str)
        tbr = new()
        tbr.deps = deps
        tbr.indeps = indeps
        tbr.Dgmm_vars = Dgmm_vars
        tbr.Lgmm_vars = Lgmm_vars
        tbr.IV_vars = IV_vars
        tbr.command_str = command_str
        return tbr
    end
end


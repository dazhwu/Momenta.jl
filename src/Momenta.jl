__precompile__(true)
module Momenta

using DataFrames
using LinearAlgebra
using Distributions
using Printf
using Random
using PrettyTables
using Base.Threads
using PrecompileTools # This is the magic tool for "precompilation"

export fit, regress, plot_irf, basic_info, model, GMM, IV, ALL_LAGS, irf, 
        bootstrap, bootstrap_results, print_summary, export_html, export_latex

include("data_struct.jl")
include("common_functions.jl")
include("command.jl")
include("panel_data.jl")
include("Instruments.jl")
include("GMM.jl")
include("test.jl")
include("irf.jl")
include("model_summary.jl")


function generate_model_info(info::basic_info, final_xy_tables::NamedTuple, z_list::Vector{Matrix{Float64}},
    na_records::Vector{Int64}, m::model, options::InternalOptions)

    # 1. Call prepare_reg (note: using .Cx and .Cy here)
    info.num_obs, info.max_obs, info.min_obs, info.avg_obs = prepare_reg(
        info,
        z_list,
        final_xy_tables.Cx,
        final_xy_tables.Cy,
        na_records,
        options.transformation,
        options.level
    )

    # 2. Get dimension information
    info.num_indep, info.num_xy_width = size(final_xy_tables.Cx[1])

end

function fit(
    df::DataFrame,
    panel_ids::Vector{String},
    model_str::String,    
    instr_str::String, opts::String="", silent::Bool=false
)

    id_time = string.(panel_ids)
    # 1. Extract ID and Time
    # === 1. Parameter validation (this must be supplemented) ===
    if length(id_time) != 2
        throw(ArgumentError("id_time parameter must contain exactly 2 elements: [panel ID, time variable]"))
    end

    # === 2. Extract variables ===
    #id_col = id_time[1]
    time_col = id_time[2]

    T = length(unique(df[!, time_col]))

    col_names = names(df)
    var_names = Vector{String}()
    positions = Vector{Int64}()


    info = basic_info()
    info.id_time=id_time

    m, the_options, info.num_dep_lags = process_command(
        model_str, instr_str, opts, col_names, var_names, positions, T
    )

    m.info=info

    p_data, info.N, info.T = export_data(df, m, the_options, positions)

    fd_data = get_first_difference_table(p_data, info.N)

    get_df_info(info, the_options, m)

    #first_difference_part=the_options.transformation==TRANSFORMATION_FOD ? get_fod_table(p_data, info.N) : fd_data

    final_xy_tables = handle_tables(p_data, fd_data, m,  info, the_options)

    z_list = get_z_table(info, p_data, fd_data, m, the_options)   


    na_records = Vector{Int64}(fill(0, info.N * size(final_xy_tables.Cy[1], 2)))

    generate_model_info(info, final_xy_tables, z_list, na_records, m, the_options)

    results, H1, xz_list, zy_list, zHz_list = regular_process(z_list, final_xy_tables.Cx, final_xy_tables.Cy, info, the_options)
    
    
    last_step = length(results)

    m.regression_result = regression_result(results[last_step].beta, results[last_step].std_err)

    m.hansen = hansen_overid(results[last_step].W_inv, results[last_step].zs, info.z_width * info.num_dep, info.num_indep * info.num_dep, info.N)

    process_residual(results[the_options.steps], na_records, info)

    m.residuals=results[the_options.steps].residual
    m.options=the_options

    m.stability = stability_test(results[the_options.steps].beta, info.num_dep_lags)
    m.cache_Cx = final_xy_tables.Cx
    m.cache_Cy = final_xy_tables.Cy
    m.cache_z_list = z_list
    m.cache_xz_list = xz_list
    m.cache_zy_list = zy_list
    m.cache_zHz_list = zHz_list
    m.cache_H1 = H1
    m.cache_na_records = na_records

    #println(m.info.num_instr)
    if !silent 
    #    print_summary(m, id_time)
        #display_summary_text(m, id_time)
    end

    return m
end


function plot_irf(args...)
    error("Please include `using Plots` before you call this function！")
end


function process_residual(result, na_records::Vector{Int64}, info::basic_info)
    # Simple parallelism, accelerate processing under large samples
    xy_width = info.num_xy_width

    Threads.@threads for n in 1:info.N
        # Precompute offset
        offset = (n - 1) * xy_width

        # Get the residual matrix reference of this individual
        res_n = result.residual[n]

        for j in 1:xy_width
            # Lookup table na_records
            if na_records[offset+j] == 1
                # Set this column to NaN
                res_n[:, j] .= NaN
            end
        end
    end
end

function bootstrap(m::model, ahead::Int64, num_draws::Int64, method::String="", seed=123)

    internal_method=irf_method(method)

    the_irf=irf(internal_method, m.residuals, m.cache_na_records, m.regression_result.beta, ahead, m.info.num_dep_lags, m.info.N)

    L, U = Bootstrapping(num_draws, ahead, 
    m.cache_z_list, m.cache_Cx, m.cache_Cy, m.cache_xz_list, m.cache_zy_list, m.cache_zHz_list, 
    m.cache_H1, m.cache_na_records, m.info, m.options, internal_method, seed)

    return bootstrap_results(ahead, num_draws, L, U, the_irf)


end
function Bootstrapping(num_draws::Int64, ahead::Int64, z_table::Vector{Matrix{Float64}}, Cx::Vector{Matrix{Float64}}, Cy::Vector{Matrix{Float64}},
    xz_list::Vector{Matrix{Float64}}, zy_list::Vector{Matrix{Float64}}, zHz_list::Vector{Matrix{Float64}}, 
    H1::Matrix{Float64}, na_records::Vector{Int64}, model_info::basic_info, the_options::InternalOptions, internal_method::Int64, seed=123)

    N = model_info.N
    Random.seed!(seed)

    # Generate random index matrix (N x Draws)
    ids = rand(1:N, N, num_draws)

    All_Mats = Vector{Matrix{Float64}}(undef, num_draws)

    xy_width = model_info.num_xy_width
    x_height = model_info.num_indep
    y_height = model_info.num_dep
    z_width = model_info.z_width
    z_height = model_info.z_height

    # --- 1. Prepare thread-local caches (key modification) ---
    # 🔴 Must use maxthreadid() instead of nthreads() 🔴
    # Thus, even if threadid jumps to 5 or 8, the array is large enough
    max_tid = Threads.maxthreadid()

    thread_xz = [zeros(x_height, z_width) for _ in 1:max_tid]
    thread_zy = [zeros(y_height, z_width) for _ in 1:max_tid]
    thread_W1 = [zeros(z_width, z_width) for _ in 1:max_tid]

    thread_pseudo_Cx = [Vector{Matrix{Float64}}(undef, N) for _ in 1:max_tid]
    thread_pseudo_Cy = [Vector{Matrix{Float64}}(undef, N) for _ in 1:max_tid]
    thread_pseudo_z = [Vector{Matrix{Float64}}(undef, N) for _ in 1:max_tid]

    thread_pseudo_na = [zeros(Int, N * xy_width) for _ in 1:max_tid]

    # --- 2. Parallel sampling loop ---
    # Removed @showprogress, since it may cause deadlock or display chaos in multithreading
    # If you really need a progress bar, it is recommended to run single-threaded, or only print log on the main thread
    Threads.@threads for i in 1:num_draws
        tid = Threads.threadid()

        # Get current thread's cache
        temp_xz = thread_xz[tid]
        fill!(temp_xz, 0.0)
        temp_zy = thread_zy[tid]
        fill!(temp_zy, 0.0)
        temp_W1 = thread_W1[tid]
        fill!(temp_W1, 0.0)

        pseudo_Cx = thread_pseudo_Cx[tid]
        pseudo_Cy = thread_pseudo_Cy[tid]
        pseudo_z = thread_pseudo_z[tid]
        pseudo_na = thread_pseudo_na[tid]

        for j in 1:N
            the_id = ids[j, i]

            pseudo_Cx[j] = Cx[the_id]
            pseudo_Cy[j] = Cy[the_id]
            pseudo_z[j] = z_table[the_id]

            # Manually copy NA records (faster than view)
            src_offset = (the_id - 1) * xy_width
            dst_offset = (j - 1) * xy_width
            for k in 1:xy_width
                pseudo_na[dst_offset+k] = na_records[src_offset+k]
            end

            # Accumulate in-place (avoid memory allocation)
            temp_xz .+= xz_list[the_id]
            temp_zy .+= zy_list[the_id]
            temp_W1 .+= zHz_list[the_id]
        end

        # Build matrices (use optimized kron_with_identity)
        _XZ_t = kron_with_identity(temp_xz, y_height)
        W1 = kron_with_identity(temp_W1, y_height)
        _ZY_t = reshape(temp_zy, :, 1)

        # Solve GMM
        beta, residual = special_process(pseudo_z, pseudo_Cx, pseudo_Cy, _XZ_t, _ZY_t, W1, H1, model_info, the_options)

        # Calculate IRF
        All_Mats[i] = irf(internal_method, residual, pseudo_na, beta, ahead, model_info.num_dep_lags, N)
    end

    

    return choose_U_L(All_Mats, num_draws, model_info.num_dep, ahead)
end

@inbounds function choose_U_L(All_Mats::Vector{Matrix{Float64}}, num_draws::Int64, num_dep::Int64, ahead::Int)
    # Determine quantile indexes
    L::Int = max(round(Int, num_draws * 0.025), 1) # Fix index calculation logic
    U::Int = min(round(Int, num_draws * 0.975), num_draws)

    upper = zeros(ahead, num_dep * num_dep)
    lower = zeros(ahead, num_dep * num_dep)

    # This is a good opportunity for parallelism, since each (i, j) is independent
    
    total_elements = num_dep * num_dep

    Threads.@threads for j in 1:total_elements
        # Thread-local cache to avoid allocations in inner loop
        temp_vec = Vector{Float64}(undef, num_draws)

        for i in 1:ahead
            # Collect all sampled results
            for m in 1:num_draws
                temp_vec[m] = All_Mats[m][i, j]
            end

            # Sort
            # For small samples (N<1000), full sort! is very fast
            sort!(temp_vec)

            lower[i, j] = temp_vec[L]
            upper[i, j] = temp_vec[U]
        end
    end

    return lower, upper
end


function calculate_MMSC_LU(hanse::Hansen_test_info, model_info::basic_info)
    log_n::Float64 = log(model_info.num_obs)
    dif::Float64 = model_info.z_width * model_info.num_dep - model_info.num_indep
    BIC = hansen.test_value - dif * log_n
    HQIC = hansen.test_value - dif * log(log_n) * 2.1
    AIC = hansen.test_value - dif * 2.0

    return MMSC_LU(BIC, HQIC, AIC)
end
@setup_workload begin
    # 1. Set sufficiently large dimension
    # GMM(n, 2:4) requires up to 4 lags, and differencing, so if T is too small error occurs
    # Recommend N=50, T=10, which is enough to support dozens of instruments, and should not error
    n_id = 50   
    n_time = 10 
    total_obs = n_id * n_time
    
    # 2. Construct simple AR(1) data, ensure matrix invertibility
    # If fully random, sometimes the instrument matrix is rank-deficient
    Random.seed!(1234) # Fixed seed, ensure consistent compilation

    # Initialize vectors
    id_vec = repeat(1:n_id, inner=n_time)
    year_vec = repeat(2000:(2000+n_time-1), outer=n_id)
    n_val = zeros(Float64, total_obs)
    w_val = zeros(Float64, total_obs)
    k_val = randn(total_obs) # Exogenous variable can be fully random

    # Generate AR(1) data: y_t = 0.5 * y_{t-1} + e_t
    # Thus lag(n) and n are correlated, so instrument is valid and won't lead to singular matrix
    for i in 1:total_obs
        # If not the first time point of the entity
        if (i > 1) && (id_vec[i] == id_vec[i-1])
            n_val[i] = 0.5 * n_val[i-1] + randn()
            w_val[i] = 0.5 * w_val[i-1] + randn()
        else
            n_val[i] = randn()
            w_val[i] = randn()
        end
    end

    df_pre = DataFrame(
        :id => id_vec,
        :year => year_vec,
        :n => n_val,
        :w => w_val,
        :k => k_val
    )

    @compile_workload begin
        # The try-catch is still retained here, but let's print out to see if there is still an error
        try
            # === 1. Univariate FD (most common) ===
            m=fit(df_pre, ["id", "year"],  "n ~ lag(n, 1:2) k", "GMM(n, 2:4) IV(k)", "", true)
            
            # === 2. Univariate FOD (your new feature) ===
            m=fit(df_pre, ["id", "year"], "n ~ lag(n, 1:2) k",  "GMM(n, 2:4) IV(k)", "fod", true)

            # === 3. Multivariate System GMM (more complex case) ===
            m=fit(df_pre, ["id", "year"],  "n w ~ lag(n, 1:2) lag(w, 1:2)",  "GMM(n w, 2:4) IV(k)", "fod", true)
            bootstrap_result=bootstrap(m, 8, 200)
            
            # If this line prints, precompilation finished successfully!
            # println("Precompilation finished successfully!") 
        catch e
            
            # For example "DimensionMismatch" or "SingularException"
            println("Precompile FAILED with error: ", e)
            
            # If SingularException, maybe data is still problematic, but at least print it out
            for (exc, bt) in Base.catch_stack()
               showerror(stdout, exc, bt)
               println()
            end
        end
    end
end
end # module PanelVar

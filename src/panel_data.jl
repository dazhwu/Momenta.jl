@inbounds function xt_set(df::DataFrame, identifiers::Vector{String})::Tuple{Int64,Int64,Vector{Int64}}

    col_individual = identifiers[1]
    col_time = identifiers[2]

    # 1. Extract the individual column, de-duplicate and sort
    individuals = df[!, col_individual]
    unique_individual = sort!(unique(individuals))
    N = length(unique_individual)

    # --- Optimization core: build a hash mapping (Value -> Index) ---
    # This changes lookup speed from O(N) to O(1)
    id_map = Dict{eltype(individuals),Int}()
    sizehint!(id_map, N)
    for (idx, val) in enumerate(unique_individual)
        id_map[val] = idx
    end
    # ---------------------------------------------

    # 2. Extract the time column
    times = df[!, col_time]
    unique_time = sort!(unique(times))
    T = length(unique_time)

    if isempty(unique_time)
        return (0, 0, Vector{Int64}(undef, 0))
    end

    min_T = unique_time[1]
    max_T = unique_time[end]

    # Check if time is continuous
    if max_T - min_T + 1 != T
        println("time is not continuous")
        return (0, 0, Vector{Int64}(undef, 0))
    end

    # 3. Compute indices
    num_rows = length(individuals)
    the_index = Vector{Int64}(undef, num_rows)

    # Now we do not need findfirst, just look up from the table
    for i in 1:num_rows
        # Assume all individuals are in the map (unique guarantees this)
        index_i = id_map[individuals[i]]

        # Compute flattened linear index (Column-Major logic)
        # The corresponding matrix size is (T, N) or similar block structure
        # The formula is: (individual index-1)*T + (time offset)
        the_index[i] = (index_i - 1) * T + (times[i] - min_T + 1)
    end

    return (N, T, the_index)
end

@inbounds function export_data(
    df::DataFrame,    
    m::model,
    the_options::InternalOptions,
    positions::Vector{Int64}       # Only pass column indices, not names
)

    # 1. Build index system
    identifiers=m.info.id_time
    N, T, the_index = xt_set(df, identifiers)
    if N == 0
        return (Matrix{Float64}(undef, 0, 0), 0, 0)
    end

    # 2. Compute matrix dimensions
    num_vars = length(positions)
    num_con = the_options.level ? 1 : 0
    num_timedumm = the_options.timedumm ? T : 0

    total_rows = num_vars + num_con + num_timedumm

    # 3. Allocate memory (Master Matrix)
    # Default is NaN, automatically handles missing values
    mat_tbr = fill(NaN, total_rows, N * T)

    # =======================================================
    # Part A: Fill user variables (User Variables)
    # =======================================================

    # Zero-copy column references
    data_columns = [df[!, c] for c in positions]

    # Parallel filling (fastest innermost loop)
    Threads.@threads for j in 1:num_vars
        col_data = data_columns[j]
        for i in 1:length(the_index)
            # Only here is memory read, others are register operations
            mat_tbr[j, the_index[i]] = col_data[i]
        end
    end

    # =======================================================
    # Part B: Fill generated variables (Numeric Only)
    # =======================================================

    current_row = num_vars + 1

    # 1. Constant term
    if the_options.level
        # Broadcast assignment, at low-level is memset/SIMD, extremely fast
        mat_tbr[current_row, :] .= 1.0
        push!(m.indeps, regular_var("_constant", current_row, 0))
        #push!(m.IV_vars, regular_var("_constant", current_row, 0))
        current_row += 1
    end

    # 2. Time dummies
    if the_options.timedumm

        prefix = identifiers[2] * "_"
        for num in 1:T
            name = prefix * string(num)
            push!(m.indeps, regular_variable(name, current_row + num - 1, 0))
            push!(m.IV_vars, regular_variable(name, current_row + num - 1, 0))
        end

        # Zero out this area first
        mat_tbr[current_row:end, :] .= 0.0

        # Parallel filling of the diagonal
        Threads.@threads for i in 1:N
            # Time block start column for individual i
            col_start = (i - 1) * T
            for t in 1:T
                # Row: current_row + t - 1 (the t-th dummy)
                # Col: col_start + t       (the t-th time point)
                mat_tbr[current_row+t-1, col_start+t] = 1.0
            end
        end
    end

    # Only return the data matrix and dimensions
    return (mat_tbr, N, T)
end

function handle_tables(
    p_data::Matrix{Float64},
    fd_data::Matrix{Float64},
    m::model,
    info::basic_info,
    options::InternalOptions
)
    deps = m.deps
    indeps = m.indeps
    T = info.T

    # 1. Compute dimensions
    Dcut_len = info.last_diff_index - info.first_diff_index + 1
    Lcut_len = info.last_level_index - info.first_level_index + 1

    h_deps = length(deps)
    h_indeps = length(indeps)

    # Matrix width
    x_width = options.level ? (Dcut_len + Lcut_len) : Dcut_len

    # 2. Allocate memory
    Cy = [fill(NaN, h_deps, x_width) for _ in 1:info.N]
    Cx = [fill(NaN, h_indeps, x_width) for _ in 1:info.N]

    idx_D_start, idx_D_end = info.first_diff_index, info.last_diff_index

    # =======================================================
    # Part A: Diff part (FD or FOD)
    # =======================================================

    if options.transformation == TRANSFORMATION_FD
        # === FD mode ===
        gen_table(fd_data, Cy, idx_D_start, idx_D_end, deps, info.N, info.T, options, true)
        gen_table(fd_data, Cx, idx_D_start, idx_D_end, indeps, info.N, info.T, options, true)

    else
        # === FOD mode (optimized - eliminate intermediate matrix) ===

        # Create temporary matrices for full data (for reading)
        source_y = [fill(NaN, h_deps, T) for _ in 1:info.N]
        source_x = [fill(NaN, h_indeps, T) for _ in 1:info.N]

        # Build complete time series from original data
        gen_table(p_data, source_y, 1, T, deps, info.N, info.T, options, true)
        gen_table(p_data, source_x, 1, T, indeps, info.N, info.T, options, true)

        # Parallel FOD calculation and directly write to cut target area
        Threads.@threads for i in 1:info.N
            # Use diff part of Cy/Cx as target (view, zero-copy)
            target_y_view = @view Cy[i][:, 1:Dcut_len]
            target_x_view = @view Cx[i][:, 1:Dcut_len]

            # Compute FOD directly to target area, no need for intermediate matrix
            get_fod_table!(target_y_view, source_y[i], idx_D_start, idx_D_end)
            get_fod_table!(target_x_view, source_x[i], idx_D_start, idx_D_end)
        end
    end

    # =======================================================
    # Part B: System GMM Level part
    # =======================================================
    if options.level
        idx_L_start, idx_L_end = info.first_level_index, info.last_level_index
        gen_table(p_data, Cy, idx_L_start, idx_L_end, deps, info.N, info.T, options, false)
        gen_table(p_data, Cx, idx_L_start, idx_L_end, indeps, info.N, info.T, options, false)
    end

    return (Cx=Cx, Cy=Cy)
end
function handle_tablesOLD(
    p_data::Matrix{Float64},
    fd_data::Matrix{Float64},
    m::model,
    info::basic_info,
    options::InternalOptions
)
    deps = m.deps
    indeps = m.indeps
    T = info.T

    # 1. Compute dimensions
    Dcut_len = info.last_diff_index - info.first_diff_index + 1
    Lcut_len = info.last_level_index - info.first_level_index + 1

    h_deps = length(deps)
    h_indeps = length(indeps)

    # Matrix width
    x_width = options.level ? (Dcut_len + Lcut_len) : Dcut_len

    # 2. Allocate memory
    Cy = [fill(NaN, h_deps, x_width) for _ in 1:info.N]
    Cx = [fill(NaN, h_indeps, x_width) for _ in 1:info.N]

    idx_D_start, idx_D_end = info.first_diff_index, info.last_diff_index

    # =======================================================
    # Part A: Diff part (FD or FOD)
    # =======================================================

    if options.transformation == TRANSFORMATION_FD
        # === FD mode ===
        gen_table(fd_data, Cy, idx_D_start, idx_D_end, deps, info.N, info.T, options, true)
        gen_table(fd_data, Cx, idx_D_start, idx_D_end, indeps, info.N, info.T, options, true)

    else
        # === FOD mode (corrected logic) ===

        copy_y = [fill(NaN, h_deps, T) for _ in 1:info.N]
        copy_x = [fill(NaN, h_indeps, T) for _ in 1:info.N]

        gen_table(p_data, copy_y, 1, T, deps, info.N, info.T, options, true)
        gen_table(p_data, copy_x, 1, T, indeps, info.N, info.T, options, true)

        Threads.@threads for i in 1:info.N

            source_y = copy_y[i]
            source_x = copy_x[i]
            target_x = similar(source_x)
            target_y = similar(source_y)
            # 3. Compute FOD
            # Input: source (read-only, raw data)
            # Output: target (write, FOD data)
            get_fod_table!(target_y, source_y)
            get_fod_table!(target_x, source_x)
            Cy[i][:, 1:Dcut_len] = @view target_y[:, idx_D_start:idx_D_end]
            Cx[i][:, 1:Dcut_len] = @view target_x[:, idx_D_start:idx_D_end]
        end
    end

    # =======================================================
    # Part B: System GMM Level part
    # =======================================================
    if options.level

        idx_L_start, idx_L_end = info.first_level_index, info.last_level_index
        gen_table(p_data, Cy, idx_L_start, idx_L_end, deps, info.N, info.T, options, false)
        gen_table(p_data, Cx, idx_L_start, idx_L_end, indeps, info.N, info.T, options, false)

    end

    return (Cx=Cx, Cy=Cy)
end

function gen_table(
    ori_data::Matrix{Float64},
    out_data::Vector{Matrix{Float64}},
    start_index::Int64,
    last_index::Int64,
    variable_list::Vector{regular_var},
    N::Int64,
    T::Int64,
    options::InternalOptions,
    is_diff_part::Bool=true
)
    # 1. Determine target column range
    width = last_index - start_index + 1
    total_cols = size(out_data[1], 2)

    dest_range_start = if options.level && !is_diff_part
        total_cols - width + 1
    else
        1
    end

    # 2. Preprocess variable info
    row_indices = [v.p_index for v in variable_list]
    lags = [v.lag for v in variable_list]
    num_vars = length(variable_list)

    # 3. Parallel fill
    Threads.@threads for i in 1:N
        t_offset = (i - 1) * T
        current_out = out_data[i]

        for j in 1:num_vars
            p_row = row_indices[j]
            lag = lags[j]

            # === Change to loop over each time point, strictly check ===
            # k is the k-th column in the target matrix (corresponding to time start_index + k - 1)
            @inbounds for k in 1:width

                # 1. Compute real physical time point t
                t_real = start_index + k - 1  # The time point to fill

                # 2. Compute absolute index in p_data
                idx_current = t_offset + t_real         # Lag = 0 (current)
                idx_lagged = t_offset + t_real - lag   # Lag = L (lagged)

                # 3. Target position
                col_idx = dest_range_start + k - 1

                # === Core logic: double validity check ===
                # Condition A: Lagged index must not exceed boundary (cannot read previous person's data)
                # Condition B: Lagged index must be > t_offset (ensure same individual)
                # Condition C: [Your requirement] Current value (Lag=0) must not be NaN

                if (t_real - lag >= 1) && (t_real - lag <= T)

                    # Get current value (Lag=0) for check
                    val_current = ori_data[p_row, idx_current]

                    # Only when the current value is valid, move the lagged value
                    if isnan(val_current)
                        current_out[j, col_idx] = NaN
                    else
                        # Move lagged value (note to check lagged value itself for NaN)
                        val_lagged = ori_data[p_row, idx_lagged]
                        current_out[j, col_idx] = val_lagged
                    end
                else
                    # If out of bounds (e.g. t=1, lag=1), fill NaN directly
                    current_out[j, col_idx] = NaN
                end
            end
        end
    end
end

function prepare_reg_fod(Diff_x::Matrix{Float64}, Diff_y::Matrix{Float64})
    col_if_nan_y = col_has_nan(Diff_y)
    col_if_nan_x = col_has_nan(Diff_x)

    #pragma omp parallel for
    for i in 1:size(Diff_y, 1)
        if (col_if_nan_y[i] || col_if_nan_x[i])
            Diff_x[:, i] .= 0
            Diff_y[:, i] .= 0
        end
    end
end


function prepare_reg(info::basic_info, z_list::Vector{Matrix{Float64}}, Cx::Vector{Matrix{Float64}}, Cy::Vector{Matrix{Float64}}, na_records::Vector{Int64}, transformation::Int64, level::Bool)

    N = info.N
    

    z_height = size(z_list[1], 1)

    # --- 1. Thread-local accumulators ---
    # 🔴 Key fix: use maxthreadid() 🔴
    # This ensures that even if threadid goes to 5, 8, or even higher, the array won't be out of bounds
    num_slots = Threads.maxthreadid()

    thread_NA_sum = zeros(Int, num_slots)
    thread_max = zeros(Int, num_slots)
    thread_min = fill(typemax(Int), num_slots)

    # --- 2. Parallel loop ---
    Threads.@threads for i in 1:N
        tid = Threads.threadid()

        # ... (no changes in core logic below) ...
        cx = Cx[i]
        cy = Cy[i]
        z = z_list[i]

        cx_rows, cx_cols = size(cx)
        cy_rows, cy_cols = size(cy)

        local_temp_NA = 0

        for j in 1:z_height
            has_nan = false
            for r in 1:cx_rows
                if isnan(cx[r, j])
                    has_nan = true
                    break
                end
            end
            if !has_nan
                for r in 1:cy_rows
                    if isnan(cy[r, j])
                        has_nan = true
                        break
                    end
                end
            end

            if has_nan
                na_records[z_height*(i-1)+j] = 1
                cx[:, j] .= 0.0
                cy[:, j] .= 0.0
                z[j, :] .= 0.0

                # Your logic: only count missingness for diff part
                if j <= info.diff_height
                    local_temp_NA += 1
                end
            end
        end

        # --- 3. Update thread-local stats ---
        # Now safe because array length is maxthreadid()
        thread_NA_sum[tid] += local_temp_NA

        if local_temp_NA > thread_max[tid]
            thread_max[tid] = local_temp_NA
        end

        if local_temp_NA < thread_min[tid]
            thread_min[tid] = local_temp_NA
        end
    end

    # --- 4. Aggregate all thread results ---
    total_num_NA = sum(thread_NA_sum)
    final_max = maximum(thread_max)
    final_min = minimum(thread_min)

    if final_min == typemax(Int)
        final_min = 0
    end

    # --- 5. Compute final statistics ---
    height = info.diff_height
    nobs = height * N - total_num_NA
    min_obs = height - final_max
    max_obs = height - final_min
    avg_obs = height * 1.0 - (total_num_NA / N)

    return (nobs, max_obs, min_obs, avg_obs)
end

function get_df_info(info::basic_info, options::InternalOptions, m::model)

    info.num_dep = length(m.deps)
    info.num_indep = length(m.indeps)
    max_iv_lag::Int64 = length(m.IV_vars) > 0 ? maximum([var.lag for var in m.IV_vars]) : 0

    max_indep_lag::Int64 = maximum([var.lag for var in m.indeps])

    max_lag = max(max_indep_lag, max_iv_lag)

    #max_Dgmm_minlag = maximum([var.min_lag for var in m.Dgmm_vars]) # not used ???

    max_Lgmm_minlag = length(m.Lgmm_vars) > 0 ? maximum([var.min_lag for var in m.Lgmm_vars]) : 0

    last_level_index = info.T    # last period as 1 based
    last_diff_index = info.T

    first_level_index = max(max_lag + 1, max_Lgmm_minlag + 1)

    if ((options.level) && (options.transformation == TRANSFORMATION_FOD))
        first_diff_index = first_level_index
    else
        first_diff_index = first_level_index + 1   # # max(max_lag + 1, max_Dgmm_minlag)
    end

    if first_diff_index + 2 > last_diff_index   # # to do: change 3 to something rated to AR(p)
        error("Not enough periods to run the model")
    end

    info.first_diff_index = first_diff_index
    info.last_diff_index = last_diff_index
    info.first_level_index = first_level_index
    info.last_level_index = last_level_index
    info.max_lag = max_lag
end

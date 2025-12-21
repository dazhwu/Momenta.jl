function calculate_z_dimension(m::model, info::basic_info, options::InternalOptions)::Dict

    # Calculate the height of diff block
    info.diff_height = info.last_diff_index - info.first_diff_index + 1
    # Calculate the height of level block
    info.level_height = info.last_level_index - info.first_level_index + 1

    info.level_width = 0

    num_Dgmm_instr, list_diff_info = prepare_Z_diff(m, info, options.level, options.transformation, options.collapse)
    info.diff_width=num_Dgmm_instr + length(m.IV_vars)
    if options.level
        num_Lgmm_instr, gmm_level_info = prepare_z_level(m, info, options.collapse)
        list_diff_info["gmm_level_info"] = gmm_level_info
        info.level_width = num_Lgmm_instr + 1
        # level_width = num_Lgmm_instr
        # if options.constant
        #     level_width = num_Lgmm_instr + 1  # for constant
        # else
        #     level_width = num_Lgmm_instr
        # end

    end

    info.z_height = info.diff_height
    info.z_width = info.diff_width

    if options.level
        info.z_height = info.diff_height + info.level_height
        info.z_width = info.diff_width + info.level_width
    end

    return list_diff_info
end

function get_z_table(info::basic_info, p_data, fd_data::Matrix{Float64}, m::model, options::InternalOptions)::Vector{Matrix{Float64}}

    gmm_info = calculate_z_dimension(m, info, options)

    z_list = Vector{Matrix{Float64}}(undef, info.N)

    # Initialize (in parallel)
    # Because build_z_diff is sparsely filled, must initialize to 0 first

    Threads.@threads for i in 1:info.N
        z_list[i] = zeros(info.z_height, info.z_width)
    end

    if options.level
        build_z_level(z_list, m, info, gmm_info, p_data, fd_data, options)
    else
        build_z_diff(z_list, m, info, gmm_info, p_data, fd_data, options)
    end

    # Previously, there was a huge O(N*T*Z) NaN checking loop here
    # Now it is handled inline during the build process (with ifelse), so it's no longer needed.

    return z_list
end

function build_z_level(
    z_list::Vector{Matrix{Float64}},
    m::model,
    info::basic_info,
    gmm_info::Dict,
    p_data,
    fd_data,
    options::InternalOptions
)
    N = info.N

    # 1. Get map
    gmm_level_info = gmm_info["gmm_level_info"]

    # 2. First build the Diff part
    build_z_diff(z_list, m, info, gmm_info, p_data, fd_data, options)

    # 3. Prepare Level part parameters
    # Level part starting row (right after the Diff part)

    # Variable lists
    Lgmm_list = m.Lgmm_vars
    IV_list = m.IV_vars
    num_Lgmm = length(Lgmm_list)
    num_IV = length(IV_list)

    # Calculate variable height for each individual in the source data
    vars_height = div(size(p_data, 2), N)

    start_col = info.diff_width + 1   # This is equivalent to Python's start_row
    start_row = info.diff_height + 1

    # 4. Parallel fill
    Threads.@threads for n in 1:N
        # Get reference to the current individual's Z matrix
        z_mat = z_list[n]

        # Pre-calculate data source views (reduce slice overhead)
        # Level GMM uses fd_data (differenced data as IV)
        array_fd = @view fd_data[:, (n-1)*vars_height+1:n*vars_height]
        # Standard IV uses p_data (raw data)
        array_p = @view p_data[:, (n-1)*vars_height+1:n*vars_height]
        z_mat[start_row:end, end] .= 1
        # Level block start column (immediately after the Diff block)
        # current_col = info.diff_width + 1

        # =======================================================
        # Part A: Fill Level GMM (System GMM Instruments)
        # =======================================================
        # Rule: Traverse Lgmm_vars, no longer distinguishing dep/pred

        # Step control: If collapse=true, should all variables be stacked in the same column?
        # Usually, Level IV does not collapse (or it is meaningless, since there is only one column).
        # Assume: Not collapsed -> one column per variable

        for k in 1:num_Lgmm
            var_struct = Lgmm_list[k]
            p_row_idx = var_struct.p_index

            for r in 1:info.level_height
                # Actual row number
                real_row = start_row + r - 1
                real_col = start_col + gmm_level_info[k, r, 1] - 1

                # Lookup table (map)
                data_idx = gmm_level_info[k, r, 2]

                # 🔴 Only read data if the map indicates valid (>0)
                # Since z_mat is already initialized as zeros, no need to write else branch
                if data_idx > 0
                    val = array_fd[p_row_idx, data_idx]
                    # If the data itself is NaN (e.g., lost due to differencing), fill 0.0
                    z_mat[real_row, real_col] = ifelse(isnan(val), 0.0, val)
                end
            end
            # current_col += col_inc
        end

        # =======================================================
        # Part B: Fill Standard IVs (Exogenous / Constants)
        # =======================================================

        if num_IV > 0
            current_col = info.diff_width - num_IV + 1
            for k in 1:num_IV
                var_struct = IV_list[k]
                p_row_idx = var_struct.p_index

                for r in 1:info.level_height
                    real_row = start_row + r - 1
                    time_idx = info.first_level_index + r - 1

                    # Calculate lagged index
                    data_idx = time_idx - var_struct.lag

                    # 🔴 Same logic: only read when index is valid
                    if data_idx >= 1
                        val = array_p[p_row_idx, data_idx]
                        z_mat[real_row, current_col] = ifelse(isnan(val), 0.0, val)
                        # endgmm_width = options.collapse ? num_gmm : num_gmm * info.level_height
                    end
                end
                current_col += 1
            end
        end
    end
end

function prepare_z_level(m::model, info::basic_info, collapse::Bool)::Tuple{Int64,Array{Int64}}
    num_gmm = length(m.Lgmm_vars)

    # gmm_width = collapse ? num_gmm : num_gmm * info.level_height

    # 2. Calculate the width of Standard IV section
    # In build_z_level, each IV variable occupies 1 column (lines 14-18)
    # num_iv = length(m.IV_vars)

    # Initialize map (use -1 to denote invalid value)
    gmm_level_info::Array{Int64} = fill(-1, num_gmm, info.level_height, 2)
    start_row = 1
    start_col = 1
    for (var_idx, var) in enumerate(m.Lgmm_vars)
        for i in 1:info.level_height
            gmm_level_info[var_idx, i, 1] = start_col
            # The time t of current Level equation
            the_time_idx = info.first_level_index + i - 1

            # Level IV logic: use differenced value at t - min_lag
            # e.g.: min_lag=1 (predetermined), index = t-1
            #       min_lag=2 (endogenous), index = t-2
            gmm_index = the_time_idx - var.min_lag

            # Only valid if index > 0
            if gmm_index >= 1
                gmm_level_info[var_idx, i, 2] = gmm_index
                if collapse
                    if i == info.level_height
                        start_col += 1
                    end
                else
                    start_col += 1
                end
            end
        end
    end

    return start_col - 1, gmm_level_info
end

@inbounds function build_z_diff(z_list::Vector{Matrix{Float64}}, m::model, info::basic_info, gmm_info::Dict,
    p_data, fd_data::Matrix{Float64}, options::InternalOptions)

    N = info.N
    Dgmm_width = div(size(p_data, 2), N)
    
    gmm_Div_width = div(size(fd_data, 2), N) # If IV exists, use fd_data
    gmm_DIV_height = size(fd_data, 1)

    diff_GMM_info::Matrix{Int} = gmm_info["diff_GMM_info"]
    

    # Determine starting row offset of Z matrix (if FOD, may skip the first row)
    start_row_offset = (options.level && options.transformation == TRANSFORMATION_FOD) ? 1 : 0 # Corresponds to relative index in info
   
    if options.transformation==TRANSFORMATION_FOD
        fod_data=similar(fd_data)
        # This can also be multithreaded since each person writes to different regions independently
        Threads.@threads for i in 1:N
            # Calculate the column range for the individual
            col_indices = (i-1)*gmm_Div_width+1 : i*gmm_Div_width
            
            # Extract view (Source & Target)
            v_source = view(p_data, :, col_indices)
            v_target = view(fod_data, :, col_indices)
            
            # Do FOD for this individual
            get_fod_table!(v_target, v_source)
        end
    end

    Threads.@threads for i in 1:N

       
        col_indices = (i-1)*gmm_Div_width+1:i*gmm_Div_width
        # Get view of the column data for the current individual
        array_gmm = @view p_data[:, (i-1)*Dgmm_width+1:i*Dgmm_width]

        array_fd_iv = if options.transformation == TRANSFORMATION_FOD
            
            view(fod_data, :, col_indices)
        else
            # === FD branch ===
            # Directly return view of the large matrix
            view(fd_data, :, col_indices)
        end
        z_mat = z_list[i]

        # === Traverse map ===
        for scan_index in 1:size(diff_GMM_info, 1)
            v_idx = diff_GMM_info[scan_index, 1]
            p_start = diff_GMM_info[scan_index, 2] # Deepest
            p_last = diff_GMM_info[scan_index, 3] # Recent
            base_col = diff_GMM_info[scan_index, 4]
            rel_row = diff_GMM_info[scan_index, 5]

            # Actual row number = relative row + offset
            the_row = rel_row + start_row_offset

            # === Core filling logic ===
            if options.collapse


                for j in p_start:p_last
                    val = array_gmm[v_idx, j]
                    # col_offset = p_last - j
                    target_col = base_col + (p_last - j)

                    z_mat[the_row, target_col] = ifelse(isnan(val), 0.0, val)
                end

            else
                # No Collapse: standard GMM (diagonal expansion)
                # base_col points to the starting column for this row
                # directly fill in order
                count = 0
                
                for j in p_start:p_last
                    val = array_gmm[v_idx, j]
                    if base_col + count > size(z_mat, 2)
                        println("p_start:$(p_start)  p_last:$(p_last)")
                    end
                    z_mat[the_row, base_col+count] = ifelse(isnan(val), 0.0, val)
                    count += 1
                end
            end
        end
        
        # diff_IV_info=gmm_info["diff_IV_info"]
        # Fill IV variables (remain unchanged)
        if length(m.IV_vars) >= 1
            diff_IV_info = gmm_info["diff_IV_info"]
            for scan_index in 1:size(diff_IV_info, 1)
                v = diff_IV_info[scan_index, 1]
                the_index = diff_IV_info[scan_index, 2]
                the_row = diff_IV_info[scan_index, 4] + start_row_offset 
                the_col=info.diff_width -length(m.IV_vars) + diff_IV_info[scan_index, 3]
                # the_col = diff_IV_info[scan_index, 3]
                val = array_fd_iv[v, the_index]
                z_mat[the_row, the_col] = ifelse(isnan(val), 0.0, val)
            end
        end
    end
end


function prepare_Z_diff(m::model, info::basic_info, level::Bool, transformation::Int64, collapse::Bool)::Tuple{Int,Dict}

    # 1. Determine the time window
    first_index = (level && transformation == TRANSFORMATION_FOD) ? (info.first_diff_index + 1) : info.first_diff_index
    last_index = info.last_diff_index
    time_height = last_index - first_index + 1 # Height of Z matrix

    # 2. Pre-estimate the size of Info matrix (maximum possible rows = number of variables * time height)
    # We'll resize or only use a part of it later
    max_rows = length(m.Dgmm_vars) * time_height
    diff_GMM_info = Matrix{Int}(undef, max_rows, 5)

    current_row = 0     # Current row in the info matrix
    current_z_col = 1   # Current column pointer of the Z matrix

    # =======================================================
    # Part A: Process GMM variables (each variable computes width separately)
    # =======================================================
    for (i, var) in enumerate(m.Dgmm_vars)

        # In collapse mode, each variable takes a fixed number of columns: (max - min + 1)
        # In no-collapse mode, columns accumulate over time t

        # Record starting column in Z matrix for this variable (needed in collapse mode)
        var_start_col = current_z_col

        for t in 1:time_height
            # Restore current time point's absolute index in p_data
            y_index = first_index + t - 1

            # --- Core logic: calculate at current t, how many valid lags for this variable ---
            # Theoretical lag range: [y_index - max_lag, y_index - min_lag]
            # Physical constraint: index must be >= 1

            idx_deepest = max(1,y_index - var.max_lag)
            idx_recent = y_index - var.min_lag

            current_row += 1

            # Fill map
            diff_GMM_info[current_row, 1] = i              # Var ID
            diff_GMM_info[current_row, 2] = idx_deepest  # p_data Start
            diff_GMM_info[current_row, 3] = idx_recent   # p_data End

            # Z Col Index computation is the key difference
            if collapse
                # Collapse: columns fixed, based on relative position of lag
                # This logic is a bit more complex: we need the map to point to the "Min Lag" column
                # In build phase, we use col + (recent - j) to backfill
                # So just record base column here
                diff_GMM_info[current_row, 4] = var_start_col
            else
                # No Collapse: current column pointer is the starting column
                diff_GMM_info[current_row, 4] = current_z_col

                # Shift pointer: this row consumes (recent - deepest + 1) columns
                current_z_col += (idx_recent - idx_deepest + 1)
            end

            diff_GMM_info[current_row, 5] = t # Z Row (Relative)
        end

        # If collapse mode, after processing all time points for a variable, shift the pointer
        if collapse
            # The total width this variable needs = max_lag - min_lag + 1            
            # To be safe, pre-allocate space according to the max-min definition
            width_needed = var.max_lag - var.min_lag + 1
            current_z_col += width_needed
        end
    end

    # Trim matrix to actual size
    final_GMM_info = diff_GMM_info[1:current_row, :]

    list_tbr = Dict{String,Array{Int64}}()
    list_tbr["diff_GMM_info"] = final_GMM_info

    # =======================================================
    # Part B: Process IV variables (standard IV)
    # ======================================================= or i in 1:100
    num_exog = length(m.IV_vars)
    if num_exog >= 1
       
        diff_IV_info = Matrix{Int}(undef, num_exog * time_height, 4)
        iv_row_count = 0

        for i in 1:num_exog
            var = m.IV_vars[i]
            for t in 1:time_height
                y_index = first_index + t - 1
                t_idx = y_index - var.lag # Lagged index of IV

                if t_idx >= 1
                    iv_row_count += 1
                    diff_IV_info[iv_row_count, 1] = m.IV_vars[i].p_index         # Var ID
                    diff_IV_info[iv_row_count, 2] = t_idx     # p_data Index
                    diff_IV_info[iv_row_count, 3] = i # current_z_col # Z Col
                    diff_IV_info[iv_row_count, 4] = t         # Z Row

                    # current_z_col += 1 # Each IV row consumes one column (diagonal matrix)
                end
            end
        end
        list_tbr["diff_IV_info"] = diff_IV_info[1:iv_row_count, :]
    end

    return current_z_col - 1, list_tbr
end


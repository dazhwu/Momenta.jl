

function parse_instruments(
    instr_str::String, 
    col_names::Vector{String},
    names::Vector{String}, 
    positions::Vector{Int64}, 
    opts::InternalOptions,  # Need opts to decide whether to generate Level GMM
    T::Int
)

    # 1. Prepare three containers
    Dgmm_list = Vector{gmm_var}() # Difference equation instruments
    Lgmm_list = Vector{gmm_var}() # Level equation instruments
    IV_list   = Vector{regular_var}() # Standard instruments

    # 2. Regex: extract GMM(...) or IV(...) blocks
    # Same logic as before: allow nested parentheses inside IV
    block_regex = r"(GMM|IV)\s*\(\s*((?:[^()]+|\([^()]*\))+)\s*\)"

    for m in eachmatch(block_regex, instr_str)
        type_str = m.captures[1]  # "GMM" or "IV"
        content  = m.captures[2]  # Content inside parentheses

        if type_str == "GMM"
            # ====================================================
            # Process GMM: "n k , 2:4" -> generate gmm_var
            # ====================================================
            parts = split(content, ",")
            if length(parts) != 2
                error("GMM format error: $content (should contain variables and lag range)")
            end

            # --- A. Parse variables ---
            vars_part = split(strip(parts[1]))
            
            # --- B. Parse range ---
            lag_str = strip(parts[2])
            lag_str = replace(lag_str, "." => string(T-1)) # Support 2:.
            
            local min_l, max_l
            try
                val = eval(Meta.parse(lag_str)) # Parse "2:4" -> UnitRange
                if val isa Number
                    min_l, max_l = val, val
                else
                    min_l, max_l = minimum(val), maximum(val)
                end
            catch
                error("GMM Lag parse failed: $lag_str")
            end

            # --- C. Iterate variables, generate gmm_var ---
            for v_str in vars_part
                v_name = string(v_str)
                # Lookup/register index
                idx = process_vars(v_name, names, positions, col_names)

                # 1. Add to Diff GMM list
                actual_max = min(max_l, T - 1)
                if min_l <= actual_max
                    push!(Dgmm_list, gmm_var(
                        v_name, idx, 
                        min_l, actual_max, 
                        opts.collapse # Temporarily use the global collapse option here
                    ))
                end

                # 2. Add to Level GMM list (System GMM logic)
                if opts.level
                    # Python logic: Level IV uses lag t-1
                    # min_lag_level = max(min_lag_diff - 1, 0)
                    lev_min = max(min_l - 1, 0)
                    lev_max = min_l # Usually Level IV only uses the most recent period
                    
                    actual_lev_max = min(lev_max, T - 1)

                    if actual_lev_max >= lev_min
                        push!(Lgmm_list, gmm_var(
                            v_name, idx,
                            lev_min, actual_lev_max, 
                            opts.collapse
                        ))
                    end
                end
            end

        elseif type_str == "IV"
            # ====================================================
            # Process IV: "z lag(y,1)" -> generate regular_var
            # ====================================================
            # Reuse the previous internal regex
            inner_regex = r"lag\s*\(([^)]+)\)|([\w\u4e00-\u9fa5]+)"

            for token in eachmatch(inner_regex, content)
                if token.captures[1] !== nothing
                    # Case 1: lag(y, 1)
                    args = split(token.captures[1], ",")
                    v_name = strip(args[1])
                    l_val  = parse(Int, strip(args[2]))
                    
                    idx = process_vars(v_name, names, positions, col_names)
                    push!(IV_list, regular_var(v_name, idx, l_val))

                elseif token.captures[2] !== nothing
                    # Case 2: plain variable z
                    # args = split(token.captures[2], ",")
                    # v_name = strip(args[1])
                    vars_part = split(strip(token.captures[2] ))
                    #if v_name == "lag" continue end # Error mitigation
                    
                    for v_str in vars_part
                        v_name = string(v_str)
                        idx = process_vars(v_name, names, positions, col_names)
                        push!(IV_list, regular_var(v_name, idx, 0)) # lag=0
                    end
                end
            end
        end
    end

    return Dgmm_list, Lgmm_list, IV_list
end


function process_vars(the_name::String, 
                      names::Vector{String}, 
                      positions::Vector{Int64}, 
                      col_names::Vector{String})
    
    # 1. Search in the already registered list
    idx = findfirst(==(the_name), names)
    
    if !isnothing(idx)
        return idx
    end

    # 2. If not found, search in the original DataFrame columns
    df_idx = findfirst(==(the_name), col_names)
    
    if isnothing(df_idx) 
        error("Variable '$the_name' not found in DataFrame columns.")
    else
        # 3. Register the new variable
        push!(names, the_name)      # Save name
        push!(positions, df_idx)    # Save its physical column position in DF
        return length(names)        # Return the new index in names
    end
end

function parse_model_string(model_str::String, 
                            col_names::Vector{String}, 
                            names::Vector{String}, 
                            positions::Vector{Int64})

    # 1. Split left and right
    if !occursin("~", model_str)
        throw(ArgumentError("Model formula must contain '~'"))
    end
    lhs_str, rhs_str = split(model_str, "~", limit=2)

    # =======================================================
    # Part A: Parse left (LHS) -> build list of dependent variables
    # =======================================================
    deps = Vector{regular_var}()
    dep_names_set = Set{String}() # Used for fast lookup if it's a dependent variable

    for name in split(lhs_str)
        if isempty(name) continue end
        s_name = string(name)
        
        idx = process_vars(s_name, names, positions, col_names)
        push!(deps, regular_var(s_name, idx, 0))
        push!(dep_names_set, s_name)
    end

    # =======================================================
    # Part B: Parse right (RHS) + real-time continuity checks
    # =======================================================
    indeps = Vector{regular_var}()
    
    # Counter: record the highest lag order parsed so far for each dep var
    # All start at 0. If lag(n, 1) is found, update to 1. Next it must find 2, etc.
    lag_tracker = Dict{String, Int}()
    for d in dep_names_set
        lag_tracker[d] = 0
    end

    # Regex: extract lag(...) or variable names
    regex = r"lag\s*\(([^)]+)\)|([\w\u4e00-\u9fa5]+)"

    for m in eachmatch(regex, rhs_str)
        if m.captures[1] !== nothing
            # === Case 1: lag(var, range) ===
            content = m.captures[1]
            args = split(content, ",")
            if length(args) != 2 error("lag format error: $content") end
            
            target_var = strip(args[1])
            lag_str    = strip(args[2])
            s_target   = string(target_var)

            # Parse lag value/range
            lag_val = try eval(Meta.parse(lag_str)) catch; error("Cannot parse lag: $lag_str") end
            lag_range = (lag_val isa Number) ? [lag_val] : lag_val
            
            # --- Register variable index ---
            idx = process_vars(s_target, names, positions, col_names)

            # --- Core loop: handle each lag ---
            for l in lag_range                
                # Only if this variable is a dep var, do strict continuity check
                if s_target in dep_names_set
                    current_count = lag_tracker[s_target]
                    
                    if l != current_count + 1
                        error("Lags are not continuous or out of order!\n" * "Variable: $(s_target)\n" * "Currently at lag $(current_count), expecting next lag $(current_count+1), but got lag $(l).\n" *
                              "Please write in order, e.g.: lag(n, 1:2)")
                    end
                    
                    # Check passed, increment counter
                    lag_tracker[s_target] = l
                end

                # Add to list
                push!(indeps, regular_var(s_target, idx, l))
            end

        elseif m.captures[2] !== nothing
            # === Case 2: plain variable ===
            var_name = m.captures[2]
            if var_name == "lag" continue end
            s_name = string(var_name)
            
            # What if user writes lag(n, 1) as a lagged n but does not use lag function?
            # Usually, plain variable default lag=0, no need for continuity check
            idx = process_vars(s_name, names, positions, col_names)
            push!(indeps, regular_var(s_name, idx, 0))
        end
    end

    # =======================================================
    # Part C: Final symmetry check
    # =======================================================
    # After the loop, lag_tracker stores the highest lag for each dep var
    # Just check if they are all equal
    
    # Take all final lag orders for each dep var
    final_orders = values(lag_tracker)
    
    if !isempty(final_orders)
        p_order = first(final_orders) # Take the first as the reference
        
        for (name, order) in lag_tracker
            if order != p_order
                error("Model is not symmetric!\n" * "All dependent variables must have the same number of lags.\n" *
                      "Check result: " * join(["$k=$v" for (k,v) in lag_tracker], ", "))
            end
        end
        
        
    else
        # Static model, p=0
        p_order = 0
    end

    return deps, indeps, p_order
end

function process_command(
    model_str::String,             # "n ~ w lag(k, 1)"
    instruments::String, # Instruments string provided by user
    opts::String,                 # Options string provided by user
    col_names::Vector{String},     # All DataFrame column names
    names::Vector{String},
    positions::Vector{Int64},

    T::Int64                       # Time length (for checking lags)
)::Tuple{model, InternalOptions, Int64}
    # 1. Initialize "Unique Registry"
    # names stores variable names, positions stores DataFrame column indices

    
    # 2. Build internal Options
    internal_opts = parse_options(opts)

    # 3. Parse formula (will populate names and positions)
    deps, indeps, p_order = parse_model_string(model_str, col_names, names, positions)

    # 4. Parse instruments (will also populate names and positions)
    # Note: must pass internal_opts and T
    Dgmm_list, Lgmm_list, IV_list = parse_instruments(
        instruments, col_names, names, positions, internal_opts, T
    )


    m = model(
        deps, indeps, Dgmm_list,
        Lgmm_list, IV_list, model_str
    )

    # 5. Return all objects
    # names and positions are your "unique vars" list
    # Later, idx in deps, indeps, etc. all point to this names list
    return m, internal_opts, p_order
end



function parse_options(options_str)::InternalOptions

    list_options::Vector{String} = split(lowercase(options_str), " ", keepempty=false)

    options=InternalOptions()

    for option in list_options
        the_option = lowercase(option)
        if the_option == "onestep"
            options.steps = 1
        elseif the_option == "nolevel"
            options.level = false
        elseif the_option == "fod"
            options.transformation = TRANSFORMATION_FOD
        elseif the_option == "timedumm"
            options.timedumm = true
        elseif the_option == "collapse"
            options.collapse = true        
        else
            throw(ArgumentError(option * ": is not a valid option"))
        end
    end
    # if options.constant && !options.level
    #     throw(ArgumentError("Options constant and nolevel are mutually exclusive"))
    # end


    return options
end



# Specially computes A ⊗ I_k
# k times faster than the native kron and does not generate garbage memory
function kron_with_identity(A::AbstractMatrix{Float64}, k::Int)
    # 1. Fastest path: if k=1 (scalar), directly return the original matrix
    if k == 1
        return A # or return copy(A) depending on whether you will modify it later
    end

    rows, cols = size(A)
    # 2. Preallocate result matrix (all zeros)
    out = zeros(Float64, rows * k, cols * k)

    # 3. Fill manually (skip all zeros)
    # Traverse each column of source matrix A (Julia is column-major, this is fastest)
    @inbounds for j in 1:cols
        for i in 1:rows
            val = A[i, j]
            # Fill only when val is not 0 (optional optimization, depending on sparsity, usually always fill is faster)
            
            # Calculate start position of target block
            # A[i,j] corresponds to a k*k diagonal block in out
            row_start = (i - 1) * k
            col_start = (j - 1) * k
            
            # Fill only the diagonal
            # @simd may attempt vectorization, even though this is strided write, modern CPUs can still optimize
            @simd for d in 1:k
                out[row_start + d, col_start + d] = val
            end
        end
    end
    
    return out
end

function lag(mat::AbstractArray{Float64,2}, lagged::Matrix{Float64}, N::Int64, lags::Int64)
    # T is the time span for each individual
    # Use div to ensure integer index
    T = div(size(mat, 2), N) 

    # Recommendation: If lagged is not pre-filled with NaN, best to fill here, or ensure logic covers all locations
    fill!(lagged, NaN)

    # Parallel processing for each individual
    Threads.@threads for i in 1:N
        # Calculate current individual's start and end column index
        # Formula: (i-1)*T + 1
        col_start = (i - 1) * T + 1
        col_end   = i * T
        
        # Only meaningful if time span > number of lags
        if T > lags
            # Source data range: from start to (end - lags)
            # Target range: from (start + lags) to end
            
            # Use view to avoid memory allocation
            # lagged[:, (col_start + lags):col_end] .= view(mat, :, col_start:(col_end - lags))
            
            # Or, a more low-level version (usually faster):
            # Target: The k-th column of lagged (relative to col_start) = the (k-lags)-th column of mat
            for c in (col_start + lags):col_end
                for r in 1:size(mat, 1)
                    lagged[r, c] = mat[r, c - lags]
                end
            end
        end
    end
    return lagged
end


function get_first_difference_table(ori_arr::Matrix{Float64}, N::Int64)::Matrix{Float64}
    # Must know T
    T = div(size(ori_arr, 2), N)
    num_rows = size(ori_arr, 1)
    
    # Preallocate result matrix
    first_diff = fill(NaN, num_rows, N * T)

    # Parallel processing for each individual
    Threads.@threads for i in 1:N
        # Start column of current individual
        start_col = (i - 1) * T + 1
        
        # Start differencing from period 2 (first period remains NaN)
        # Corresponding column indices are from start_col + 1 to start_col + T - 1
        for t in 2:T
            curr_idx = start_col + t - 1
            prev_idx = curr_idx - 1
            
            for r in 1:num_rows
                first_diff[r, curr_idx] = ori_arr[r, curr_idx] - ori_arr[r, prev_idx]
            end
        end
    end
    
    return first_diff
end

function get_fod_table!(
    tbr::AbstractMatrix{Float64}, 
    ori_arr::AbstractMatrix{Float64},
    start_idx::Int64,
    end_idx::Int64
)
    """
    Compute FOD and only output results for the specified time range

    Arguments:
        tbr: target matrix (num_vars × output_width) - rows=variables, columns=time
        ori_arr: source data matrix (num_vars × T) - complete time series
        start_idx: output start time index (in ori_arr)
        end_idx: output end time index (in ori_arr)
    """
    
    # Check parameters
    num_vars_src, T = size(ori_arr)
    num_vars_dst, output_width = size(tbr)
    
    if num_vars_src != num_vars_dst
        throw(DimensionMismatch("Number of variables mismatch: ori_arr has $num_vars_src variables, tbr has $num_vars_dst variables"))
    end
    
    expected_width = end_idx - start_idx + 1
    if output_width != expected_width
        throw(DimensionMismatch("Output width error: expected $expected_width columns, actual $output_width columns"))
    end
    
    if start_idx < 1 || end_idx > T || start_idx > end_idx
        throw(ArgumentError("Invalid index range: start=$start_idx, end=$end_idx, T=$T"))
    end
    
    fill!(tbr, NaN)
    
    # Preallocate accumulators (one for each variable)
    acc_sum = zeros(Float64, num_vars_src)
    acc_count = zeros(Int, num_vars_src)
    
    # Backward from the second to last column (still need to traverse all time points to accumulate correctly)
    for t in (T - 1):-1:1
        @inbounds for i in 1:num_vars_src
            # Step A: accumulate value for t+1
            next_val = ori_arr[i, t+1]
            if !isnan(next_val)
                acc_sum[i] += next_val
                acc_count[i] += 1
            end
            
            # Step B: compute FOD at t
            # Write to position t+1
            output_t = t + 1
            
            # Only compute and write when t+1 is in output range
            if output_t >= start_idx && output_t <= end_idx
                if acc_count[i] > 0
                    current_val = ori_arr[i, t]
                    if !isnan(current_val)
                        future_mean = acc_sum[i] / acc_count[i]
                        adjustment = sqrt(acc_count[i] / (acc_count[i] + 1))
                        
                        # Map to target matrix column index
                        col_in_tbr = output_t - start_idx + 1
                        tbr[i, col_in_tbr] = (current_val - future_mean) * adjustment
                    end
                end
            end
        end
    end
    
    return tbr
end

# Keep the original version as a convenience interface
function get_fod_table!(tbr::AbstractMatrix{Float64}, ori_arr::AbstractMatrix{Float64})
    """Full conversion version (backward compatible)"""
    if size(tbr) != size(ori_arr)
        throw(DimensionMismatch("size mismatch"))
    end
    
    _, T = size(ori_arr)
    return get_fod_table!(tbr, ori_arr, 1, T)
end
 
function col_has_nan(mat::AbstractArray{Float64,2})::Vector{Bool}
    num_cols = size(mat, 2)
    num_rows = size(mat, 1)
    tbr = Vector{Bool}(undef, num_cols)

    # Parallel check each column
    Threads.@threads for j in 1:num_cols
        has_nan = false
        # As soon as a NaN is found, break
        for i in 1:num_rows
            if isnan(mat[i, j])
                has_nan = true
                break
            end
        end
        tbr[j] = has_nan
    end
    return tbr
end

function is_dep_NAs(deps::AbstractArray{Float64,2})::Matrix{Float64}
    num_rows, num_cols = size(deps)
    # Initialize as 0.0 (Float64)
    tbr = zeros(Float64, num_rows, num_cols)

    # Parallel scan
    Threads.@threads for j in 1:num_cols
        for i in 1:num_rows
            if isnan(deps[i, j])
                tbr[i, j] = 1.0
            end
        end
    end
    return tbr
end


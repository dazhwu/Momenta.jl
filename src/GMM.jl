const TOL = sqrt(eps(Float64))


function special_process(z_list::Vector{Matrix{Float64}}, Cx::Vector{Matrix{Float64}}, Cy::Vector{Matrix{Float64}},
    _XZ_t::Matrix{Float64}, _Zy_t::Matrix{Float64}, W1::Matrix{Float64}, H1::Matrix{Float64}, model_info::basic_info, options::InternalOptions)

    results = core_GMM(Cy, Cx, z_list, _XZ_t, _Zy_t, W1, H1, model_info.N)
    return (results[options.steps].beta, results[options.steps].residual)

end

function regular_process(z_list::Vector{Matrix{Float64}}, Cx::Vector{Matrix{Float64}}, Cy::Vector{Matrix{Float64}}, model_info::basic_info, options::InternalOptions)

    N = model_info.N
    T = model_info.T
    num_dep_lags = model_info.num_dep_lags    

    H1 = get_H1(model_info.z_height, model_info.diff_height, T, num_dep_lags, options.transformation, options.level)

    _XZ_t, _Zy_t, W1, xz_list, zy_list, W_list = calculate_basic(z_list, Cx, Cy, H1, model_info.num_dep,
        model_info.num_indep, model_info.z_width, N)

    results = core_GMM(Cy, Cx, z_list, _XZ_t, _Zy_t, W1, H1, N)
    
    results[1].vcov = N * (results[1]._M_XZ_W * results[1].W_next * results[1]._M_XZ_W')
    results[1].std_err = vcov2stderr(results[1].vcov, model_info.num_dep)

    if options.steps == 2
        #println("Windmeijer:")
        results[2].vcov = Windmeijer(results[2].M, results[2]._M_XZ_W, results[2].W_inv, results[2].qs, results[1].vcov, xz_list, results[1].uz_list, model_info.num_dep,
            model_info.num_indep, model_info.z_width, N)

        results[2].std_err = vcov2stderr(results[2].vcov, model_info.num_dep)   #     

    end

    return results, H1, xz_list, zy_list, W_list

end



function vcov2stderr(vcov::Matrix{Float64}, num_dep::Int)::Matrix{Float64}
    #
    num_rows::Int = size(vcov, 1) / num_dep
    tbr::Matrix{Float64} = fill(NaN, num_dep, num_rows)
    diag_index::Int = 0
    @inbounds for i in 1:num_rows
        for j in 1:num_dep
            diag_index += 1

            tbr[j, i] = sqrt(vcov[diag_index, diag_index])
        end
    end
    return tbr
end

@inbounds function Windmeijer(
    M2::Matrix{Float64},
    _M2_XZ_W2::Matrix{Float64},
    W2_inv::Matrix{Float64},
    zs2::Matrix{Float64},
    vcov_step1::Matrix{Float64},
    xz_list::Vector{Matrix{Float64}},
    uz_list::Vector{Matrix{Float64}},
    num_dep::Int64,
    num_indep::Int64,
    z_width::Int64,
    N::Int
)
    total_width = z_width * num_dep # Pre-calculate
    D = zeros(total_width, total_width)
    D_W = zeros(size(M2)) # Assuming size(M2) is (total_width, num_indep * num_dep) or similar

    W2_zs_vec = W2_inv * reshape(zs2, :, 1)

    num_threads = Threads.nthreads() # The result is 4
    # Get the maximum thread ID that may occur in the current environment
    max_tid = Threads.maxthreadid()

    # Allocate space according to the maximum ID; may waste 1 slot but is absolutely safe
    thread_zx_rows = [zeros(total_width) for _ in 1:max_tid]
    thread_D_parts = [zeros(total_width, total_width) for _ in 1:max_tid]

    for j = 1:num_indep
        for k = 1:num_dep
            # Reset D_parts for each (j,k) iteration for all threads
            for part in thread_D_parts
                fill!(part, 0.0)
            end

            # Parallelize the loop over N (observations/samples)
            Threads.@threads for i = 1:N
                tid = Threads.threadid()
                current_zx_row = thread_zx_rows[tid] # Get thread-local buffer
                current_D_part = thread_D_parts[tid] # Get thread-local D buffer

                fill!(current_zx_row, 0.0) # Reset for current iteration i for this thread
                for m in 1:z_width
                    # Note: xz_list[i] is a matrix. Accessing [j, m]
                    # Ensure j and m are valid indices for dimensions of xz_list[i]
                    current_zx_row[(m-1)*num_dep+k] = xz_list[i][j, m]
                end

                # Accumulate into the lower triangle of the thread-local D_part
                # The scaling by (-1.0 / N) will be done after summing up all parts
                for c = 1:total_width
                    for r = c:total_width # Iterate c first, then r from c
                        # Ensure uz_list[i] is a matrix. Accessing [r,1] and [c,1]
                        val_uz_r = uz_list[i][r, 1]
                        val_uz_c = uz_list[i][c, 1]
                        val_zx_r = current_zx_row[r]
                        val_zx_c = current_zx_row[c]
                        current_D_part[r, c] += (val_uz_r * val_zx_c + val_uz_c * val_zx_r)
                    end
                end
            end # End of threaded i loop

            # Aggregate D_parts from all threads into the main D matrix
            fill!(D, 0.0)
            for part in thread_D_parts
                D .+= part
            end

            # Apply the scaling factor
            D .*= (-1.0 / N)

            # Symmetrize D (copy lower triangle to upper triangle)
            for c = 1:total_width
                for r = (c+1):total_width # r starts from c+1 to avoid diagonal and redundant ops
                    D[c, r] = D[r, c]
                end
            end

            
            
            col_idx = (j - 1) * num_dep + k
            # D_W[:, col_idx] = (-1.0) * _M2_XZ_W2 * D * W2_zs_vec

            temp_vec = D * vec(W2_zs_vec) 
            
            # Get the view of the target column (this is a vector)
            dest_view = view(D_W, :, col_idx)

            # 3. Use 5-parameter mul! to do it in one go
            # mul!(Y, A, B, alpha, beta) => Y = A * B * alpha + Y * beta
            # Here:  dest_view = _M2_XZ_W2 * temp_vec * (-1.0) + dest_view * 0.0
            # Note: vec(temp_vec) ensures second multiplicand is a vector matching dest_view's dimension
            mul!(dest_view, _M2_XZ_W2, vec(temp_vec), -1.0, 0.0)
            
            
        end # End of k loop
    end # End of j loop

    # Final calculation
    term1 = N * M2
    term2 = D_W * term1
   

    final_result = term1 + term2 + (term1 + D_W * vcov_step1) * D_W'
    return final_result
end


function core_GMM(Cy::Vector{Matrix{Float64}}, Cx::Vector{Matrix{Float64}}, z_list::Vector{Matrix{Float64}}, _XZ_t::Matrix{Float64}, _Zy_t::Matrix{Float64}, W1::Matrix{Float64}, H1::Matrix{Float64}, N::Int64, express::Bool=false)::Vector{step_result}

    

    result1::step_result = GMM_step(N, z_list, Cx, Cy, _XZ_t, _Zy_t, W1)

    result2::step_result = GMM_step(N, z_list, Cx, Cy, _XZ_t, _Zy_t, result1.W_next, express)

    results = Vector{step_result}()
    push!(results, result1)
    push!(results, result2)
    return results
end


@inbounds function GMM_step(N::Int64, z_list::Vector{Matrix{Float64}}, Cx_list::Vector{Matrix{Float64}}, Cy_list::Vector{Matrix{Float64}}, _XZ_t::Matrix{Float64}, _Zy_t::Matrix{Float64}, W_table::Matrix{Float64}, express::Bool=false)::step_result


    W_inv = pinv(Symmetric(W_table), TOL)  #The size of  W_table is num_dep*z_width, num_dep*z_width
    #W_inv=fast_symmetric_pinv(W_table, TOL)
    #@time  W_inv = smart_inverse(W_table, TOL)

    _XZ_W = _XZ_t * W_inv # the size of _xz_w is x_height * num_dep, num_dep*z_width
    _M_inv = _XZ_W * _XZ_t' # the size of M_inv is x_height * num_dep, x_height * num_dep
    
    M = pinv(Symmetric(_M_inv), TOL)
    #M =fast_symmetric_pinv(_M_inv, TOL)
    
    _M_XZ_W = M * _XZ_W

    beta = reshape((_M_XZ_W * _Zy_t), size(Cy_list[1], 1), :)  # the size of beta is num_dep, num_indep
    residual = Vector{Matrix{Float64}}(undef, N)

    for i in eachindex(Cy_list)
        residual[i] = Cy_list[i] - (beta * Cx_list[i])
    end

    #_residual_t = residual'


    if express
        return step_result(residual, beta)
    else
        qs, zs, uz_list, W_next = calculate_ZuuZ(N, z_list, residual)
        return step_result(residual, _XZ_W, W_table, W_inv, W_next, _M_XZ_W, zs, qs, uz_list, M, beta)

    end
end


@inbounds function calculate_ZuuZ(N::Int64, z_list::Vector{Matrix{Float64}}, residual::Vector{Matrix{Float64}})

    num_dep = size(residual[1], 1)
    diag_dep = Matrix{Float64}(I, num_dep, num_dep)
    z_width::Int = size(z_list[1], 2)
    r_width::Int = size(residual[1], 2)

    zs_width = z_width * num_dep

    zs = zeros(1, zs_width)
    qs = zeros(num_dep, z_width)
    ZuuZ = zeros(zs_width, zs_width)
    uz_list = Vector{Matrix{Float64}}(undef, N)

    for i = 1:N
        rz = residual[i] * z_list[i]
        uz_list[i] = reshape(rz, :, 1)

        qs += rz
        for r in 1:zs_width
            zs[1, r] += uz_list[i][r, 1]
            for c in r:zs_width
                ZuuZ[r, c] += (uz_list[i][r, 1] * uz_list[i][c, 1]) / N
            end
        end
    end

    for r in 1:zs_width

        for c in r:zs_width
            ZuuZ[c, r] = ZuuZ[r, c]

        end
    end

    return qs, zs, uz_list, ZuuZ
end



@inbounds function calculate_basic(
    z_list::Vector{Matrix{Float64}},
    Cx_list::Vector{Matrix{Float64}},
    Cy_list::Vector{Matrix{Float64}},
    H::Matrix{Float64},
    num_dep::Int64,
    num_indep::Int64,
    z_width::Int64,
    N::Int)

    zx_list = Vector{Matrix{Float64}}(undef, N)
    zy_list = Vector{Matrix{Float64}}(undef, N)
    zHz_list = Vector{Matrix{Float64}}(undef, N)

    temp_xz = zeros(num_indep, z_width)
    
    temp_zy = zeros(num_dep, z_width)
    temp_W = zeros(z_width, z_width)

    for i in eachindex(z_list)

        zx_list[i] = Cx_list[i] * z_list[i]
        zy_list[i] = Cy_list[i] * z_list[i]
        zHz_list[i] = z_list[i]' * H * z_list[i]
        temp_xz .+= zx_list[i]
        temp_zy .+= zy_list[i]
        temp_W .+= zHz_list[i]

    end

   
    # 1. Replace kron(temp_xz, I)
    tbr_xz = kron_with_identity(temp_xz, num_dep)
    
    # 2. Replace kron(temp_W, I)
    W1 = kron_with_identity(temp_W, num_dep)

    # tbr_zy does not require kron, just reshape directly    
    tbr_zy = reshape(temp_zy, :, 1)

    return tbr_xz, tbr_zy, W1, zx_list, zy_list, zHz_list
end

function get_H1(width::Int64, diff_height::Int64, T::Int64, num_dep_lags::Int64, transformation::Int64, level::Bool)
    if transformation == TRANSFORMATION_FD
        # Preallocate the matrix
        tbr = zeros(width, width)
        
        # 1. Fill main part (differencing part)
        # This is an extremely fast loop, a single thread is enough
        @inbounds @simd for i = 1:diff_height
            tbr[i, i] = 2.0
            if i >= 2
                tbr[i-1, i] = -1.0
            end
            if i < diff_height
                tbr[i+1, i] = -1.0
            end
        end

        # 2. Fill System GMM's level part (if width > diff_height)
        if width > diff_height
            # Set diagonal to 1
            @inbounds @simd for i = (diff_height+1):width
                tbr[i, i] = 1.0
            end
            
            # Fill lower left block (low_left)
            # The memory access pattern here is scattered, but since it's simple assignment of -1/1, single thread is still fastest
            # Directly operating tbr indices is faster than creating a view
            @inbounds for i = 1:diff_height
                # Corresponds to view: low_left[i, i] = -1
                tbr[diff_height+i, i] = -1.0
                # Corresponds to view: low_left[i+1, i] = 1
                # Note boundary check: diff_height+i+1 should not exceed width
                if diff_height+i+1 <= width
                    tbr[diff_height+i+1, i] = 1.0
                end
            end
            
            # Fill upper right block (up_right)
            @inbounds for i = 1:diff_height
                # Corresponds to view: up_right[i, i] = -1
                tbr[i, diff_height+i] = -1.0
                # Corresponds to view: up_right[i, i+1] = 1
                if diff_height+i+1 <= width
                    tbr[i, diff_height+i+1] = 1.0
                end
            end
        end
        return tbr
    else
        return get_H1_fod(width, diff_height, T, num_dep_lags, level)
    end
end

function get_H1_fod(width::Int64, diff_width::Int64, T::Int64, num_dep_lags::Int64, level::Bool)
    # Optimization: generate_D_matrix does allocation inside, so we don't optimize D_up creation again here
    # Assume generate_D_matrix is very fast (usually T is small)
    D_up = generate_D_matrix(diff_width, T, num_dep_lags)
    
    if level
        # Directly construct D matrix to avoid repeated slicing
        D = zeros(width, T)
        # Copy D_up into top-left corner
        D[1:diff_width, 1:T] = D_up
        
        # Fill the lower-right (identity matrix part for system GMM level equation)
        # start_col is to align with time T
        start_col = T - (width - diff_width) + 1
        
        # Ensure the indices are valid
        if start_col >= 1
             @inbounds for i = 1:(width-diff_width)
                if (start_col + i - 1) <= T
                    D[diff_width+i, start_col+i-1] = 1.0
                end
            end
        end
        
        return D * D'
    else
        return D_up * D_up'
    end
end

# generate_D_matrix usually doesn't need to be changed, unless T is extremely large.
# The current O(T^2) logic is nanosecond-level when T<100.
function generate_D_matrix(height::Int64, T::Int64, num_dep_lags::Int)
    # Directly compute the part of D needed, avoid first calculating a large temp and then slicing
    # This reduces a T*T memory allocation
    D = zeros(height, T)
    
    start_row = num_dep_lags - 1 # Corresponds to the starting slice index in the original code temp
    
    @inbounds for r = 1:height
        # Map back to original row index of temp matrix
        original_i = start_row + r 
        
        # According to original logic: temp[i, j] calculation
        # Only need to compute for j >= original_i
        if original_i <= T
            # Precompute constants
            # temp[i, i] = sqrt((T-i)/(T-i+1))
            val_diag = sqrt((T - original_i) * 1.0 / (T - original_i + 1))
            D[r, original_i] = val_diag
            
            # temp[i, j] = -sqrt(...)
            if original_i < T
                 val_off = -sqrt(1.0 / ((T - original_i + 1) * 1.0 * (T - original_i)))
                 for j = (original_i + 1):T
                     D[r, j] = val_off
                 end
            end
        end
    end
    return D
end



function PAR1_matrix(beta::Matrix{Float64}, lags::Int64)
    num_dep = size(beta, 1)
    num_indep = num_dep * lags
    tbp = zeros(num_indep, num_indep)  #beta's size is num_dep x num_indep
    if lags == 1
        tbp = @view beta[:, 1:num_indep]

    else
        tbp[1:num_dep, :] = @view beta[:, 1:num_indep]   #[1:num_indep, 1:num_dep]
        for i = 1:num_indep-num_dep
            tbp[i+num_dep, i] = 1
        end

    end

    return tbp

end


function IRF_matrix(beta::Matrix{Float64}, ahead::Int64, lags::Int64)
    num_dep = size(beta, 1)
    num_indep = num_dep * lags
    #iden = Matrix{Float64}(I, num_indep, num_indep)
    par1 = PAR1_matrix(beta, lags)  # size is num_indep x num_indep
    J = zeros(num_dep, num_indep)
    for i = 1:num_dep
        J[i, i] = 1
    end
    tbr = Vector{Matrix{Float64}}(undef, ahead)
    tbr[1] = Matrix{Float64}(I, num_dep, num_dep)
    temp = Matrix{Float64}(I, num_indep, num_indep)
    for i = 2:ahead
        temp = temp * par1

        new_mat = J * temp * J'
        ##writedlm("new_mat" * string(i) *".csv", new_mat)
        tbr[i] = new_mat #temp[1:num_dep, 1:num_dep]
    end
    return tbr
end


function vcov_residual(residual::Vector{Matrix{Float64}}, num_indep, is_na::Vector{Int64}, N::Int64)
    num_obs = length(is_na) - sum(is_na) #is_na is a binary vector width 1 indicating a missing value
    
    residual_height, residual_width= size(residual[1])
    

    nona_residual = zeros(residual_height, num_obs)
    new_col = 1
    for n=1:N
        for i = 1:residual_width # column by column 
            if is_na[(n-1)*residual_width+i] == 0
                nona_residual[:,new_col] = residual[n][:, i]
                new_col += 1
            end
        end
    end
    
    #centered = nona_residual .- mean(nona_residual, dims=2)
    cov = (nona_residual * nona_residual') / (num_obs - num_indep)
    
    return cov
end

function irf_method(method::String)
    
    internal_methord = lowercase(method)=="oirf" ?  IRF_oirf : IRF_girf
    return internal_methord
end

function irf(m::model, ahead::Int64, method::String="")
    if !isdefined(m, :regression_result)
        error("Model not fitted.")
    end    

    return irf(irf_method(method), m.residuals, m.cache_na_records, m.regression_result.beta, ahead, m.info.num_dep_lags, m.info.N)

end

function irf(method::Int64, residual::Vector{Matrix{Float64}}, is_na::Vector{Int64}, beta::Matrix{Float64}, ahead::Int64, lags::Int64, N::Int64)
    if method == IRF_girf
        return girf(residual, is_na, beta, ahead, lags, N)
    else
        return oirf(residual, is_na, beta, ahead, lags, N)
    end
end


function oirf(residual::Vector{Matrix{Float64}}, is_na::Vector{Int64}, beta::Matrix{Float64}, ahead::Int64, lags::Int64, N::Int64)
    ma_phi = IRF_matrix(beta, ahead, lags) 
    num_dep = size(residual[1], 1)
    cov = vcov_residual(residual, size(beta, 1), is_na, N)


    p = cholesky(cov).L 
    
    MA_Phi_P = Vector{Matrix{Float64}}(undef, ahead)
    for i0 = 1:ahead
        MA_Phi_P[i0] = ma_phi[i0] * p 
    end
    return calculate_irf_matrix(num_dep, ahead, MA_Phi_P)
end

function girf(residual::Vector{Matrix{Float64}}, is_na::Vector{Int64}, beta::Matrix{Float64}, ahead::Int64, lags::Int64, N::Int64)
    ma_phi = IRF_matrix(beta, ahead, lags)
    num_dep = size(residual[1], 1)
    cov = vcov_residual(residual, size(beta, 1), is_na, N)

    sigmas = [1 / sqrt(cov[i, i]) for i = 1:num_dep]
    MA_Phi_P = Vector{Matrix{Float64}}(undef, ahead)
    
    for i0 = 1:ahead

        temp_mat = ma_phi[i0] * cov
        
        for r = 1:size(temp_mat, 1)
            for col = 1:size(temp_mat, 2)
                
                temp_mat[r, col] = temp_mat[r, col] * sigmas[col]
            end
        end
        MA_Phi_P[i0] = temp_mat
    end
    return calculate_irf_matrix(num_dep, ahead, MA_Phi_P)
end

function calculate_irf_matrix(num_dep::Int64, ahead::Int64, MA_Phi_P::Vector{Matrix{Float64}})
    tbr = zeros(ahead, num_dep * num_dep)
    for i0 = 1:num_dep
        for i = 1:ahead
            tbr[i, (i0-1)*num_dep+1:i0*num_dep] = MA_Phi_P[i][i0, :]
        end
    end
    return tbr

end

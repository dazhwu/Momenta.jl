function stability_test(beta::Matrix{Float64}, lags::Int)
    tbp = PAR1_matrix(beta, lags)
    
    eivals = eigen(tbp).values
    return abs.(eivals)
    
end


function hansen_overid(W2_inv, zs, num_instru, num_indep, N)
    hansen_test = (zs * W2_inv * zs')  * (1.0 / N)
    df = num_instru - num_indep
    
    k2 = Chisq(df)
    crit = quantile(k2, 0.95)
    return Hansen_test_info(hansen_test[1], df,  1 - cdf(k2, hansen_test[1]), crit)
end


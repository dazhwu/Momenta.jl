
function print_summary(m::model, identifiers::Vector{String})
    the_options=m.options
    if the_options.steps == 2
        strSteps = "two-step "
    elseif the_options.steps == 1
        strSteps = "one-step "
    else
        strSteps = "m-step "
    end
    if the_options.level
        str_gmm = "system GMM"
    else
        str_gmm = "difference GMM"
    end
    to_print = []
    push!(to_print, " Dynamic Panel VAR Estimation, " * strSteps * str_gmm)
    # push!(to_print, basic_information(model))
    # push!(to_print, regression_table(model, reg_tables))
    # push!(to_print, test_results(model))
    for line in to_print
        println(line)
    end
    basic_information(m, the_options, identifiers)
    regression_table( m)
    test_results(m)
end


function basic_information(m::model, the_options::InternalOptions, identifiers::Vector{String})
    basic_table=Matrix{String}(undef, 4,3)
    fill!(basic_table, "")
    middle_space="        "
    basic_table[1,1]="Group variable: " * identifiers[1]
    basic_table[:,2:2] .=middle_space
    basic_table[1,3]="Number of obs = " * string(m.info.num_obs)
    basic_table[2,1]="Time variable: " * identifiers[2]
    #basic_table[2,2]=middle_space
    basic_table[2,3]="Min obs per group = " * string(m.info.min_obs)
    basic_table[3,1]="Number of instruments = " * string(m.info.z_width * m.info.num_dep) 
    basic_table[3,3]="Max obs per group = " * string(m.info.max_obs)
    basic_table[4,1]="Numbwer of groups = " * string(m.info.N)
    basic_table[4,3]="Avg obs per group = " * string(round(m.info.avg_obs; digits=2))

    pretty_table(basic_table, tf = tf_borderless; header=[" ", " ", " "])
end

function test_results(m::model)
    str_toprint = "Hansen test of overid. restrictions: chi(" * string(m.hansen.df) * ") = " * string(round(m.hansen.test_value, digits=3))
    str_toprint = str_toprint * " Prob > Chi2 = " * string(round(m.hansen.P_value, digits=3)) * '\n'
    if maximum(m.stability) <= 1
        str_toprint = str_toprint * "All the eigenvalues lie inside the unit circle." * '\n'
        str_toprint = str_toprint * "PVAR satisfies stability condition." * '\n'
    else
        str_toprint = str_toprint * "Not all the eigenvalues lie inside the unit circle." * '\n'
        str_toprint = str_toprint * "PVAR does not satisfy stability condition." * '\n'
    end
    println(str_toprint)
end


function regression_table(m::model)
    indeps = m.indeps
    num_rows = m.info.num_dep *(1+ m.info.num_indep)
    r_table = Matrix{String}(undef, num_rows, 7)
    fill!(r_table, " ")
    for i = 1:m.info.num_dep
        for j = 0:m.info.num_indep
            if j == 0
                r_table[i+(i-1)*m.info.num_indep,1]= m.deps[i].name
            else
                #r_table[1+j+(i-1)*m.info.num_indep, 2]= indep[j+1]
                the_var= indeps[j]
                prefix=the_var.lag>0 ? "L"*string(the_var.lag)*"." : ""
                r_table[j+i+(i-1)*m.info.num_indep, 2] = prefix * the_var.name 
                r_table[j+i+(i-1)*m.info.num_indep, 3] = string(round(m.regression_result.beta[i,j], digits=4))
                r_table[j+i+(i-1)*m.info.num_indep, 4] = string(round(m.regression_result.std_err[i,j], digits=4))
                r_table[j+i+(i-1)*m.info.num_indep, 5] = string(round(m.regression_result.z_stats[i,j], digits=4))
                r_table[j+i+(i-1)*m.info.num_indep, 6] = string(round(m.regression_result.p_values[i,j], digits=4))

                star=""
                if m.regression_result.p_values[i,j] <= 0.001
                    star="***"
                elseif m.regression_result.p_values[i,j] <= 0.01
                    star="**"
                elseif m.regression_result.p_values[i,j] <= 0.05
                    star="*"                
                end

                r_table[j+i+(i-1)*m.info.num_indep, 7] = star

            end
        end
    end
    pretty_table(r_table; header=["Equation", "   ", "coef.", "Corrected Std. Err.", "z", "P>|z|", " "],max_num_of_rows = -1, crop = :none)

end

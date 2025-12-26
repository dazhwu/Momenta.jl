function print_summary(m::model)
    
    the_options = m.options
    
    # Determine step type
    if the_options.steps == 2
        strSteps = "two-step "
    elseif the_options.steps == 1
        strSteps = "one-step "
    else
        strSteps = "m-step "
    end
    
    # Determine GMM type
    str_gmm = the_options.level ? "system GMM" : "difference GMM"
    
    # 1. Print title
    println("Dynamic Panel VAR Estimation, " * strSteps * str_gmm)
    
    # 2. Print basic information
    basic_information(m, the_options)
    
    # 3. Print regression results
    regression_table(m)
    
    # 4. Print test results
    test_results(m)
end

function basic_information(m::model, the_options::InternalOptions)
    identifiers=m.info.id_time
    basic_table = Matrix{String}(undef, 4, 3)
    fill!(basic_table, "")
    
    basic_table[1, 1] = "Group variable: " * identifiers[1]
    basic_table[1, 3] = "Number of obs = " * string(m.info.num_obs)
    basic_table[2, 1] = "Time variable: " * identifiers[2]
    basic_table[2, 3] = "Min obs per group = " * string(m.info.min_obs)
    basic_table[3, 1] = "Number of instruments = " * string(m.info.z_width * m.info.num_dep)
    basic_table[3, 3] = "Max obs per group = " * string(m.info.max_obs)
    basic_table[4, 1] = "Number of groups = " * string(m.info.N)
    basic_table[4, 3] = "Avg obs per group = " * string(round(m.info.avg_obs; digits=2))
    
    pretty_table(basic_table, tf = tf_borderless, header = ["", "", ""], 
                 show_header = false, alignment = :l)
end

function regression_table(m::model)
    indeps = m.indeps
    num_rows = m.info.num_dep * (1 + m.info.num_indep)
    r_table = Matrix{Any}(undef, num_rows, 7)
    fill!(r_table, "")
    
    for i = 1:m.info.num_dep
        for j = 0:m.info.num_indep
            row_idx = j + i + (i - 1) * m.info.num_indep
            
            if j == 0
                r_table[row_idx, 1] = m.deps[i].name
            else
                the_var = indeps[j]
                prefix = the_var.lag > 0 ? "L" * string(the_var.lag) * "." : ""
                r_table[row_idx, 2] = prefix * the_var.name
                r_table[row_idx, 3] = round(m.regression_result.beta[i, j], digits=4)
                r_table[row_idx, 4] = round(m.regression_result.std_err[i, j], digits=4)
                r_table[row_idx, 5] = round(m.regression_result.z_stats[i, j], digits=4)
                r_table[row_idx, 6] = round(m.regression_result.p_values[i, j], digits=4)
                
                # Star marking
                star = ""
                if m.regression_result.p_values[i, j] <= 0.001
                    star = "***"
                elseif m.regression_result.p_values[i, j] <= 0.01
                    star = "**"
                elseif m.regression_result.p_values[i, j] <= 0.05
                    star = "*"
                end
                r_table[row_idx, 7] = star
            end
        end
    end
    
    pretty_table(r_table, 
                 header = ["Equation", "", "coef.", "Corrected Std. Err.", "z", "P>|z|", ""],
                 alignment = [:l, :l, :r, :r, :r, :r, :c],
                 max_num_of_rows = -1,
                 crop = :none)
end

function test_results(m::model)
    str_toprint = "Hansen test of overid. restrictions: chi(" * string(m.hansen.df) * ") = " * 
                  string(round(m.hansen.test_value, digits=3))
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

# If exporting to LaTeX or HTML is needed, use these functions:
function export_summary_table(m::model, identifiers::Vector{String})
    """Return the complete table data matrix for export"""
    return build_complete_table(m, identifiers)
end

function build_complete_table(m::model, identifiers::Vector{String})
    the_options = m.options
    
    # Determine step type and GMM type
    if the_options.steps == 2
        strSteps = "two-step "
    elseif the_options.steps == 1
        strSteps = "one-step "
    else
        strSteps = "m-step "
    end
    str_gmm = the_options.level ? "system GMM" : "difference GMM"
    
    # Build all rows
    all_rows = []
    
    # Title
    push!(all_rows, ["Dynamic Panel VAR Estimation, " * strSteps * str_gmm, "", "", "", "", "", ""])
    push!(all_rows, ["", "", "", "", "", "", ""])
    
    # Basic information
    push!(all_rows, ["Group variable: " * identifiers[1], "", "Number of obs = " * string(m.info.num_obs), "", "", "", ""])
    push!(all_rows, ["Time variable: " * identifiers[2], "", "Min obs per group = " * string(m.info.min_obs), "", "", "", ""])
    push!(all_rows, ["Number of instruments = " * string(m.info.z_width * m.info.num_dep), "", "Max obs per group = " * string(m.info.max_obs), "", "", "", ""])
    push!(all_rows, ["Number of groups = " * string(m.info.N), "", "Avg obs per group = " * string(round(m.info.avg_obs; digits=2)), "", "", "", ""])
    push!(all_rows, ["", "", "", "", "", "", ""])
    
    # Regression table header
    push!(all_rows, ["Equation", "Variable", "coef.", "Corrected Std. Err.", "z", "P>|z|", ""])
    
    # Regression results
    indeps = m.indeps
    for i = 1:m.info.num_dep
        for j = 0:m.info.num_indep
            if j == 0
                push!(all_rows, [m.deps[i].name, "", "", "", "", "", ""])
            else
                the_var = indeps[j]
                prefix = the_var.lag > 0 ? "L" * string(the_var.lag) * "." : ""
                var_name = prefix * the_var.name
                
                coef = string(round(m.regression_result.beta[i,j], digits=4))
                std_err = string(round(m.regression_result.std_err[i,j], digits=4))
                z_stat = string(round(m.regression_result.z_stats[i,j], digits=4))
                p_val = string(round(m.regression_result.p_values[i,j], digits=4))
                
                star = ""
                if m.regression_result.p_values[i,j] <= 0.001
                    star = "***"
                elseif m.regression_result.p_values[i,j] <= 0.01
                    star = "**"
                elseif m.regression_result.p_values[i,j] <= 0.05
                    star = "*"
                end
                
                push!(all_rows, ["", var_name, coef, std_err, z_stat, p_val, star])
            end
        end
    end
    
    # Test results
    push!(all_rows, ["", "", "", "", "", "", ""])
    push!(all_rows, ["Hansen test of overid. restrictions: chi(" * string(m.hansen.df) * ") = " * 
                     string(round(m.hansen.test_value, digits=3)) * 
                     " Prob > Chi2 = " * string(round(m.hansen.P_value, digits=3)), "", "", "", "", "", ""])
    
    if maximum(m.stability) <= 1
        push!(all_rows, ["All the eigenvalues lie inside the unit circle.", "", "", "", "", "", ""])
        push!(all_rows, ["PVAR satisfies stability condition.", "", "", "", "", "", ""])
    else
        push!(all_rows, ["Not all the eigenvalues lie inside the unit circle.", "", "", "", "", "", ""])
        push!(all_rows, ["PVAR does not satisfy stability condition.", "", "", "", "", "", ""])
    end
    
    # Convert to matrix
    output_matrix = Matrix{String}(undef, length(all_rows), 7)
    for (i, row) in enumerate(all_rows)
        output_matrix[i, :] = row
    end
    
    return output_matrix
end

# LaTeX 
function export_latex(m::model,  filename::String)
    identifiers=m.info.id_time
    table_data = export_summary_table(m, identifiers)
    open(filename, "w") do io
        pretty_table(io, table_data, backend = Val(:latex), 
                     header = ["", "", "", "", "", "", ""],
                     show_header = false)
    end
end

# HTML
function export_html(m::model,  filename::String)
    identifiers=m.info.id_time
    table_data = export_summary_table(m, identifiers)
    open(filename, "w") do io
        pretty_table(io, table_data, backend = Val(:html), 
                     header = ["", "", "", "", "", "", ""],
                     show_header = false)
    end
end

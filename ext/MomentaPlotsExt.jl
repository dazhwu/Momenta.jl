module MomentaPlotsExt

using Plots
using Momenta

function Momenta.plot_irf(m::Momenta.model, boot_res::Momenta.bootstrap_results)
    
    num_vars = m.info.num_dep
    # Assume irf_len can be directly obtained from data dimensions, or from m.options
    irf_len = size(boot_res.lower, 1) 
    x = 1:irf_len
    
    # Get variable names
    var_names = [v.name for v in m.deps[1:num_vars]]

    # === 1. Initialize an array to store subplots in order ===
    # Plots.jl fills by row by default, so our outer loop is Row (Response), inner is Col (Impulse)
    plot_list = Vector{Plots.Plot}()

    plots_dict = Dict{String, Any}()

    for affected = 1:num_vars      # Row: Response (affected)
        for affecting = 1:num_vars # Col: Impulse (impulse)
            
            idx = (affected-1)*num_vars + affecting
            
            # Prepare data
            y = boot_res.irf[:, idx] 
            y_lower = boot_res.lower[:, idx]
            y_upper = boot_res.upper[:, idx]

            # Calculate the Y-axis range (for aesthetics, leave some margin)
            max_val = maximum(y_upper)
            min_val = minimum(y_lower)
            padding = (max_val - min_val) * 0.1
            if padding == 0 padding = 0.1 end
            y_lims = (min(min_val, 0) - padding, max_val + padding)

            # Construct title: Show impulse var only in first row, response var in first column, or simply show in all
            # For clarity, each subplot shows a short title
            title_str = "$(var_names[affecting]) → $(var_names[affected])"
            key_str = "$(var_names[affecting]) on $(var_names[affected])"

            # Generate the individual plot
            p = plot(x, y, 
                label="",          # Remove label to avoid legend in each subplot
                lw=2, 
                color=:blue,
                ribbon=(y .- y_lower, y_upper .- y), 
                fillalpha=0.2,
                fillcolor=:blue,
                title=title_str,
                titlefontsize=10,
                framestyle=:box,     # Recommend box style for better combined layout
                grid=:true,
                gridalpha=0.3,
                gridlinestyle=:dot,
                ylims=y_lims,
                xlims=(1, irf_len),
                # Slightly adjust margins
                margin=2Plots.mm 
            )

            # Decoration: zero line
            hline!(p, [0], color=:black, lw=1, linestyle=:dash, label="")

            # Add to list
            push!(plot_list, p)
            plots_dict[key_str] = p
        end
    end

    # === 2. Auto-compute full figure size ===
    # Assume each subplot width 300px, height 200px
    total_width = 300 * num_vars
    total_height = 200 * num_vars

    # === 3. Generate full grid of plots ===
    # layout = (number of rows, number of columns)
    full_plot = plot(plot_list..., 
        layout = (num_vars, num_vars), 
        size = (total_width, total_height),
        plot_title = "Impulse Response Functions (Rows: Response, Cols: Impulse)",
        plot_titlefontsize = 12
    )

    # === 4. Automatically display ===
    display(full_plot)
    plots_dict["full"]=full_plot

    # Return the full plot object (convenient for users to savefig(p, "irf.png"))
    return plots_dict
end

end
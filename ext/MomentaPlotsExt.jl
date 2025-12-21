module MomentaPlotsExt

using Plots
using Momenta

function Momenta.plot_irf(m::Momenta.model, boot_res::Momenta.bootstrap_results)
    
    num_vars = m.info.num_dep
    # 假设 irf_len 可以从数据维度直接获取，或者从 m.options 获取
    irf_len = size(boot_res.lower, 1) 
    x = 1:irf_len
    
    # 获取变量名
    var_names = [v.name for v in m.deps[1:num_vars]]

    # === 1. 初始化数组来存储按顺序排列的子图 ===
    # Plots.jl 默认按行填充，所以我们外层循环是 Row (Response), 内层是 Col (Impulse)
    plot_list = Vector{Plots.Plot}()

    plots_dict = Dict{String, Any}()

    for affected = 1:num_vars      # Row: Response (受影响)
        for affecting = 1:num_vars # Col: Impulse (冲击)
            
            idx = (affected-1)*num_vars + affecting
            
            # 准备数据
            y = boot_res.irf[:, idx] 
            y_lower = boot_res.lower[:, idx]
            y_upper = boot_res.upper[:, idx]

            # 计算 Y 轴范围 (为了美观，稍微留点白)
            max_val = maximum(y_upper)
            min_val = minimum(y_lower)
            padding = (max_val - min_val) * 0.1
            if padding == 0 padding = 0.1 end
            y_lims = (min(min_val, 0) - padding, max_val + padding)

            # 构建标题: 仅在第一行显示冲击变量，仅在第一列显示响应变量，或者简单点每个都显示
            # 这里为了清晰，每个子图都显示简短标题
            title_str = "$(var_names[affecting]) → $(var_names[affected])"
            key_str = "$(var_names[affecting]) on $(var_names[affected])"

            # 生成单张图
            p = plot(x, y, 
                label="",          # 去掉 label 避免图例占地
                lw=2, 
                color=:blue,
                ribbon=(y .- y_lower, y_upper .- y), 
                fillalpha=0.2,
                fillcolor=:blue,
                title=title_str,
                titlefontsize=10,
                framestyle=:box,     # 推荐用 box 风格，拼图更好看
                grid=:true,
                gridalpha=0.3,
                gridlinestyle=:dot,
                ylims=y_lims,
                xlims=(1, irf_len),
                # 稍微调整一下边距
                margin=2Plots.mm 
            )

            # 装饰: 零线
            hline!(p, [0], color=:black, lw=1, linestyle=:dash, label="")

            # 加入列表
            push!(plot_list, p)
            plots_dict[key_str] = p
        end
    end

    # === 2. 自动计算大图尺寸 ===
    # 假设每个子图宽 300px, 高 200px
    total_width = 300 * num_vars
    total_height = 200 * num_vars

    # === 3. 生成整张大图 ===
    # layout = (行数, 列数)
    full_plot = plot(plot_list..., 
        layout = (num_vars, num_vars), 
        size = (total_width, total_height),
        plot_title = "Impulse Response Functions (Rows: Response, Cols: Impulse)",
        plot_titlefontsize = 12
    )

    # === 4. 自动显示 ===
    display(full_plot)
    plots_dict["full"]=full_plot

    # 返回大图对象 (方便用户 savefig(p, "irf.png"))
    return plots_dict
end

end
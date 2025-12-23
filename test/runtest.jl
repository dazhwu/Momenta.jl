using Test
using Momenta
using CSV
using DataFrames
using Plots

# 获取当前文件所在的路径，定位到 csv 文件
const DATA_FILE = joinpath(@__DIR__, "data", "dahlberg_data.csv")

@testset "Momenta.jl" begin

    @testset "Quick Start Example" begin
        # 1. 确保文件存在
        @test isfile(DATA_FILE)

        # 2. 这里把 README 里的代码复制进来运行一遍
        
        df = CSV.read(DATA_FILE, DataFrame)


        m = Momenta.fit(df,
            ["id", "year"],
            "n w  ~ lag(n, 1:2) lag(w, 1:2) k",
            "GMM(n w ,2:4) IV(k)",
            ""
        )

        irf = Momenta.irf(m, 8)

        bootstrap_result=Momenta.bootstrap(m, 8, 1000, "girf")
        all_plots = Momenta.plot_irf(m, bootstrap_result)

        println("Quick Start example ran successfully!")
    end


    # @testset "Internal Functions" begin

    # end
end
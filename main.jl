using DataFrames
using CSV
using Dates
#using BenchmarkTools

using Plots
using Momenta

#include("./src/PanelVar.jl")


df=CSV.read("data.csv", DataFrame)

#@time m=Momenta.regress(df, "n ", ["id", "year"],2, "w","k"," nolevel collapse  oirf", 8, 200);
#m=Momenta.fit(df, "n  ~ lag(n, 1:2) lag(w, 1:2) lag(k, 1:2)", ["id", "year"],  "GMM(n w ,2:4) IV(k)", "nolevel")
#start_t=now()

#m=Momenta.fit(df, "n  ~ lag(n, 1:2) ", ["id", "year"],  "GMM(n ,2:4)", "fod")
m = Momenta.fit(df, 
        ["id", "year"],  
        "n w  ~ lag(n, 1:2) lag(w, 1:2) k", 
        "GMM(n w ,2:4) IV(k)", 
        "fod" 
)


#println(now()-start_t)

irf = Momenta.irf(m, 8)
bootstrap_result=Momenta.bootstrap(m, 8, 200)

all_plots=Momenta.plot_irf(m, bootstrap_result)
display(all_plots["n on w"])


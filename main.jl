using DataFrames
using CSV
using Dates
#using BenchmarkTools

using Plots
using Momenta

#include("./src/PanelVar.jl")


df=CSV.read("dahlberg_data.csv", DataFrame)

#@time m=Momenta.regress(df, "n ", ["id", "year"],2, "w","k"," nolevel collapse  oirf", 8, 200);
#m=Momenta.fit(df, "n  ~ lag(n, 1:2) lag(w, 1:2) lag(k, 1:2)", ["id", "year"],  "GMM(n w ,2:4) IV(k)", "nolevel")
start_t=now()

#m=Momenta.fit(df, "n  ~ lag(n, 1:2) ", ["id", "year"],  "GMM(n ,2:4)", "fod")
for i in 1:100
        Momenta.fit(df, 
                ["id", "year"],  
                "n w  ~ lag(n, 1:2) lag(w, 1:2) k", 
                "GMM(n w ,2:4) IV(k)", 
                "" 
        )
end

println(now()-start_t) #2.1 seconds. Mac 1.57 Win 3.0" 


m=Momenta.fit(df, 
        ["id", "year"],  
        "n w  ~ lag(n, 1:2) lag(w, 1:2) k", 
        "GMM(n w ,2:4) IV(k)", 
        "" 
)

Momenta.print_summary(m)
Momenta.export_html(m,"2.html")

irf = Momenta.irf(m, 8)

start_t=now()
bootstrap_result=Momenta.bootstrap(m, 8, 1000, "girf")
println(now()-start_t) #1.8 sec. Mac: 1.9 sec  Win: 2.5 sec
all_plots=Momenta.plot_irf(m, bootstrap_result)
display(all_plots["n on w"])


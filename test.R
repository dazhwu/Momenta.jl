library(panelvar)
library(data.table)

df <- fread("data.csv")
df <- as.data.frame(df)


start_time <- Sys.time()
for (i in 1:100) {
  p1 <- pvargmm(
    dependent_vars = c("n", "w"),
    lags = 2,
    exog_vars = c("k"),
    transformation = "fd",
    data = df,
    panel_identifier = c("id", "year"),
    steps = c("twostep"),
    system_instruments = TRUE, 
    system_constant = TRUE,
    max_instr_dependent_vars = 4,
    min_instr_dependent_vars = 2,
    min_instr_predet_vars = 1L,
    max_instr_predet_vars = 5L,
    # collapse = TRUE,
    progressbar = FALSE
  )

}
summary(p1)
time_diff_secs <- difftime(Sys.time(), start_time, units = "secs")

print(time_diff_secs)

start_time <- Sys.time()
p1_boot=bootstrap_irf(p1, "GIRF",8, 1000, 0.95, 1)
time_diff_secs <- difftime(Sys.time(), start_time, units = "secs")

print(time_diff_secs)


# > source("/home/laohu/Seafile/Momenta.jl/test.R", encoding = "UTF-8")
# Welcome to panelvar! Please cite our package in your publications -- see citation("panelvar")
# data.table 1.17.8 using 4 threads (see ?getDTthreads).  Latest news: r-datatable.com
# Time difference of 523.4 secs  windows: 621.2 secs Mac: 209.7 secs
# Time difference of 1756.1 secs. Windows: 7510.8 secs Mac 2609.9 secs
#Error in parallel::mclapply(1:nof_Nstar_draws, function(i0) { : 
#  'mc.cores' > 1 is not supported on Windows

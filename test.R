library(panelvar)
library(data.table)
df <- fread("data.csv")

start_time <- Sys.time()
p1 <- pvargmm(
  dependent_vars = c("n", "w", "k"),
  lags = 2,
  #predet_vars = c("w","k"),
  #exog_vars=c("revenues"),
  transformation = "fod",
  data = df,
  panel_identifier = c("id", "year"),
  steps = c("twostep"),
  system_instruments =FALSE, #TRUE, #FALSE, #TRUE,
  system_constant = TRUE,
  max_instr_dependent_vars = 4,
  min_instr_dependent_vars = 2,
  #max_instr_predet_vars = 3,
  #min_instr_dependent_vars = 1L,
  min_instr_predet_vars = 1L,
  max_instr_predet_vars = 5L,
  #collapse = TRUE,
  progressbar = TRUE
)
summary(p1)


time_diff_secs <- difftime(Sys.time(), start_time, units = "secs")

print(time_diff_secs)

# p1_boot=bootstrap_irf(p1, "OIRF",8,200, 0.95,4)
# time_diff_secs <- difftime(Sys.time(), start_time, units = "secs")

# print(time_diff_secs)
# the_oirf=oirf(p1, n.ahead=8)
# plot(the_oirf, p1_boot)
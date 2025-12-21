using LinearAlgebra
using BenchmarkTools
using Random
using Printf

# Set random seed to ensure reproducibility of results each time
Random.seed!(1234)

# --- Test parameters ---
N = 500          # Matrix dimension (500x500)
Iterations = 5   # Number of loop iterations (for checking numerical differences)

println("=== Begin test: pinv(A) vs pinv(Symmetric(A)) ===")
println("Matrix size: $N x $N\n")

# --- 1. Numerical consistency test ---
println("--- 1. Numerical accuracy check (loop $Iterations times) ---")
max_diff = 0.0

for i in 1:Iterations
    # 1. Generate a random symmetric matrix
    # First generate a random matrix, then force symmetry via A + A'
    local A_raw = rand(N, N)
    local A_sym_val = A_raw + A_raw' 
    
    # 2. Method A: Ordinary pinv (uses SVD)
    # Julia does not know it is symmetric, so treats it as a general matrix
    P1 = pinv(A_sym_val)
    
    # 3. Method B: Symmetric wrapper (uses Eigen)
    # Explicitly tell Julia it is symmetric
    P2 = pinv(Symmetric(A_sym_val))
    
    # 4. Compute difference (Frobenius norm)
    diff = norm(P1 - P2)
    global max_diff = max(max_diff, diff)
end

@printf "Maximum numerical difference (Norm): %.2e\n" max_diff
if max_diff < 1e-12
    println("Conclusion: Results are almost identical (numerical error can be ignored).\n")
else
    println("Conclusion: There are minor numerical differences (this is normal, as algorithms differ).\n")
end

# --- 2. Performance benchmark test ---
println("--- 2. Performance benchmark test (BenchmarkTools) ---")

# Prepare a fixed matrix for benchmarking
A_bench = rand(N, N)
A_bench = A_bench + A_bench'

println("\nTesting: pinv(A) [general matrix -> SVD algorithm] ...")
# @btime runs the code many times and takes the minimum, very accurate
t1 = @benchmark pinv($A_bench)
display(t1)

println("\nTesting: pinv(Symmetric(A)) [symmetric matrix -> Eigen algorithm] ...")
t2 = @benchmark pinv(Symmetric($A_bench))
display(t2)

# --- 3. Summary ---
t1_med = median(t1.times)
t2_med = median(t2.times)
speedup = t1_med / t2_med

println("\n=== Final results ===")
@printf "General pinv time: %.2f ms\n" (t1_med / 1e6)
@printf "Symmetric pinv time: %.2f ms\n" (t2_med / 1e6)
@printf "🚀 Speedup factor: %.2f x\n" speedup
println("===========================================")
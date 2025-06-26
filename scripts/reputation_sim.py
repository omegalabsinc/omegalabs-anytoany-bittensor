import numpy as np

# Parameters
alpha = 0.01
rep_0 = 0.5
epsilon_values = [ 0.05, 0.01, 0.001, 0.0001]  # For 95%, 99%, 99.9%

# Function to compute number of intervals needed to reach (1 - epsilon)
def compute_intervals(alpha, rep_0, epsilon):
    target = 1 - epsilon
    lhs = np.log(epsilon / (1 - rep_0))
    rhs = np.log(1 - alpha)
    t = lhs / rhs
    return int(np.ceil(t))

# Compute for different thresholds
results = {
    f"{100 * (1 - eps)}% target - days": compute_intervals(alpha, rep_0, eps) *40/(60*24)
    for eps in epsilon_values
}

print(results)

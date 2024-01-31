import numpy as np
import random

# defining the intervals for Q2
intervals = [[0, 2], [4.5, 10]]

# weighted random selection
def choose_random_from_intervals(intervals):
    return random.uniform(*random.choices(intervals, weights=[r[1]-r[0] for r in intervals])[0])

def generate_x_vals(p_val, num_samples):
    x = np.zeros(num_samples)
    bernoulli_trials = np.random.binomial(1, p_val, num_samples)
    for j in range(num_samples):
        z = bernoulli_trials[j]
        x1 = np.random.uniform(2, 4.5) 
        x2 = choose_random_from_intervals(intervals) 
        x[j] = z * x1 + (1 - z) * x2
    return x

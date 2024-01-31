import torch
import numpy as np
from data_generation import generate_x_vals
from density_ratio_estimation import train_logistic_regression, estimate_density_ratio
from training import train_standard_model, train_RU_model
import matplotlib.pyplot as plt

# constants
P_TRAIN = 0.1
P_TEST = 0.9
NUM_PTS = 100
gamma_values = [2, 6, 18]

# data Generation
x_train = generate_x_vals(P_TRAIN, NUM_PTS)
x_test = generate_x_vals(P_TEST, NUM_PTS)
X_train = torch.from_numpy(x_train).float().reshape(-1, 1)
Y_train = torch.from_numpy(np.log10(x_train) + np.sin(x_train) + np.sqrt(2 * x_train)).float().reshape(-1, 1)
X_test = torch.from_numpy(x_test).float().reshape(-1, 1)
Y_test = torch.from_numpy(np.log10(x_test) + np.sin(x_test) + np.sqrt(x_test)).float().reshape(-1, 1)

# train logistic regression for density ratio estimation
classifier = train_logistic_regression(x_train, x_test)

# train standard model
std_model, std_loss = train_standard_model(X_train, Y_train, classifier)

# train RU models
ru_models, ru_loss = train_RU_model(X_train, Y_train, gamma_values, classifier)

# plotting standard model loss
plt.figure(figsize=(10, 5))
plt.plot(std_loss, label='Standard Model Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Standard Model Training Loss')
plt.legend()
plt.show()

# plotting RU model loss
plt.figure(figsize=(10, 5))
for gamma, loss in ru_loss.items():
    plt.plot(loss, label=f'RU Model Loss (Gamma={gamma})')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('RU Model Training Loss')
plt.legend()
plt.show()

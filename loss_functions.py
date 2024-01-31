import torch

def weighted_mse_loss(input, target, weight):
    squared_errors = (input - target) ** 2
    weighted_squared_errors = squared_errors * weight
    return weighted_squared_errors.mean()

def RU_loss(z, a, gamma, y, weights):
    L = (z - y) ** 2
    weighted_RU_loss = (
        (L / gamma)
        + (1 - (1 / gamma)) * a
        + (gamma - (1 / gamma)) * torch.nn.functional.relu(L - a)
    ) * weights
    return torch.mean(weighted_RU_loss)


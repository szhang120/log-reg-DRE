import torch
import torch.nn as nn

def standard_model():
    return nn.Sequential(nn.Linear(1, 1))

def h_net():
    return nn.Sequential(nn.Linear(1, 1))

def a_net():
    return nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

import torch
import torch.optim as optim
from models import standard_model, h_net, a_net
from loss_functions import weighted_mse_loss, RU_loss

def train_standard_model(X_train, Y_train, classifier, n_epochs=100, batch_size=10):
    model = standard_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    LOSS = []

    for epoch in range(n_epochs):
        for i in range(0, len(X_train), batch_size):
            Xbatch = X_train[i:i+batch_size]
            ybatch = Y_train[i:i+batch_size]

            density_ratios = torch.tensor([estimate_density_ratio(x, classifier) for x in Xbatch.numpy()], dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(Xbatch)
            loss = weighted_mse_loss(y_pred, ybatch, density_ratios)
            loss.backward()
            optimizer.step()

            LOSS.append(loss.item())

    return model, LOSS

def train_RU_model(X_train, Y_train, gamma_values, classifier, n_epochs=100, batch_size=10):
    LOSS = {}
    models = {}

    for gamma in gamma_values:
        LOSS[gamma] = []
        h_model = h_net()
        a_model = a_net()
        optimizer = optim.Adam(list(h_model.parameters()) + list(a_model.parameters()), lr=1e-2)

        for epoch in range(n_epochs):
            for i in range(0, len(X_train), batch_size):
                Xbatch = X_train[i:i+batch_size]
                ybatch = Y_train[i:i+batch_size]

                density_ratios = torch.tensor([estimate_density_ratio(x, classifier) for x in Xbatch.numpy()], dtype=torch.float32)

                optimizer.zero_grad()
                h_pred = h_model(Xbatch)
                a_pred = a_model(Xbatch)
                loss = RU_loss(h_pred, a_pred, gamma, ybatch, density_ratios)
                loss.backward()
                optimizer.step()

                LOSS[gamma].append(loss.item())

        models[gamma] = (h_model, a_model)

    return models, LOSS


import torch
from torch.utils.data import DataLoader
from deep_ensemble.training.losses import gaussian_nll
from deep_ensemble.models.mve_model import MVEModel

def train_single_model(model, train_loader, epochs=30, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            mu, var = model(batch_x)
            loss = gaussian_nll(mu, var, batch_y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def train_ensemble(num_models, input_dim, train_dataset, hidden_dim=128, **kwargs):
    """
    Train K independent networks.
    """
    ensemble = []
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for i in range(num_models):
        print(f"Training model {i+1}/{num_models}...")
        model = MVEModel(input_dim, hidden_dim)
        model = train_single_model(model, train_loader, **kwargs)
        ensemble.append(model)

    return ensemble

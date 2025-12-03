import torch
import numpy as np
import matplotlib.pyplot as plt

from deep_ensemble.models.mve_model import MVEModel
from deep_ensemble.inference.ensemble_predict import ensemble_predict

def load_ensemble(num_models, input_dim=1):
    ensemble = []
    for i in range(num_models):
        model = MVEModel(input_dim)
        model.load_state_dict(torch.load(f"model_{i}.pth", map_location="cpu"))
        model.eval()
        ensemble.append(model)
    return ensemble

if __name__ == "__main__":
    ensemble = load_ensemble(5)
    
    xs = torch.linspace(-3, 3, 200).view(-1, 1)

    mu, var, epistemic, aleatoric = ensemble_predict(ensemble, xs)

    mu = mu.squeeze().numpy()
    std = var.sqrt().squeeze().numpy()
    xs_np = xs.squeeze().numpy()

    plt.plot(xs_np, mu, label="Predictive mean")
    plt.fill_between(xs_np, mu - 2*std, mu + 2*std, alpha=0.3, label="Uncertainty")
    plt.legend()
    plt.show()

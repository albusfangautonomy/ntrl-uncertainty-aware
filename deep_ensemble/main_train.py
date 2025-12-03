import numpy as np
from deep_ensemble.training.train_ensemble import train_ensemble
from deep_ensemble.utils.data_utils import SimpleDataset

def synthetic_data():
    X = np.linspace(-3, 3, 400).reshape(-1, 1)
    y = np.sin(X) + 0.2*np.random.randn(*X.shape)
    return X, y

if __name__ == "__main__":
    X, y = synthetic_data()
    dataset = SimpleDataset(X, y)

    ensemble = train_ensemble(
        num_models=5,
        input_dim=1,
        train_dataset=dataset,
        epochs=40,
        lr=1e-3,
        hidden_dim=128,
        device="cpu"
    )

    # Save ensemble
    for idx, model in enumerate(ensemble):
        torch.save(model.state_dict(), f"model_{idx}.pth")

    print("Training complete!")

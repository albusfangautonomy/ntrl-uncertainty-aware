import torch

@torch.no_grad()
def ensemble_predict(ensemble, x):
    """
    Combine ensemble predictions to compute predictive mean and uncertainty.

    Returns:
        mu_pred:     predictive mean
        var_total:   total uncertainty = epistemic + aleatoric
        epistemic:   variance across ensemble means
        aleatoric:   average predicted variance
    """
    mus = []
    vars_ = []

    for model in ensemble:
        mu, var = model(x)
        mus.append(mu)
        vars_.append(var)

    mus = torch.stack(mus)        # (K, batch, 1)
    vars_ = torch.stack(vars_)    # (K, batch, 1)

    # Predictive mean
    mu_pred = mus.mean(dim=0)

    # Epistemic uncertainty = variance of ensemble means
    epistemic = mus.var(dim=0)

    # Aleatoric uncertainty = mean of predicted variances
    aleatoric = vars_.mean(dim=0)

    # Total uncertainty
    var_total = epistemic + aleatoric

    return mu_pred, var_total, epistemic, aleatoric

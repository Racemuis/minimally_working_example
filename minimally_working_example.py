# Common libraries
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt

# Used for GPs
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize

# Set seeds
np.random.seed(0)
torch.manual_seed(0)


def main():
    """
    Test a GP model on (approximately) the data from Figure 5.11.

    Returns:
        None.
    """
    # Set hyperparameters
    n_samples = 58
    n_random = 8
    n_edge = 8

    sampler = qmc.Sobol(d=1, )

    # Create data domain (append an axis to create an array of shape [n_samples, 1] format requested by my GP)
    x_train = np.linspace(0, 1, n_samples)[:, np.newaxis]

    # Compute loc and scale as functions of input data
    loc = np.sin(2 * np.pi * x_train)

    # Create the noisy training outcomes
    x_initial = sampler.random(8).squeeze()  # The initial samples spaced according to a Sobol sequence.
    x_edge = np.random.normal(0.02, 0.005, size=n_edge)  # The samples around the edge of the domain.
    x_eval = np.random.normal(0.25, 0.01, size=n_samples-n_random-n_edge)  # The blob around the maximum.

    # Combine the data
    x_sample = np.concatenate((x_initial, x_eval, x_edge))[:, np.newaxis]
    y_sample = np.random.normal(np.sin(2 * np.pi * x_sample), np.abs(0.5 * np.sin(x_sample * 2 * np.pi)))

    # Fit the Gaussian process [paste your own models here :)]
    model = SingleTaskGP(train_X=torch.tensor(x_sample), train_Y=torch.tensor(y_sample),
                         input_transform=Normalize(d=1), outcome_transform=Standardize(m=1))

    # Get posterior mean and std
    mu = model.posterior(torch.tensor(x_train)).mean.detach().numpy().squeeze()
    std = np.sqrt(model.posterior(torch.tensor(x_train)).variance.detach().numpy()).squeeze()

    # Plot the distribution
    fig, axes = plt.subplots(1, 1, figsize=(8, 3))
    axes.plot(x_train, mu, label="$\mu(x)$", color="tab:blue",)
    axes.plot(x_train, loc, label="$f(x)$", color="tab:orange", linestyle='--', )
    axes.scatter(x_sample, y_sample, label="Observed data, y(x)", color="orange", s=10)
    axes.fill_between(
        x_train.squeeze(),
        mu - 1.96 * std, mu + 1.96 * std,
        color="tab:blue",
        alpha=0.2,
        label="95% confidence interval"
    )
    axes.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import torch
import torch.distributions as D
import torch.nn as nn
from numpy.typing import ArrayLike
from torch import Tensor


class NoisingLookAheadForwardKernel(nn.Module):
    def __init__(self, betas: ArrayLike, t: int):
        """Initialize a look-ahead forward (noising) kernel. Given a noise
        schedule defined by the betas, returns and samples from
        N(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I), i.e.
        Eq. 4 of DDPM paper.

        Args:
            betas: a list of scalars
            t: integer of what stage we are looking ahead to in the noising process
        """
        super().__init__()
        self.betas = betas
        if type(self.betas) is not Tensor:
            self.betas = torch.tensor(self.betas)
        self.t = t
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _realize(self, x_0):
        qxt = D.Normal(
            torch.sqrt(self.alpha_bars[self.t]) * x_0, (1 - self.alpha_bars[self.t])
        )
        return qxt

    def sample(self, x_0, rparam: bool = True):
        qxt = self._realize(x_0)
        if rparam:
            shaped_zeros = torch.zeros(qxt.loc.shape)
            shaped_ones = torch.ones(qxt.scale.shape)
            std_gaussian = D.Normal(shaped_zeros, shaped_ones)
            noise = std_gaussian.sample().to(x_0.device)
            samples = qxt.loc + noise * qxt.scale
            return noise, samples
        else:
            return qxt.sample()


class DenoisingOneStepBackwardKernel(nn.Module):
    def __init__(self, t: int, mean_network: nn.Module, cov_network: nn.Module):
        """Initialize a stage_t backward (denoising) kernel.

        Args:
            t: stage of the process, passed to the mean and cov networks.
            mean_network: neural network taking in x_t and stage t and returning mean for distribution on x_{t-1}
            cov_network: same but for covariance (we use even simpler suggestion of DDPM)
        """
        super().__init__()
        self.t = t
        self.mean_network = mean_network
        self.cov_network = cov_network

    def _realize(self, x_t: Tensor):
        mean = self.mean_network.predict_mean(x_t, self.t)
        cov = self.cov_network(x_t, self.t)
        qxtminus1 = D.Normal(mean, cov)
        return qxtminus1

    def sample(self, x_t: Tensor):
        qxtminus1 = self._realize(x_t)
        return qxtminus1.sample()


class CondCovariance(nn.Module):
    def __init__(self, betas, naive=True):
        """Naive conditional covariance module. No learnable parameters.
        Given a stage $t$ and observatoin $x_t$, returns sigma_t^2 I for a
        choice of sigma_t^2 depending only on $t$ and the beta_t values of the forward noising process.

        naive=True: \sigma_t^2 = \beta_t
        naive=False: \sigma_t^2 = (1-\bar{\alpha}_{t-1})/(1-\bar{\alpha}_t) * \beta_t

        Args:
            betas: array-like of scalars
            naive: flag determining what \sigma_t^2 to return as above
        """
        super().__init__()
        self.betas = betas
        self.naive = naive
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x_t, t):
        if self.naive:
            return torch.sqrt(self.betas)[t]
        else:
            sigma2 = (
                (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            )
            return torch.sqrt(sigma2)


class CondMean(nn.Module):
    def __init__(
        self,
        betas,
        shape: torch.Size,
        hidden_dim: int = 256,
        n_hidden_layers: int = 4,
        net=None,
    ):
        """Conditional mean module. Parameterizes the mean of x_{t-1} given x_t
        via a neural network that takes t and x_t as inputs.

        Args:
            betas: array-like a betas for this model
            shape: relevant shape of x_t
            hidden_dim: hidden dimension of network
            n_hidden_layers: number of hidden layers in network
            net: if not None, user-supplied conditional network, used
                instead of dense network.
        """
        super().__init__()
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.shape = shape
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        if not net:
            layers = [nn.Linear(shape[-1], hidden_dim), nn.ReLU()]
            for _ in range(n_hidden_layers):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers += [nn.Linear(hidden_dim, shape[-1]), nn.ReLU()]
            self.network = nn.Sequential(*layers)
        else:
            self.network = net

    def _mu_tilde(self, x_t, x_0, t):
        """Computes \tilde{\mu}_t (Eq. 6 of DDPM)

        Args:
            x_t: observation at time t
            x_0: observation at time 0
            t: integer stage of the diffusion

        Returns:
            mu_tilde (from Eq. 6 of DDPM)
        """

        mult1 = (torch.sqrt(self.alpha_bars[t - 1]) * self.betas[t]) / (
            1 - self.alpha_bars[t]
        )
        mult2 = (torch.sqrt(self.alpha_bars[t]) * 1 - self.alpha_bars[t - 1]) / (
            1 - self.alpha_bars[t]
        )
        return mult1 * x_t + mult2 * x_0

    def _beta_tilde(self, t):
        return (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]

    def _compute_x_t(self, x_0, eps, t):
        return (
            torch.sqrt(self.alpha_bars[t]) * x_0
            + torch.sqrt(1 - self.alpha_bars[t]) * eps
        )

    def predict_noise(self, x_t, t):
        return self.network(x_t, t)

    def predict_mean(self, x_t, t):
        pred_noise = self.predict_noise(x_t, t)
        scaled_noise = (
            self.betas[t] / (torch.sqrt(1 - self.alpha_bars[t]))
        ) * pred_noise
        diff = x_t - scaled_noise
        to_return = diff / (torch.sqrt(self.alphas[t]))
        return to_return

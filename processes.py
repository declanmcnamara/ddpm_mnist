import torch
import torch.nn as nn
from numpy.typing import ArrayLike

from modules import DenoisingOneStepBackwardKernel, NoisingLookAheadForwardKernel


class ForwardProcess(nn.Module):
    """Initialize a class representing the forward process.

    Only functionality really needed to is sample, i.e. take the observation
    x_0 and noise it up. This is very simple
    as a result; we don't even need to compute likelihoods.

    Args:
        betas: array-like of noise parameters that defines the conditional
            distributions q(x_t | x_{t-1}) \sim N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t I).
    """

    def __init__(
        self,
        betas: ArrayLike,
    ):
        super().__init__()
        self.betas = betas

    def realize(self, x_0, t):
        kernel = NoisingLookAheadForwardKernel(self.betas, t)
        return kernel._realize(x_0)

    def sample(self, x_0, t):
        kernel = NoisingLookAheadForwardKernel(self.betas, t)
        noise, samples = kernel.sample(x_0)
        return noise, samples


class BackwardProcess(nn.Module):
    """Initialize a class representing the backward (denoising) process.

    Class needs only to support sampling functionality, likelihoods not necessary.
    Main task is predicting noise at each step.
    Accepts betas from forward process to pass to subcomponents.

    Args:
        betas: array-like of noise parameters that defines the conditional
            distributions q(x_t | x_{t-1}) \sim N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t I).
        base_dist: univariate Gaussian, shapes handled with .sample()
        shape: tells us the shape of observation
        mean_network: neural network taking in x_t and stage t and returning mean for distribution on x_{t-1}
        cov_network: same but for covariance (we use even simpler suggestion of DDPM)
    """

    def __init__(
        self,
        betas: ArrayLike,
        base_dist: torch.distributions.Distribution,
        shape: torch.Size,
        mean_network: nn.Module,
        cov_network: nn.Module,
    ):

        super().__init__()
        self.betas = betas
        self.base_dist = base_dist
        self.shape = shape
        self.T = betas.shape[0]
        self.mean_network = mean_network
        self.cov_network = cov_network

    def sample(self, t):
        x_T = self.base_dist.sample(self.shape)
        draws = [x_T]
        for j in reversed(range(t, self.T)):
            kernel = DenoisingOneStepBackwardKernel(
                j, self.mean_network, self.cov_network
            )
            next = kernel.sample(draws[-1])
            draws.append(next)

        return draws

    def sample_K(self, t, K):
        to_sample = list(self.shape)
        to_sample[0] = K

        x_T = self.base_dist.sample(to_sample)
        draws = [x_T]
        for j in reversed(range(t, self.T)):
            kernel = DenoisingOneStepBackwardKernel(
                j, self.mean_network, self.cov_network
            )
            next = kernel.sample(draws[-1])
            draws.append(next)

        return draws

import math
import os

import hydra
import lightning as L
import torch
from lightning import Trainer
from omegaconf import DictConfig
from torch import utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from diffusion import DDPM


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    seed = cfg.seed
    L.seed_everything(seed)

    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    images = dataset.data / 255.0 * 2 - 1
    train_loader = utils.data.DataLoader(images, cfg.training.batch_size)

    beta_min = cfg.forward_process.beta_min
    beta_max = cfg.forward_process.beta_max
    T = cfg.forward_process.T
    min_exp = math.log10(beta_min)
    max_exp = math.log10(beta_max)
    betas = torch.logspace(min_exp, max_exp, T)
    device_name = "cuda:{}".format(cfg.training.gpu_num) if torch.cuda.is_available() else "cpu"
    acc = "gpu" if torch.cuda.is_available() else "cpu"
    betas = betas.to(device_name)

    diffusion_model = DDPM(
        betas,
        device_name,
    )

    
    trainer = Trainer(
        devices=[cfg.training.gpu_num],
        accelerator=acc,
        default_root_dir="./{}".format(cfg.logging.log_dir),
        max_epochs=100,
    )
    trainer.fit(diffusion_model, train_dataloaders=train_loader)

    return


if __name__ == "__main__":
    main()

import lightning as L
import torch
import torch.distributions as D
from torch.optim import Adam

from modules import CondCovariance, CondMean
from network import UNet
from processes import BackwardProcess, ForwardProcess


class DDPM(L.LightningModule):
    def __init__(self, betas, device):
        super().__init__()
        self.betas = betas
        self.device_name = device
        self.T = betas.shape[0]
        self.shape = torch.Size([1, 1, 28, 28])
        self.base_dist = D.Normal(
            torch.tensor(0.0).to(self.device_name),
            torch.tensor(1.0).to(self.device_name),
        )
        self.mean_network = CondMean(
            betas,
            self.shape,
            net=UNet(1, 32, [1, 2, 4, 8]).to(self.device_name),
        )
        self.cov_network = CondCovariance(betas)
        self.forward_process = ForwardProcess(betas)

        self.backward_process = BackwardProcess(
            betas,
            base_dist=self.base_dist,
            shape=self.shape,
            mean_network=self.mean_network,
            cov_network=self.cov_network,
        )

    def configure_optimizers(self):
        optimizer = Adam(self.backward_process.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Loss",
            },
        }

    def training_step(self, batch, batch_idx):
        # Loss computation
        images = batch  # mb_size x 28 x 28
        images = images.reshape(-1, 1, 28, 28)
        t = torch.randint(low=0, high=self.T, size=(1,)).item()
        noise = self.base_dist.sample(images.shape)

        x_t = self.mean_network._compute_x_t(images, noise, t)
        pred_noise = self.mean_network.predict_noise(x_t, t)
        loss = torch.square(torch.norm(noise - pred_noise, dim=(-2, -1))).mean()

        # Logging
        self.log("Loss", loss.item(), on_step=True, logger=True)

        return loss

import torch
from torch import nn

from src.config import (
    box_dim,
    height_limits,
    latent_dim,
    length_limits,
    requirement_dim,
    width_limits,
)
from src.models.components import AuxiliaryPredictor

embedding_dim = 1024  # Embedded previously generated boxes

MAIN_INPUT_DIM = latent_dim + requirement_dim + embedding_dim + 1


# -------------------------------------------------
#     Auto Regressive Box Generator (Main Model)
# -------------------------------------------------


class AutoRegressiveBoxGenerator(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")

        # embedding of requirements
        self.embeddor = nn.Sequential(
            nn.Linear(box_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, embedding_dim),
        )

        # main MLP to generate box predictions
        self.create_box = nn.Sequential(
            nn.Linear(MAIN_INPUT_DIM, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, box_dim),
        )

        self.auxiliary_predictor = AuxiliaryPredictor(
            embedding_dim, device=self.device
        )  # so we can initialize a single model

    def forward(self, volume, latent_z):
        assert len(volume.shape) == 2, (
            "volume must be a 2D tensor"
        )  # [batchsize, seq_len]
        assert len(latent_z.shape) == 2, (
            "latent_z must be a 2D tensor"
        )  # [batchsize, latent_dim]

        batch_size, seq_len = volume.shape
        device = volume.device  # Get device from input tensor

        embedding = torch.zeros(batch_size, embedding_dim, device=device)
        boxes = torch.zeros(batch_size, seq_len, box_dim, device=device)
        z_prediction = torch.zeros(batch_size, latent_dim, device=device)
        seq_len_tensor = seq_len * torch.ones(batch_size, 1, device=device)

        for i in range(seq_len):
            # 1) concatenate embedding (info about previous boxes) with latent_z and previous box predictions
            y = torch.cat(
                [embedding, latent_z, seq_len_tensor, volume[:, i].unsqueeze(-1)],
                dim=-1,
            )

            # 2) generate box[i] prediction
            box_prediction = self.create_box(y)  # batch_size, 6

            # Apply ReLU to first 3 components (size) but keep positions as is
            box_prediction_modified = torch.cat(
                [
                    torch.sigmoid(box_prediction[:, :3])
                    + 1e-4,  # Apply ReLU to first 3 components: dimensions need to be positive
                    box_prediction[
                        :, 3:
                    ],  # Keep last 3 components unchanged: locations can be positive or negative
                ],
                dim=1,
            )

            box_prediction_modified[:, 0] = (
                box_prediction_modified[:, 0] * (length_limits[1] - length_limits[0])
                + length_limits[0]
            )
            box_prediction_modified[:, 1] = (
                box_prediction_modified[:, 1] * (width_limits[1] - width_limits[0])
                + width_limits[0]
            )
            box_prediction_modified[:, 2] = (
                box_prediction_modified[:, 2] * (height_limits[1] - height_limits[0])
                + height_limits[0]
            )

            box_prediction_modified[:, 4] = 0  # enforces y-axis symmetry

            # 3) add generated box prediction [i] to output
            boxes[:, i, :] = box_prediction_modified

            # 4) Get auxiliary prediction for current step, i.e. guess the next box arrangement only from latent codes z and embedding
            z_add_prediction = self.auxiliary_predictor(
                box_prediction_modified[:, 3:], embedding, seq_len_tensor
            )  # predict z_add from x, y, z of the given box, overall sequence length, and i
            z_prediction[:] += z_add_prediction

            # 5) add generated box prediction [i] to embedding
            emb = self.embeddor(box_prediction_modified)
            embedding += emb

        return (
            boxes,
            z_prediction,
        )  # Return both main predictions and auxiliary predictions

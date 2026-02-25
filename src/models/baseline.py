import torch
from torch import nn

latent_dim = 4
requirement_dim = 1  # Volume
embedding_dim = 1024  # Embedded previously generated boxes

box_dim = 6

length_limits = [0.1, 1.5]  # m
width_limits = [0.1, 0.5]  # m
height_limits = [0.1, 0.5]  # m

volume_limits = [
    0.005,
    0.1,
]  # m3    --> 5 - 100 L, which is still quite large for a UAV

MAIN_INPUT_DIM = latent_dim + requirement_dim + embedding_dim + 1
AUX_INPUT_DIM = 3 + embedding_dim + 1  # xyz, seq_len, i


class AuxiliaryPredictor(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")

        self.predictor = nn.Sequential(
            nn.Linear(AUX_INPUT_DIM, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x, emb, seq_len):
        feed = torch.cat([x, emb, seq_len], dim=-1)
        return self.predictor(feed)


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
            device=self.device
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


# -------------------------------------------------
#            Mutual Information Loss
# -------------------------------------------------


def mi_loss(latent_codes, latent_code_guess, temperature=0.1):

    batch_size = latent_codes.shape[0]

    # Normalize the vectors
    latent_codes = torch.nn.functional.normalize(latent_codes, dim=1)
    latent_code_guess = torch.nn.functional.normalize(latent_code_guess, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(latent_code_guess, latent_codes.T) / temperature

    # InfoNCE loss
    labels = torch.arange(batch_size, device=latent_codes.device)
    loss = torch.nn.functional.cross_entropy(sim_matrix, labels)

    return loss


# -------------------------------------------------
#            DPP losses
# -------------------------------------------------


def DPP_loss(x, sigma=1, eps=1e-6):

    batch_size, seq_len, box_dim = x.shape
    assert seq_len > 1, (
        "seq_len must be greater than 1: cannot compute DPP loss for a single box"
    )

    x_flat = x.reshape(
        batch_size, -1
    )  # [batch_size, seq_len * box_dim]  --> since we only want to compare similarity across batch, not within box sequence!

    # Similarity
    mean, std = torch.mean(x_flat, dim=0), torch.std(x_flat, dim=0)
    x_flat_normalized = (x_flat - mean) / (std + 1e-6)

    D2 = torch.cdist(x_flat_normalized, x_flat_normalized).pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))

    try:
        identity = torch.eye(batch_size, device=K.device)
        K = K + identity * eps
        L_c = torch.linalg.cholesky(K)
        log_det = 2 * torch.log(torch.diagonal(L_c)).sum()
        # Normalize by matrix size
        log_det = log_det / batch_size
    except Exception:
        # Move to CPU for eigenvalue calculation
        L_cpu = K.cpu()
        eigenvalues = torch.linalg.eigvalsh(L_cpu)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        log_det = torch.log(eigenvalues).sum()
        # Normalize by matrix size
        log_det = log_det / batch_size
        log_det = log_det.to(K.device)

    return -log_det


# -------------------------------------------------
#        DPP loss
# -------------------------------------------------


def DPP_AR_loss(x, sigma=1, eps=1e-6):

    # Collapse (n, 2) → (2*n) so we get batchsize vectors of size 2*n
    x = x.view(x.size(0), -1)

    D2 = torch.cdist(x, x).pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))

    try:
        identity = torch.eye(K.shape[0], device=K.device)
        K = K + identity * eps
        L_c = torch.linalg.cholesky(K)
        log_det = 2 * torch.log(torch.diagonal(L_c)).sum()
        # Normalize by matrix size
        log_det = log_det / K.shape[0]
    except Exception:
        # Move to CPU for eigenvalue calculation
        L_cpu = K.cpu()
        eigenvalues = torch.linalg.eigvalsh(L_cpu)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        log_det = torch.log(eigenvalues).sum()
        # Normalize by matrix size
        log_det = log_det / K.shape[0]
        log_det = log_det.to(K.device)

    return -log_det


def DPP_loss_adjacency(A, sigma=1, eps=1e-6):

    batch_size, n1, n2 = A.shape
    A_flat = A.reshape(batch_size, -1)

    assert n1 == n2, "A must be a square matrix"
    assert A_flat.shape[0] == batch_size and A_flat.shape[1] == n1 * n2, (
        "Flattening operation not correct"
    )

    D2 = torch.cdist(A_flat, A_flat).pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))

    try:
        identity = torch.eye(batch_size, device=K.device)
        K = K + identity * eps
        L_c = torch.linalg.cholesky(K)
        log_det = 2 * torch.log(torch.diagonal(L_c)).sum()
        # Normalize by matrix size
        log_det = log_det / batch_size
    except Exception:
        # Move to CPU for eigenvalue calculation
        L_cpu = K.cpu()
        eigenvalues = torch.linalg.eigvalsh(L_cpu)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        log_det = torch.log(eigenvalues).sum()
        # Normalize by matrix size
        log_det = log_det / batch_size
        log_det = log_det.to(K.device)

    return -log_det

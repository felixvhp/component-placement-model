#  Shared constants

latent_dim = 4
requirement_dim = 1  # volume
box_dim = 6  # (l, w, h, x, y, z)

length_limits = [0.1, 1.5]
width_limits = [0.1, 0.5]
height_limits = [0.1, 0.5]

volume_limits = [
    0.005,
    0.1,
]


# -------------------------------------------------
#  Training
# -------------------------------------------------

batch_size = 256
epochs = 2500
lr = 5e-4
max_seq_length = 10

# loss weight schedule
loss_weight_schedule = {
    "ramp_start": 500,
    "ramp_end": 1500,
    "volume": {"initial": 5, "final": 10},
    "overlap": {"initial": 4, "final": 20},
    "compactness": {"initial": 2, "final": 4},
    "dpp_aspect_ratios": {"initial": 4, "final": 1},
    "ldv": {"initial": 5e-1, "final": 5e-1},
    "adj_vol_sensitivity": {"initial": 0, "final": 0},
    "infonce": {"initial": 1.0, "final": 2.0},
    "adjacency": {"initial": 1.0, "final": 30.0},
}

def get_loss_weights(epoch):
    s = loss_weight_schedule
    ramp_start = s["ramp_start"]
    ramp_end = s["ramp_end"]

    keys = [
        "overlap",
        "volume",
        "compactness",
        "ldv",
        "dpp_aspect_ratios",
        "adj_vol_sensitivity",
        "infonce",
        "adjacency",
    ]

    if epoch < ramp_start:
        return tuple(s[k]["initial"] for k in keys)
    elif epoch < ramp_end:
        t = (epoch - ramp_start) / (ramp_end - ramp_start)
        return tuple(
            s[k]["initial"] + t * (s[k]["final"] - s[k]["initial"]) for k in keys
        )
    else:
        return tuple(s[k]["final"] for k in keys)

# -------------------------------------------------
#  Projector settings (LM step + PAL-EMA)
# -------------------------------------------------

projector_config = {
    "delta": 1e-3,          # YF damping scale: lam = delta * ||c||^2
    "lam_min": 1e-6,        # damping floor
    "feas_tol": 1e-7,       # post-projection residual tolerance for EMA
    "disp_tol": 1e-1,       # displacement tolerance for EMA
    "ema_max_decades": 1.0,
    "ema_rate": 1e-2,
    "grad_clip": 20.0,      # gradient clip norm (supervisor default)
}


# -------------------------------------------------
#  Fixed objective weights
# -------------------------------------------------

objective_weights = {
    "conform": 30.0,
    "compactness": 3.0,
    "dpp_aspect_ratios": 2.0,
    "ldv": 0.5,
    "infonce": 0.1,
}

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
    "infonce": {"initial": 0.1, "final": 0.1},
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

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Box:
    def __init__(self, lwh):

        assert lwh.ndim == 1 and lwh.shape[0] == 3
        if lwh.any() < 0:
            raise ValueError("lwh must be positive")

        self.lwh = lwh  # [3]

    def getCornerPoints(self, xyz_center, mode="positive"):

        assert mode in ["positive", "full"], "mode must be either 'positive' or 'full'"

        # returns four points for box, side with y > 0

        if mode == "positive":
            assert xyz_center[1] == 0, (
                "xyz_center must be on the xz plane if positive only mode is to be used. (Boxes must be in center of Aircraft to only check one side of the box)"
            )

            return_points = np.zeros((4, 3))
            return_points[0, :] = xyz_center + np.array(
                [self.lwh[0] / 2, self.lwh[1] / 2, self.lwh[2] / 2]
            )
            return_points[1, :] = xyz_center + np.array(
                [self.lwh[0] / 2, self.lwh[1] / 2, -self.lwh[2] / 2]
            )
            return_points[2, :] = xyz_center + np.array(
                [-self.lwh[0] / 2, self.lwh[1] / 2, self.lwh[2] / 2]
            )
            return_points[3, :] = xyz_center + np.array(
                [-self.lwh[0] / 2, self.lwh[1] / 2, -self.lwh[2] / 2]
            )

            return_points[:, 1] = (
                np.abs(return_points[:, 1]) + 1e-8
            )  # since mode is positive, we need to make sure y is positive

        # returns eight points for full box
        elif mode == "full":
            return_points = np.zeros((8, 3))
            return_points[0, :] = xyz_center + np.array(
                [self.lwh[0] / 2, self.lwh[1] / 2, self.lwh[2] / 2]
            )
            return_points[1, :] = xyz_center + np.array(
                [self.lwh[0] / 2, -self.lwh[1] / 2, self.lwh[2] / 2]
            )
            return_points[2, :] = xyz_center + np.array(
                [-self.lwh[0] / 2, self.lwh[1] / 2, self.lwh[2] / 2]
            )
            return_points[3, :] = xyz_center + np.array(
                [-self.lwh[0] / 2, -self.lwh[1] / 2, self.lwh[2] / 2]
            )
            return_points[4, :] = xyz_center + np.array(
                [self.lwh[0] / 2, self.lwh[1] / 2, -self.lwh[2] / 2]
            )
            return_points[5, :] = xyz_center + np.array(
                [self.lwh[0] / 2, -self.lwh[1] / 2, -self.lwh[2] / 2]
            )
            return_points[6, :] = xyz_center + np.array(
                [-self.lwh[0] / 2, self.lwh[1] / 2, -self.lwh[2] / 2]
            )
            return_points[7, :] = xyz_center + np.array(
                [-self.lwh[0] / 2, -self.lwh[1] / 2, -self.lwh[2] / 2]
            )

        return return_points

    def getVolume(self):
        return self.lwh[0] * self.lwh[1] * self.lwh[2]

    def getBoxSurface(self, xyz_center, ax, alpha, color, linewidth=0.5):

        points = self.getCornerPoints(xyz_center, mode="full")

        # Define the faces of the box using the corner points
        faces = [
            [points[0], points[1], points[3], points[2]],  # Bottom face
            [points[4], points[5], points[7], points[6]],  # Top face
            [points[0], points[1], points[5], points[4]],  # Front face
            [points[2], points[3], points[7], points[6]],  # Back face
            [points[0], points[2], points[6], points[4]],  # Left face
            [points[1], points[3], points[7], points[5]],  # Right face
        ]

        # Create a 3D polygon collection with improved transparency and depth sorting
        collection = Poly3DCollection(
            faces, alpha=alpha, linewidths=linewidth, edgecolor="black"
        )
        collection.set_facecolor(color)
        collection.set_sort_zpos(
            xyz_center[2]
        )  # Use the z-coordinate of center for depth sorting --> prevent rendering issues

        ax.add_collection3d(collection)


# -------------------------------------------------------------
#                   Batched Box Class
# -------------------------------------------------------------


class BatchBox:
    def __init__(self, lwh):

        assert lwh.ndim == 2 and lwh.shape[1] == 3, (
            "lwh must be a 2D array with shape [batch_size, 3]"
        )
        self.batch_size = lwh.shape[0]

        if torch.any(lwh < 0):
            raise ValueError("lwh must be positive")

        self.lwh = lwh  # [batch_size, 3]

    def getCornerPoints(self, xyz_center, mode="positive"):

        assert mode in ["positive", "full"], "mode must be either 'positive' or 'full'"
        assert (
            xyz_center.ndim == 2
            and xyz_center.shape[0] == self.batch_size
            and xyz_center.shape[1] == 3
        ), "xyz_center must be a 2D array with shape [batch_size, 3]"

        # returns four points for box, side with y > 0 (4 points per batch element, with 3 coordinates (xyz) each)
        if mode == "positive":
            assert xyz_center[:, 1].all() == 0, (
                "xyz_center must be on the xz plane if positive only mode is to be used. (Boxes must be in center of Aircraft to only check one side of the box)"
            )

            return_points = torch.zeros(
                (self.batch_size, 4, 3), device=self.lwh.device
            )  # [batch_size, 4, 3]

            half_l = self.lwh[:, 0] / 2
            half_w = torch.abs(self.lwh[:, 1] / 2) + 1e-8
            half_h = self.lwh[:, 2] / 2

            return_points[:, 0, :] = xyz_center + torch.stack(
                [half_l, half_w, half_h], dim=1
            )
            return_points[:, 1, :] = xyz_center + torch.stack(
                [half_l, half_w, -half_h], dim=1
            )
            return_points[:, 2, :] = xyz_center + torch.stack(
                [-half_l, half_w, half_h], dim=1
            )
            return_points[:, 3, :] = xyz_center + torch.stack(
                [-half_l, half_w, -half_h], dim=1
            )

        # returns eight points for full box
        elif mode == "full":
            return_points = torch.zeros(
                (self.batch_size, 8, 3), device=self.lwh.device
            )  # [batch_size, 8, 3]

            half_l = self.lwh[:, 0] / 2
            half_w = torch.abs(self.lwh[:, 1] / 2) + 1e-8
            half_h = self.lwh[:, 2] / 2

            return_points[:, 0, :] = xyz_center + torch.stack(
                [half_l, half_w, half_h], dim=1
            )
            return_points[:, 1, :] = xyz_center + torch.stack(
                [half_l, -half_w, half_h], dim=1
            )
            return_points[:, 2, :] = xyz_center + torch.stack(
                [-half_l, half_w, half_h], dim=1
            )
            return_points[:, 3, :] = xyz_center + torch.stack(
                [-half_l, -half_w, half_h], dim=1
            )
            return_points[:, 4, :] = xyz_center + torch.stack(
                [half_l, half_w, -half_h], dim=1
            )
            return_points[:, 5, :] = xyz_center + torch.stack(
                [half_l, -half_w, -half_h], dim=1
            )
            return_points[:, 6, :] = xyz_center + torch.stack(
                [-half_l, half_w, -half_h], dim=1
            )
            return_points[:, 7, :] = xyz_center + torch.stack(
                [-half_l, -half_w, -half_h], dim=1
            )

        return return_points

    def getVolume(self):
        volumes = self.lwh[:, 0] * self.lwh[:, 1] * self.lwh[:, 2]
        return volumes

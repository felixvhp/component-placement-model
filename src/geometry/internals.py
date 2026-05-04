import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .box import BatchBox, Box
from .unique_with_grad import UniqueWithGrad

default_safety_margin = 0.02


class Internals:
    def __init__(self, centers, dimensions, safety_margin=default_safety_margin):

        # assert centers and dimensions are of shape  [seq_len, 3]
        assert centers.ndim == 2 and centers.shape[1] == 3
        assert dimensions.ndim == 2 and dimensions.shape[1] == 3
        assert centers.shape[0] == dimensions.shape[0]

        self.centers = centers  # xyz coordinates of box centers
        self.n_boxes = centers.shape[0]  # number of boxes
        self.safety_margin_value = safety_margin  # Store the safety margin

        # Create boxes (using original dimensions for volume, visualization, etc.)
        self.boxes = [Box(dimensions[i, :]) for i in range(self.n_boxes)]

    def checkOverlap(self):

        # build a n_boxes x n_boxes matrix
        overlap_matrix = np.zeros((self.n_boxes, self.n_boxes))

        for i in range(self.n_boxes):
            for j in range(self.n_boxes):
                if i == j:
                    overlap_matrix[i, j] = 0
                else:
                    # New calculation with safety margin:
                    # The term (dim_i + dim_j)/2 is the sum of half-dimensions.
                    # Adding self.safety_margin_value means overlap is detected sooner.
                    overlap_x = np.maximum(
                        0,
                        (self.boxes[i].lwh[0] + self.boxes[j].lwh[0]) / 2
                        + self.safety_margin_value
                        - abs(self.centers[i, 0] - self.centers[j, 0]),
                    )
                    overlap_y = np.maximum(
                        0,
                        (self.boxes[i].lwh[1] + self.boxes[j].lwh[1]) / 2
                        + self.safety_margin_value
                        - abs(self.centers[i, 1] - self.centers[j, 1]),
                    )
                    overlap_z = np.maximum(
                        0,
                        (self.boxes[i].lwh[2] + self.boxes[j].lwh[2]) / 2
                        + self.safety_margin_value
                        - abs(self.centers[i, 2] - self.centers[j, 2]),
                    )
                    overlap_matrix[i, j] = overlap_x * overlap_y * overlap_z

        return overlap_matrix

    def getBoxVolumes(self):
        return [box.getVolume() for box in self.boxes]

    def getAdjacency(self):
        # Reshape centers to enable broadcasting
        centers_i = self.centers[:, np.newaxis, :]  # Shape: (n_boxes, 1, 3)
        centers_j = self.centers[np.newaxis, :, :]  # Shape: (1, n_boxes, 3)

        # Calculate differences
        diff = centers_i - centers_j

        # Use smooth sign approximation with tanh
        alpha = 1.0  # Steepness factor - higher values make it closer to sign()
        smooth_sign = np.tanh(alpha * diff)

        # Split into separate matrices for each axis
        Ax = smooth_sign[..., 0]  # Shape: (n_boxes, n_boxes)
        Ay = smooth_sign[..., 1]
        Az = smooth_sign[..., 2]

        return Ax, Ay, Az

    def getConvexHullPoints(self, view):

        # Returns points of the convex hull in the specified view
        # Source: https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Python

        # Collect corner points from all boxes
        points = []
        for i, box in enumerate(self.boxes):
            # Get all 8 corner points for each box
            corner_points = box.getCornerPoints(self.centers[i, :], mode="full")
            points.extend(corner_points)

        # Convert to numpy array and remove duplicates
        points = np.array(points)

        if len(points) == 1:
            raise ValueError(
                "Only one point found or all points are the same, cannot compute convex hull"
            )

        # Extract relevant coordinates based on view
        if view == "frontal":
            points = points[:, [1, 2]]  # y, z coordinates
        elif view == "side":
            points = points[:, [0, 2]]  # x, z coordinates
        elif view == "top":
            points = points[:, [0, 1]]  # x, y coordinates
        else:
            raise ValueError("Invalid view. Must be 'frontal', 'side', or 'top'")

        # Sort points lexicographically
        points = points[np.lexsort((points[:, 1], points[:, 0]))]

        points = np.unique(points, axis=0)

        # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
        # Returns a positive value, if OAB makes a counter-clockwise turn,
        # negative for clockwise turn, and zero if the points are collinear.
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenation of the lower and upper hulls gives the convex hull.
        # Last point of each list is omitted because it is repeated at the beginning of the other list.
        return lower[:-1] + upper[:-1]

    def getViewArea(self, view):

        hull_points = self.getConvexHullPoints(view)

        area = 0

        for i in range(len(hull_points)):
            j = (i + 1) % len(hull_points)
            area += (
                hull_points[i][0] * hull_points[j][1]
                - hull_points[j][0] * hull_points[i][1]
            )

        return abs(area) / 2

    def getRoughVolume(self):

        # Collect corner points from all boxes
        points = []
        for i, box in enumerate(self.boxes):
            # Get all 8 corner points for each box
            corner_points = box.getCornerPoints(self.centers[i, :], mode="full")
            points.extend(corner_points)

        points = np.array(points)

        x_lims = np.array([points[:, 0].min(), points[:, 0].max()])
        y_lims = np.array([points[:, 1].min(), points[:, 1].max()])
        z_lims = np.array([points[:, 2].min(), points[:, 2].max()])

        volume = np.abs(
            (x_lims[1] - x_lims[0]) * (y_lims[1] - y_lims[0]) * (z_lims[1] - z_lims[0])
        )

        return volume

    def getRoughViewArea(self, view):

        # Collect corner points from all boxes
        points = []
        for i, box in enumerate(self.boxes):
            # Get all 8 corner points for each box
            corner_points = box.getCornerPoints(self.centers[i, :], mode="full")
            points.extend(corner_points)

        points = np.array(points)

        x_lims = np.array([points[:, 0].min(), points[:, 0].max()])
        y_lims = np.array([points[:, 1].min(), points[:, 1].max()])
        z_lims = np.array([points[:, 2].min(), points[:, 2].max()])

        if view == "frontal":
            area = np.abs((y_lims[1] - y_lims[0]) * (z_lims[1] - z_lims[0]))
        elif view == "side":
            area = np.abs((x_lims[1] - x_lims[0]) * (z_lims[1] - z_lims[0]))
        elif view == "top":
            area = np.abs((x_lims[1] - x_lims[0]) * (y_lims[1] - y_lims[0]))
        else:
            raise ValueError("Invalid view. Must be 'frontal', 'side', or 'top'")

        return area

    def getAllShiftedCornerPoints(self, internalsbase_xyz):
        corner_points = []
        for i, box in enumerate(self.boxes):
            corner_points.extend(box.getCornerPoints(self.centers[i, :], mode="full"))
        shifted_corner_points = np.array(corner_points) + internalsbase_xyz
        return shifted_corner_points

    def getBoxSurfaces(
        self,
        ax,
        internalsbase_xyz,
        alpha=1,
        colors=[
            "#F4FF1A",
            "#E3F20F",
            "#D2E504",
            "#BDD600",
            "#A7C200",
            "#90AD00",
            "#799800",
            "#628300",
            "#4B6E00",
            "#345900",
        ],
        linewidth=0.5,
    ):
        box_surfaces = []
        for i, box in enumerate(self.boxes):
            # Add the base shift to each box's center position
            shifted_box_centers = self.centers[i, :] + internalsbase_xyz
            box_surfaces.append(
                box.getBoxSurface(
                    shifted_box_centers,
                    ax,
                    alpha,
                    colors[i % len(colors)],
                    linewidth=linewidth,
                )
            )

        return box_surfaces


# -------------------------------------------------
#       Batch Internals
# -------------------------------------------------


class BatchInternals:
    def __init__(
        self,
        centers: torch.Tensor,
        dimensions: torch.Tensor,
        safety_margin: float = default_safety_margin,
    ):
        # Assert inputs are PyTorch tensors
        assert isinstance(centers, torch.Tensor) and isinstance(
            dimensions, torch.Tensor
        )

        # Assert proper shapes: [batch_size, seq_len, 3]
        assert centers.ndim == 3 and centers.shape[-1] == 3
        assert dimensions.ndim == 3 and dimensions.shape[-1] == 3
        assert centers.shape == dimensions.shape

        self.batch_size = centers.shape[0]  # number of sequences in the batch
        self.n_boxes = centers.shape[1]  # number of boxes per sequence
        self.safety_margin_value = safety_margin  # Store the safety margin

        self.centers = centers  # xyz coordinates of box centers
        self.dimensions = dimensions  # store original dimensions tensor

        # Create a single BatchBox object for all boxes in the batch (uses original dimensions)
        self.boxes = BatchBox(dimensions.reshape(-1, 3)).lwh.reshape(
            self.batch_size, self.n_boxes, 3
        )

    def checkOverlap(self):
        # Initialize overlap matrix for the batch
        overlap_matrix = torch.zeros(
            (self.batch_size, self.n_boxes, self.n_boxes), device=self.centers.device
        )

        # Compute pairwise differences for each dimension
        # Reshape centers to enable broadcasting
        centers_i = self.centers.unsqueeze(2)  # [batch, n_boxes, 1, 3]
        centers_j = self.centers.unsqueeze(1)  # [batch, 1, n_boxes, 3]

        dims_i = self.dimensions.unsqueeze(2)  # [batch, n_boxes, 1, 3]
        dims_j = self.dimensions.unsqueeze(1)  # [batch, 1, n_boxes, 3]

        # Compute overlaps along each dimension using relu, incorporating safety_margin_value
        overlap_x = torch.relu(
            (dims_i[..., 0] + dims_j[..., 0]) / 2
            + self.safety_margin_value
            - torch.abs(centers_i[..., 0] - centers_j[..., 0])
        )

        overlap_y = torch.relu(
            (dims_i[..., 1] + dims_j[..., 1]) / 2
            + self.safety_margin_value
            - torch.abs(centers_i[..., 1] - centers_j[..., 1])
        )

        overlap_z = torch.relu(
            (dims_i[..., 2] + dims_j[..., 2]) / 2
            + self.safety_margin_value
            - torch.abs(centers_i[..., 2] - centers_j[..., 2])
        )

        # Compute overlap volumes
        overlap_matrix = overlap_x * overlap_y * overlap_z

        # Set diagonal to zero (box doesn't overlap with itself)
        overlap_matrix.diagonal(dim1=1, dim2=2).zero_()

        return overlap_matrix

    def getBoxVolumes(self):
        # Compute volumes directly using the dimensions tensor
        return torch.prod(self.dimensions, dim=-1)  # [batch_size, n_boxes]

    def getEuclideanCenterDistance(self, xyz_center):
        # Reshape centers to (batch_size, n_boxes, 1, 3) and (batch_size, 1, n_boxes, 3)
        # This allows for broadcasting in the subtraction
        centers_i = xyz_center.unsqueeze(2)  # Shape: (batch_size, n_boxes, 1, 3)
        centers_j = xyz_center.unsqueeze(1)  # Shape: (batch_size, 1, n_boxes, 3)

        # Calculate pairwise differences and compute Euclidean norm
        # The result will be (batch_size, n_boxes, n_boxes)
        euclidian_distance_matrix = torch.norm(centers_i - centers_j, dim=3)

        # Create a mask for the diagonal (1s on diagonal, 0s elsewhere)
        diagonal_mask = ~torch.eye(
            self.n_boxes, dtype=torch.bool, device=xyz_center.device
        ).expand(self.batch_size, -1, -1)

        # Zero out diagonal using multiplication instead of in-place operation
        euclidian_distance_matrix = euclidian_distance_matrix * diagonal_mask

        return euclidian_distance_matrix

    def getAdjacency(self):

        centers_i = self.centers.unsqueeze(2)  # Shape: (batch, n_boxes, 1, 3)
        centers_j = self.centers.unsqueeze(1)  # Shape: (batch, 1, n_boxes, 3)

        # Use smooth sign approximation with tanh
        alpha = 1.0  # Steepness factor - higher values make it closer to sign()
        diff = centers_i - centers_j
        smooth_sign = torch.tanh(alpha * diff)

        # Split into separate matrices for each axis
        Ax = smooth_sign[..., 0]  # Shape: (batch, n_boxes, n_boxes)
        Ay = smooth_sign[..., 1]
        Az = smooth_sign[..., 2]

        return Ax, Ay, Az

    def getConvexHullPoints(self, view):

        # Collect corner points from all boxes in a batched way
        # Reshape boxes to [batch_size * n_boxes, 3]
        boxes_flat = self.boxes.reshape(-1, 3)
        centers_flat = self.centers.reshape(-1, 3)

        # Get corner points for all boxes at once
        corner_points = BatchBox(boxes_flat).getCornerPoints(centers_flat, mode="full")

        # Reshape back to [batch_size, n_boxes * 8, 3]
        # (8 corner points per box)
        points = corner_points.reshape(self.batch_size, self.n_boxes * 8, 3)

        # Extract relevant coordinates based on view
        if view == "frontal":
            points = points[:, :, [1, 2]]  # y, z coordinates
        elif view == "side":
            points = points[:, :, [0, 2]]  # x, z coordinates
        elif view == "top":
            points = points[:, :, [0, 1]]  # x, y coordinates
        else:
            raise ValueError("Invalid view. Must be 'frontal', 'side', or 'top'")

        # remove duplicates + sort
        points = UniqueWithGrad.apply(
            points, 1
        )  # weed out duplicate points along dim = 1 (in every batch element)

        points_sorted_tuple = torch.sort(points, dim=1, stable=True)
        points_sorted = points_sorted_tuple.values

        # Convex Hull Algorithm
        n_points = points_sorted.shape[1]

        P_i = points_sorted.unsqueeze(2).unsqueeze(3)  # [batch_size, n_points, 1, 1, 2]
        P_j = points_sorted.unsqueeze(1).unsqueeze(3)  # [batch_size, 1, n_points, 1, 2]
        P_k = points_sorted.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, n_points, 2]

        # compute cross = (Pj−Pi) × (Pk−Pi) for every (i,j,k)
        v_ij = P_j - P_i  # [batch_size, n_points, n_points, 1, 2]
        v_ik = P_k - P_i  # [batch_size, n_points, 1, n_points, 2]

        # broadcast so both are [batch_size, n_points, n_points, n_points, 2]
        v_ij = v_ij.expand(-1, -1, -1, n_points, -1)
        v_ik = v_ik.expand(-1, -1, n_points, -1, -1)

        cross = (
            v_ij[..., 0] * v_ik[..., 1] - v_ij[..., 1] * v_ik[..., 0]
        )  # [batch_size, n_points, n_points, n_points]

        # For each (i,j), check if all cross >=0 or all <=0 over k:
        ge = (cross >= 0).all(dim=-1)  # [batch_size, n_points, n_points]
        le = (cross <= 0).all(dim=-1)  # [batch_size, n_points, n_points]
        edge = ge | le  # [batch_size, n_points, n_points]

        # zero out edge[i,i] so that i==j never counts
        I = torch.eye(
            n_points, device=points_sorted.device, dtype=torch.bool
        )  # [n_points, n_points]
        edge = edge & ~I.unsqueeze(0)  # [batch_size, n_points, n_points]

        # point i is on the hull if ∃ j≠i: edge[b,i,j] is True
        on_hull = edge.any(dim=2)  # [batch_size, n_points]

        # Compute centroid over hull‐points (avoid div by zero)
        count_hull = on_hull.sum(dim=1)  # [batch_size]
        denom = count_hull.clamp(min=1).view(self.batch_size, 1)  # [batch_size, 1]
        cen = (points_sorted * on_hull.unsqueeze(-1)).sum(
            dim=1
        ) / denom  # [batch_size, 2]

        # Compute angle rel to centroid for every point, set non‐hull to +inf --> used for sorting counter-clockwise
        rel = points_sorted - cen.unsqueeze(1)  # [batch_size, n_points, 2]
        ang = torch.atan2(rel[:, :, 1], rel[:, :, 0])  # [batch_size, n_points]
        ang = torch.where(
            on_hull, ang, torch.tensor(float("inf"), device=points_sorted.device)
        )

        # Sort by angle along dim=1, gather points accordingly
        idx_sorted = ang.argsort(dim=1)  # [batch_size, n_points]

        # count hull points per batch and find the maximum
        n_h = on_hull.sum(dim=1)  # [batch_size]
        max_nh = int(n_h.max().item())

        # create a zero‐padded tensor
        hull_tensor = points_sorted.new_zeros(
            (self.batch_size, max_nh, 2)
        )  # [batch_size, max_nh, 2]

        # fill each batch‐row with its hull points
        for b in range(self.batch_size):
            nh = int(n_h[b].item())
            inds = idx_sorted[b, :nh]
            hull_tensor[b, :nh] = points_sorted[b, inds]

        return hull_tensor, n_h  # shape: [batch_size, max_nh, 2], zeros are padding

    def getViewArea(self, view):

        hull_points, n_h = self.getConvexHullPoints(view)
        m = hull_points.shape[1]

        idx = torch.arange(m, device=hull_points.device)
        valid = idx.unsqueeze(0) < n_h.unsqueeze(
            1
        )  # [batchsize, m] mask of real points
        next_idx = idx.unsqueeze(0) + 1  # [batchsize, m]
        next_idx = next_idx.masked_fill(
            next_idx >= n_h.unsqueeze(1), 0
        )  # wrap‐around at each n_h[b]

        x = hull_points[..., 0]  # [batchsize, m]
        y = hull_points[..., 1]  # [batchsize, m]
        x_next = x.gather(1, next_idx)
        y_next = y.gather(1, next_idx)

        cross = x * y_next - x_next * y
        area = 0.5 * (cross * valid).sum(dim=1).abs()  # [batchsize]
        return area

    def getRoughVolume(self):
        # Collect corner points from all boxes in a batched way
        # Reshape boxes to [batch_size * n_boxes, 3]
        boxes_flat = self.boxes.reshape(-1, 3)
        centers_flat = self.centers.reshape(-1, 3)

        # Get corner points for all boxes at once
        corner_points = BatchBox(boxes_flat).getCornerPoints(centers_flat, mode="full")

        # Reshape back to [batch_size, n_boxes * 8, 3]
        points = corner_points.reshape(self.batch_size, self.n_boxes * 8, 3)

        # Find min and max along each dimension for each batch
        # Shape: [batch_size, 3]
        mins = points.min(dim=1)[0]  # min over all points for each batch
        maxs = points.max(dim=1)[0]  # max over all points for each batch

        # Calculate dimensions of bounding box
        dimensions = maxs - mins  # [batch_size, 3]

        # Calculate volume
        volume = torch.abs(dimensions.prod(dim=1))  # [batch_size]

        return volume

    def getRoughViewArea(self, view):
        # Collect corner points from all boxes in a batched way
        # Reshape boxes to [batch_size * n_boxes, 3]
        boxes_flat = self.boxes.reshape(-1, 3)
        centers_flat = self.centers.reshape(-1, 3)

        # Get corner points for all boxes at once
        corner_points = BatchBox(boxes_flat).getCornerPoints(centers_flat, mode="full")

        # Reshape back to [batch_size, n_boxes * 8, 3]
        points = corner_points.reshape(self.batch_size, self.n_boxes * 8, 3)

        # Extract relevant coordinates based on view
        if view == "frontal":
            points = points[:, :, [1, 2]]  # y, z coordinates
        elif view == "side":
            points = points[:, :, [0, 2]]  # x, z coordinates
        elif view == "top":
            points = points[:, :, [0, 1]]  # x, y coordinates
        else:
            raise ValueError("Invalid view. Must be 'frontal', 'side', or 'top'")

        # Find min and max along each dimension for each batch
        # Shape: [batch_size, 2]
        mins = points.min(dim=1)[0]  # min over all points for each batch
        maxs = points.max(dim=1)[0]  # max over all points for each batch

        # Calculate dimensions of bounding box
        dimensions = maxs - mins  # [batch_size, 2]

        # Calculate area
        area = torch.abs(dimensions.prod(dim=1))  # [batch_size]

        return area

    def getAllShiftedCornerPoints(self, internalsbase_xyz, mode="positive"):
        if mode not in ["positive", "full"]:
            raise ValueError("Invalid mode. Must be 'positive' or 'full'")

        k = 4 if mode == "positive" else 8

        # Collect corner points from all boxes in a batched way
        # Reshape boxes to [batch_size * n_boxes, 3]
        boxes_flat = self.boxes.reshape(-1, 3)
        centers_flat = self.centers.reshape(-1, 3)

        # Get corner points for all boxes at once
        corner_points = BatchBox(boxes_flat).getCornerPoints(centers_flat, mode=mode)

        # Reshape back to [batch_size, n_boxes * k, 3]
        points = corner_points.reshape(self.batch_size, self.n_boxes * k, 3)

        # Add the base shift to each box's center position
        shifted_corner_points = points + internalsbase_xyz.unsqueeze(1)

        return shifted_corner_points

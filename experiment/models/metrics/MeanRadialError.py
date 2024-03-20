import torch
from torch import nn


class MeanRadialError(nn.Module):
    def __init__(
        self,
        resized_image_size: tuple[int, int],
        original_image_size_mm: tuple[int, int],
    ):
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.resized_image_size = resized_image_size
        self.original_image_size_mm = original_image_size_mm

        self.init_ratios()

    def init_ratios(self):
        height, width = self.resized_image_size
        height_mm, width_mm = self.original_image_size_mm

        self.height_ratio = height_mm / height
        self.width_ratio = width_mm / width

    def px_to_mm(self, px: torch.Tensor) -> torch.Tensor:

        px[:, 0] *= self.height_ratio
        px[:, 1] *= self.width_ratio

        return px

    def forward(
        self,
        predicted_points: torch.Tensor,
        ground_truth_points: torch.Tensor,
    ) -> torch.Tensor:
        difference = predicted_points - ground_truth_points
        difference_mm = self.px_to_mm(difference)

        distance = (difference_mm ** 2).sum(dim=-1).sqrt()

        mask = (ground_truth_points > 0).prod(-1)

        distance *= mask

        return distance.mean(dim=0)

    def percent_under_n_mm(
        self,
        unreduced_mre: torch.Tensor,
        targets: torch.Tensor,
        n_mm: int
    ) -> torch.Tensor:
        mask = (targets > 0).prod(-1)
        under_n_mm = (unreduced_mre < n_mm).sum().float()
        total = mask.sum().float()

        return under_n_mm / total

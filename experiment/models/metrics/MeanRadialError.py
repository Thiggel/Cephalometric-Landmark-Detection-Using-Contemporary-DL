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

        self.resized_image_size = torch.tensor(resized_image_size).to(self.device) \
            .view(1, 1, -1).float()
        self.original_image_size_mm = torch.tensor(original_image_size_mm) \
            .to(self.device).view(1, 1, -1).float()

    def forward(
        self,
        predicted_points: torch.Tensor,
        ground_truth_points: torch.Tensor,
    ) -> torch.Tensor:
        difference = predicted_points - ground_truth_points
        diff_btwn_zero_one = difference / self.resized_image_size.flip(-1)
        difference_mm = diff_btwn_zero_one * self.original_image_size_mm.flip(-1)

        distance = (difference_mm ** 2).sum(dim=-1).sqrt()

        mask = (ground_truth_points > 0).prod(-1)

        distance *= mask

        return distance

    def percent_under_n_mm(
        self,
        unreduced_mre: torch.Tensor,
        targets: torch.Tensor,
        n_mm: int
    ) -> torch.Tensor:
        mask = (targets > 0).prod(-1)
        under_n_mm = (unreduced_mre <= n_mm).sum().float()
        total = mask.sum().float()

        return under_n_mm / total

import torch
from torch import nn
import torch.nn.functional as F


class MaskedWingLoss(nn.Module):
    def __init__(
        self,
        w: float = 10,
        epsilon: float = 2,
        old_px_per_m: int = 7_756,
        original_image_size: tuple[int, int] = (1360, 1840),
        resize_to: tuple[int, int] = (450, 450),
    ):
        super().__init__()

        self.w = w
        self.epsilon = epsilon
        self.C = w - w * torch.tensor(1 + w / epsilon).log()

        self.px_per_m = self._get_px_per_m(
            old_px_per_m=old_px_per_m,
            original_image_size=original_image_size,
            resize_to=resize_to
        )

    def _get_px_per_m(
        self,
        old_px_per_m: int,
        original_image_size: tuple[int, int],
        resize_to: tuple[int, int],
    ) -> torch.Tensor:
        old_px_per_m = torch.tensor(old_px_per_m)
        original_image_size = torch.tensor(original_image_size)
        original_image_size_m = original_image_size / old_px_per_m
        resize_to = torch.tensor(resize_to)
        new_px_per_m = original_image_size_m / resize_to

        return new_px_per_m

    def wing_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        magnitude = F.l1_loss(predictions, targets, reduction='none')

        loss = torch.where(
            magnitude < self.w,
            self.w * torch.log(1 + magnitude / self.epsilon),
            magnitude - self.C
        )

        return loss, magnitude

    def magnitude_to_mm(
        self,
        magnitude: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        m_to_mm = 1000
        mm_error = magnitude * self.px_per_m * m_to_mm
        masked_mm_error = mm_error * mask
        masked_mm_error = masked_mm_error.squeeze().mean(0)

        return masked_mm_error

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        with_mm_error: bool = False
    ) -> torch.Tensor:
        loss, magnitude = self.wing_loss(predictions, targets)

        mask = (targets > 0).prod(-1)
        masked_loss = (loss * mask).mean()

        masked_unreduced_mm_error = self.magnitude_to_mm(magnitude, mask) \
            if with_mm_error else None

        return masked_loss, masked_unreduced_mm_error

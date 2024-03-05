import torch
from torch import nn


class MaskedWingLoss(nn.Module):
    def __init__(
        self,
        w: float = 10,
        epsilon: float = 2,
        old_px_per_m: int = 7_756,
        original_image_size: tuple[int, int] = (1360, 1840),
        resized_images_shape: tuple[int, int] = (224, 224),
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * torch.tensor(1 + w / epsilon, device=self.device).log()

        self.px_per_m = self._get_px_per_m(
            old_px_per_m=old_px_per_m,
            original_image_size=original_image_size,
            resized_images_shape=resized_images_shape
        )

    def _get_px_per_m(
        self,
        old_px_per_m: int,
        original_image_size: tuple[int, int],
        resized_images_shape: tuple[int, int],
    ) -> torch.Tensor:
        old_px_per_m = torch.tensor(old_px_per_m, device=self.device)
        original_image_size = torch.tensor(original_image_size, device=self.device)
        original_image_size_m = original_image_size / old_px_per_m
        resized_images_shape = torch.tensor(resized_images_shape, device=self.device)
        new_px_per_m = original_image_size_m / resized_images_shape

        return new_px_per_m

    def difference_to_magnitude(
        self,
        difference: torch.Tensor
    ) -> torch.Tensor:
        return (difference ** 2).sum(-1).sqrt()

    def wing_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        magnitude = self.difference_to_magnitude(targets - predictions)

        loss = torch.where(
            magnitude < self.w,
            self.w * torch.log(1 + magnitude / self.epsilon),
            magnitude - self.C
        )

        return loss, magnitude

    def mm_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        m_to_mm = 1000
        difference = predictions - targets
        difference_mm = difference * self.px_per_m * m_to_mm
        magnitude = self.difference_to_magnitude(difference_mm)

        masked_mm_error = magnitude * mask
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

        masked_unreduced_mm_error = self.mm_error(
            predictions, targets, mask
        ) if with_mm_error else None

        return masked_loss, masked_unreduced_mm_error

    def percent_under_n_mm(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_mm: int
    ) -> torch.Tensor:
        m_to_mm = 1000
        difference = predictions - targets
        difference_mm = difference * self.px_per_m * m_to_mm
        magnitude = self.difference_to_magnitude(difference_mm)

        mask = (targets > 0).prod(-1)
        masked_magnitude = magnitude * mask

        under_n_mm = (masked_magnitude < n_mm).sum().float()
        total = mask.sum().float()

        return under_n_mm / total

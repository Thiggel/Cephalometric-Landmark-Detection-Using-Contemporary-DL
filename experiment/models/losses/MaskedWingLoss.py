import torch
from torch import nn


class MaskedWingLoss(nn.Module):
    def __init__(
        self,
        w: float = 10,
        epsilon: float = 2,
        px_to_mm: int = 0.1,
        original_image_size: tuple[int, int] = (1360, 1840),
        resized_image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * torch.tensor(1 + w / epsilon, device=self.device).log()

        self.px_to_mm = self._get_px_to_mm(
            px_to_mm=px_to_mm,
            original_image_size=original_image_size,
            resized_image_size=resized_image_size,
        )

    def _get_px_to_mm(
        self,
        px_to_mm: int,
        original_image_size: tuple[int, int],
        resized_image_size: tuple[int, int],
    ) -> torch.Tensor:
        px_to_mm = torch.tensor(px_to_mm, device=self.device)
        original_image_size = torch.tensor(original_image_size, device=self.device)
        resized_image_size = torch.tensor(resized_image_size, device=self.device)

        size_ratio = original_image_size / resized_image_size

        new_px_to_mm = px_to_mm * size_ratio

        return new_px_to_mm

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
        difference = predictions - targets
        difference_mm = difference * self.px_to_mm
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
        difference = predictions - targets
        difference_mm = difference * self.px_to_mm
        magnitude = self.difference_to_magnitude(difference_mm)

        mask = (targets > 0).prod(-1)
        masked_magnitude = magnitude * mask

        under_n_mm = (masked_magnitude < n_mm).sum().float()
        total = mask.sum().float()

        return under_n_mm / total

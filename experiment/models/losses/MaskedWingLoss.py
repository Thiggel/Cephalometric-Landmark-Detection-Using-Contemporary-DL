import torch
from torch import nn


class MaskedWingLoss(nn.Module):
    def __init__(
        self,
        w: float = 10,
        epsilon: float = 2,
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * torch.tensor(1 + w / epsilon, device=self.device).log()

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

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        loss, magnitude = self.wing_loss(predictions, targets)

        mask = (targets > 0).prod(-1)
        masked_loss = (loss * mask).mean()

        return masked_loss

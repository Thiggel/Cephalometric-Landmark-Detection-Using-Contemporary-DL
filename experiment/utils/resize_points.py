import torch

def resize_points(
    points: torch.Tensor,
    resize_to: tuple[int, int],
    resize_points_to_aspect_ratio: tuple[int, int]
) -> torch.Tensor:
    resize_factor = (
        torch.tensor(resize_to, device=points.device)
        / torch.tensor(resize_points_to_aspect_ratio, device=points.device)
    ).unsqueeze(0).unsqueeze(0)
    resized_points = points * resize_factor

    return resized_points

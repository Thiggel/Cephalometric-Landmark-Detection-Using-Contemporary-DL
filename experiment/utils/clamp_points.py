import torch


def clamp_points(
    points: torch.Tensor,
    images: torch.Tensor
) -> torch.Tensor:
    image_height, image_width = images.shape[-1], images.shape[-2]

    points = torch.clamp(points, min=0)
    points[..., 0] = torch.clamp(points[..., 0], max=image_width)
    points[..., 1] = torch.clamp(points[..., 1], max=image_height)

    return points

import torch


def resize_points(
    points: torch.Tensor,
    resized_image_size: tuple[int, int],
    resized_point_reference_frame_size: tuple[int, int]
) -> torch.Tensor:
    resize_factor = (
        torch.tensor(resized_image_size, device=points.device)
        / torch.tensor(
            resized_point_reference_frame_size,
            device=points.device
        )
    ).unsqueeze(0).unsqueeze(0)
    resized_points = points * resize_factor

    return resized_points

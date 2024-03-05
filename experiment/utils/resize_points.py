import torch


def resize_points(
    points: torch.Tensor,
    resized_images_shape: tuple[int, int],
    resized_points_reference_frame_shape: tuple[int, int]
) -> torch.Tensor:
    resize_factor = (
        torch.tensor(resized_images_shape, device=points.device)
        / torch.tensor(
            resized_points_reference_frame_shape,
            device=points.device
        )
    ).unsqueeze(0).unsqueeze(0)
    resized_points = points * resize_factor

    return resized_points

import torch


class CanExtractPatches:
    def _extract_patch(
        self,
        image: torch.Tensor,
        x: int,
        y: int,
        debug: bool = False
    ) -> torch.Tensor:
        image_height, image_width = image.shape[-2:]
        patch_height, patch_width = self.patch_size

        x = min(image_width, max(0, x.round().int()))
        y = min(image_height, max(0, y.round().int()))

        x_offset = patch_width // 2
        y_offset = patch_height // 2

        x_min = max(0, x - x_offset)
        x_max = min(image_width, x + x_offset)
        y_min = max(0, y - y_offset)
        y_max = min(image_height, y + y_offset)

        x_offset = max(0, x_offset - x)
        y_offset = max(0, y_offset - y)

        patch = torch.zeros(patch_width, patch_height)

        patch[
            y_offset:y_offset + y_max - y_min,
            x_offset:x_offset + x_max - x_min,
        ] = image[
            ...,
            y_min:y_max,
            x_min:x_max,
        ]

        return patch

    def _extract_patches(
        self,
        images: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_points, _ = coords.shape

        patches = torch.zeros(
            batch_size,
            num_points,
            *self.patch_size
        )

        for i in range(batch_size):
            for j in range(num_points):
                x, y = coords[i, j]
                patch = self._extract_patch(images[i], x, y)
                patches[i, j] = patch

        return patches.unsqueeze(2)

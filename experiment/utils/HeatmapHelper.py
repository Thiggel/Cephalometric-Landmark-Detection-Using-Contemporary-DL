import torch
import torch.nn.functional as F


class HeatmapHelper:
    def __init__(
        self,
        original_image_size: tuple[int, int],
        resized_image_size: tuple[int, int],
        resized_point_reference_frame_size: tuple[int, int],
        patch_size: tuple[int, int],
    ):
        self.original_image_size = original_image_size
        self.resized_image_size = resized_image_size
        self.resized_point_reference_frame_size = resized_point_reference_frame_size
        self.patch_size = patch_size
        self.resized_patch_size = self._get_resized_patch_size()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_highest_points(
        self,
        heatmaps: torch.Tensor
    ) -> torch.Tensor:
        """
        For heatmaps of shape (batch_size, num_points, height, width),
        this function returns the highest points in the shape
        (batch_size, num_points, 2)
        """
        batch_size, num_points, height, width = heatmaps.shape
        reshaped_heatmaps = heatmaps.reshape(
            batch_size, num_points, -1
        )

        argmax_indices_flat = torch.argmax(reshaped_heatmaps, dim=2)
        y_offset = argmax_indices_flat // width
        x_offset = argmax_indices_flat % width
        argmax_indices = torch.stack([x_offset, y_offset], dim=2)

        return argmax_indices

    def paste_heatmaps(
        self,
        global_heatmaps: torch.Tensor,
        local_heatmaps: torch.Tensor,
        point_predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Paste local heatmaps back into the global heatmaps.
        It is assumed that each local heatmap is a patch that
        corresponds to a point in point_predictions. Hence,
        each local heatmap is pasted back into the global heatmap
        at the position of the corresponding point.
        """
        resized_local_heatmaps = F.interpolate(
            local_heatmaps,
            self.resized_patch_size
        )

        batch_size, num_points, _ = point_predictions.shape

        padding_height, padding_width = self._get_padding_size(
            self.resized_patch_size
        )

        padded_global_heatmaps = self._pad_images(
            global_heatmaps,
            (padding_height, padding_width)
        )

        adjusted_points = self._adjust_points(
            point_predictions,
            (padding_height, padding_width)
        )

        y_indices, x_indices = self._get_patch_positions_in_images(
            adjusted_points,
            (padding_height, padding_width),
            padded_global_heatmaps,
            self.resized_patch_size
        )

        horizontal_strip = self._cut_out_horizontal_strip(
            padded_global_heatmaps,
            y_indices
        )

        strip_with_patch = self._paste_heatmaps_into_horizontal_strip(        
            horizontal_strip,
            resized_local_heatmaps,
            x_indices
        )

        heatmap_with_patch = self._paste_horizontal_strip_into_global_heatmaps(
            padded_global_heatmaps,
            strip_with_patch,
            y_indices
        )

        final_heatmaps = self._remove_padding_from_heatmaps(
            heatmap_with_patch,
            (padding_height, padding_width)
        )

        return final_heatmaps

    def _remove_padding_from_heatmaps(
        self,
        heatmaps: torch.Tensor,
        padding_size: tuple[int, int]
    ) -> torch.Tensor:
        padding_height, padding_width = padding_size
        return heatmaps[
            :, :, padding_height:-padding_height, padding_width:-padding_width
        ]

    def _paste_horizontal_strip_into_global_heatmaps(
        self,
        global_heatmaps: torch.Tensor,
        horizontal_strip: torch.Tensor,
        y_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        The horizontal strip is pasted back into the global heatmaps
        """
        return torch.scatter(
            global_heatmaps,
            -2,
            y_indices,
            horizontal_strip
        )

    def _paste_heatmaps_into_horizontal_strip(
        self,
        horizontal_strip: torch.Tensor,
        local_heatmaps: torch.Tensor,
        x_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        The local heatmaps are pasted into the horizontal strip
        """
        strip_with_patch = torch.scatter(
            horizontal_strip,
            -1,
            x_indices,
            local_heatmaps
        )

    def _cut_out_horizontal_strip(
        self,
        images: torch.Tensor,
        y_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        A horizontal strip of the global heatmaps is extracted
        at the position of the points. The strip is as wide as
        the padded_global_heatmaps and as high as the patch.
        """
        return images.gather(2, y_indices)

    def create_heatmaps(
        self,
        points: torch.Tensor,
        gaussian_sd: float = 1
    ) -> torch.Tensor:
        """
        Create heatmaps for target points.
        The resulting tensor's shape will be
        (batch_size, num_points, image_height, image_width).
        A mask is returned alongside for points where
        one coordinate is negative. These can then be filtered out
        of the loss.
        """
        batch_size, num_points, _ = points.shape

        mask = (points[..., 0] >= 0) & (points[..., 1] >= 0)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(
                self.resized_point_reference_frame_size[0],
                device=self.device
            ),
            torch.arange(
                self.resized_point_reference_frame_size[1],
                device=self.device
            ),
        )

        y_grid = y_grid.unsqueeze(0).unsqueeze(0)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)

        x, y = points.split(1, dim=-1)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

        heatmaps = torch.exp(
            -0.5 * ((y_grid - y) ** 2 + (x_grid - x) ** 2) / (gaussian_sd ** 2)
        )

        return heatmaps, mask.unsqueeze(-1).unsqueeze(-1)

    def _get_resized_patch_size(self) -> torch.Tensor:
        """
        For some methods, a patch is extracted from the original image
        which is a lot larger than the resized image. This function
        calculates to what size this large patch should be resized to
        be in the same aspect ratio as the resized image.
        That way, the local heatmaps can be pasted back into the
        global heatmaps at the respective position of the refined
        point prediction.
        """
        resize_factor = self.resized_image_size[0] / self.original_image_size[0]
        resized_patch_size = (
            int(resize_factor * self.resized_image_size[0]),
            int(resize_factor * self.resized_image_size[1])
        )

        return resized_patch_size

    def _pad_images(
        self,
        images: torch.Tensor,
        padding_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Pad images with zeros, so that patches can be extracted
        from the images even at the border.
        """
        padding_height, padding_width = padding_size

        return F.pad(
            images,
            (padding_width, padding_width, padding_height, padding_height)
        )

    def _repeat_images(
        self,
        images: torch.Tensor,
        num_points: int
    ) -> torch.Tensor:
        """
        Repeat images for each point so that multiple patches can be
        extracted from them.
        """
        return images.repeat(1, num_points, 1, 1)

    def _adjust_points(
        self,
        points: torch.Tensor,
        padding_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Adjust points to the padded image.
        """
        return points + torch.tensor(
            padding_size[::-1],
            device=points.device
        )

    def _get_padding_size(
        self,
        patch_size: tuple[int, int]
    ) -> tuple[int, int]:
        patch_height, patch_width = patch_size
        padding_height = patch_height // 2
        padding_width = patch_width // 2

        return padding_height, padding_width

    def _pad_and_repeat_images(
        self,
        images: torch.Tensor,
        points: torch.Tensor,
        padding_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_height, padding_width = padding_size
        _, num_points, _ = points.shape

        padded_images = self._pad_images(
            images, (padding_height, padding_width)
        )
        repeated_images = self._repeat_images(padded_images, num_points)
        adjusted_points = self._adjust_points(
            points, (padding_height, padding_width)
        )

        return repeated_images, adjusted_points

    def _get_patch_positions_in_images(
        self,
        points: torch.Tensor,
        padding_size: tuple[int, int],
        images: torch.Tensor,
        patch_size
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each point, this function returns the x and y indices
        at which a patch of size patch_size is located in an image
        padded with padding_size.
        This can be used to paste local heatmaps back into the
        global heatmaps.
        """
        padding_height, padding_width = padding_size
        image_height, image_width = images.shape[-2:]
        patch_height, patch_width = patch_size

        y_indices = (
            points[:, :, 1].unsqueeze(2) +
            torch.arange(
                patch_height,
                0,
                step=-1,
                device=images.device
            ) - padding_height - 1
        ).unsqueeze(-1).repeat(1, 1, 1, image_width)

        x_indices = (
            points[:, :, 0].unsqueeze(2) +
            torch.arange(
                patch_width,
                device=images.device
            ) - padding_width
        ).unsqueeze(-2).repeat(1, 1, patch_height, 1)

        return y_indices, x_indices

    def extract_patches(self, images, points):
        padding_height, padding_width = self._get_padding_size(self.patch_size)

        padded_images, adjusted_points = self._pad_and_repeat_images(
            images, points, (padding_height, padding_width)
        )

        padded_height, padded_width = padded_images.shape[-2:]

        y_indices, x_indices = self._get_patch_positions_in_images(
            adjusted_points,
            (padding_height, padding_width),
            padded_images,
            self.patch_size
        )

        patches = padded_images.gather(2, y_indices).gather(3, x_indices)

        return patches

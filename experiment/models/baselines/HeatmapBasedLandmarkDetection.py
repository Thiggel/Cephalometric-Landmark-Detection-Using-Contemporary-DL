import torch
import torch.nn.functional as F


class HeatmapBasedLandmarkDetection:
    @property
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_highest_points(
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

    def _paste_heatmaps(
        self,
        global_heatmaps: torch.Tensor,
        local_heatmaps: torch.Tensor,
        point_predictions: torch.Tensor,
    ) -> torch.Tensor:
        resized_local_heatmaps = F.interpolate(
            local_heatmaps,
            self.patch_resize_to
        )

        batch_size, num_points, _ = point_predictions.shape

        padding_height, padding_width = self._get_padding_size(
            self.patch_resize_to
        )

        padded_global_heatmaps = self._pad_images(
            global_heatmaps.clone(),
            (padding_height, padding_width)
        )

        adjusted_points = self._adjust_points(
            point_predictions,
            (padding_height, padding_width)
        )

        y_indices, x_indices = self._get_indices(
            adjusted_points,
            (padding_height, padding_width),
            padded_global_heatmaps,
            self.patch_resize_to
        )

        horizontal_strip = padded_global_heatmaps.gather(2, y_indices)

        horizontal_strip.scatter_(-1, x_indices, resized_local_heatmaps)

        padded_global_heatmaps.scatter_(-2, y_indices, horizontal_strip)

        final_heatmaps = padded_global_heatmaps[
            :, :, padding_height:-padding_height, padding_width:-padding_width
        ]

        return final_heatmaps

    def _create_heatmaps(
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
            torch.arange(self.resize_to[0], device=self.device),
            torch.arange(self.resize_to[1], device=self.device),
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

    def _get_patch_resize_to(self) -> torch.Tensor:
        """
        For some methods, a patch is extracted from the original image
        which is a lot larger than the resized image. This function
        calculates to what size this large patch should be resized to
        be in the same aspect ratio as the resized image.
        That way, the local heatmaps can be pasted back into the
        global heatmaps at the respective position of the refined
        point prediction.
        """
        resize_factor = self.resize_to[0] / self.original_image_size[0]
        patch_resize_to = (
            int(resize_factor * self.resize_to[0]),
            int(resize_factor * self.resize_to[1])
        )

        return patch_resize_to

    def _pad_images(
        self,
        images: torch.Tensor,
        padding_size: tuple[int, int]
    ) -> torch.Tensor:
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
        return images.repeat(1, num_points, 1, 1)

    def _adjust_points(
        self,
        points: torch.Tensor,
        padding_size: tuple[int, int]
    ) -> torch.Tensor:
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

    def _get_indices(
        self,
        points: torch.Tensor,
        padding_size: tuple[int, int],
        images: torch.Tensor,
        patch_size
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _extract_patches(self, images, points):
        padding_height, padding_width = self._get_padding_size(self.patch_size)

        padded_images, adjusted_points = self._pad_and_repeat_images(
            images, points, (padding_height, padding_width)
        )

        padded_height, padded_width = padded_images.shape[-2:]

        y_indices, x_indices = self._get_indices(
            adjusted_points,
            (padding_height, padding_width),
            padded_images,
            self.patch_size
        )

        patches = padded_images.gather(2, y_indices).gather(3, x_indices)

        return patches

    def forward_batch(
        self,
        images: torch.Tensor,
        patch_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, channels, height, width = images.shape

        output = self.global_module(
            images
        )

        offset_maps = output[:, :2 * self.num_points].view(
            batch_size, self.num_points, 2, height, width
        ) if self.use_offset_maps else None

        global_heatmaps = output[:, 2 * self.num_points:] \
            if self.use_offset_maps else output

        point_predictions = self._get_highest_points(
            global_heatmaps
        )

        regions_of_interest = self._extract_patches(
            images, point_predictions
        )

        patch_height, patch_width = patch_size

        local_heatmaps = self.local_module(
            regions_of_interest.view(
                batch_size * self.num_points,
                channels,
                patch_height,
                patch_width
            )
        ).view(
            batch_size,
            self.num_points,
            patch_height,
            patch_width
        )

        local_heatmaps = self._paste_heatmaps(
            global_heatmaps,
            local_heatmaps,
            point_predictions,
        )

        refined_point_predictions = self._get_highest_points(
            local_heatmaps
        )

        return (
            global_heatmaps,
            local_heatmaps,
            refined_point_predictions,
            offset_maps
        )

    def _create_offset_maps(
        self,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_points, _ = targets.shape

        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.resize_to[0], device=self.device),
            torch.arange(self.resize_to[1], device=self.device),
        )

        y_grid = y_grid.unsqueeze(0).unsqueeze(0)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)

        x, y = targets.split(1, dim=-1)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        x_offset_maps = (x - x_grid).abs().unsqueeze(2)
        y_offset_maps = (y - y_grid).abs().unsqueeze(2)

        offset_maps = torch.cat([
            x_offset_maps,
            y_offset_maps
        ], dim=2) / self.offset_map_radius

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(offset_maps[0, 0, 0].detach().cpu().numpy())
        ax[1].imshow(offset_maps[0, 0, 1].detach().cpu().numpy())
        ax[0].scatter(targets[0, 0, 0], targets[0, 0, 1], c='r')
        ax[1].scatter(targets[0, 0, 0], targets[0, 0, 1], c='r')
        plt.show()


        return offset_maps

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, targets = batch

        (
            global_heatmaps,
            local_heatmaps,
            predictions,
            offset_maps
        ) = self.forward_with_heatmaps(images)

        target_heatmaps, mask = self._create_heatmaps(targets)
        target_offset_maps = self._create_offset_maps(targets) \
            if self.use_offset_maps else None

        offset_loss = self.loss(
            offset_maps,
            target_offset_maps
        ).mean(2) if self.use_offset_maps else 0

        loss = self.loss(
            global_heatmaps,
            target_heatmaps,
        ) + self.loss(
            local_heatmaps,
            target_heatmaps,
        ) + offset_loss

        masked_loss = loss * mask

        return (
            masked_loss.mean(),
            predictions,
            global_heatmaps,
            local_heatmaps,
            target_heatmaps
        )

    def validation_test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, predictions, _, _, _ = self.step(batch)
        targets = batch[1]

        _, unreduced_mm_error = self.mm_error(
            predictions,
            targets,
            with_mm_error=True
        )

        return loss, unreduced_mm_error, predictions, targets

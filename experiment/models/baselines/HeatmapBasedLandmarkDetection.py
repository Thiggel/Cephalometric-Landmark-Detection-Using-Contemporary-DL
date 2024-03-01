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
        reshaped_heatmaps = heatmaps.view(
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
        point_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        each global heatmap has a shape of (44, 256, 256)
        it contains one heatmap for each point

        each local heatmap was created from patch of the original
        unresized image, so a patch of 256x256 pixels was cut out
        around the respective point prediction at from there on
        a refined prediction was created.
        The local heatmaps tensor thus has a shape of (44, 256, 256)
        but each of the 44 heatmaps refers to a patch of the original
        image.
        This function pastes the local heatmaps back into the global
        heatmaps at the respective position of the refined point
        specifically, it first resizes the local heatmaps to be in the
        same aspect ratio as the global heatmaps.
        it then takes the point prediction from the point_predictions
        tensor, and pastes the resized patch for each point into the
        global heatmap at the respective position of the refined
        point prediction.
        """
        resized_local_heatmaps = F.interpolate(
            local_heatmaps,
            self.patch_resize_to,
        )

        batch_size, num_points, _ = point_predictions.shape
        patch_height, patch_width = self.patch_resize_to

        final_heatmaps = global_heatmaps.clone()

        for datapoint_idx in range(batch_size):
            for point_idx in range(num_points):
                point = point_predictions[datapoint_idx][point_idx]

                (
                    image_x1,
                    image_x2,
                    image_y1,
                    image_y2,

                    patch_y1,
                    patch_y2,
                    patch_x1,
                    patch_x2,
                ) = self._calculate_bounding_boxes(
                    point[0],
                    point[1],
                    self.patch_resize_to,
                    self.resize_to
                )

                final_heatmaps[
                    datapoint_idx,
                    point_idx,
                    image_y1:image_y2,
                    image_x1:image_x2,
                ] = resized_local_heatmaps[
                    datapoint_idx,
                    point_idx,
                    patch_y1:patch_y2,
                    patch_x1:patch_x2,
                ]

        torch.testing.assert_allclose(
            final_heatmaps,
            self._paste_heatmaps_efficient(
                global_heatmaps,
                local_heatmaps,
                point_predictions
            )
        )

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

        return torch.tensor(
            patch_resize_to,
            device=self.device
        )

    def _calculate_bounding_boxes(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        patch_size: torch.Tensor,
        image_size: tuple[int, int]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        """
        This function takes in a point (x, y) and a patch size and
        calculates the bounding boxes for the patch and the image
        that the patch is extracted from.
        If the patch would be outside of the image, the bounding
        boxes are adjusted accordingly.
        Hence, it can be used to calculate the bounding boxes for
        patch extraction from an image as well as pasting a patch
        into an image.
        """
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        x = min(image_width, max(0, x.round().int()))
        y = min(image_height, max(0, y.round().int()))

        x_offset = patch_width // 2
        y_offset = patch_height // 2

        image_y1 = max(0, y - y_offset)
        image_y2 = min(image_height, y + y_offset)
        image_x1 = max(0, x - x_offset)
        image_x2 = min(image_width, x + x_offset)

        x_offset = max(0, x_offset - x)
        y_offset = max(0, y_offset - y)

        patch_y1 = y_offset
        patch_y2 = y_offset + image_y2 - image_y1
        patch_x1 = x_offset
        patch_x2 = x_offset + image_x2 - image_x1

        return (
            image_x1,
            image_x2,
            image_y1,
            image_y2,

            patch_y1,
            patch_y2,
            patch_x1,
            patch_x2,
        )

    def _extract_patches(self, images, points):
        patch_height, patch_width = self.patch_size
        # Get batch size, number of channels, height, and width of images
        batch_size, channels, height, width = images.size()
        _, num_points, _ = points.size()

        # Calculate padding amount
        padding_height = patch_height // 2
        padding_width = patch_width // 2

        # Pad images with zeros
        padded_images = F.pad(images, (padding_width, padding_width, padding_height, padding_height)) \
            .repeat(1, num_points, 1, 1)

        _, _, padded_height, padded_width = padded_images.size()

        adjusted_points = points + torch.tensor([padding_width, padding_height], device=images.device)

        y_indices = (
            adjusted_points[:, :, 1].unsqueeze(2) +
            torch.arange(padding_height, -padding_height, step=-1, device=images.device)
        ).unsqueeze(-1).repeat(1, 1, 1, padded_width)

        x_indices = (
            adjusted_points[:, :, 0].unsqueeze(2) +
            torch.arange(-padding_width, padding_width, device=images.device)
        ).unsqueeze(-2).repeat(1, 1, patch_height, 1)

        patches = padded_images.gather(2, y_indices).gather(3, x_indices)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, batch_size)
        for image_idx in range(batch_size):
            ax[image_idx].imshow(images[image_idx].permute(1, 2, 0).cpu())
            for point_idx in range(5):
                x, y = points[image_idx, point_idx]
                ax[image_idx].imshow(
                    patches[image_idx, point_idx].unsqueeze(-1).cpu(),
                    extent=(x - padding_width, x + padding_width, y - padding_height, y + padding_height),
                    cmap='gray',
                    alpha=0.5
                )
                ax[image_idx].scatter(x, y)

        plt.show()
        exit()

        return patches

    def forward_batch(
        self,
        images: torch.Tensor,
        patch_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, channels, _, _ = images.shape

        global_heatmaps = self.global_module(
            images
        )

        point_predictions = self._get_highest_points(
            global_heatmaps
        )

        regions_of_interest = self._extract_patches(
            images, point_predictions
        )

        print(regions_of_interest.shape)
        exit()

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
            point_predictions
        )

        refined_point_predictions = self._get_highest_points(
            local_heatmaps
        )

        return global_heatmaps, local_heatmaps, refined_point_predictions

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, targets = batch

        (
            global_heatmaps,
            local_heatmaps,
            predictions,
        ) = self.forward_with_heatmaps(images)

        target_heatmaps, mask = self._create_heatmaps(targets)

        loss = self.loss(
            global_heatmaps,
            target_heatmaps,
        ) + self.loss(
            local_heatmaps,
            target_heatmaps,
        )

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

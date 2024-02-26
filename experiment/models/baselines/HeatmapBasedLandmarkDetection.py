import torch
from torchvision.transforms.functional import resize
import torch.nn.functional as F
import gc


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
        resized_local_heatmaps = resize(
            local_heatmaps,
            self.patch_resize_to,
        )

        batch_size, num_points, _ = point_predictions.shape
        patch_height, patch_width = self.patch_resize_to

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

                global_heatmaps[
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

        return global_heatmaps

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

        if not hasattr(self, 'x_grid') or not hasattr(self, 'y_grid'):
            self.y_grid, self.x_grid = torch.meshgrid(
                torch.arange(self.resize_to[0], device=self.device),
                torch.arange(self.resize_to[1], device=self.device),
            )

            self.y_grid = self.y_grid.unsqueeze(0).unsqueeze(0)
            self.x_grid = self.x_grid.unsqueeze(0).unsqueeze(0)

        x, y = points.split(1, dim=-1)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

        self.heatmaps = torch.exp(
            -0.5 * ((self.y_grid - y) ** 2 + (self.x_grid - x) ** 2) / (gaussian_sd ** 2)
        )

        return self.heatmaps, mask.unsqueeze(-1).unsqueeze(-1)

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

    def _extract_patch(
        self,
        image: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        debug: bool = False
    ) -> torch.Tensor:
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
            x,
            y,
            self.patch_size,
            image.shape[-2:]
        )

        self.patch = torch.zeros(
            *self.patch_size,
            device=self.device
        )

        self.patch[
            patch_y1:patch_y2,
            patch_x1:patch_x2,
        ] = image[
            ...,
            image_y1:image_y2,
            image_x1:image_x2,
        ]

        return self.patch

    def _extract_patches(
        self,
        images: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_points, _ = coords.shape

        self.patches = torch.zeros(
            batch_size,
            num_points,
            *self.patch_size,
            device=self.device
        )

        for i in range(batch_size):
            for j in range(num_points):
                x, y = coords[i, j]
                self.patches[i, j] = self._extract_patch(images[i], x, y)

        return self.patches.unsqueeze(2)

    def forward_batch(
        self,
        images: torch.Tensor,
        patch_size: tuple[int, int],
    ) -> torch.Tensor:
        torch.cuda.empty_cache()
        gc.collect()

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

        del (
            regions_of_interest,
            point_predictions,
        )

        self._free_memory()

        return global_heatmaps, local_heatmaps, refined_point_predictions

    def _free_memory(self):
        if hasattr(self, 'patches'):
            del self.patches

        if hasattr(self, 'patch'):
            self.patch

        if hasattr(self, 'heatmaps'):
            self.heatmaps

        if hasattr(self, 'x_grid'):
            self.x_grid

        if hasattr(self, 'y_grid'):
            self.y_grid

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, targets = batch

        (
            global_heatmaps,
            local_heatmaps,
            predictions
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

        return masked_loss.mean(), predictions

    def validation_test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, predictions = self.step(batch)
        targets = batch[1]

        _, unreduced_mm_error = self.mm_error(
            predictions,
            targets,
            with_mm_error=True
        )

        return loss, unreduced_mm_error, predictions, targets

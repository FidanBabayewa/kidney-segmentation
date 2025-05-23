import torch
import random

class SegmentationPuzzleMix:
    def __init__(self, alpha=0.5):
        """
        PuzzleMix for full 3D volume mixing.
        :param alpha: Mixing ratio between 0 and 1
        """
        self.alpha = alpha

    def __call__(self, img1, mask1, img2, mask2):
        assert img1.shape == img2.shape, "Image shapes must match"
        assert mask1.shape == mask2.shape, "Mask shapes must match"

        mixed_img = self.alpha * img1 + (1 - self.alpha) * img2
        mixed_mask = (self.alpha * mask1 + (1 - self.alpha) * mask2) > 0.5
        mixed_mask = mixed_mask.float()

        return mixed_img, mixed_mask

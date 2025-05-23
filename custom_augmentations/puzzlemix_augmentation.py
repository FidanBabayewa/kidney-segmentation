import torch
import numpy as np
import random
import torch.nn.functional as F

class PuzzleMixAugmentation:
    def __init__(self, alpha=0.3, beta=0.3, patch_size=32):
        """
        PuzzleMix-style augmentation for 3D medical images.
        
        :param alpha: Mixup blending strength.
        :param beta: Randomized patch mixup strength.
        :param patch_size: Size of patches to shuffle within the image.
        """
        self.alpha = alpha
        self.beta = beta
        self.patch_size = patch_size

    def __call__(self, data, target):
        """
        Applies PuzzleMix augmentation.

        :param data: Input 3D image tensor (C, H, W, D)
        :param target: Corresponding segmentation mask (H, W, D)
        :return: Augmented image and mask
        """
        if random.random() > 0.5:  # Apply augmentation randomly
            return data, target

        # Get shape
        C, H, W, D = data.shape
        
        # Select random region for patch mixing
        px = random.randint(0, H - self.patch_size)
        py = random.randint(0, W - self.patch_size)
        pz = random.randint(0, D - self.patch_size)
        
        # Select another random region to swap
        px2 = random.randint(0, H - self.patch_size)
        py2 = random.randint(0, W - self.patch_size)
        pz2 = random.randint(0, D - self.patch_size)

        # Extract patches
        patch1 = data[:, px:px+self.patch_size, py:py+self.patch_size, pz:pz+self.patch_size]
        patch2 = data[:, px2:px2+self.patch_size, py2:py2+self.patch_size, pz2:pz2+self.patch_size]
        
        # Mix patches
        lambda_mix = np.random.beta(self.alpha, self.beta)
        mixed_patch = lambda_mix * patch1 + (1 - lambda_mix) * patch2

        # Replace patches in the original images
        data[:, px:px+self.patch_size, py:py+self.patch_size, pz:pz+self.patch_size] = mixed_patch

        # Apply same transformation to the segmentation mask
        target_patch1 = target[px:px+self.patch_size, py:py+self.patch_size, pz:pz+self.patch_size]
        target_patch2 = target[px2:px2+self.patch_size, py2:py2+self.patch_size, pz2:pz2+self.patch_size]
        mixed_target_patch = lambda_mix * target_patch1 + (1 - lambda_mix) * target_patch2
        target[px:px+self.patch_size, py:py+self.patch_size, pz:pz+self.patch_size] = mixed_target_patch

        return data, target

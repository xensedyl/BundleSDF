# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
import os
import numpy as np

class Segmenter():
    def __int__(self):
        return

    def run(self, mask_file=None):
        mask = cv2.imread(mask_file, -1)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask file: {mask_file}")

        if mask.ndim == 3:
            mask = np.any(mask > 0, axis=-1)
        else:
            mask = mask > 0

        return np.ascontiguousarray(mask.astype(np.uint8))

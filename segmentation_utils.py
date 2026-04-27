# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
import os
import numpy as np

class Segmenter():
    def __int__(self):
        return

    def run(self, mask_file=None):
        return (cv2.imread(mask_file, -1)>0).astype(np.uint8)

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <torch/extension.h>
#include "common.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sampleRaysUniformOccupiedVoxels", &sampleRaysUniformOccupiedVoxels);
    m.def("postprocessOctreeRayTracing", &postprocessOctreeRayTracing);
    m.def("rayColorToTextureImageCUDA", &rayColorToTextureImageCUDA);
}
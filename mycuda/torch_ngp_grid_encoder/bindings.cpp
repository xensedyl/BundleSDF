/*
 * Originally derived from torch-ngp grid encoder
 *   Copyright (c) 2022 hawkey. Licensed under the MIT License.
 *   See mycuda/torch_ngp_grid_encoder/LICENSE.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <torch/extension.h>

#include "gridencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_encode_forward", &grid_encode_forward, "grid_encode_forward (CUDA)");
    m.def("grid_encode_backward", &grid_encode_backward, "grid_encode_backward (CUDA)");
}
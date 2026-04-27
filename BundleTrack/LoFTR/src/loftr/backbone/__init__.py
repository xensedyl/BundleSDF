# Originally from LoFTR (Detector-Free Local Feature Matching with Transformers)
#   Copyright (c) ZJU3DV. Licensed under the Apache License, Version 2.0.
#   See BundleTrack/LoFTR/LICENSE.
#
# SPDX-License-Identifier: Apache-2.0

from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4


def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")

from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.modeling import ViT, SimpleFeatUpPyramid, SimpleFeatUpPyramid3
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .mask_rcnn_fpn import model
from ..data.constants import constants

model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"

# Base
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeatUpPyramid3)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    ),
    # upsampler_path="/mnt/haiwen/pretrained_models/FeatUp/checkpoints/pixelimplicit_sam_hr/HR_896_n_freqs_20_EMA_True_pretrained_True_crop_upsampler_True_vitdet_pixelimplicit_saccade_False_sam_reg_0.0_sam_hr_alpha_0.5_sam_hr_reg_0.0_sa1b_attention_crf_0.001_ent_0.0_tv_0.0_rec_1.0a100_2.ckpt",
    # upsampler_path="/mnt/haiwen/pretrained_models/FeatUp/checkpoints/pixelimplicit_sam_hr/HR_896_n_freqs_20_EMA_True_pretrained_True_crop_upsampler_True_vitdet_pixelimplicit_saccade_False_sam_reg_1.0_sam_hr_alpha_0.8_sam_hr_reg_0.0_sa1b_attention_crf_0.001_ent_0.0_tv_0.0_rec_1.0a100_2.ckpt",
    upsampler_path="/mnt/haiwen/pretrained_models/FeatUp/checkpoints/sam_mask_featup/vitdet_pixelimplicit_depth2_catLR_False_sam_reg_0.0_saccade_sam_mask_0.8_loadsize_224_upsample_size_224_sa1b_attention_crf_0.001_tv_0.0_ent_0.0a100_2.ckpt",
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(8.0, 4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    # square_pad=1024,
    square_pad=448,
)

model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "LN"

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

model.proposal_generator.in_features = ["p1", "p2", "p3", "p4", "p5", "p6"]
model.proposal_generator.anchor_generator.sizes = [[16], [32], [64], [128], [256], [512]]
model.proposal_generator.anchor_generator.strides = [2, 4, 8, 16, 32, 64]

model.roi_heads.box_in_features = ["p1", "p2", "p3", "p4", "p5"]
model.roi_heads.box_pooler.scales = (1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32)
model.roi_heads.mask_in_features = ["p1", "p2", "p3", "p4", "p5"]
model.roi_heads.mask_pooler.scales = (1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32)

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

import torch
import torch.nn as nn
import sys

from torchvision import transforms
import torch.nn.functional as F
from einops import rearrange

import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import numpy as np

class ChannelNorm(torch.nn.Module):
    ### Not compatible with DDP

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        new_x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return new_x

class MinMaxScaler(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = x.shape[1]
        flat_x = x.permute(1, 0, 2, 3).reshape(c, -1)
        flat_x_min = flat_x.min(dim=-1).values.reshape(1, c, 1, 1)
        flat_x_scale = flat_x.max(dim=-1).values.reshape(1, c, 1, 1) - flat_x_min
        return ((x - flat_x_min) / flat_x_scale.clamp_min(0.0001)) - .5

class ImplicitFeaturizer(torch.nn.Module):

    def __init__(self, color_feats=True, n_freqs=10, learn_bias=False, time_feats=False, lr_feats=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_feats = color_feats
        self.time_feats = time_feats
        self.n_freqs = n_freqs
        self.learn_bias = learn_bias

        self.dim_multiplier = 2

        if self.color_feats:
            self.dim_multiplier += 3

        if self.time_feats:
            self.dim_multiplier += 1

        if self.learn_bias:
            self.biases = torch.nn.Parameter(torch.randn(2, self.dim_multiplier, n_freqs).to(torch.float32))
        
        self.low_res_feat = lr_feats

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]).unsqueeze(0)
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        if self.color_feats:
            feat_list = [feats, original_image]
        else:
            feat_list = [feats]

        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)) \
            .reshape(1, self.n_freqs, 1, 1, 1) # torch.Size([1, 30, 1, 1, 1])
        feats = (feats * freqs) # torch.Size([1, 30, 5, 224, 224])

        if self.learn_bias:
            sin_feats = feats + self.biases[0].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1) # torch.Size([1, 30, 5, 224, 224])
            cos_feats = feats + self.biases[1].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1) # torch.Size([1, 30, 5, 224, 224])
        else:
            sin_feats = feats
            cos_feats = feats

        sin_feats = sin_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w) # torch.Size([1, 150, 224, 224])
        cos_feats = cos_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w) # torch.Size([1, 150, 224, 224])

        if self.color_feats:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats), original_image]
        else:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats)]

        if self.low_res_feat is not None:
            upsampled_feats = F.interpolate(self.low_res_feat, size=(h, w), mode='bilinear', align_corners=False)
            all_feats.append(upsampled_feats)

        return torch.cat(all_feats, dim=1)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)  # Norm for query
        self.norm_kv = nn.LayerNorm(dim)  # Norm for key/value
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)

    def forward(self, query, key, value):
        # Apply layer normalization
        query = self.norm_q(query)
        key = self.norm_kv(key)
        value = self.norm_kv(value)

        # Multi-head attention takes (sequence_length, batch_size, embedding_dim)
        query = query.permute(1, 0, 2)  # (seq_len, batch_size, dim)
        key = key.permute(1, 0, 2)      # (seq_len, batch_size, dim)
        value = value.permute(1, 0, 2)  # (seq_len, batch_size, dim)

        # Apply multi-head attention (cross-attention)
        attn_output, _ = self.attention(query, key, value)

        # Return to original format (batch_size, seq_len, dim)
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output

class CATransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CrossAttentionLayer(dim, heads=heads, dim_head=dim_head, dropout=dropout),  # Cross-Attention
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, query, key_value):
        for cross_attn, ff in self.layers:
            query = cross_attn(query, key_value, key_value) + query  # Cross-Attention
            # query = cross_attn(query, key_value, key_value) ## Because we are transforming imgs to features, we don't need to add the query back
            query = ff(query) + query  # Feed-Forward residual connection

        return self.norm(query)

class CAImplicitUpsampler(nn.Module):
    """
    Inspired by Pixel Nerf, we use Fourier features of images as inputs, and do cross attention with the LR features, the output is the HR features.
    """
    def __init__(self, dim, color_feats=True, cat_lr_feats=True, add_lr_feats=False, n_freqs=20, num_heads=4, num_layers=1, num_conv_layers=1, lr_size=16):
        super(CAImplicitUpsampler, self).__init__()

        if color_feats:
            start_dim = 5 * n_freqs * 2 + 3
        else:
            start_dim = 2 * n_freqs * 2
        if cat_lr_feats:
            start_dim += dim
        self.cat_lr_feats = cat_lr_feats
        self.add_lr_feats = add_lr_feats
        if add_lr_feats:  
            self.channel_norm = ChannelNorm(dim)
        self.dropout = nn.Dropout(p=.2)
        
        num_patches = lr_size * lr_size
        self.positional_encoding_lr = nn.Parameter(torch.randn(1, num_patches, dim))

        self.fourier_feat = torch.nn.Sequential(
                                MinMaxScaler(),
                                ImplicitFeaturizer(color_feats, n_freqs=n_freqs, learn_bias=True),
                            )
        self.first_conv = torch.nn.Sequential(
                                ChannelNorm(start_dim),
                                nn.Conv2d(start_dim, dim, kernel_size=1)
                            )

        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])


    def forward(self, lr_feats, img):
        # Step 1: Extract Fourier features from the input image
        x = self.fourier_feat(img) # Output shape: (B, dim, H, W)
        b, c, h, w = x.shape

        ## Resize and add LR feats to x? 
        if self.cat_lr_feats:
            lr_feats_resize = F.interpolate(lr_feats, size=(h, w), mode='bicubic')
            x = torch.cat([x, lr_feats_resize], dim=1)
        x = self.first_conv(x)
    
        # Reshape for attention (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        if self.add_lr_feats:
            lr_feats_norm = self.channel_norm(lr_feats)
            lr_feats_resize = F.interpolate(lr_feats_norm, size=(h, w), mode='bicubic')
            lr_feats_resize = lr_feats_resize.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            x = x + lr_feats_resize

        # Step 2: Process LR features for keys and values
        b, c_lr, h_lr, w_lr = lr_feats.shape

        lr_feats = lr_feats.flatten(2).permute(0, 2, 1) # Shape: (B, H_lr*W_lr, C)
        dim = self.positional_encoding_lr.shape[2]
        # Interpolate positional encoding to the size of lr_feats if needed
        if lr_feats.shape[1] != self.positional_encoding_lr.shape[1]:
            len_pos_old = int(math.sqrt(self.positional_encoding_lr.shape[1]))
            # first reshape the positional encoding to (1, C, H_lr, W_lr)
            pe = self.positional_encoding_lr.reshape(1, len_pos_old, len_pos_old, dim).permute(0, 3, 1, 2)
            # then interpolate it to the size of lr_feats
            pe = F.interpolate(pe, size=(h_lr, w_lr), mode='bicubic')
            # reshape back to (1, C, H_lr * W_lr)
            pe = pe.reshape(1, dim, h_lr * w_lr).permute(0, 2, 1)
            lr_feats = lr_feats + pe
        else:
            lr_feats = lr_feats + self.positional_encoding_lr

        for ca_layer in self.cross_attention:
            x, _ = ca_layer(query=x, key=lr_feats, value=lr_feats)

        # Reshape back to (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        hr_feats = x

        return hr_feats

class PixImplicitUpsampler(nn.Module):
    """
    Inspired by Pixel Nerf, we use Fourier features of images as inputs, and do cross attention with the LR features, the output is the HR features.
    """
    def __init__(self, dim, color_feats=True, cat_lr_feats=True, add_lr_feats=False, n_freqs=20, num_heads=4, num_layers=2, num_conv_layers=1, lr_size=16):
        super(PixImplicitUpsampler, self).__init__()

        if color_feats:
            start_dim = 5 * n_freqs * 2 + 3
        else:
            start_dim = 2 * n_freqs * 2
        if cat_lr_feats:
            start_dim += dim
        self.cat_lr_feats = cat_lr_feats
        self.add_lr_feats = add_lr_feats
        if add_lr_feats:  
            self.channel_norm = ChannelNorm(dim)
        self.dropout = nn.Dropout(p=.2)
        
        num_patches = lr_size * lr_size
        self.positional_encoding_lr = nn.Parameter(torch.randn(1, num_patches, dim))

        self.fourier_feat = torch.nn.Sequential(
                                MinMaxScaler(),
                                ImplicitFeaturizer(color_feats, n_freqs=n_freqs, learn_bias=True),
                            )
        self.first_conv = torch.nn.Sequential(
                                ChannelNorm(start_dim),
                                nn.Conv2d(start_dim, dim, kernel_size=1)
                            )

        # Cross-attention layers
        # self.cross_attention = nn.ModuleList([
        #     nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        #     for _ in range(num_layers)
        # ])
        self.ca_transformer = CATransformer(dim, depth=num_layers, heads=num_heads, dim_head=dim//num_heads, mlp_dim=dim, dropout=0.)


    def forward(self, lr_feats, img):
        # Step 1: Extract Fourier features from the input image
        x = self.fourier_feat(img) # Output shape: (B, dim, H, W)
        b, c, h, w = x.shape

        ## Resize and add LR feats to x? 
        if self.cat_lr_feats:
            lr_feats_resize = F.interpolate(lr_feats, size=(h, w), mode='bicubic')
            x = torch.cat([x, lr_feats_resize], dim=1)
        x = self.first_conv(x)
    
        # Reshape for attention (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        if self.add_lr_feats:
            lr_feats_norm = self.channel_norm(lr_feats)
            lr_feats_resize = F.interpolate(lr_feats_norm, size=(h, w), mode='bicubic')
            lr_feats_resize = lr_feats_resize.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            x = x + lr_feats_resize

        # Step 2: Process LR features for keys and values
        b, c_lr, h_lr, w_lr = lr_feats.shape

        lr_feats = lr_feats.flatten(2).permute(0, 2, 1) # Shape: (B, H_lr*W_lr, C)
        dim = self.positional_encoding_lr.shape[2]
        # Interpolate positional encoding to the size of lr_feats if needed
        if lr_feats.shape[1] != self.positional_encoding_lr.shape[1]:
            len_pos_old = int(math.sqrt(self.positional_encoding_lr.shape[1]))
            # first reshape the positional encoding to (1, C, H_lr, W_lr)
            pe = self.positional_encoding_lr.reshape(1, len_pos_old, len_pos_old, dim).permute(0, 3, 1, 2)
            # then interpolate it to the size of lr_feats
            pe = F.interpolate(pe, size=(h_lr, w_lr), mode='bicubic')
            # reshape back to (1, C, H_lr * W_lr)
            pe = pe.reshape(1, dim, h_lr * w_lr).permute(0, 2, 1)
            lr_feats = lr_feats + pe
        else:
            lr_feats = lr_feats + self.positional_encoding_lr

        # for ca_layer in self.cross_attention:
        #     x, _ = ca_layer(query=x, key=lr_feats, value=lr_feats)
        #     x = self.dropout(x)       
        # 
        x = self.ca_transformer(x, lr_feats)     

        # Reshape back to (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        hr_feats = x

        return hr_feats
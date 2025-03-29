import torch
import torch.nn as nn
import torch.nn.functional as F

# Efficient Channel Attention (ECA) Module
class ECA(nn.Module):
    def __init__(self):
        super(ECA, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, feature_map):

        gap_feature_map = F.adaptive_avg_pool2d(feature_map, output_size=(1, 1))
        gap_feature_map = gap_feature_map.squeeze(dim=3)
        gap_feature_map = gap_feature_map.permute(0, 2, 1)

        conv_feature_map = self.conv1d(gap_feature_map)
        conv_feature_map = conv_feature_map.permute(0, 2, 1)
        conv_feature_map = conv_feature_map.squeeze(dim=2)

        # Apply sigmoid to obtain attention weights in [0, 1]
        weight_feature_map = torch.sigmoid(conv_feature_map)

        return weight_feature_map


# Spatial Attention Module
class spatial_attention(nn.Module):
    def __init__(self):
        super(spatial_attention, self).__init__()

    def forward(self, concat_feature_map):
        batch_size, c, h, w = concat_feature_map.size()

        # Channel average pooling (CAP feature)
        cap_feature = torch.mean(concat_feature_map, dim=1, keepdim=True)

        # Channel max pooling (CMP feature)
        cmp_feature, _ = torch.max(concat_feature_map, dim=1, keepdim=True)

        # Apply sigmoid and expand back to original shape
        cap_feature = torch.sigmoid(cap_feature).expand(batch_size, c, h, w)
        cmp_feature = torch.sigmoid(cmp_feature).expand(batch_size, c, h, w)

        # Element-wise multiplication with attention maps
        element_wise_mul_result = concat_feature_map * cap_feature * cmp_feature

        # Split features along channel dimension
        split_tensors = torch.split(element_wise_mul_result, c // 2, dim=1)
        RGB_Feature_Maps = split_tensors[0]
        IR_feature_map = split_tensors[1]

        # Fuse the two feature sets by averaging
        fused_feature_map = (RGB_Feature_Maps + IR_feature_map) / 2

        return fused_feature_map


# Channel Switching Module
class channel_switching(nn.Module):
    def __init__(self):
        super(channel_switching, self).__init__()
        self.RGB_ECA = ECA()
        self.IR_ECA = ECA()

    def forward(self, RGB_Feature_Maps, IR_Feature_Maps):
        # Compute channel attention weights for both inputs
        RGB_weight = self.RGB_ECA(RGB_Feature_Maps).unsqueeze(-1).unsqueeze(-1)
        IR_weight = self.IR_ECA(IR_Feature_Maps).unsqueeze(-1).unsqueeze(-1)

        # Switch channels based on attention weights
        switch_RGB_Feature_Maps = torch.where(RGB_weight > 0.5, RGB_Feature_Maps, IR_Feature_Maps)
        switch_IR_feature_map = torch.where(IR_weight > 0.5, IR_Feature_Maps, RGB_Feature_Maps)

        # Concatenate both switched feature maps
        output = torch.cat((switch_RGB_Feature_Maps, switch_IR_feature_map), dim=1)

        return output


# Combined Channel Switching and Spatial Attention (CSSA) Module
class CSSA(nn.Module):
    def __init__(self):
        super(CSSA, self).__init__()
        self.channel_switching = channel_switching()
        self.spatial_attention = spatial_attention()

    def forward(self, RGB_Feature_Maps, IR_Feature_Maps):
        # Stage 1: Channel switching
        channel_switching_feature_maps = self.channel_switching(RGB_Feature_Maps, IR_Feature_Maps)

        # Stage 2: Spatial attention
        fused_feature_maps = self.spatial_attention(channel_switching_feature_maps)

        return fused_feature_maps

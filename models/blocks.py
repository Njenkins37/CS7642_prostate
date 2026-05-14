import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Standard UNet building block: Two sequential Convolution -> BatchNorm -> ReLU layers.
    """
    def __init__(self, in_channels, out_channels):
        # Here, the channel is 3 slides from the spatial Z-axis when using 2.5D U-net.
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # Normalize each channel independently with 2 parameters.
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CrossAttentionBlock(nn.Module):
    """
    Spatial Cross-Attention bottleneck.
    T2 provides the Query (Anatomy). ADC provides the Key and Value (Function).
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        # Reduce channels by a factor of 8 for Q and K to save memory. References:
        # https://arxiv.org/pdf/1711.07971
        # https://arxiv.org/pdf/1805.08318
        # Typical values are 2, 4, 8 for the reduction ratio.
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable parameter to scale the attention output before adding the residual.
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = nn.Parameter(torch.tensor([1e-4]))

    def forward(self, t2_features, adc_features):
        # Batch size, Channels, Height, Width.
        B, C, H, W = t2_features.size()
        input_dtype = t2_features.dtype

        # A plain .float() cast is NOT sufficient inside torch.amp.autocast:
        # autocast intercepts Conv2d and bmm (which are on its fp16 dispatch list)
        # and casts their inputs back down to fp16 even if you passed float32 tensors.
        # The only correct escape hatch is a nested autocast with enabled=False,
        # which forces all ops in this block to run in the dtype of their inputs.
        with torch.autocast(device_type='cuda', enabled=False): # The only way for APM to work.
            t2_f  = t2_features.float()
            adc_f = adc_features.float()

            # Generate Q, K, V.
            # Flatten spatial dimensions (H, W) into a single sequence (N).
            # Attention only understands 1D, so .view() changes it to (B,C,N).
            proj_query = self.query_conv(t2_f).view(B, -1, H * W).permute(0, 2, 1)  # (B, N, C')
            proj_key   = self.key_conv(adc_f).view(B, -1, H * W)                    # (B, C', N)
            proj_value = self.value_conv(adc_f).view(B, -1, H * W)                  # (B, C, N)

            # Calculate Attention Scores (Q * K^T).
            energy = torch.bmm(proj_query, proj_key)  # (B, N, N), batch matrix multiplication.
            scale_factor = proj_query.size(-1) ** 0.5
            energy = energy / scale_factor
            # Each row is t2, so apply softmax for each row, look at ADC images, apply softmax row-wise.
            attention_map = F.softmax(energy, dim=-1)

            # Apply Attention to ADC Values.
            out = torch.bmm(proj_value, attention_map.permute(0, 2, 1))  # (B, C, N)
            out = out.view(B, C, H, W)

        # Cast back so the residual add and all downstream layers stay in the
        # precision that the outer autocast context expects.
        out = out.to(input_dtype)

        # Residual Connection: Add original T2 anatomy back into the attended functional features.
        fused_out = self.gamma.to(input_dtype) * out + t2_features

        return fused_out, attention_map
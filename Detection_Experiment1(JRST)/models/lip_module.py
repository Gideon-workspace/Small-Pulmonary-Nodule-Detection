import torch
import torch.nn as nn
import torch.nn.functional as F


class LIPModule(nn.Module):
    """
    Local Importance-based Pooling (LIP)
    Based on Gao et al., ICCV 2019.

    Used to replace the ResNet stem MaxPool:
        kernel_size = 3
        stride = 2
        padding = 1
    """

    def __init__(self, channels, kernel_size=3, stride=2, padding=1):
        super().__init__()

        # Paper's logit module for the MaxPool replacement:
        # a single 3x3 convolution
        self.logit = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Affine Instance Normalization
        self.norm = nn.InstanceNorm2d(
            channels,
            affine=True
        )

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize LIP to behave like average pooling
        nn.init.zeros_(self.logit.weight)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x):

        # 1. Learn local importance logits
        logit = self.logit(x)

        # 2. Instance normalization
        logit = self.norm(logit)

        # 3. Amplified sigmoid
        logit = 12.0 * torch.sigmoid(logit)

        # 4. Convert logits into positive importance weights
        weight = torch.exp(logit)

        # 5. Importance-weighted local aggregation
        numerator = F.avg_pool2d(
            x * weight,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        denominator = F.avg_pool2d(
            weight,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        # 6. Local normalization
        output = numerator / (denominator + 1e-8)

        return output

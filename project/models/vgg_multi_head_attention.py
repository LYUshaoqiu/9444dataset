import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.d_k = in_channels // num_heads
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.fc = nn.Linear(in_channels, in_channels)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.conv1(x).view(batch_size, self.num_heads, self.d_k, width * height)
        proj_key = self.conv2(x).view(batch_size, self.num_heads, self.d_k, width * height)
        proj_value = self.conv3(x).view(batch_size, self.num_heads, self.d_k, width * height)

        energy = torch.einsum('bnhd,bmhd->bnhm', proj_query, proj_key)  # B x H x N x N
        attention = self.softmax(energy)
        
        out = torch.einsum('bnhm,bmhd->bnhd', attention, proj_value)  # B x H x N x C
        out = out.contiguous().view(batch_size, -1, width, height)  # B x C x W x H

        out = self.fc(out.view(batch_size, -1, width * height).permute(0, 2, 1)).permute(0, 2, 1).view(batch_size, C, width, height)
        out = out + x  # Residual connection

        return out

class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = features
        # Add multi-head attention method
        self.attention = MultiHeadAttention(512, 16)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # Add multi-head attention method
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg: list, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def vgg_multi_head_attention(num_classes: int = 1000, init_weights: bool = True, batch_norm: bool = False) -> VGG:
    model = VGG(make_layers(cfg, batch_norm=batch_norm), num_classes=num_classes, init_weights=init_weights)
    return model

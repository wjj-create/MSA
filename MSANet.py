
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class EnhancedCCA(nn.Module):
    def __init__(self, F_g, F_x):
        super(EnhancedCCA, self).__init__()
        
        self.mlp_x = nn.Sequential( 
            Flatten(),
            nn.Linear(F_x * 2, F_x),  
            nn.ReLU(),
            nn.Linear(F_x, F_x)
        )

        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g * 2, F_x), 
            nn.ReLU(),
            nn.Linear(F_x, F_x)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool_x = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        max_pool_g = F.max_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))

        
        combined_pool_x = torch.cat([avg_pool_x, max_pool_x], dim=1)
        combined_pool_g = torch.cat([avg_pool_g, max_pool_g], dim=1)

        channel_att_x = self.mlp_x(combined_pool_x)
        channel_att_g = self.mlp_g(combined_pool_g)

        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = EnhancedCCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def reshape_downsample(x):
    b, c, h, w = x.shape
    ret = torch.zeros_like(x)
    ret = ret.reshape(b, c * 4, h // 2, -1)
    ret[:, 0::4, :, :] = x[:, :, 0::2, 0::2]
    ret[:, 1::4, :, :] = x[:, :, 0::2, 1::2]
    ret[:, 2::4, :, :] = x[:, :, 1::2, 0::2]
    ret[:, 3::4, :, :] = x[:, :, 1::2, 1::2]
    return ret


def reshape_upsample(x):
    b, c, h, w = x.shape
    ret = torch.zeros_like(x)
    ret = ret.reshape(b, c // 4, h * 2, w * 2)
    ret[:, :, 0::2, 0::2] = x[:, 0::4, :, :]
    ret[:, :, 0::2, 1::2] = x[:, 1::4, :, :]
    ret[:, :, 1::2, 0::2] = x[:, 2::4, :, :]
    ret[:, :, 1::2, 1::2] = x[:, 3::4, :, :]
    return ret


class DownFuseBlock(nn.Module):
    def __init__(self, base_channels, dropout_rate=0.1):
        super(DownFuseBlock, self).__init__()
        self.eca = ECA(base_channels * 2)
        self.down = reshape_downsample
        self.conv1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, groups=base_channels)
        self.norm1 = nn.BatchNorm2d(base_channels * 2)
        self.fuse_conv = ConvBatchNorm(base_channels * 2, base_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, fp1, fp2):
        down = self.down(fp1)
        down = self.conv1(down)
        down = self.relu(self.norm1(down))
        fp2 = self.fuse_conv(fp2 * 0.75 + down * 0.25) + fp2
        fp2 = self.eca(fp2)
        return fp2


class UpFuseBlock(nn.Module):
    def __init__(self, base_channels, dropout_rate=0.1):
        super(UpFuseBlock, self).__init__()
        self.eca = ECA(base_channels)
        self.up = reshape_upsample
        self.conv1 = nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=1, padding=1,
                               groups=base_channels // 2)
        self.norm1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU()
        self.fuse_conv = ConvBatchNorm(base_channels, base_channels)

    def forward(self, fp1, fp2):
        up = self.up(fp2)
        up = self.conv1(up)
        up = self.relu(self.norm1(up))
        fp1 = self.fuse_conv((fp1 * 0.75 + up * 0.25)) + fp1
        fp1 = self.eca(fp1)
        return fp1

class FuseBlock(nn.Module):
    def __init__(self, base_channels):
        super(FuseBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(base_channels)
        self.norm2 = nn.BatchNorm2d(base_channels * 2)
        self.norm3 = nn.BatchNorm2d(base_channels * 4)
        self.norm4 = nn.BatchNorm2d(base_channels * 8)

        self.up3 = UpFuseBlock(base_channels=base_channels * 4)
        self.up2 = UpFuseBlock(base_channels=base_channels * 2)
        self.up1 = UpFuseBlock(base_channels=base_channels)

        self.down1 = DownFuseBlock(base_channels=base_channels)
        self.down2 = DownFuseBlock(base_channels=base_channels * 2)
        self.down3 = DownFuseBlock(base_channels=base_channels * 4)

    def forward(self, fp1, fp2, fp3, fp4):
        fp4 = self.norm4(fp4)
        fp3 = self.norm3(fp3)
        fp2 = self.norm2(fp2)
        fp1 = self.norm1(fp1)

        fp2 = self.down1(fp1, fp2)
        fp3 = self.down2(fp2, fp3)
        fp4 = self.down3(fp3, fp4)

        fp3 = self.up3(fp3, fp4)
        fp2 = self.up2(fp2, fp3)
        fp1 = self.up1(fp1, fp2)

        return fp1, fp2, fp3, fp4


class FuseModule(nn.Module):
    def __init__(self, base_channel, nb_blocks: int = 2):
        super(FuseModule, self).__init__()
        self.base_channel = base_channel
        self.blocks = nn.ModuleList()
        nb_blocks = max(1, nb_blocks)
        for _ in range(nb_blocks):
            self.blocks.append(FuseBlock(base_channel))

    def forward(self, fp1, fp2, fp3, fp4):
        for block in self.blocks:
            fp1, fp2, fp3, fp4 = block(fp1, fp2, fp3, fp4)

        return fp1, fp2, fp3, fp4


class ModifiedMSAG(nn.Module):
    def __init__(self, channel):
        super(ModifiedMSAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv2 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv4 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=4, stride=1, dilation=4, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv8 = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=8, stride=1, dilation=8, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 5, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv2(x)
        x4 = self.dilationConv4(x)
        x5 = self.dilationConv8(x)
        _x = self.relu(torch.cat((x1, x2, x3, x4, x5), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x



class MFUnet(nn.Module):
    def __init__(self, in_channels=3, n_cls=10, base_channels=64, aggre_depth=2):
        super(MFUnet, self).__init__()
        self.in_channel = in_channels
        self.n_cls = n_cls

        self.inc = ConvBatchNorm(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(base_channels * 8, base_channels * 8, nb_Conv=2)

        self.fuse = FuseModule(base_channel=base_channels, nb_blocks=aggre_depth)

        self.up4 = UpBlock_attention(base_channels * 16, base_channels * 4, nb_Conv=2)
        self.up3 = UpBlock_attention(base_channels * 8, base_channels * 2, nb_Conv=2)
        self.up2 = UpBlock_attention(base_channels * 4, base_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(base_channels * 2, base_channels, nb_Conv=2)

        self.msag4 = ModifiedMSAG(base_channels * 8)
        self.msag3 = ModifiedMSAG(base_channels * 8)
        self.msag2 = ModifiedMSAG(base_channels * 4)
        self.msag1 = ModifiedMSAG(base_channels * 2)

        self.outc = nn.Conv2d(base_channels, n_cls, kernel_size=1, stride=1)
        self.last_activation = nn.Softmax()



    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.msag1(x2)
        x3 = self.down2(x2)
        x3 = self.msag2(x3)
        x4 = self.down3(x3)
        x4 = self.msag3(x4)
        x5 = self.down4(x4)
        x5 = self.msag4(x5)

        x1, x2, x3, x4 = self.fuse(x1, x2, x3, x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits


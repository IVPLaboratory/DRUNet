from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


'''
# ===================================
# Useful blocks
# --------------------------------
# conv (+ normaliation + relu)
# concat
# sum
# resblock (ResBlock)
# resdenseblock (ResidualDenseBlock_5C)
# resinresdenseblock (RRDB)
# ===================================
'''


# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR'):
    L = []
    for t in mode:
        # 普通卷积，用于网络前半段的下采样操作
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  bias=bias))
        # 扩张卷积用于USRNet
        elif t == 'D':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=2, dilation=2, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            # L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
            L.append(nn.BatchNorm2d(out_channels, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


'''
FFT block
'''

class FFTBlock(nn.Module):
    def __init__(self, channel=64):
        super(FFTBlock, self).__init__()
        self.conv_fc = nn.Sequential(
                nn.Conv2d(1, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, 1, 1, padding=0, bias=True),
                nn.Softplus()
        )

    def forward(self, x, u, d, sigma):
        rho = self.conv_fc(sigma)
        x = torch.irfft(self.divcomplex(u + rho.unsqueeze(-1)*torch.rfft(x, 2, onesided=False), d + self.real2complex(rho)), 2, onesided=False)
        return x

    def divcomplex(self, x, y):
        a = x[..., 0]
        b = x[..., 1]
        c = y[..., 0]
        d = y[..., 1]
        cd2 = c**2 + d**2
        return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)

    def real2complex(self, x):
        return torch.stack([x, torch.zeros(x.shape).type_as(x)], -1)




# -------------------------------------------------------
# Concat the output of a submodule to its input
# -------------------------------------------------------
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + 'concat'


# -------------------------------------------------------
# Elementwise sum the output of a submodule to its input
# -------------------------------------------------------
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x))), USRNet中的残差块前后特征图分辨率一样大，不会变小
# -------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]
            "mode= DRD"
        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)

    def forward(self, x):
        res = self.res(x)
        return x + res

"Depthwise Sparable Convolution"
class DSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSC, self).__init__()
        "先做DepthWise Convolution"
        self.depthConv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3 // 2, groups=in_channels)
        self.depthRelu = nn.ReLU(inplace=False)  # 设置为True会报错
        "再做PointWise Convolution"
        self.pointConv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pointRelu = nn.ReLU(inplace=True)

    def forward(self, x):
        depthResult = self.depthRelu(self.depthConv(x))
        pointResult = self.pointRelu(self.pointConv(depthResult))
        return pointResult


# -------------------------------------------------------
# DenseLayer:
# -------------------------------------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        # "in_channel随着layers增加而增加，out_channels不变"
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        # self.relu = nn.ReLU(inplace=True)

        self.dsc = DSC(in_channels, out_channels)

    def forward(self, x):
        # output = torch.cat([x, self.relu(self.conv(x))], 1)
        # print(output.shape)
        # return torch.cat([x, self.relu(self.conv(x))], 1)
        return torch.cat([x, self.dsc(x)], 1)
# -------------------------------------------------------
# DenseLayer:
# -------------------------------------------------------
# class DenseLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DenseLayer, self).__init__()
#         "in_channel随着layers增加而增加，out_channels不变"
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         # output = torch.cat([x, self.relu(self.conv(x))], 1)
#         # print(output.shape)
#         return torch.cat([x, self.relu(self.conv(x))], 1)


# -------------------------------------------------------
# ResDen Block:
# -------------------------------------------------------
class ResDenBlock(nn.Module):
     def __init__(self, in_channels, growth_rate, num_layers):
         super(ResDenBlock, self).__init__()

         self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

         self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, in_channels, kernel_size=1)

     def forward(self, x):
        return x + self.lff(self.layers(x))
        # y=x + self.lff(self.layers(x))
        # return torch.cat([x, y], 1)


# -------------------------------------------------------
# CSP Splite Block:
# -------------------------------------------------------
class Channel_Split(nn.Module):
    def __init__(self, in_channels):
        super(Channel_Split, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


# -------------------------------------------------------
# 上采样阶段Dec Block:
# -------------------------------------------------------
# class DecBlock(nn.Module):
#     def __init__(self, nc1=[512,256,128,64]):
#         super(DecBlock, self).__init__()
#         self.convtran1 = conv(nc1[0], nc1[1], kernel_size=2,stride=2, padding=0, bias=False, mode='T')
#         # self.z1=conv(nc1[1]*2, nc1[1], kernel_size=3, stride=1, padding=1, mode='CB')
#
#         self.z1 = conv(nc1[1] * 2, nc1[1], kernel_size=1, stride=1, padding=0, mode='CR')
#         self.z11 = conv(nc1[1], nc1[1], kernel_size=3, stride=1, padding=1, mode='CR')
#
#         self.convtran2 = conv(nc1[1], nc1[2], kernel_size=2,stride=2, padding=0,bias=False, mode='T')
#         self.convtran3 = conv(nc1[1], nc1[2], kernel_size=2,stride=2, padding=0,bias=False, mode='T')
#
#         # self.z2 = conv(nc1[2]*2, nc1[2], kernel_size=3, stride=1, padding=1, mode='CB')
#         self.z2 = conv(nc1[2]*2, nc1[2], kernel_size=1, stride=1, padding=0, mode='CR')
#         self.z21 = conv(nc1[2], nc1[2], kernel_size=3, stride=1, padding=1, mode='CR')
#
#
#         # self.z3 = conv(nc1[2]*3, nc1[2], kernel_size=3,  stride=1, padding=1,mode='CB')
#         self.z3 = conv(nc1[2]*3, nc1[2], kernel_size=1, stride=1, padding=0, mode='CR')
#         self.z31 = conv(nc1[2], nc1[2], kernel_size=3, stride=1, padding=1, mode='CR')
#
#         self.convtran4 = conv(nc1[2], nc1[3], kernel_size=2,stride=2, padding=0, bias=False,mode='T')
#         self.convtran5 = conv(nc1[2], nc1[3], kernel_size=2,stride=2, padding=0, bias=False,mode='T')
#         self.convtran6 = conv(nc1[2], nc1[3], kernel_size=2,stride=2, padding=0, bias=False,mode='T')
#
#         # self.z4 = conv(nc1[3]*2, nc1[3], kernel_size=3, stride=1, padding=1, mode='CB')
#         self.z4 = conv(nc1[3]*2, nc1[3], kernel_size=1, stride=1, padding=0, mode='CR')
#         self.z41 = conv(nc1[3], nc1[3], kernel_size=3, stride=1, padding=1, mode='CR')
#
#         # self.z5 = conv(nc1[3]*3, nc1[3], kernel_size=3, stride=1, padding=1, mode='CB')
#         self.z5 = conv(nc1[3]*3, nc1[3], kernel_size=1, stride=1, padding=0, mode='CR')
#         self.z51 = conv(nc1[3], nc1[3], kernel_size=3, stride=1, padding=1, mode='CR')
#
#         # self.z6 = conv(nc1[3]*4,nc1[3], kernel_size=3, stride=1, padding=1, mode='CB')
#         self.z6 = conv(nc1[3]*4, nc1[3], kernel_size=1, stride=1, padding=0, mode='CR')
#         self.z61 = conv(nc1[3], nc1[3], kernel_size=3, stride=1, padding=1, mode='CR')
#         #
#         # self.convtran7 = conv(nc1[0], nc1[1], kernel_size=2 ,stride=2, padding=0, bias=False,mode='TR')
#         # self.convtran7_1 = conv(nc1[1], nc1[2], kernel_size=2, stride=2, padding=0, bias=False, mode='TR')
#         # self.convtran7_2 = conv(nc1[2], nc1[3], kernel_size=2, stride=2, padding=0, bias=False, mode='TR')
#         #
#         # self.convtran8 = conv(nc1[1], nc1[2], kernel_size=2, stride=2, padding=0, bias=False, mode='TR')
#         # self.convtran8_1 = conv(nc1[2], nc1[3], kernel_size=2, stride=2, padding=0, bias=False, mode='TR')
#         #
#         # self.convtran9 = conv(nc1[2], nc1[3], kernel_size=2, stride=2, padding=0, bias=False, mode='TR')
#
#         # self.z7 = conv(nc1[3]*4, nc1[3], kernel_size=1, stride=1, padding=0, mode='C')
#         # self.z8 = conv(nc1[3], nc1[3], kernel_size=3, stride=1, padding=1, mode='C')
#
#         self.z7 = conv(nc1[3]*3, nc1[3], kernel_size=1, stride=1, padding=0, mode='C')
#
#         self.z8=conv(nc1[3] , nc1[3], kernel_size=3, stride=1, padding=1, mode='C')
#
#     def forward(self,a4,a3,a2,a1):
#         y1 = self.convtran1(a4)
#         z1 = self.z1(torch.cat((a3, y1), 1))
#         z11 = self.z11(z1)
#
#         y2 = self.convtran2(a3)
#         y3 = self.convtran3(z11)
#
#         z2 = self.z2(torch.cat((a2, y2), 1))
#         z21 = self.z21(z2)
#
#         z3 = self.z3(torch.cat((a2, z21, y3), 1))
#         z31 = self.z31(z3)
#
#         y4 = self.convtran4(a2)
#         y5 = self.convtran5(z21)
#         y6 = self.convtran6(z31)
#
#         z4 = self.z4(torch.cat((a1, y4), 1))
#         z41=self.z41(z4)
#
#         z5 = self.z5(torch.cat((a1, z41, y5), 1))
#         z51 = self.z51(z5)
#
#         z6 = self.z6(torch.cat((a1, z41, z51, y6), 1))
#         z61 = self.z61(z6)
#
#         # y7 = self.convtran7(a4)
#         # y71=self.convtran7_1(y7)
#         # y72=self.convtran7_2(y71)
#         #
#         # y8 = self.convtran8(z11)
#         # y81 = self.convtran8_1(y8)
#         #
#         # y9 = self.convtran9(z31)
#
#         # z = self.z7(torch.cat((y72, y81, y9, z61), 1))
#         # z = self.z8(z)
#
#         z = self.z7(torch.cat((z41,z51,z61),1))
#         z = self.z8(z)
#
#         return z

#
# -------------------------------------------------------
# 上采样阶段Dec Block:
# -------------------------------------------------------
# class DecBlock(nn.Module):
#     def __init__(self, nc1=[256,128,64]):
#         super(DecBlock, self).__init__()
#         self.convtran1 = conv(nc1[0], nc1[1], kernel_size=2,stride=2, padding=0, bias=False, mode='T')
#         # self.z1=conv(nc1[1]*2, nc1[1], kernel_size=3, stride=1, padding=1, mode='CB')
#
#         self.z1 = conv(nc1[1] * 2, nc1[1], kernel_size=1, stride=1, padding=0, mode='CR')
#         self.z11 = conv(nc1[1], nc1[1], kernel_size=3, stride=1, padding=1, mode='CR')
#
#         self.convtran2 = conv(nc1[1], nc1[2], kernel_size=2,stride=2, padding=0,bias=False, mode='T')
#         self.convtran3 = conv(nc1[1], nc1[2], kernel_size=2,stride=2, padding=0,bias=False, mode='T')
#
#         # self.z2 = conv(nc1[2]*2, nc1[2], kernel_size=3, stride=1, padding=1, mode='CB')
#         self.z2 = conv(nc1[2]*2, nc1[2], kernel_size=1, stride=1, padding=0, mode='CR')
#         self.z21 = conv(nc1[2], nc1[2], kernel_size=3, stride=1, padding=1, mode='CR')
#
#
#         # self.z3 = conv(nc1[2]*3, nc1[2], kernel_size=3,  stride=1, padding=1,mode='CB')
#         self.z3 = conv(nc1[2]*3, nc1[2], kernel_size=1, stride=1, padding=0, mode='CR')
#         self.z31 = conv(nc1[2], nc1[2], kernel_size=3, stride=1, padding=1, mode='CR')
#
#
#         self.z7 = conv(nc1[2]*2, nc1[2], kernel_size=1, stride=1, padding=0, mode='C')
#
#         self.z8=conv(nc1[2] , nc1[2], kernel_size=3, stride=1, padding=1, mode='C')
#
#     def forward(self,a3,a2,a1):
#         y1 = self.convtran1(a3)
#         z1 = self.z1(torch.cat((a2, y1), 1))
#         z11 = self.z11(z1)
#
#         y2 = self.convtran2(a2)
#         y3 = self.convtran3(z11)
#
#         z2 = self.z2(torch.cat((y2, a1), 1))
#         z21 = self.z21(z2)
#
#
#         z3=self.z3(torch.cat((y3, a1, z21), 1))
#         z31=self.z31(z3)
#
#         z = self.z7(torch.cat((z21,z31),1))
#         z = self.z8(z)
#
#         return z


class DecBlock(nn.Module):
    def __init__(self, nc1=[128,64]):
        super(DecBlock, self).__init__()
        self.convtran1 = conv(nc1[0], nc1[1], kernel_size=2,stride=2, padding=0, bias=False, mode='T')
        # self.z1=conv(nc1[1]*2, nc1[1], kernel_size=3, stride=1, padding=1, mode='CB')

        self.z1 = conv(nc1[1] * 2, nc1[1], kernel_size=1, stride=1, padding=0, mode='CR')
        self.z11 = conv(nc1[1], nc1[1], kernel_size=3, stride=1, padding=1, mode='CR')

        self.z7 = conv(nc1[1], nc1[1], kernel_size=1, stride=1, padding=0, mode='C')

        self.z8=conv(nc1[1] , nc1[1], kernel_size=3, stride=1, padding=1, mode='C')

    def forward(self,x,x11):
        y1 = self.convtran1(x)
        z1 = self.z1(torch.cat((y1, x11), 1))
        z11 = self.z11(z1)



        z = self.z7(z11)
        z = self.z8(z)

        return z

# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return x * y


# -------------------------------------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        return res + x


# -------------------------------------------------------
# Residual Channel Attention Group (RG)
# -------------------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16, nb=12):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding, bias, mode, reduction)  for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        self.rg = nn.Sequential(*RG)  # self.rg = ShortcutBlock(nn.Sequential(*RG))

    def forward(self, x):
        res = self.rg(x)
        return res + x


# -------------------------------------------------------
# Residual Dense Block
# style: 5 convs
# -------------------------------------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR'):
        super(ResidualDenseBlock_5C, self).__init__()

        # gc: growth channel
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode)
        self.conv2 = conv(nc+gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv3 = conv(nc+2*gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv4 = conv(nc+3*gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv5 = conv(nc+4*gc, nc, kernel_size, stride, padding, bias, mode[:-1])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


class ResidualDenseBlock_8C(nn.Module):
    def __init__(self, nc=64, gc=16, kernel_size=3, stride=1, padding=1, bias=True, mode='CR'):
        super(ResidualDenseBlock_8C, self).__init__()

        # gc: growth channel
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode)
        self.conv2 = conv(nc+gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv3 = conv(nc+2*gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv4 = conv(nc+3*gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv5 = conv(nc + 4 * gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv6 = conv(nc + 5 * gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv7 = conv(nc + 6 * gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv8 = conv(nc+7*gc, nc, kernel_size, stride, padding, bias, mode[:-1])
        self.lff = conv(nc+8*gc,gc,kernel_size=1,mode='C')
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1))
        x8 = self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1))
        return x+self.lff(x8)

        #return x8.mul_(0.2) + x

# -------------------------------------------------------
# Residual in Residual Dense Block
# 3x5c
# -------------------------------------------------------
class RRDB(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR'):
        super(RRDB, self).__init__()

        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x


'''
# ======================
# Upsampler
# ======================
'''


# -------------------------------------------------------
# conv + subp + relu
# -------------------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode)
    return up1


# -------------------------------------------------------
# nearest_upsample + conv + relu
# -------------------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=2, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode)
    return up1


# -------------------------------------------------------
# convTranspose + relu
# -------------------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    # mode = mode.replace(mode[0], 'U')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return up1


'''
# ======================
# Downsampler
# ======================
'''


# -------------------------------------------------------
# strideconv + relu，等同于maxpool
# -------------------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return down1


# -------------------------------------------------------
# maxpooling + conv + relu
# -------------------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:])
    return sequential(pool, pool_tail)


# -------------------------------------------------------
# averagepooling + conv + relu
# -------------------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:])
    return sequential(pool, pool_tail)


'''
# ======================
# NonLocalBlock2D: 
# embedded_gaussian
# +W(softmax(thetaXphi)Xg)
# ======================
'''


# -------------------------------------------------------
# embedded_gaussian
# -------------------------------------------------------
class NonLocalBlock2D(nn.Module):
    def __init__(self, nc=64, kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='maxpool'):

        super(NonLocalBlock2D, self).__init__()

        inter_nc = nc // 2
        self.inter_nc = inter_nc
        self.W = conv(inter_nc, nc, kernel_size, stride, padding, bias, mode='C'+act_mode)
        self.theta = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

        if downsample:
            if downsample_mode == 'avgpool':
                downsample_block = downsample_avgpool
            elif downsample_mode == 'maxpool':
                downsample_block = downsample_maxpool
            elif downsample_mode == 'strideconv':
                downsample_block = downsample_strideconv
            else:
                raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
            self.phi = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
            self.g = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
        else:
            self.phi = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')
            self.g = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')


    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
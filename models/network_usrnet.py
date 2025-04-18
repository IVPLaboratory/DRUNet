import torch
import torch.nn as nn
import models.basicblock as B
import numpy as np
from utils import utils_image as util


"""
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}
# --------------------------------------------
"""


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""


def splits(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxWxHx2
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


def c2c(x):
    return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)


def crdiv(x, y):
    # complex/real division
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a/y, b/y], -1)


def csum(x, y):
    # complex + real
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    # modulus of a complex number
    return torch.pow(x[..., 0]**2+x[..., 1]**2, 0.5)


def cabs2(x):
    return x[..., 0]**2+x[..., 1]**2


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def rfft(t):
    # Real-to-complex Discrete Fourier Transform
    return torch.rfft(t, 2, onesided=False)


def irfft(t):
    # Complex-to-real Inverse Discrete Fourier Transform
    return torch.irfft(t, 2, onesided=False)


def fft(t):
    # Complex-to-complex Discrete Fourier Transform
    return torch.fft(t, 2)


def ifft(t):
    # Complex-to-complex Inverse Discrete Fourier Transform
    return torch.ifft(t, 2)


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""

#定义网络模型，需继承torch.nn.Module基类
class ResUNet(nn.Module):
    #输入图像通道数in_nc；
    #输出图像通道数out_nc；
    #每个stage图像的通道数分别为64,128,256,512，存在列表nc中nc=[64, 128, 256, 512];
    #每个stage的残差卷积块nb=2;
    #下采样downsample_mode = 'strideconv'
    #上采样upsample_mode='convtranspose' 反卷积


    def __init__(self, in_nc=4, out_nc=3, nc=[16,32,64], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        #调用父类构造函数
        super(ResUNet, self).__init__()

        # model = net(in_nc=1, out_nc=1, nc=[64, 128, 256, 512],
        #             nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

        "第一个卷积，输入图像channel=1， conv之后变为64"
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv   #C
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        "ResBlock(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC' )"
        "downsample_block(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2')"
        "将ResBlock中的mode改为DRD，即dilated_conv + ReLU + dilated_conv"
        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='D'+act_mode+'D') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='D'+act_mode+'D') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        # self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='D'+act_mode+'D') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='D' + act_mode + 'D') for _ in range(nb)])
        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose  #T
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        "upsample_block(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R')"
        "ResBlock(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC' )"
        "将ResBlock中的mode改为DRD，即dilated_conv + ReLU + dilated_conv"
        # self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        "最后一个卷积，输入特征图channel=64，结果图channel=1"
        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        # print('x.shape=', x.shape)

        x1 = self.m_head(x)    #in_channel=1, out_channel=64
        # print('x1.shape=', x1.shape)
        x2 = self.m_down1(x1)  #downsample=2，in_channel=64, out_channel=128
        # print('x2.shape=', x2.shape)
        x3 = self.m_down2(x2)  #downsample=4，in_channel=128, out_channel=256
        # print('x3.shape=', x3.shape)
        # x4 = self.m_down3(x3)  #downsample=6，in_channel=256, out_channel=512
        # # print('x4.shape=', x4.shape)
        x = self.m_body(x3)    #channel=512
        # print('x.shape=', x.shape)
        # x = self.m_up3(x+x4)   #in_channel=512, out_channel=256, upsample=2
        x = self.m_up2(x+x3)   #in_channel=256, out_channel=128, upsample=4
        x = self.m_up1(x+x2)   #in_channel=128, out_channel=64, upsample=6
        x = self.m_tail(x+x1)  #in_channel=64, out_channel=1

        x = x[..., :h, :w]

        return x

"----------------"
#定义网络模型，需继承torch.nn.Module基类
class ResDenUNet(nn.Module):


    def __init__(self, in_channels=1, num_features=64, growth_rate=16, num_blocks=1, num_layers=8,nc=[64,128,256], downsample_mode='strideconv', upsample_mode='convtranspose',nb=2, act_mode='R'):
        #调用父类构造函数
        super(ResDenUNet, self).__init__()

        self.G0 = num_features  #64
        self.G = growth_rate  # 16
        self.D = num_blocks  # 2
        self.C = num_layers  # 8

        "第一个卷积，输入图像channel=1， conv之后变为16"
        self.m_head = B.conv(in_channels, num_features, bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        # self.m_down1 = B.sequential(*[B.ResDenBlock(nc[0], self.G, self.C) for _ in range(num_blocks)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        # self.m_down2 = B.sequential(*[B.ResDenBlock(nc[1], self.G, self.C) for _ in range(num_blocks)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        # self.m_down3 = B.sequential(*[B.ResDenBlock(nc[2], self.G, self.C) for _ in range(num_blocks)], downsample_block(nc[2], nc[3], bias=False, mode='2'))
        # self.m_body = B.sequential(*[B.ResDenBlock(nc[3],self.G, self.C) for _ in range(num_blocks)])

        # self.m_chansp1 = B.sequential(B.Channel_Split(nc[0]))
        self.m_rdb1 = B.sequential(*[B.ResDenBlock(nc[0], self.G, self.C) for _ in range(num_blocks)])
        self.m_down1 = B.sequential(downsample_block(nc[0], nc[1], bias=False, mode='2'))

        # self.m_body = B.sequential(*[B.ResDenBlock(nc[1], self.G, self.C) for _ in range(num_blocks)])

        #self.m_chansp2 = B.sequential(B.Channel_Split(nc[1]))
        self.m_rdb2 = B.sequential(*[B.ResDenBlock(nc[1], self.G, self.C) for _ in range(num_blocks)])
        self.m_down2 = B.sequential(downsample_block(nc[1], nc[2], bias=False, mode='2'))


        # # self.m_chansp3 = B.sequential(B.Channel_Split(nc[2]))
        # self.m_rdb3 = B.sequential(*[B.ResDenBlock(nc[2], self.G, self.C) for _ in range(num_blocks)])
        # self.m_down3 = B.sequential(downsample_block(nc[2], nc[3], bias=False, mode='2'))
        #
        # # self.m_body = B.sequential(*[B.ResDenBlock(nc[2], self.G, self.C) for _ in range(num_blocks)])
        # # self.m_chansp4 = B.sequential(B.Channel_Split(nc[3]))
        self.m_body = B.sequential(*[B.ResDenBlock(nc[2], self.G, self.C) for _ in range(num_blocks)])
        #



        # RDB消融实验
        "upsample"
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose  # T
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        "upsample_block(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R')"
        "ResBlock(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC' )"
        # self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
        #                           *[B.ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[B.ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])


        #
        # upsample
        # if upsample_mode == 'upconv':
        #     upsample_block = B.upsample_upconv
        # elif upsample_mode == 'pixelshuffle':
        #     upsample_block = B.upsample_pixelshuffle
        # elif upsample_mode == 'convtranspose':
        #     upsample_block = B.upsample_convtranspose  #T
        # else:
        #     raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        #
        # "upsample_block(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R')"
        # self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResDenBlock(nc[2], self.G, self.C) for _ in range(num_blocks)])
        # self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResDenBlock(nc[1], self.G, self.C) for _ in range(num_blocks)])
        # self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResDenBlock(nc[0], self.G, self.C) for _ in range(num_blocks)])

        # self.my_up=B.sequential(*[B.DecBlock(nc1=[128,64])])
        # self.my_up = B.sequential(*[B.DecBlock(nc1=[512,256, 128, 64])])

        "最后一个卷积，输入特征图channel=16，结果图channel=1"
        self.m_tail = B.conv(nc[0], in_channels, bias=False, mode='C')

    def forward(self, x):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        temp = x

        # print('x.shape=', x.shape)


        # downsampling
        x10 = self.m_head(x)  # in_channel=1, out_channel=64

        # print('x10.shape=', x10.shape)
        x11 = self.m_rdb1(x10)
        # print('x11.shape=', x11.shape)
        x12 = self.m_down1(x11)
        # print('x12.shape=', x12.shape)

        x21 = self.m_rdb2(x12)
        # print('x21.shape=', x21.shape)
        x22 = self.m_down2(x21)
        # # print('x22.shape=', x22.shape)
        # #
        # x31 = self.m_rdb3(x22)
        # # print('x31.shape=', x31.shape)
        # x32 = self.m_down3(x31)
        # # print('x32.shape=', x32.shape)
        x = self.m_body(x22)
        # print('x.shape=', x.shape)
        # x = self.m_body(x12)
# ---------------------------------
        # downsampling +csp
        # x10 = self.m_head(x)  # in_channel=1, out_channel=64
        #
        # x11 = self.m_chansp1(x10)
        # # print('x10.shape=', x10.shape)
        # x12 = self.m_rdb1(x11)
        # # print('x11.shape=', x11.shape)
        # x13 = self.m_down1(x12)
        # # print('x12.shape=', x12.shape)
        #
        # x21 = self.m_chansp2(x13)
        # x22 = self.m_rdb2(x21)
        # # print('x21.shape=', x21.shape)
        # x23 = self.m_down2(x22)
        # # print('x22.shape=', x22.shape)
        #
        # x31 = self.m_chansp3(x23)
        # x32 = self.m_rdb3(x31)
        # # print('x31.shape=', x31.shape)
        # x33 = self.m_down3(x32)
        # # print('x32.shape=', x32.shape)
        # x = self.m_body(x33)

        # upsampling
        # r=self.my_up(x,x31,x21,x11)
        # # print('r.shape=',r.shape)
        # result = self.m_tail(r)
        # print('result.shape=',result.shape)
        # result =result[..., :h, :w]
        # return result


       # 最终输出
       #  r = self.my_up(x,x31,x21, x11)
       #  r = self.my_up(x, x11)
       #  r = r + x10
       #  result = self.m_tail(r)
       #  result2 = temp + result
       #  result2 = result2[..., :h, :w]
       #  return result2

        # #RDB消融实验
        # x = self.m_up3(x+x32)   #in_channel=512, out_channel=256, upsample=2
        x = self.m_up2(x+x22)   #in_channel=256, out_channel=128, upsample=4
        x = self.m_up1(x+x12)   #in_channel=128, out_channel=64, upsample=6
        x = self.m_tail(x+x10)  #in_channel=64, out_channel=1

        x = x[..., :h, :w]

        return x

"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""


class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):
        FR = FBFy + torch.rfft(alpha*x, 2, onesided=False)
        x1 = cmul(FB, FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = cdiv(FBR, csum(invW, alpha))
        FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
        FX = (FR-FCBinvWBR)/alpha.unsqueeze(-1)
        Xest = torch.irfft(FX, 2, onesided=False)

        return Xest


"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""


class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


"""
# --------------------------------------------
# main USRNet
# deep unfolding super-resolution network
# --------------------------------------------
"""


class USRNet(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        #必须要写这一步
        super(USRNet, self).__init__()

        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=n_iter*2, channel=h_nc)
        self.n = n_iter

    def forward(self, x, k, sf, sigma):
        '''
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''

        # initialization & pre-calculation
        w, h = x.shape[-2:]
        FB = p2o(k, (w*sf, h*sf))
        FBC = cconj(FB, inplace=False)
        F2B = r2c(cabs2(FB))
        STy = upsample(x, sf=sf)
        FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))
        x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')

        # hyper-parameter, alpha & beta
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))

        # unfolding
        for i in range(self.n):

            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i+1, ...], sf)
            x = self.p(torch.cat((x, ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1))

        return x

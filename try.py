# from models import RDN
from models.network_usrnet import ResDenUNet as neta
import os
import numpy as np
import torch

from torch.utils.data.dataloader import DataLoader
from thop import profile
from thop import clever_format
#
# train_dataset = TrainDataset('train/train.h5', patch_size=32, scale=1)
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=8,
#                               shuffle=True,
#                               num_workers=0,
#                               pin_memory=True)
model = neta(in_channels=1, num_features=64, growth_rate=16, num_blocks=1, num_layers=8,
                    nc=[64,128], downsample_mode='strideconv', upsample_mode="convtranspose")
print(model)
input = torch.randn(1, 1, 256, 320)
flops, params = profile(model, inputs=(input, ))
flops, params = clever_format([flops, params], '%.3f')
print(flops, params)


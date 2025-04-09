'''from "test" to "results"(just background)'''

import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# from models import DnCNN
from utils import *
import time
# import xlwt
# # import xlrd
# from xlutils.copy import copy
from PIL import Image
from models.network_usrnet import ResUNet as net
from models.network_usrnet import ResDenUNet as neta
import shutil

from matplotlib import pyplot as plt

time_start = time.time()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="USRNet_Test")
parser.add_argument('--weights-file', type=str, default='weights/g16/epoch_187.pth')
# parser.add_argument('--weights-file', type=str, default='weights/best.pth')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="model_zoo", help='path of log files')
parser.add_argument("--test_data", type=str, default='test', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()


def normalize(data):
    return data / 255.


def main():

    # Build model
    print('Loading model ...\n')
    # model = net(in_nc=1, out_nc=1, nc=[16,32,64,128],
    #             nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    # model = neta(in_channels=1, num_features=64, growth_rate=16, num_blocks=2, num_layers=8,
    #              nc=[64, 128, 256, 512], downsample_mode='strideconv', upsample_mode="convtranspose")
    # model = neta(in_channels=1, num_features=64, growth_rate=16, num_blocks=1, num_layers=8,
    #              nc=[64,128,256,512], downsample_mode='strideconv', upsample_mode="convtranspose")
    model = neta(in_channels=1, num_features=64, growth_rate=16, num_blocks=1, num_layers=8,
                    nc=[64,128,256], downsample_mode='strideconv', upsample_mode="convtranspose")
    # model = net(in_nc=1, out_nc=1, nc=[16, 32, 64],
    #             nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # model = nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.load_state_dict(torch.load(opt.weights_file))
    model.eval()
    model = model.to(device)


    "可取消的代码，residual_result用来保存网络输出的残差图像"
    residual_path = './residual result/'
    if os.path.exists(residual_path):
        shutil.rmtree(residual_path)
        os.mkdir(residual_path)
    else:
        os.mkdir(residual_path)

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('testsets', opt.test_data, '*.bmp'))
    files_source.sort()

    # process data
    count = 0
    save_path = "./results/"  # 保存图像帧的相对路径. image_frames_bmp手动创建
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    name = []
    for i in os.listdir("testsets/test"):
        name.append(i)
    name.sort()

    time_test = []
    test_size = []

    for f in files_source:
        time_test_start = time.time()
        test_size.append(str(Image.open("testsets/test/" + name[count]).size))
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:, :, 0]))
        # Img = normalize(np.float32(Img[:, :]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        # ISource = torch.Tensor(Img)
        INoisy = torch.Tensor(Img)

        # noise
        # noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)

        # noisy image
        # INoisy = ISource + noise
        # ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        INoisy = Variable(INoisy.cuda())

        with torch.no_grad():  # this can save much memory
            Out = torch.clamp(INoisy - model(INoisy), 0., 1.)
            "可取消的代码，网络的输出残差图像"
            residual_out = torch.clamp(model(INoisy), 0., 1.)

        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        # psnr = batch_PSNR(Out, ISource, 1.)
        # psnr_test += psnr
        # print("%s PSNR %f" % (f, psnr))
        # print(Out)

        # plt.savefig()
        # Img1 = Out.cuda().data.cpu().numpy()
        # cv2.imwrite(save_path + "%d.bmp" % count, Img1)  # 保存图像帧, 可为bmp，png, bmp等

        "小目标图像"
        img = Out.mul(255).byte()

        "可取消的代码，网络输出的残差图像"
        residual_img = residual_out.mul(255).byte()

        "小目标图像"
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))

        "可取消的代码，网络输出的残差图像"
        residual_img = residual_img.cpu().numpy().squeeze(0).transpose((1, 2, 0))

        "保存小目标图像"
        cv2.imwrite(save_path + name[count], img)
        "可取消的代码，保存网络输出的残差图像"
        cv2.imwrite(residual_path + name[count], residual_img)

        time_test.append(time.time() - time_test_start)
        print(name[count] + "\t" + "cost %.4f seconds " % time_test[count])
        time_test[count] = str("%.4f" % time_test[count])
        count += 1
    print("totally cost %.3f seconds ! " % (time.time() - time_start))

    data = [["测试样本"] + name, ["时间/s"] + time_test, ["图片大小"] + test_size]
    print(data)

    # workbook = xlrd.open_workbook("data/data_50epoch.xls")  # 打开工作簿
    # sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    # worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    # rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    # new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    # new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    # for i in range(4, 7):
    #     for j in range(count - 1):
    #         new_worksheet.write(i, j, data[i - 4][j])
    # new_workbook.save("data/data.xls")  # 保存工作簿

    # psnr_test /= len(files_source)
    # print("\nPSNR on test data %f" % psnr_test)


if __name__ == "__main__":
    main()

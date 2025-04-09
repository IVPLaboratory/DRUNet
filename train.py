import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.network_usrnet import ResUNet as net
from models.network_usrnet import ResDenUNet as neta
import time
from matplotlib import pyplot as plt
from utils.dataset_dncnn import DatasetDnCNN as D
import math
import logging
import torch.distributed as dist
from utils.dataset import Dataset, prepare_data
import shutil
import glob
import cv2

import tools as tl
from indicator import *

def normalize(data):
    return data / 255.
#获取当前系统时间
time_start = time.time()


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # GPU  指定显卡

#argparse模块的作用是用于解析命令行参数
#ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
#创建解析对象
parser = argparse.ArgumentParser(description="ResUnet")
#添加参数
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="model_zoo", help='path of log files')
# parser.add_argument('--valid-dir', type=str, default='valid/valid/')  # 原始验证图像
parser.add_argument('--outputs-dir', type=str, default='weights')  # 存放权重文件
# parser.add_argument('--results-valid', type=str, default='results_valid/')  #存放模型在验证集上的残差图像
# parser.add_argument('--accumulate', type=int, default=4, help='')
parser.add_argument('--growthrate', type=int, default=16)

#ArgumentParser 通过 parse_args() 方法解析参数。
opt = parser.parse_args()


def main():
    "create the dir to save weight files"
    opt.outputs_dir = os.path.join(opt.outputs_dir, 'g{}'.format(opt.growthrate))
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    # Load dataset
    print('Loading dataset ...\n')
    prepare_data(data_path='trainsets')
    dataset_train = Dataset()
    #为后面网络提供不同数据形式
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=False)

    print("# of training samples: %d\n" % int(len(dataset_train)))

    # load noise

    #Build model
    # model = net(in_nc=1, out_nc=1, nc=[16,32,64,128],
    #             nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

    # # #model for ResDenUNet
    # model = neta(in_channels=1, num_features=64, growth_rate=16, num_blocks=1, num_layers=8,
    #             nc=[64,128,256,512], downsample_mode='strideconv', upsample_mode="convtranspose")

    model = neta(in_channels=1, num_features=64, growth_rate=16, num_blocks=1, num_layers=8,
                    nc=[64,128,256], downsample_mode='strideconv', upsample_mode="convtranspose", nb=2, act_mode="R",)
    #计算均方损失误差，size_average = False，返回 loss.sum()，不取平均值，此参数不重要;
    criterion = nn.L1Loss(size_average=False)
    model.eval()

    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:9999',
                                world_size=1,# 定义进程通信方式
                                rank=0
                                )
        model = torch.nn.parallel.DistributedDataParallel(model)

    best_weights = model.state_dict()
    best_epoch = 0
    best_CON_OUT = 0.0
    best_sigma_out = 1000.0
    CONout = []
    sigmaOut = []

    # Optimizer 构造一个优化器对象Optimizer这个对象能保存当前的参数状态并且基于计算梯度更新参数，用model.parameters()找出训练中所有的参数
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # training
    writer = SummaryWriter(opt.outf)
    step = 0

    loss_list = []
    milestone = [opt.milestone]

    for epoch in range(opt.epochs):
        loss_i = []  # 每一个迭代开始时置零

        # if len(milestone) > 0:
        #     if epoch + 1 < milestone[0]:
        #         current_lr = opt.lr
        #     # elif epoch + 1 == milestone[0]:  # 在epoch=10时 lr*2
        #     #     milestone = milestone[1:]
        #     #     current_lr = opt.lr / 2.
        #     #     opt.lr = opt.lr / 2.
        #     else:
        #         milestone = milestone[1:]
        #         if epoch + 1 == 30:
        #             current_lr = opt.lr / 2.
        #             opt.lr = opt.lr / 2.
        #         if epoch + 1 == 50:
        #             current_lr = opt.lr / 10.
        #             opt.lr = opt.lr / 10.
        #
        #         if epoch + 1 == 80:
        #             current_lr = opt.lr / 10.
        #             opt.lr = opt.lr / 10.
        #         if milestone == []:
        #             milestone = [opt.epochs]  # 保证其大于epoch

        # set learning rate
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = current_lr
        # print('learning rate %f' % current_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr * (0.1 ** (epoch // int(opt.epochs * 0.8)))
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            # 对模型参数的梯度置0
            model.zero_grad()
            # # optimizer.zero_grad()的作用是清除所有优化的torch.Tensor的梯度
            optimizer.zero_grad()

            # img_train = data
            imgn_train = data[0]
            # print("imgn_train.shape = ", imgn_train.shape)
            noise = data[1]
            # print(imgn_train.shape)
            # print(noise.shape)
            # print(np.sum(imgn_train[0][0][0] - noise[0][0][0]))
            # if np.sum(imgn_train[0][0][0] - noise[0][0][0]) != 0:
            #     raise Exception("Mismatch of train and back")

            imgn_train = Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)

            loss = criterion(out_train, noise) / (imgn_train.size()[0] * 2)
            # 误差反向传播
            loss.backward()
            #
            # if (i + 1) % opt.accumulate == 0 or (i + 1) == len(loader_train):
            #     optimizer.step()
            #     optimizer.zero_grad()

            # # 所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            optimizer.step()
            #.item()将矩阵变为数值
            loss_i.append(loss.item())

            # results
            # model.eval()
            # out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)

            print("[epoch %d/%d][%d/%d] loss: %.4f " %
                      (epoch + 1, opt.epochs, i + 1, len(loader_train), loss.item()))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                # writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('PSNR on training data', step)
            step += 1

        # # the end of each epoch
        # model.eval()
        # print("-----------Loss_i的值------------")
        # print(loss_i)
        print("-----------Loss_i的长度------------")
        print(len(loss_i))
        print("-----------Loss_i的平均值------------")
        print(np.mean(loss_i))
        loss_list.append(np.mean(loss_i))
        print("-----------Loss_list的值------------")
        print(loss_list)
        print("loss_list长度为%d" %len(loss_list))
        print("=" * 100 + "\nepoch = %d  average loos = %f\n" % (epoch + 1, np.mean(loss_i)) + "=" * 100)

        with open('loss.txt', 'a+') as f:
            f.write(str(epoch) + ": " + str(np.mean(loss_i)) + '\n')
        f.close()

        # log the images
        #torch.clamp(input, min, max, out=None)将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量
        out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
        # Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        #
        # # save model
        # torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))



        # if (epoch + 1) % 5 == 0:
        #     plt.figure(figsize=(5, 5), dpi=600)
        #     plt.plot(range(len(loss_list)), loss_list, label="loss")
        #     plt.title("small object", fontsize="large")
        #     plt.xlabel("epoch", fontsize="large")
        #     plt.ylabel("loss", rotation=90, fontsize="large")
        #     plt.xlim(0, epoch)
        #     plt.ylim(0, 5)
        #     plt.grid()
        #     plt.legend()
        #     plt.savefig("./%d.png" % (epoch + 1))


        "Every epoch save the weight file"

        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, 'epoch_{}.pth'.format(epoch)))


        "拿当前权重在验证集上计算CG，更新最优CG"
        model.eval()

        "创建大目录存放所有验证集的残差图像, results_path = results_valid/"
        results_path='results_valid/'
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
            os.mkdir(results_path)
        else:
            os.mkdir(results_path)

        "创建大目录存放所有验证集的残差图像正则化后的结果, normalize_path = results_valid_normalize/"
        normalize_path = 'results_valid_normalize/'
        if os.path.exists(normalize_path):
            shutil.rmtree(normalize_path)
            os.mkdir(normalize_path)
        else:
            os.mkdir(normalize_path)

        "遍历validsets目录，依次取出每一个验证序列"
        for dir in os.listdir('validsets'):

            "加载验证集"
            print('Loading ' + dir + ' Images...')
            valid_dir = 'validsets/' + dir + '/valid'
            file_source = glob.glob(os.path.join(valid_dir, '*.bmp'))
            file_source.sort()

            path_out = results_path + dir
            if os.path.exists(path_out):
                shutil.rmtree(path_out)
                os.mkdir(path_out)
            else:
                os.mkdir(path_out)
            normalize_out = normalize_path + dir
            if os.path.exists(normalize_out):
                shutil.rmtree(normalize_out)
                os.mkdir(normalize_out)
            else:
                os.mkdir(normalize_out)

            count = 0
            name = []
            for i in os.listdir(valid_dir):
                name.append(i)
            name.sort()

            "模型处理验证集"
            for file in file_source:
                image = cv2.imread(file)
                image = normalize(np.float32(image[:, :, 0]))
                image = np.expand_dims(image, 0)
                image = np.expand_dims(image, 1)
                valid_img = torch.tensor(image).cuda()

                with torch.no_grad():
                    out = torch.clamp(valid_img - model(valid_img), 0., 1.)

                img = out.mul(255).byte()
                img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))

                cv2.imwrite(path_out + '/' + name[count], img)
                count += 1

            "计算CG, img_valid = validsets/***/valid, path_out = results_valid/***"
            img_valid = valid_dir

            "对残差图像做归一化处理"
            tl.normalize(path_out, normalize_out)

            img_out = []
            img_in = []
            con = []
            sigmal = []

            "path_out = results_valid/***; img_out = results_valid_normalize/***/*.bmp"
            for i in os.listdir(path_out):
                img_out.append(normalize_path + dir + '/' + i)

            "img_valid = validsets/***/valid; img_in = validsets/***/valid/*.bmp"
            for j in os.listdir(img_valid):
                img_in.append(img_valid + '/' + j)

            img_out.sort()
            img_in.sort()

            "计算每一张图的检测结果的CONout和sigmaOut"
            for i in range(len(img_in)):
                con.append(SBC(img_out[i], img_in[i])[0])
                sigmal.append(SBC(img_out[i], img_in[i])[1])

            CON = float("{:.2f}".format(np.mean(con)))
            Sigma_CON = float("{:.2f}".format(np.mean(sigmal)))
            CONout.append(CON)
            sigmaOut.append(Sigma_CON)

            print('validset {}: CONout = {}'.format(dir, CON))
            print('validset {}: sigmaOut = {}'.format(dir, Sigma_CON))
            with open('loss.txt', 'a+') as f:
                f.write(str(dir) + ", CONout: " + str(CON) + ", sigmaOut: " + str(Sigma_CON) + '; ')
            f.close()

            print("-----------------" + dir + " Done!--------------------")

        with open('loss.txt', 'a+') as f:
            f.write('\n')
        f.close()
        "选择最优CG的迭代期"
        # all_CON = float("{:.2f}".format(np.mean(CONout)))
        # all_Sigma = float("{:.2f}".format(np.mean(sigmaOut)))
        # if all_CON > best_CON_OUT and all_Sigma <= best_sigma_out:
        #     best_CON_OUT = all_CON
        #     print('best_CON_OUT:', best_CON_OUT)
        #     best_sigma_out = all_Sigma
        #     print('best_sigma_out:', best_sigma_out)
        #     best_epoch = epoch
        #     torch.save(model.state_dict(), 'weights/best.pth')
        #
        # print('best epoch: {}, CON: {}, sigmal_out: {}'.format(best_epoch, best_CON_OUT, best_sigma_out))

        # "创建目录存放残差图像, opt.results_valid = results_valid/"
        # save_path = opt.results_valid
        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)
        #     os.mkdir(save_path)
        # else:
        #     os.mkdir(save_path)
        #
        # "加载验证集, opt.valid_dir = valid/valid/"
        # print('Loading valid images...\n')
        # file_source = glob.glob(os.path.join(opt.valid_dir, '*.bmp'))
        # file_source.sort()
        #
        # count = 0
        # name = []
        # for i in os.listdir(opt.valid_dir):
        #     name.append(i)
        # name.sort()
        #
        # "模型处理验证集"
        # for file in file_source:
        #     image = cv2.imread(file)
        #     image = normalize(np.float32(image[:, :, 0]))
        #     image = np.expand_dims(image, 0)
        #     image = np.expand_dims(image, 1)
        #     valid_img = torch.tensor(image).cuda()
        #
        #     with torch.no_grad():
        #         out = torch.clamp(valid_img - model(valid_img), 0., 1.)
        #
        #     img = out.mul(255).byte()
        #     img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        #
        #     cv2.imwrite(save_path + name[count], img)
        #     count += 1
        # print('Done!')
        #
        # "计算CG, img_valid = valid/valid/; path_out = results_valid/"
        # img_valid = opt.valid_dir
        # path_out = save_path
        #
        # "对残差图像做归一化处理"
        # tl.normalize(path_out, 'results_valid_normalize/')
        #
        # img_out = []
        # img_in = []
        # # cg = []
        # con = []
        # sigmal = []
        #
        # "path_out = results_valid/; img_out = results_valid_normalize/***.bmp"
        # for i in os.listdir(path_out):
        #     img_out.append('results_valid_normalize/' + i)
        #
        # "img_valid = valid/valid/; img_in = valid/valid/***.bmp"
        # for j in os.listdir(img_valid):
        #     img_in.append(img_valid + j)
        #
        # img_out.sort()
        # img_in.sort()
        #
        # "计算每一个检测结果的CG"
        # for i in range(len(img_in)):
        #     con.append(SBC(img_out[i], img_in[i])[0])
        #     sigmal.append(SBC(img_out[i], img_in[i])[1])
        #     # cg.append(SBC(img_out[i], img_in[i]))
        #
        # CON = float("{:.2f}".format(np.mean(con)))
        # Sigma_CON = float("{:.2f}".format(np.mean(sigmal)))
        #
        # print('eval CON: %g' % CON)
        # print('eval Sigma_CON: %g' % Sigma_CON)
        # with open('loss.txt', 'a+') as f:
        #     f.write("CONout: " + str(CON) + ", sigmaOut: " + str(Sigma_CON) + '\n')
        # f.close()

        # print("-----------------1个结束用验证集测试结束--------------------")
        # "选择最优CG的迭代期"
        # if CON > best_CON_OUT and Sigma_CON < best_sigma_out:
        #     best_CON_OUT = CON
        #     print('best_CON_OUT:',best_CON_OUT)
        #     best_sigma_out = Sigma_CON
        #     print('best_sigma_out:', best_sigma_out)
        #     best_epoch = epoch
        #     torch.save(model.state_dict(), 'weights/best.pth')

    # print('best epoch: {}, CON: {}, sigmal_out: {}'.format(best_epoch, best_CON_OUT,best_sigma_out))

    plt.figure(figsize=(5, 5), dpi=600)
    plt.plot(range(len(loss_list)), loss_list, label="loss")
    plt.xlabel("epoch", fontsize="large")
    plt.ylabel("loss", rotation=90, fontsize="large")
    plt.xlim(0, opt.epochs)
    plt.ylim(0, 5)
    plt.grid()
    plt.legend()
    plt.savefig("ResUnet_small.eps", bbox_inches='tight', pad_inches=0, dpi=600)  # 保存结果为600dpi的eps格式, 并且去除白边
    plt.savefig("ResUnet_small.pdf", bbox_inches='tight', pad_inches=0, dpi=600)  # 保存结果为600dpi的pdf格式，并且去除白
    plt.title("small object", fontsize="large")
    plt.savefig("./loss.png")
    print('totally cost %.3f hours ! ' % ((time.time() - time_start) / 3600))


if __name__ == "__main__":
    main()

# ================================================================
# lr = 0.00015  milestone = 10  lr/10     =30,50,70     lr = opt.lr/10
# ================================================================

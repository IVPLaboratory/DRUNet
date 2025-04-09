import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
# from utils.utils import data_augmentation


def normalize(data):
    return data / 255.


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data(data_path):
    # train
    print('process training data')
    #os.path.join()将多个路径组合返回，读取训练集的所有图片
    #glob.glob()返回所有匹配的文件路径列表
    files = glob.glob(os.path.join(data_path, "train", '*.bmp'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        Img = np.expand_dims(img[:, :, 0].copy(), 0)
        # Img = Img.reshape(1, 1, Img.shape[1], Img.shape[2])
        Img = np.float32(normalize(Img))
        data = Img.copy()
        h5f.create_dataset(str(train_num), data=data)
        train_num += 1

        # patches = Im2Patch(Img, win=patch_size, stride=stride)
        # for n in range(patches.shape[3]):
        #     data = patches[:, :, :, n].copy()
        #     h5f.create_dataset(str(train_num), data=data)
        #     train_num += 1
        #     for m in range(aug_times - 1):
        #         data_aug = data_augmentation(data, np.random.randint(1, 8))
        #         h5f.create_dataset(str(train_num) + "_aug_%d" % (m + 1), data=data_aug)
        #         train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'background_train', '*.bmp'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:, :, 0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    # print('training set, # samples %d\n' % train_num)
    # print('val set, # samples %d\n' % val_num)

#加载数据
#Dataset是Pytorch中的一个数据读取类,提供一种方式去读取数据以及label
class Dataset(udata.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        h5f = h5py.File('train.h5', 'r')
        h5f_back = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        self.keys_back = list(h5f_back.keys())  # self.keys==self.keys_back
        # print("self.keys = ", self.keys)
        # random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)
#__getitem__函数的作用是根据索引index遍历数据，一般返回image的Tensor形式和对应标注。当然也可以多返回一些其它信息，这个根据需求而定
    def __getitem__(self, index):
        h5f = h5py.File('train.h5', 'r')
        h5f_back = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        data_back = np.array(h5f_back[key])
        h5f.close()
        h5f_back.close()
        return [torch.Tensor(data), torch.Tensor(data_back)]

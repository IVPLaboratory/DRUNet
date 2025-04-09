import xml.etree.cElementTree as et
import cv2
import os
import shutil
import numpy as np


def normalize(path, path_save):
    path = path
    path_save = path_save
    for i in os.listdir(path):
        img = path + '/' + i
        im = cv2.imread(img)
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        # _, thresh = cv2.threshold(im_gray, 238, 255, cv2.THRESH_BINARY)
        Img = cv2.normalize(im_gray, im_gray, 0, 255, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(path_save + '/' + i, Img)



def xml_parse(path):
    tree = et.parse(path)  # 传入待解析的xml文件的地址
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)
    pList = []
    for Object in root.findall('object'):
        tempList = []
        name = Object.find('name').text
        bndbox = Object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        tempList.append(xmin)  # 左上角横坐标
        tempList.append(ymin)  # 左上角纵坐标
        tempList.append(xmax-xmin)  # 宽
        tempList.append(ymax-ymin)  # 高
        # tempList.append(width)  # 4
        # tempList.append(height)  # 5
        # tempList.append(name)  # 6s
        pList.append(tempList)
    return pList  # 返回一个二维数组


if __name__ == "__main__":
    path = "results/"
    path_save1 = "test_cloud1_results/"
    if os.path.exists(path_save1):
        shutil.rmtree(path_save1)
        os.mkdir(path_save1)
    else:
        os.mkdir(path_save1)
    for i in os.listdir(path):
        img = path + i
        im = cv2.imread(img)
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        # _, thresh = cv2.threshold(im_gray, 238, 255, cv2.THRESH_BINARY)
        Img = cv2.normalize(im_gray, im_gray, 0, 255, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(path_save1 + i, Img)

    # path = "test_cloud2_results/"
    # # path_save2 = "cloud1Th/"
    # path_save2 = "cloud2Th/"
    # if os.path.exists(path_save2):
    #     shutil.rmtree(path_save2)
    #     os.mkdir(path_save2)
    # else:
    #     os.mkdir(path_save2)
    # for i in os.listdir(path):
    #     img = path + i
    #     im = cv2.imread(img)
    #     im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    #     th1 = np.mean(im_gray) + 30 * np.std(im_gray)
    #
    #     # _, thresh = cv2.threshold(im_gray, 238, 255, cv2.THRESH_BINARY)
    #     _, thresh = cv2.threshold(im_gray, th1, 255, cv2.THRESH_BINARY)
    #     cv2.imwrite(path_save2 + i, thresh)
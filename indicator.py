import cv2
import numpy as np
from tools import xml_parse


def SBC(img_out, img_in):
    # List_out中存着：CON_out, std背景区域的标准差, std_Img, status
    List_out = Img_out(img_out)
    "目标位置信息，左上角坐标和宽高"
    status = List_out[3]
    "List_in中存着：CON_in, std背景区域的标准差, std_Img"
    List_in = Img_in(img_in, status)
    CON_out = List_out[0]
    sigmal_bg = List_out[2]

    # CG = CON_out / CON_in

    # return CG
    return [CON_out,sigmal_bg]

"img = results_valid_normalize/***/*.bmp"
def Img_out(img):
    im = cv2.imread(img)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    img_xml = 'validsets/' + img.split('/')[1] + '/XML/' + img.split('/')[2].replace('bmp', 'xml')
    status = xml_parse(img_xml)[0]  # GT,标准框 二维列表[[x,y,w,h]] 左上角坐标和宽高
    ob = im_gray[status[1]:status[1] + status[3], status[0]:status[0] + status[2]]  # 取图像框中的部分
    mean_t = np.mean(ob)  # 目标区域平均像素值

    "判断GT是否处在图像边缘"
    if status[1] - 20 < 0:
        ob_large = im_gray[0:status[1] + status[3] + 20, status[0] - 20: status[0] + status[2] + 20]  # 到顶
    else:
        ob_large = im_gray[status[1] - 20:status[1] + status[3] + 20,
                   status[0] - 20: status[0] + status[2] + 20]  # 框各边加20

    lar = np.array(ob_large).astype(np.int)

    "以目标区域GT为中间，向两边扩充20个像素，作为背景区域"
    if status[1] - 20 < 0:
        lar[status[1]:status[3], 20:20 + status[2]] = np.ones(
            lar[status[1]:status[3], 20:20 + status[2]].shape) * -1  # 将框内的像素值全部置-1
    else:
        lar[20:20 + status[3], 20:20 + status[2]] = np.ones(
            lar[20:20 + status[3], 20:20 + status[2]].shape) * -1  # 将框内的像素值全部置-1
    std = np.std(lar[lar > -1])  # 背景区域的标准差
    mean_b = np.mean(lar[lar > -1])  # 背景区域的平均像素值
    # print("\tmean_t = %f\tmean_b = %f" % (mean_t, mean_b))

    im_gray = im_gray.astype(int)
    im_gray[status[1]:status[1] + status[3], status[0]: status[0] + status[2]] = np.ones(
        im_gray[status[1]:status[1] + status[3], status[0]: status[0] + status[2]].shape) * -1  # 将框内的像素值全部置-1
    std_Img = np.std(im_gray[im_gray > -1])  # 整图,除目标框外的标准差

    # CON_out = abs(mean_t - mean_b)
    CON_out = mean_t - mean_b

    return [CON_out, std, std_Img, status]


"img = validsets/***/valid/*.bmp"
def Img_in(img, status):
    im = cv2.imread(img)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    ob = im_gray[status[1]:status[1] + status[3], status[0]:status[0] + status[2]]  # 取图像框中的部分
    "计算原始图像中目标区域的平均像素值"
    mean_t = np.mean(ob)

    if status[1] - 20 < 0:
        ob_large = im_gray[0:status[1] + status[3] + 20, status[0] - 20: status[0] + status[2] + 20]  # 到顶
    else:
        ob_large = im_gray[status[1] - 20:status[1] + status[3] + 20,
                   status[0] - 20: status[0] + status[2] + 20]  # 框各边加20

    lar = np.array(ob_large).astype(np.int)
    lar[20:20 + status[3], 20: 20 + status[2]] = np.ones(
        lar[20:20 + status[3], 20: 20 + status[2]].shape) * -1  # 将框内的像素值全部置-1

    "背景区域的标准差和平均像素值"
    std = np.std(lar[lar > -1])
    mean_b = np.mean(lar[lar > -1])

    # im_gray = im_gray.astype(int)
    im_gray[status[1]:status[1] + status[3], status[0]: status[0] + status[2]] = np.ones(
        im_gray[status[1]:status[1] + status[3], status[0]: status[0] + status[2]].shape) * -1  # 将框内的像素值全部置-1

    std_Img = np.std(im_gray[im_gray > -1])

    return [abs(mean_t - mean_b), std, std_Img]
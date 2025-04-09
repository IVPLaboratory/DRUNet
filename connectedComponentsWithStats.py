import cv2
import numpy as np
import os


# img = np.array([
#     [0, 255, 0, 0],
#     [0, 0, 0, 255],
#     [0, 0, 0, 255],
#     [255, 0, 0, 0]], np.uint8)
# img = np.array([
#     [0, 0, 0, 0],
#     [0, 0, 0, 255],
#     [0, 0, 255, 255],
#     [0, 0, 0, 0]], np.uint8)
# nccomps = cv2.connectedComponentsWithStats(img)
# print(nccomps[2])
# a

def cen(status):
    centroids_x = status[0] + status[2] / 2
    centroids_y = status[1] + status[3] / 2
    return np.array([centroids_x, centroids_y])


img_path = "./DnCNN/test_strong_results_old/"
path = "./rectangle_DnCNN_old/"
if not os.path.exists(path):
    os.mkdir(path)

img = []
name = []
for i in os.listdir(img_path):
    img.append(img_path + i)
    name.append(i)
img.sort()
name.sort()

p = 0
false = []
emp = []

with open("All.txt", 'w') as f:
    k = 10
    f.write("k = %d" % k + "\r\n")
    f.write("th1 = np.mean(im_gray) + k * np.std(im_gray)" + "\r\n")
    for i in img:
        k = 10
        im = cv2.imread(i)
        w, h, n = im.shape
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        th1 = np.mean(im_gray) + k * np.std(im_gray)
        th1 = 0
        f.write("\r\n" + name[p] + "\r\n")
        f.write("阈值th1 = %.3f" % th1 + "\r\n")

        _, thresh = cv2.threshold(im_gray, th1, 255, cv2.THRESH_BINARY)
        # cv2.imshow('th', thresh)
        nccomps = cv2.connectedComponentsWithStats(thresh)  # labels,stats,centroids
        num_labels = nccomps[0]  # 连通域数量
        labels = nccomps[1]  # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
        centroids = nccomps[3]  # 连通域的中心点
        status = nccomps[2]  # 连通域的信息：对应各个轮廓的x、y、width、height和面积
        background = []
        for row in range(status.shape[0]):
            if (status[row, :][0] == 0 and status[row, :][1] == 0):
                background.append(row)
            else:
                continue

        status = np.delete(status, background, axis=0)
        # status = status
        # if len(status) != 0:
        #     rec_value_max = np.asarray(status[:, 4].max())
        #     re_value_max_position = np.asarray(status[:, 4].argmax())
        #     h = np.asarray(labels, 'uint8')
        #     h[h == (re_value_max_position + 1)] = 255
        # for single in range(centroids.shape[0]):  # 打印中心点坐标
        #     print(tuple(map(int, centroids[single])))
        # position = tuple(map(int,centroids[single]))
        # cv2.circle(h, position, 1, (255,255,255), thickness=0,lineType=8)

        # cv2.rectangle(image, 左上角坐标, 右下角坐标, color(BRG), 线条粗度)，示例：
        # cv2.rectangle(img, (240, 0), (480, 375), (0, 255, 0), 2)
        f.write("目标个数：%d" % len(status) + "\r\n")
        f.write("status = " + "\r\n")
        print(i.split("/")[3], "\t", "status = ", status, "\n")
        for Numpy in status:
            f.write(str(Numpy) + "\r\n")

        if len(status) == 0:  # 没有检测到目标
            emp.append(i.split("/")[2])
            cv2.imwrite(path + name[p], im)
            p += 1

        # while len(status) > 1:
        #     status = status[np.lexsort(status.T)][::-1]  # 排序，像素个数从大到小，目标一定是最亮的，为第一个或前两个
        #     flag = 0
        #     jj = 1
        #     num_rec = 0
        #     for ss in status[1:]:
        #         if all(abs(cen(status[0]) - cen(ss)) < 15) == True:  # 中心点坐标小于15,认为是一个目标（x,y的差都小于15）
        #             j, k = status[0], ss
        #             status[0] = np.array(
        #                 [min(j[0], k[0]), min(j[1], k[1]), max(j[0] + j[2], k[0] + k[2]) - min(j[0], k[0]),
        #                  max(j[1] + j[3], k[1] + k[3]) - min(j[1], k[1]), 100])  # 让status[0]为，目标左上角坐标与宽高，像素个数无所谓，置100
        #             status = np.delete(status, jj, axis=0)
        #             flag = 1
        #             num_rec += 1
        #             break
        #         else:
        #             jj += 1
        #         if num_rec == 2:
        #             break
        #
        #     if flag == 0 or len(status) == 1:  # 全部画框
        #         if len(status) > 1: false.append(i.split("/")[2])
        #         print(i.split("/")[2], "\t", "status = ", status, "\n")
        #         for j in status:
        #             draw_1 = cv2.rectangle(im, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (0, 255, 0), 1)
        #             im = draw_1
        #         cv2.imwrite(path + name[p], draw_1)
        #         p += 1
        #         break
        elif len(status) == 1:  # 仅有一个目标
            for j in status:  # j=status[0]
                draw_1 = cv2.rectangle(im, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (0, 255, 0), 1)
                cv2.imwrite(path + name[p], draw_1)
                p += 1
                break
        else:  # 有多个目标
            false.append(i.split("/")[2])
            for j in status:  # j=status[0]
                draw_1 = cv2.rectangle(im, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (0, 255, 0), 1)
                cv2.imwrite(path + name[p], draw_1)
            p += 1
    emp.sort()
    false.sort()
    print("没有目标的个数 = ", len(emp), "\n")
    for i in emp:
        print(i)
    print("\n有误检的图片个数 = ", len(false), "\n")
    for i in false:
        print(i)

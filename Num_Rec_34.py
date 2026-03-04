import cv2    # 导入opencv
import numpy as np  # 导入numpy 作为 np
import torch
from LeNet5_34 import model        # 从LeNet5文件中 引入 model
# 在Python 导入opencv的名称 叫做cv2


def show_img(title, img, flag=False):
    """
    函数的作用  用来展示图片
    :param title:  对应的窗口名称
    :param img:  要展示的图片
    :param flag: 开关 默认为true 用布尔 方便理解
    :return:
    """
    if flag:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)   # 命名一个窗口， 命名为title
        # cv2.WINDOW_NORMAL  就是让窗口 可以调节大小
        cv2.imshow(title, img)  # 对应展示的窗口名  img就是对应的图片


# 1. 数据输入
# a = "images/img.png"   # 来自内容根的路径
img_path = r"./images/05_6.png"
input_img = cv2.imread(img_path)  # imread函数想要一个 filename参数  img_path 就赋给filename
origin_img = input_img.copy()  # 复制一份 原图像
img_h, img_w = input_img.shape[0:2]  # Python取值的方法 左闭右开0 1

# Shape[H,W,C] --->  [Height(高), Width(宽), Channel(通道)]
hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)    # convert 2:to 转换色彩空间 BGR变为HSV
show_img("hsv_img", hsv_img, flag=False)  # 展示 HSV

hsv_low = [100, 43, 46]  # 黄色hsv最小的范围  是一个list
hsv_high = [124, 255, 255]   # 黄色hsv最大的范围
# array表示数组  np.array把列表转为数组
# 最低值 和 最高值 中间的值 变为 白色
filter_hsv_img = cv2.inRange(hsv_img, np.array(hsv_low), np.array(hsv_high))    # 找出对应范围
show_img("filter_hsv_img", filter_hsv_img, flag=True)

# 利用中值滤波 去 过滤
median_filter_img = cv2.medianBlur(filter_hsv_img, ksize=7)     # ksize=7 让中值的范围变大一点
show_img("median_filter_img", median_filter_img, False)

# 开运算 (先腐蚀再膨胀膨胀)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 拿到结构元素  cv2.MORPH_RECT表示矩形 Size=5*5
# 形态学运算函数  1参: 要修改的图像  2参: 表示开运算 3参: 开运算对应的结构元素
op_img = cv2.morphologyEx(median_filter_img, cv2.MORPH_OPEN, kernel)
show_img("op_img", op_img, False)


# 寻找轮廓
num_contours = []   # 创建一个 存储外接矩形的  list
# 找轮廓  参数2: 只找外轮廓  参数3: 保留4个顶点坐标
# 不想要的变量 可以用 下划线 来命名  这里下划线对应的变量  表示 轮廓等级
# cts 是一个list 存储找到的轮廓
cts, _ = cv2.findContours(op_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 画出轮廓
# cv2.drawContours(origin_img, cts, -1, (0, 0, 255), 3)
# show_img("contour_img", origin_img, True)

# 遍历轮廓
for contour in cts: # 找每一个轮廓
    area = cv2.contourArea(contour) # 计算轮廓面积
    if area < 1000:     # 面积小于1000的不要了
        continue

    # 获取最小外接矩形 中心点坐标，宽高，旋转角度
    rect = cv2.minAreaRect(contour)
    # rect 有三个值 ---> (0 中心点坐标，1 宽高，2 旋转角度)

    # 要求矩形区域长宽比在0.8到1.2之间，其余的矩形排除
    area_w, area_h = rect[1]
    # 谁大除另一个
    # if area_w < area_h:
    #     wh_ratio = area_h / area_w
    # else:
    #     wh_ratio = area_w / area_h
    wh_ratio = area_h / area_w if area_w < area_h else area_w / area_h
    if wh_ratio > 1.2:
        num_contours.append(rect)
        # 画轮廓函数  参1: 画哪个图像上 参2: 要画的轮廓 参3: 闭合  参4: 颜色  参5： 框粗细
        # cv2.drawContours(origin_img, contour, -1, (0, 0, 255), 3)     # 直接画有散点
        # boxPoints函数 可以讲 外接矩形 转换为 4个坐标  np.intp函数 作用将坐标转换为 int
        # tmp_box = np.intp(cv2.boxPoints(rect))
        # 画轮廓函数  参1: 画哪个图像上 参2: 要画的轮廓加了括号 参3: 闭合  参4: 颜色  参5： 框粗细
        # cv2.drawContours(origin_img, [tmp_box], -1, (0, 0, 255), 3)

show_img("contour_img", origin_img, True)


# 矫正数字块
correction_imgs = []        # 创建一个list  用于存放图片
label_location = []         # 创建一个list  用于存放 位置
# enumerate 函数 可以将可迭代的数据类型 ---> list  包装起来 返回两个值  第一个值为idx, 第二个值为内容
for rect in num_contours:       # 里面存储了 符合条件的 外接矩形
    angle = rect[2]         # 旋转角度
    box = np.intp(cv2.boxPoints(rect))      # 获取 带有旋转矩形块的 四个顶点坐标  ---> 转换为 整型
    # box 是4个点 同时 他是一个ndarray  idx=index的缩写  index表示索引
    x_min, x_max = np.min(box[:, 0]), np.max(box[:, 0])     # 所有行idx=0取最小  所有行idx=0取最大
    y_min, y_max = np.min(box[:, 1]), np.max(box[:, 1])     # 所有行idx=1取最小  所有行idx=1取最大
    crop_img = filter_hsv_img[y_min:y_max, x_min:x_max]
    show_img("crop_img", crop_img, False)        # 倾斜 且 带有黑边的 二值化图像
    # crop.shape ---> h, w  ---> crop_img_center = (w/2, h/2)
    crop_img_center = (crop_img.shape[1] / 2, crop_img.shape[0] / 2)   # 坐标系 是 相对crop_img 小图而言的
    label_location.append([x_min, y_min + 20])            # 左上角坐标

    if angle == 90:
        img_fin = crop_img
        print(img_fin.shape)
        # show_img("final_img", img_fin, True)
        correction_imgs.append(img_fin)
    else:
        box = np.intp(cv2.boxPoints(rect))
        # lambda表示式   box :(4, 2)  每一行 (x, y) ---> 新的x变量  取新x变量的idx=0
        # 按照X 从小到大进行排序 赋给sorted_arr
        sorted_arr = sorted(box, key=lambda x: x[0])  # 按照X进行排序
        # 在变为array
        box = np.array(sorted_arr)
        # 找到第一个点  和 最后一个点  -1代表最后一个
        left_point, right_point = box[0], box[-1]
        if left_point[1] <= right_point[1]:  # 正角度, 左低右高
            # 放射变化  找出旋转矩阵
            # getRotationMatrix2D  参1: 要中心, 参2: 角度  参3: 缩放大小 最后返回一个旋转矩阵
            mat = cv2.getRotationMatrix2D(crop_img_center, angle - 90, 1)       # 求旋转矩阵
        else:  # 负角度, 左高右低
            mat = cv2.getRotationMatrix2D(crop_img_center, angle, 1)

        # 先W  在H
        # 应用放射变化 参1: 要转的图像 参2: 旋转的矩阵 参数3：(W, H)
        final_img = cv2.warpAffine(crop_img, mat, (crop_img.shape[1], crop_img.shape[0]))  # 转回去
        # print(crop_img.shape, final_img.shape)
        show_img("rotated_img", final_img, False)

        tmp_h, tmp_w = final_img.shape[:2]
        white_ratio = 0.1
        final_img[0:int(white_ratio*tmp_h), :] = 255             # 上
        final_img[int((1-white_ratio) * tmp_h):, :] = 255       # 下
        final_img[:, :int(white_ratio * tmp_w)] = 255           # 左
        final_img[:, int((1 - white_ratio) * tmp_w):] = 255     # 右
        show_img("final_img", final_img, False)
        correction_imgs.append(final_img)

# 开始预测
predict_result = []     # 创建了一个预测结果的列表
model_img = []
# zip整合一下 包起来  顺序是一一对应
for correction_img, label_pos in zip(correction_imgs, label_location):
    # resize改变大小 参1: 要改变的图片 参2: 目标尺寸
    correction_img_28 = cv2.resize(correction_img, (28, 28))    # 尺寸变换位 28*28
    # threshold 二值化函数 ---> 变成黑白图
    # 参1: 要变的图像 参2: 低阈值  参3: 高阈值 参4: 转变方式cv2.THRESH_BINARY_INV 低于127变255 高于127 变0
    ret, img_thresh = cv2.threshold(correction_img_28, 127, 255, cv2.THRESH_BINARY_INV)     # 二值化
    model_img = torch.tensor(img_thresh,dtype=torch.float32).unsqueeze(0)
    show_img("img_thresh", img_thresh, True)
    index = model.predict(img_thresh)
    predict_result.append(index)                # 列表添加
    # 参1: 要修改的图像 参2: 添加的文字 参3: 文字位置 参4: 文字字体 参5: 缩放 参6: 文字颜色 参7: 文字粗细
    origin_img = cv2.putText(origin_img, f'{index}', label_pos,
                             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)    # 图像 文字内容 文字位置 字体 缩放 文字颜色 文字粗细
show_img("result", origin_img, True)
print("Done")

cv2.waitKey(0)  # 防止窗口一闪而过 等待

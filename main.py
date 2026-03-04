import torch as tc
import cv2
import numpy as np


from LeNet5_34 import model
import os


def show_img(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name,img)
    cv2.waitKey(0)




def red_img(path):
    chepai = []
    img=cv2.imread(path)
    n = 1
    img_width = img.shape[0]
    img_height = img.shape[1]

    img_resize_width = round(n * img_width)
    img_resize_height = round(n * img_height)
    new_img_1 = cv2.resize(img, (img_resize_height, img_resize_width))
    gray = cv2.cvtColor(new_img_1, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=3)  # 水平梯度
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=3)  # 垂直梯度
    # 计算梯度大小
    edge = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # 归一化到0-255
    edge = np.clip(edge, 0, 255).astype(np.uint8)
    # show_img('edge',edge)
    kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
    op_img = cv2.morphologyEx(edge,cv2.MORPH_CROSS,kernel_x,iterations=2)
    kernel_y= cv2.getStructuringElement(cv2.MORPH_RECT,(3,9))
    op_img = cv2.morphologyEx(op_img,cv2.MORPH_OPEN,kernel_y,iterations=2)
    # show_img('op',op_img)
    ret, th = cv2.threshold(op_img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # show_img('th',th)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,5))
    op_img = cv2.erode(th,kernel_y,iterations=2)
    op_img = cv2 .dilate(op_img,kernel,iterations=1)
    op_img =cv2.erode(op_img,kernel_y,iterations=2)
    med_img =cv2.medianBlur(op_img,5)
    # show_img("med",med_img)
    contours, h = cv2.findContours(med_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 寻找轮廓
    # 定义车牌的标准宽高比范围
    plate_aspect_ratio_min = 2.9  # 最小宽高比
    plate_aspect_ratio_max = 5 # 最大宽高比

    # 遍历每个轮廓并检查其宽高比
    for contour in contours:
        # 获取最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        # 计算宽高比
        aspect_ratio = w/h if w>h else h/w
        # show_img('1',new_img_1[y:y+h,x:x+w])
        # print(aspect_ratio,w*h)
        # 检查宽高比是否在车牌比例范围内
        if plate_aspect_ratio_min <= aspect_ratio <= plate_aspect_ratio_max and w*h>5000:
            # 添加到列表中
            print(w, h, w / h,w*h)
            chepai.append([(x, y, w, h),new_img_1[y:y+h,x:x+w]])
            # 绘制矩形
            cv2.rectangle(new_img_1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    show_img('img',new_img_1)
    return chepai
def prect(char):
   # show_img('in',input_img)
   out = []
   for input_img in char:
    gray = cv2.cvtColor(input_img[1],cv2.COLOR_BGR2GRAY)
    correction_img_28 = cv2.resize(gray, (28, 28))
    model_img = tc.tensor(correction_img_28, dtype=tc.float32).unsqueeze(0)
    index = model.predict(model_img)
    out.append(index)
   return out

def fengechar(image):
    char = []
    img = cv2.resize(image, (200, 100))
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img)}")
    if img.size == 0:
        raise ValueError("Empty image array")
    #     # 原有 resize 操作（确保图像是3通道）
    if len(img.shape) == 2:  # 灰度图转3通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_ot= cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    binary = cv2.erode(binary_ot,kernel)
    binary = cv2.dilate(binary,kernel,iterations=3)
    # show_img('b',binary)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤掉不符合条件的轮廓
    min_height = 10
    min_width = 10
    max_height = 100
    max_width = 100
    characters = []
    out_img =img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        aspect_ratio = w / h
        if (min_width <= w <= max_width and min_height <= h <= max_height and
                0.1 <= aspect_ratio <= 1.0):  # 假设字符宽高比在 0.1到 1.0 之间
            characters.append((x, y, w, h))

    # 根据 x 坐标对字符进行排序
    characters = sorted(characters, key=lambda char: char[0])
    # 裁剪每个字符并显示
    for (x, y, w, h) in characters:
        character = img[y:y + h, x:x + w]
        # show_img('1',character)
        char.append([(x, y, w, h), character])
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 显示结果
    out = prect(char)
    print(out)
    cv2.imshow('Segmented Characters', out_img)
    cv2.waitKey(0)


def find_waves(threshold, histogram):
    """ 根据设定的阈值和图片直方图，找出波峰，用于分隔字符 """
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def remove_upanddown_border(img):
    """ 去除车牌上下无用的边缘部分，确定上下边界 """
    plate_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, plate_binary_img = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和
    row_min = np.min(row_histogram)
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)
    # 挑选跨度最大的波峰
    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1] - wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    # show_img("plate_binary_img", plate_binary_img)
    return plate_binary_img
path=[]
def redpath(dir_path):
    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)
        path.append(subdir_path)
redpath("./images")
for img_path in path:
    chepai = red_img(img_path)
    for img in chepai:
        nextimg = remove_upanddown_border(img[1])
        fengechar(nextimg)
# red_img("images/img3-zh_ji_T_I_5-8-8-8.png")
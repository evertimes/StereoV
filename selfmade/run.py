import os.path
from threading import Thread
import math
import cv2
import numpy as np
import sys


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def sobel_filter(image):
    height, width = image.shape
    result = np.zeros((height, width))  # Empty result
    sobel_x_mask = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))  # Sobel horizontal mask
    sobel_y_mask = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))  # Sobel vertical mask
    for y in range(2, width - 2):  # Bounds y
        for x in range(2, height - 2):  # Bounds x
            cx, cy = 0, 0
            for y_slip in range(0, 3):
                for x_slip in range(0, 3):
                    pix = image[x + x_slip - 1, y + y_slip - 1]
                    if x_slip != 1:
                        cx += pix * sobel_x_mask[x_slip, y_slip]
                    if y_slip != 1:
                        cy += pix * sobel_y_mask[x_slip, y_slip]
            out_pix = math.sqrt(cx ** 2 + cy ** 2)  # Calculating gradient
            result[x, y] = out_pix if out_pix > 0 else 0  # Filtered result
    np.putmask(result, result > 255, 255)  # Normailizng
    return result


def calc_left_disparity(gray_left, gray_right, num_disparity, block_size):
    height, width = gray_right.shape
    disparity_matrix = np.zeros((height, width), dtype=np.float32)
    half_block = block_size // 2

    for i in range(half_block, height - half_block):

        for j in range(half_block, width - half_block):
            ##Вырезаем окно blocksizexblocksize
            left_block = gray_left[i - half_block:i + half_block, j - half_block:j + half_block]
            diff_sum = 32767  # Большое значение что бы любое другое было меньше
            disp = 0  # Изначальный диспаритет для???
            for d in range(0, min(j - half_block - 1, num_disparity)):
                # Вырезаем окно на правом изображении со смещением от 0 до j-пол блока-1 или максимальное смещение
                right_block = gray_right[i - half_block:i + half_block, j - half_block - d:j + half_block - d]
                sad_val = np.sum(np.sum(np.abs(right_block - left_block)))  # Разность блоков по модулю и сумма матрицы.
                # Самая минимальная разность будет соотвествовать наиболее одинаковым блокам
                # Если sad меньше чем прошлая сумма, то записываем. Диспаритетом становится
                # Тот самый диспартет из цикла
                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d
            # После конца цикла записываем для соотвествующего пикселя соотвествующую диспаритетность
            disparity_matrix[i - half_block, j - half_block] = disp
    return disparity_matrix


# Calculate right disparity


def calc_right_disparity(gray_left, gray_right, num_disparity, block_size):
    height, width = gray_right.shape
    disparity_matrix = np.zeros((height, width), dtype=np.float32)
    half_block = block_size // 2

    for i in range(half_block, height - half_block):
        print("%d%% " % (i * 100 // height), end=' ', flush=True)

        for j in range(half_block, width - half_block):
            right_block = gray_right[i - half_block:i + half_block, j - half_block:j + half_block]
            diff_sum = 32767
            disp = 0

            for d in range(0, min(width - j - half_block, num_disparity)):

                left_block = gray_left[i - half_block:i +
                                                      half_block, j - half_block + d:j + half_block + d]
                sad_val = np.sum(np.sum(np.abs(right_block - left_block)))

                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d

            disparity_matrix[i - half_block, j - half_block] = disp
    print('100%')
    return disparity_matrix


def cross_check(disp_left, disp_right):
    height, width = disp_left.shape
    out_image = disp_left

    for h in range(1, height - 1):
        for w in range(1, width - 1):
            left = int(disp_left[h, w])
            if w - left > 0:
                right = int(disp_right[h, w - left])
                dispDiff = left - right
                if dispDiff < -1:
                    out_image[h, w] = 0
                elif dispDiff > 1:
                    out_image[h, w] = 0
    return out_image


def getDisparityMap(left_image_path, right_image_path, num_disparity, block_size, save_intermediate):
    left = cv2.imread(left_image_path, 0)
    right = cv2.imread(right_image_path, 0)
    t = ThreadWithReturnValue(target=sobel_filter, args=[left])
    t2 = ThreadWithReturnValue(target=sobel_filter, args=[right])
    print("Start sobel")
    t.start()
    t2.start()
    filtered_left = t.join()
    filtered_right = t2.join()
    t = ThreadWithReturnValue(target=calc_left_disparity,
                              args=[filtered_left, filtered_right, num_disparity, block_size])
    t2 = ThreadWithReturnValue(target=calc_right_disparity,
                               args=[filtered_left, filtered_right, num_disparity, block_size])
    print("Start disparity")
    t.start()
    t2.start()
    disparity_left = t.join()
    disparity_right = t2.join()
    disparity = cross_check(disparity_left, disparity_right)
    if save_intermediate:
        cv2.imwrite('output/sobel_left.png', filtered_left)
        cv2.imwrite('output/sobel_right.png', filtered_right)
        cv2.imwrite('output/disparity_left.png', disparity_left)
        cv2.imwrite('output/disparity_right.png', disparity_right)
    return disparity


def getDepthMap(disparity, baseline, fx):
    depth = np.zeros(shape=disparity.shape).astype(float)
    depth[disparity > 0] = (fx * baseline) / (disparity[disparity > 0] + 40)
    return depth


def run(left_image, right_image, num_disparity, block_size, baseline, fx, save):
    if not os.path.exists(left_image):
        print("Path "+left_image+"doesnt exists")
        return
    if not os.path.exists(right_image):
        print("Path "+right_image+"doesnt exists")
        return
    #disparity = getDisparityMap(left_image, right_image, num_disparity, block_size, save)
    disparity = cv2.imread("../images/znewkuba.png")
    depth = getDepthMap(disparity, baseline, fx)
    cv2.imwrite("output/disparity.png", disparity)
    cv2.imwrite("output/depth.png", depth)


run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), bool(sys.argv[7]))

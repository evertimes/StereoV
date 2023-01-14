import os.path
import math
import cv2
import numba
import numpy as np
import sys

from numba import jit, prange, int16

@jit
def empty(height, width):
    #Матрица из нулей
    return np.zeros((height, width), dtype=numba.types.int16)


@jit(int16[:, :](int16[:, :]), fastmath=True)
def sobel_filter(image):
    height, width = image.shape
    result = empty(height, width)
    #Горизонтальная маска Собеля
    sobel_x_mask = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]), dtype=int16)
    #Вертикальная маска Собеля
    sobel_y_mask = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype=int16)  # Sobel vertical mask
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
            out_pix = math.sqrt(cx ** 2 + cy ** 2)  # Вычисление градиента
            if out_pix>255:
                out_pix=255
            result[x, y] = out_pix if out_pix > 0 else 0
    return result


@jit(int16[:, :](int16[:, :], int16[:, :], int16, int16),parallel=True, fastmath=True)
def calc_left_disparity(gray_left, gray_right, num_disparity, block_size):
    height = gray_left.shape[0]
    width = gray_left.shape[1]
    disparity_matrix = empty(height, width)
    half_block = block_size // 2

    for i in prange(half_block, height - half_block):

        for j in prange(half_block, width - half_block):
            #Вырезаем окно blocksizexblocksize
            left_block = gray_left[i - half_block:i + half_block, j - half_block:j + half_block]
            diff_sum = 3276755  # Большое значение что бы любое другое было меньше
            disp = 0
            range_for_disparity = min(j - half_block - 1, num_disparity)
            for d in prange(0, range_for_disparity):
                # Вырезаем окно на правом изображении со смещением от 0 до j-пол блока-1 или максимальное смещение
                right_block = gray_right[i - half_block:i + half_block, j - half_block - d:j + half_block - d]
                #Сумма абсолютных разностей
                sad_val = (np.sum(np.abs(right_block - left_block)))
                #Обновление минимума метрики
                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d
            # Запись диспаратности для пикселя
            disparity_matrix[i - half_block, j - half_block] = disp
    return disparity_matrix



@jit(int16[:, :](int16[:, :], int16[:, :], int16, int16),parallel=True ,fastmath=True)
def calc_right_disparity(gray_left, gray_right, num_disparity, block_size):
    # Аналогично calc_left_disparity
    height = gray_left.shape[0]
    width = gray_left.shape[1]
    disparity_matrix = empty(height, width)
    half_block = block_size // 2

    for i in prange(half_block, height - half_block):
        for j in prange(half_block, width - half_block):
            right_block = gray_right[i - half_block:i + half_block, j - half_block:j + half_block]
            diff_sum = 3276755
            disp = 0
            for d in prange(0, min(width - j - half_block, num_disparity)):
                left_block = gray_left[i - half_block:i + half_block, j - half_block + d:j + half_block + d]
                sad_val = (np.sum(np.abs(right_block - left_block)))
                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d
            disparity_matrix[i - half_block, j - half_block] = disp
    return disparity_matrix

#Кросс-проверка
@jit(int16[:, :](int16[:, :], int16[:, :]),fastmath=True)
def cross_check(disp_left, disp_right):
    height, width = disp_left.shape
    out_image = disp_left

    for h in range(1, height - 1):
        for w in range(1, width - 1):
            left = int(disp_left[h, w])
            if w - left > 0:
                right = int(disp_right[h, w - left])
                dispDiff = left - right #Разность диспаратности
                if abs(dispDiff) > 1: #Проверка различия диспаратности
                    out_image[h, w] = 0
    return out_image
@jit
def imageToNumba(image):
    return image.astype('int16')


def getDisparityMap(left_image_path, right_image_path, num_disparity, block_size, disable_cross=False):
    #Считывание изображений
    left = imageToNumba(cv2.imread(left_image_path, 0))
    right = imageToNumba(cv2.imread(right_image_path, 0))
    print("Start sobel")
    #Фильтрация Собеля
    filtered_left = sobel_filter(left)
    filtered_right = sobel_filter(right)
    cv2.imwrite('output/sobel_left.png', filtered_left)
    cv2.imwrite('output/sobel_right.png', filtered_right)
    print("Start disparity")
    #Значение диспаратности относительно левого изображения
    disparity_left = calc_left_disparity(filtered_left, filtered_right, num_disparity, block_size)
    cv2.imwrite('output/disparity_left.png', disparity_left)
    if disable_cross:
        return disparity_left #Возврат карты при отключенной проверке
    #Значение диспаратности относительно правого изображения
    disparity_right = calc_right_disparity(filtered_left, filtered_right, num_disparity, block_size)
    #Получение итоговой карты диспаратности из изображения
    disparity = cross_check(disparity_left, disparity_right)
    cv2.imwrite('output/disparity_right.png', disparity_right)
    return disparity


def getDepthMap(disparity, baseline, fx):
    depth = np.zeros(shape=disparity.shape).astype(int)
    #Переход от карты диспаратности к карте глубины
    depth[disparity > 0] = (fx * baseline) / (disparity[disparity > 0])
    return depth

def run(left_image, right_image, num_disparity, block_size, baseline, fx, disable_cross):
    #Проверка присутсвия изображений
    if not os.path.exists(left_image):
        print("Path " + left_image + "doesnt exists")
        return
    if not os.path.exists(right_image):
        print("Path " + right_image + "doesnt exists")
        return
    #Вычисление карты диспаратности
    disparity = getDisparityMap(left_image, right_image, num_disparity, block_size, disable_cross)
    #Переход к карте глубины
    depth = getDepthMap(disparity, baseline, fx)
    #Запись карты глубины и диспаратности
    cv2.imwrite("output/disparity.png", disparity)
    cv2.imwrite("output/depth.png", depth)

#Точка входа в приложение
skip = sys.argv[7] == "True"
run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),skip)
print("Results written in output directory")

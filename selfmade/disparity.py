import math
import time

import cv2
import numpy as np


def sobel_filter(image):
    height, width = image.shape
    filtered_image = np.zeros((height, width))
    sobel_matrix_x = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))  # idk
    sobel_matrix_y = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))  # idk
    for y in range(2, width - 2):
        for x in range(2, height - 2):
            cx, cy = 0, 0
            for offset_y in range(0, 3):
                for offset_x in range(0, 3):
                    pix = image[x + offset_x - 1, y + offset_y - 1]
                    if offset_x != 1:
                        cx += pix * sobel_matrix_x[offset_x, offset_y]
                    if offset_y != 1:
                        cy += pix * sobel_matrix_y[offset_x, offset_y]
            out_pix = math.sqrt(cx ** 2 + cy ** 2)
            filtered_image[x, y] = out_pix if out_pix > 0 else 0
    np.putmask(filtered_image, filtered_image > 255, 255)
    return filtered_image


def calc_left_disparity(gray_left, gray_right, num_disparity=128, block_size=11):
    height, width = gray_right.shape
    disparity_matrix = np.zeros((height, width), dtype=np.float32)
    half_block = block_size // 2

    for i in range(half_block, height - half_block):
        print("%d%% " % (i * 100 // height), end=' ', flush=True)

        for j in range(half_block, width - half_block):
            ##Вырезаем окно. 12 на 12, т.к block-size=13
            left_block = gray_left[i - half_block:i + half_block, j - half_block:j + half_block]
            diff_sum = 32767  # Большое значение что бы любое другое было меньше
            disp = 0  # Изначальный диспаритет для???
            for d in range(0, min(j - half_block - 1, num_disparity)):
                # Вырезаем окно на правом изображении со смещением от 0 до j-пол блока-1 или максимальное смещение
                right_block = gray_right[i - half_block:i + half_block, j - half_block - d:j + half_block - d]
                # right_block = right_block[i - half_block:i + half_block,j - half_block - d+1:j + half_block - d]
                # right_block = cv2.hconcat(right_block,)
                sad_val = np.sum(np.sum(np.abs(right_block - left_block)))  # Разность блоков по модулю и сумма матрицы.
                # Самая минимальная разность будет соотвествовать наиболее одинаковым блокам
                # Disparity Optimization maybe
                # Если sad меньше чем прошлая сумма, то записываем. Диспаритетом становится
                # Тот самый диспартет из цикла
                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d
            # После конца цикла записываем для соотвествующего пикселя соотвествующую диспаритетность
            disparity_matrix[i - half_block, j - half_block] = disp
            # print(disp)
    print('100%')
    return disparity_matrix


# Calculate right disparity


def calc_right_disparity(gray_left, gray_right, num_disparity=128, block_size=11):
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


def left_right_check(disp_left, disp_right):
    height, width = disp_left.shape
    out_image = disp_left

    for h in range(1, height - 1):
        for w in range(1, width - 1):
            left = int(disp_left[h, w])
            if w - left > 0:
                right = int(disp_right[h, w - left])
                dispDiff = left - right
                if dispDiff < 0:
                    dispDiff = -dispDiff
                elif dispDiff > 1:
                    out_image[h, w] = 0
    return out_image

def getDisparityMap(left_image_path,right_image_path,num_diparity,block_size,save_intermediate):
    start_time = time.time()
    left = cv2.imread(left_image_path, 0)
    right = cv2.imread(right_image_path, 0)
    filtered_left = sobel_filter(left)
    filtered_right = sobel_filter(right)


    disparity_left = calc_left_disparity(filtered_left, filtered_right, num_disparity, block_size)
    disparity_right = calc_right_disparity(
        filtered_left, filtered_right, num_disparity, block_size)
    disparity = left_right_check(disparity_left, disparity_right)
    print('Duration: %s seconds\n' % (time.time() - start_time))
    if save_intermediate==True:
        cv2.imwrite('output/sobel_left.bmp', filtered_left)
        cv2.imwrite('sobel_right.bmp', filtered_right)
        cv2.imwrite('output/disparity_left.bmp', disparity_left)
        cv2.imwrite('disparity_right.bmp', disparity_left)
        disparity_left_color = cv2.applyColorMap(cv2.convertScaleAbs(
            disparity_left, alpha=256 / num_disparity), cv2.COLORMAP_JET)
        cv2.imwrite('output/disparity_leftRGB.bmp', disparity_left_color)
        disparity_right_color = cv2.applyColorMap(cv2.convertScaleAbs(
            disparity_right, alpha=256 / num_disparity), cv2.COLORMAP_JET)
        cv2.imwrite('output/disparity_rightRGB.bmp', disparity_right_color)
    #cv2.imwrite('disparity.bmp', disparity)
    return disparity
num_disparity = 64 #48
block_size = 13






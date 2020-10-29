# coding:utf-8
import os

import cv2
from tensorflow import image


if __name__ == '__main__':
    path = os.getcwd()
    # bmp图片数据集的位置，图片需储存在当前文件夹下的data文件夹中，图片按从1到n进行编号（如1.bmp）
    data_path = path + '\\data\\'
    log_file = path + '\\log_total.txt'     # 储存计算psnr值的位置

    n = 100     # 压缩100张bmp图片
    lf = open(log_file, 'w')

    for i in range(n):
        bmp_file = data_path + str(i+1) + '.bmp'    # 第i+1张bmp图片
        bmp_img = cv2.imread(bmp_file)

        lf.write('{}.jpg\n'.format(i+1))

        for qf in range(1, 101):    # 对1-100范围内的qf计算bpp和psnr的值
            lf.write('qf: {}\n'.format(qf))
            buffer = cv2.imencode('.jpg', bmp_img, [int(cv2.IMWRITE_JPEG_QUALITY), qf])[1]
            bpp_val = (buffer.size * 8) / (1280 * 720)  # 该图片的码率bpp

            jpg_img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            psnr_val = image.psnr(bmp_img, jpg_img, max_val=255)    # 该图片的psnr

            psnr_bpp_val = psnr_val / bpp_val   # 遗留问题，本来打算作为其它用途，但是好像并没有用到，避免删改出现不必要的bug

            lf.write('psnr: {}\n'.format(psnr_val))
            lf.write('bpp: {}\n'.format(bpp_val))
            lf.write('psnr_bpp: {}\n'.format(psnr_bpp_val))

        lf.write('\n\n')

    lf.close()

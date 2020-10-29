# coding:utf-8
from bmp2jpg import *


def load_model(model_file):
    with open(model_file, 'r') as fp:
        model = fp.readlines()

    qf_list = model[1].split(', ')[:100]

    return list(map(int, qf_list))


if __name__ == '__main__':
    original_image_path = 'E:\\project\\python_project\\bmp2jpg\\data\\'
    result_image_path = 'E:\\project\\python_project\\bmp2jpg\\jpg\\'
    model_file = 'E:\\project\\python_project\\bmp2jpg\\saved_model\\extern\\best.txt'

    psnr_total = 0
    bpp_total = 0
    qf_list = load_model(model_file)

    for i in tqdm(range(1, 101)):
        bmp_file = original_image_path + str(i) + '.bmp'
        jpg_file = result_image_path + str(i) + '.jpg'
        qf = qf_list[i-1]

        bmp2jpg(bmp_file, jpg_file, quality=qf)
        psnr_total += compute_psnr(bmp_file, jpg_file)
        bpp_total += bpp(jpg_file)

    print(psnr_total.numpy() / 100)
    print(bpp_total / 100)

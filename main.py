# coding:utf-8
import os

from GA import GA

import matplotlib.pyplot as plt


if __name__ == '__main__':
    path = os.getcwd()
    original_image_path = path + '\\data'
    result_image_path = path + '\\jpg'
    load_file = path + '\\saved_individual\\10-23\\individual_040.txt'
    ga = GA(op=original_image_path, rp=result_image_path, load_file=load_file)
    fits = ga.run()

    plt.plot(list(range(len(fits))), fits)
    plt.show()

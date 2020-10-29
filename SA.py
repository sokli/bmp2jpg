# coding:utf-8
import os

import numpy as np
import cv2
from tensorflow import image


def PSNR(o_img, r_img_org):
    width, height = 1280, 720
    r_img = cv2.resize(r_img_org, (width, height), cv2.INTER_CUBIC)

    return image.psnr(o_img, r_img, max_val=255).numpy()


def BPP(size, width, height):
    return (size * 8) / (width * height)


def encode(qf, width, height):
    return (qf << 22) | (width << 11) | height


def decode(gen):
    qf = gen >> 22
    width = (gen >> 11) % (2 ** 11)
    height = gen % (2**11)

    return int(qf), int(width), int(height)


def sigmoid80(x):
    return 1 / (1 + np.exp(80 - x))


def metro(delta_fit, t):
    """
    :param delta_fit: fit_new - fit
    :param t: temperature
    :return:
    """
    return np.exp(delta_fit / (0.1 * t))


class SA:
    def __init__(self, op, rp, t_max=10000, t_min=1, r=0.97, im=10, qf_ur=1.0, model_file=None):
        self.original_path = op         # bmp image
        self.result_path = rp           # target path for jpg image
        self.t_max = t_max              # initial temperature
        self.t_min = t_min              # minimal temperature (one of the ending conditions)
        self.rate = r                   # rate of temperature drop
        self.iterate_max = im           # number of iterations when temperature is fixed
        self.qf_update_rate = qf_ur     # update rate of qf

        self.elitist = {'qf_width_height': [], 'fitness': 0, 'temperature': 0}     # global best individual

        self.individual = []   # solution
        self.init_individual(model_file)

        self.update_rate = []
        self.init_update_rate(model_file)

        # 初始化操作
        # 将所有bmp图片读到内存中
        self.img_bmps = []
        self.bmp_width_org, self.bmp_height_org = 1280, 720
        self.bmps_size = 100
        self.read_bmps()

    def init_individual(self, model_file=None):
        if model_file is None:
            for i in range(100):
                qf = np.random.randint(1, 101)
                # 因为对于大多数图片来说，不改变分辨率是一个较好的选择，所以分辨率先都初始化为1280x720附近
                width = 1280 - abs(int(np.random.randn() * 10))
                height = 720 - abs(int(np.random.randn() * 10))
                self.individual.append(encode(qf, width, height))
        else:
            with open(model_file, 'r') as mf:
                qf_list = mf.readlines()

            qf_list = qf_list[1].split(', ')[:100]
            qf_list = (list(map(int, qf_list)))
            self.individual = [encode(qf, 1280, 720) for qf in qf_list]

    def init_update_rate(self, model_file=None):
        if model_file is not None:
            for ind in self.individual:
                qf, width, height = decode(ind)
                rate = sigmoid80(qf)
                self.update_rate.append([self.qf_update_rate, rate, rate])
        else:
            self.update_rate = [[self.qf_update_rate, self.qf_update_rate, self.qf_update_rate] for i in range(100)]

    def read_bmps(self):
        for i in range(1, 101):
            file_name = self.original_path + '\\' + str(i) + '.bmp'
            img = cv2.imread(file_name)
            self.img_bmps.append(img)

    def evaluate(self, individual):
        qf_list = []
        width_list = []
        height_list = []
        for ind in individual:
            qf, width, height = decode(ind)
            qf_list.append(qf)
            width_list.append(width)
            height_list.append(height)

        psnr_total = 0
        size_total = 0
        for i in range(len(individual)):
            qf = qf_list[i]
            width = width_list[i]
            height = height_list[i]
            img_bmp_org = self.img_bmps[i]
            img_bmp = cv2.resize(img_bmp_org, (width, height), cv2.INTER_LANCZOS4)
            buffer = cv2.imencode('.jpg', img_bmp, [int(cv2.IMWRITE_JPEG_QUALITY), qf])[1]
            size_total += buffer.size

            img_jpg = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            psnr_total += PSNR(img_bmp_org, img_jpg)

        bpp_avg = (size_total * 8) / (self.bmps_size * self.bmp_height_org * self.bmp_width_org)
        psnr_avg = psnr_total / self.bmps_size
        if bpp_avg >= 1.:
            fit = 0
        else:
            fit = psnr_avg

        return fit

    def generate_new_solution(self, individual):
        new_individual = []

        p = np.random.rand()
        for i, ind in enumerate(individual):
            qf, width, height = decode(ind)
            new_qf = qf
            new_width = width
            new_height = height
            qf_ur, width_ur, height_ur = self.update_rate[i]

            if p < qf_ur:
                delta = int(np.random.randn() * 10)
                new_qf = qf + delta
                if new_qf <= 0 or new_qf > 100:
                    new_qf = qf - delta
                while new_qf <= 0 or new_qf > 100:
                    new_qf = qf + int(np.random.randn() * 10)

            if p < width_ur:
                delta = int(np.random.randn() * 100)
                new_width = width + delta
                if new_width <= 0 or new_width > 1280:
                    new_width = width - delta
                while new_width <= 0 or new_width > 1280:
                    new_width = width + int(np.random.randn() * 100)

            if p < height_ur:
                delta = int(np.random.randn() * 100)
                new_height = height + delta
                if new_height <= 0 or new_height > 720:
                    new_height = height - delta
                while new_height <= 0 or new_height > 720:
                    new_height = height + int(np.random.randn() * 100)

            new_individual.append(encode(new_qf, new_width, new_height))

        return new_individual

    def solve(self):
        individual = self.individual[:]
        fitness = self.evaluate(individual)
        t = self.t_max

        while t > self.t_min:
            print(t)
            for i in range(self.iterate_max):
                new_individual = self.generate_new_solution(individual)
                new_fitness = self.evaluate(new_individual)
                delta_fit = new_fitness - fitness

                if delta_fit > 0:
                    individual = new_individual[:]
                    fitness = new_fitness
                else:
                    p = metro(delta_fit, t)
                    if np.random.rand() < p:
                        individual = new_individual[:]
                        fitness = new_fitness

            if fitness > self.elitist['fitness']:
                self.elitist['qf_width_height'] = individual
                self.elitist['fitness'] = fitness
                self.elitist['temperature'] = t

                self.save()

            t *= self.rate

    def save(self):
        """将当前全局最好的个体写到文件中"""
        path = os.getcwd()
        file_name = path + '\\saved_model\\SA\\model.txt'
        with open(file_name, 'w') as fn:
            fn.write('Best individual:\n')
            fn.write('fitness: ' + str(self.elitist['fitness']) + '\n\n')
            for i in range(100):
                fn.write('{}.bmp\n'.format(i+1))
                qf, width, height = decode(self.elitist['qf_width_height'][i])
                fn.write('QF: {}\n'.format(qf))
                fn.write('width: {}\n'.format(width))
                fn.write('height: {}\n\n'.format(height))


if __name__ == '__main__':
    path = os.getcwd()
    original_image_path = path + '\\data'
    result_image_path = path + '\\jpg'
    load_file = path + '\\saved_model\\extern\\best.txt'
    sa = SA(op=original_image_path, rp=result_image_path, t_max=10, t_min=1e-14, im=100, qf_ur=0.8, model_file=load_file)
    sa.solve()

# coding:utf-8
import os
import copy

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


class Individual:
    def __init__(self, length=100, is_random_init=True, chromosome=None):
        self.length = length
        self.chromosome = []

        if is_random_init:
            self.random_init()
        else:
            self.chromosome = chromosome

    def random_init(self):
        for i in range(self.length):
            qf = np.random.randint(1, 101)
            # 因为对于大多数图片来说，不改变分辨率是一个较好的选择，所以分辨率先都初始化为1280x720附近
            width = 1280 - abs(int(np.random.randn() * 10))
            height = 720 - abs(int(np.random.randn() * 10))
            self.chromosome.append(encode(qf, width, height))

    def get_phe(self, idx):
        return decode(self.chromosome[idx])


class GA:
    def __init__(self, op, rp, size=50, cp=0.9, mp=0.5, rr=0.1, rsr=0.8, im=0.7, gen_max=150, load_file=None):
        self.original_path = op
        self.result_path = rp
        self.size = size    # 种群规模
        self.crossover_probability = cp
        self.mutation_probability = mp
        self.retain_rate = rr   # 设置强者的定义概率，即种群前10%为强者
        self.random_select_rate = rsr
        self.improve_rate = im  # 对个体进行适当扰动的概率
        self.generation_max = gen_max

        self.individuals = []   # 种群，个体使用二进制编码，qf | width | height
        self.fitness = []
        self.age = 0    # 当前世代数
        self.elitist = {'qf_width_height': [], 'fitness': 0, 'age': 0}     # 当前全局最优个体

        # 初始化操作
        # 将所有bmp图片读到内存中
        self.img_bmps = []
        self.read_bmps()
        self.bmp_width_org, self.bmp_height_org = 1280, 720
        self.bmps_size = 100

        # 初始化种群
        self.init_population(load_file)

    def read_bmps(self):
        for i in range(1, 101):
            file_name = self.original_path + '\\' + str(i) + '.bmp'
            img = cv2.imread(file_name)
            self.img_bmps.append(img)

    def init_population(self, load_file):
        if load_file is None:
            # 产生随机个体
            for i in range(self.size):
                self.individuals.append(Individual())
        else:
            self.load_individuals(load_file)

    def load_individuals(self, file_name):
        with open(file_name, 'r') as fn:
            individuals_list = fn.readlines()

        length = len(individuals_list)
        for i in range(0, length, 303):
            chromosome = []
            for j in range(2, 300, 3):
                qf = int(individuals_list[i+j][4:-1])
                width = int(individuals_list[i+j+1][7:-1])
                height = int(individuals_list[i+j+2][8:-1])
                chromosome.append(encode(qf, width, height))
            self.individuals.append(Individual(len(chromosome), False, chromosome))

    def fitness_func(self, individual):
        qf_list = []
        width_list = []
        height_list = []
        for i in range(individual.length):
            qf, width, height = individual.get_phe(i)
            qf_list.append(qf)
            width_list.append(width)
            height_list.append(height)

        psnr_total = 0
        size_total = 0
        for i in range(individual.length):
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

    def evaluate(self):
        for idx, individual in enumerate(self.individuals):
            fit = self.fitness_func(individual)
            self.fitness.append((fit, idx))

        self.fitness.sort(reverse=True)
        sorted_individuals = [self.individuals[idx] for _, idx in self.fitness]
        self.individuals = sorted_individuals[:]

    def select(self):
        retain_length = int(self.size * self.retain_rate)
        parents = self.individuals[:retain_length]
        rsr = self.random_select_rate
        for individual in self.individuals[retain_length:]:
            if np.random.rand() < rsr:
                parents.append(individual)
            rsr *= 0.95  # 适应度越高的个体被选中的概率越大

        return parents

    def cross(self, individual1, individual2):
        child1 = copy.deepcopy(individual1)
        child2 = copy.deepcopy(individual2)
        if np.random.rand() < self.crossover_probability:
            poi = np.random.randint(0, 100)

            child1.chromosome = individual1.chromosome[:poi] + individual2.chromosome[poi:]
            child2.chromosome = individual2.chromosome[poi:] + individual1.chromosome[:poi]

        return child1, child2

    '''
    def cross(self, individual1, individual2):
        if np.random.rand() < self.crossover_probability:
            qf1, width1, height1 = decode(individual1)
            qf2, width2, height2 = decode(individual2)

            poi = np.random.randint(0, 7)
            left1 = (qf1 >> poi) << poi
            left2 = (qf2 >> poi) << poi
            right1 = left1 ^ qf1
            right2 = left2 ^ qf2
            qf1 = left2 | right1
            qf2 = left1 | right2

            poi = np.random.randint(0, 11)
            left1 = (width1 >> poi) << poi
            left2 = (width2 >> poi) << poi
            right1 = left1 ^ width1
            right2 = left2 ^ width2
            width1 = left2 | right1
            width2 = left1 | right2

            poi = np.random.randint(0, 11)
            left1 = (height1 >> poi) << poi
            left2 = (height2 >> poi) << poi
            right1 = left1 ^ height1
            right2 = left2 ^ height2
            height1 = left2 | right1
            height2 = left1 | right2

            individual1 = encode(qf1, width1, height1)
            individual2 = encode(qf2, width2, height2)

        return individual1, individual2
    '''

    def mutate(self, individual):
        child = copy.deepcopy(individual)
        if np.random.rand() < self.mutation_probability:
            num = np.random.randint(1, 10)
            pois = np.random.randint(0, 100, num)
            for poi in pois:
                qf_org, width_org, height_org = child.get_phe(poi)

                qf = qf_org + int(np.random.randn() * 5)
                while qf <= 0 or qf > 100:
                    qf = qf_org + int(np.random.randn() * 5)

                width = width_org + int(np.random.randn() * 10)
                while width <= 0 or width > 1280:
                    width = width_org + int(np.random.randn() * 10)

                height = height_org + int(np.random.randn() * 10)
                while height <= 0 or height > 720:
                    height = height_org + int(np.random.randn() * 10)
                '''
                poi1 = np.random.randint(0, 7)
                poi2 = np.random.randint(0, 11, size=2)

                mask = 1 << poi1
                qf ^= mask

                mask = 1 << poi2[0]
                width ^= mask

                mask = 1 << poi2[1]
                height ^= mask
                '''
                child.chromosome[poi] = encode(qf, width, height)

        return child

    '''
    def improve(self, individual):
        if np.random.rand() < self.improve_rate:
            qf, width, height = decode(individual)
            qf = qf + int(np.random.randn() * 3)
            width = width + int(np.random.randn() * 10)
            height = height + int(np.random.randn() * 10)
            individual = encode(qf, width, height)

        return individual
    '''

    def multiply(self, parents):
        parents_size = len(parents)
        target_count = self.size - parents_size
        children = []

        i = 0
        while i < target_count:
            ind_idx1, ind_idx2 = np.random.randint(0, parents_size, size=2)
            father = parents[ind_idx1]
            mother = parents[ind_idx2]
            child1, child2 = self.cross(father, mother)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            children.append(child1)
            children.append(child2)
            i += 2

        while len(children) < target_count:
            ind_idx = np.random.randint(0, parents_size)
            child = parents[ind_idx]
            child = self.mutate(child)
            # child = self.improve(child)
            children.append(child)

        return children

    def save(self):
        """将当前全局最好的个体写到文件中"""
        path = os.getcwd()
        file_name = path + '\\saved_model\\model_{:03d}.txt'.format(self.age)
        with open(file_name, 'w') as fn:
            fn.write('Best individual:\n')
            fn.write('age: ' + str(self.elitist['age']) + '\n')
            fn.write('fitness: ' + str(self.elitist['fitness']) + '\n\n')
            for i in range(100):
                fn.write('{}.bmp\n'.format(i+1))
                qf, width, height = self.elitist['qf_width_height'][i]
                fn.write('QF: {}\n'.format(qf))
                fn.write('width: {}\n'.format(width))
                fn.write('height: {}\n\n'.format(height))

    def save_individual(self):
        path = os.getcwd()
        file_name = path + '\\saved_individual\\individual_{:03d}.txt'.format(self.age)
        with open(file_name, 'w') as fn:
            for i, individual in enumerate(self.individuals):
                fn.write('Individual {}\n'.format(i))
                fn.write('fitness: {}\n'.format(self.fitness[i][0]))
                for j in range(100):
                    qf, width, height = individual.get_phe(j)
                    fn.write('QF: {}\n'.format(qf))
                    fn.write('width: {}\n'.format(width))
                    fn.write('height: {}\n'.format(height))
                fn.write('\n')

    def evolve(self):
        self.evaluate()
        best_fit = self.fitness[0][0]
        best_individual = self.individuals[0]
        if best_fit > self.elitist['fitness']:
            qf_width_height_list = []
            for i in range(best_individual.length):
                qf, width, height = best_individual.get_phe(i)
                qf_width_height_list.append((qf, width, height))
            self.elitist['qf_width_height'] = qf_width_height_list
            self.elitist['fitness'] = best_fit
            self.elitist['age'] = self.age

        if self.age % 10 == 0:
            self.save_individual()

        parents = self.select()
        children = self.multiply(parents)

        self.individuals = parents[:] + children[:]
        self.fitness = []

        # for individual in parents + children:
        #     self.individuals.append(self.improve(individual))

        return best_fit

    def run(self):
        fits = []
        for i in range(self.generation_max):
            print('age: {}'.format(i))
            self.age = i
            # self.mutation_probability *= np.power(0.98, i)
            best_fit = self.evolve()
            self.save()
            fits.append(best_fit)

        # qf, width, height = self.elitist['QF'], self.elitist['width'], self.elitist['height']
        # for img_bmp_org in self.img_bmps:
        #     img_bmp = img_bmp_org.resize((width, height), Image.BICUBIC)
        #     img_bmp.save(self.result_path, 'jpeg', quality=qf)

        return fits

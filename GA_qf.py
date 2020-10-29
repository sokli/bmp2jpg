import random

import matplotlib.pyplot as plt


class Single:
    """某一张图片压缩时所用的参数，计算所得的psnr和bpp值以及psnr_bpp的值"""
    def __init__(self, qf, psnr, bpp):
        self.qf = qf
        self.psnr = psnr
        self.bpp = bpp

        self.psnr_bpp = psnr / bpp


class Mul:
    """某一张图片所有压缩压缩参数的情况下，psnr和bpp的取值"""
    def __init__(self, single_list):
        self.single_list = single_list[:]

        self.psnr_bpp_avg = 0
        for s in single_list:
            self.psnr_bpp_avg += s.psnr_bpp
        self.psnr_bpp_avg = self.psnr_bpp_avg / len(single_list)

    def get_single(self, idx):
        return self.single_list[idx]

    def __lt__(self, other):
        return self.psnr_bpp_avg < other.psnr_bpp_avg


def load_log_total(file_name):
    """加载所有图片的所有参数下的psnr和bpp的值"""
    with open(file_name, 'r') as fp:
        model = fp.readlines()

    mul_list = []

    length = len(model)
    for i in range(0, length, 403):
        single_list = []
        for j in range(1, 400, 4):
            single_list.append(Single(int(model[i+j][4:]), float(model[i+j+1][6:]), float(model[i+j+2][5:])))
        mul_list.append(Mul(single_list))

    return mul_list


class GA:
    def __init__(self, file_name, record_file, n=100, num=100, size=200, cp=0.9, mp=0.1, rr=0.05, rsr=0.5,
                 gen_max=20000, load_file=None):
        self.n = n          # 需要压缩的图片个数，默认为100
        self.num = num      # 每张图片所取的参数qf的个数，默认为100，表示每张图片的压缩参数在[1,100]都有取值
        self.size = size    # 种群规模
        self.crossover_probability = cp     # 交叉概率
        self.mutation_probability = mp      # 变异概率
        self.retain_rate = rr   # 设置精英的定义概率
        self.random_select_rate = rsr       # 随机选择的概率
        self.generation_max = gen_max       # 最大迭代次数
        self.record_file = record_file      # 将迭代过程中的最优个体（参数）记录并保存下来

        self.model = load_log_total(file_name)  # 将所有图片所有参数情况下的psnr和bpp取值加载进来
        self.individuals = []   # 种群
        self.fitness = []       # 适应度
        self.age = 0            # 当前世代数
        self.elitist = {'fitness': 0, 'individual': [0 for i in range(n)]}     # 当前全局最优个体

        if load_file is None:
            # 随机初始化种群
            for i in range(size):
                individual = [random.randint(0, num-1) for j in range(n)]
                self.individuals.append(individual)
        else:
            # 加载文件中的模型（将之前运行过程中保存的个体加载进来并继续迭代）
            with open(load_file, 'r') as fp:
                model = fp.readlines()
            qf_list = model[1].split(', ')[:n]
            intdividual = (list(map(int, qf_list)))
            self.individuals.append([i - 1 for i in intdividual])

            for i in range(size-1):
                individual = [random.randint(0, num-1) for j in range(n)]
                self.individuals.append(individual)

    def fitness_func(self, individual):
        """计算个体的适应度"""
        bpp_total = 0
        psnr_total = 0

        for i, j in enumerate(individual):
            bpp_total += self.model[i].get_single(j).bpp
            psnr_total += self.model[i].get_single(j).psnr

        if bpp_total > 100.0:
            fit = 0
        else:
            fit = psnr_total

        return fit

    def evaluate(self):
        """评估种群所有个体的适应度，并按适应度降序对种群进行排序"""
        for idx, individual in enumerate(self.individuals):
            fit = self.fitness_func(individual)
            self.fitness.append((fit, idx))

        self.fitness.sort(reverse=True)
        sorted_individuals = [self.individuals[idx] for _, idx in self.fitness]
        self.individuals = sorted_individuals[:]

    def select(self):
        """从种群中选择合适的个体保留下来"""
        # 保留精英
        retain_length = int(self.size * self.retain_rate)
        parents = self.individuals[:retain_length]
        # 从非精英中随机选择个体保存
        for individual in self.individuals[retain_length:]:
            if random.random() < self.random_select_rate:
                parents.append(individual)

        return parents

    def cross(self, individual1, individual2):
        """交叉算子，单点交叉"""
        if random.random() < self.crossover_probability:
            poi = random.randint(0, self.n-1)

            child1 = individual1[:poi] + individual2[poi:]
            child2 = individual2[:poi] + individual1[poi:]

            return child1, child2
        else:
            return individual1[:], individual2[:]

    def mutate(self, individual):
        """变异算子，单点变异"""
        if random.random() < self.mutation_probability:
            poi = random.randint(0, self.n-1)
            return individual[:poi] + [random.randint(0, self.num-1)] + individual[poi+1:]
        else:
            return individual[:]

    def multiply(self, parents):
        """种群繁衍，从旧种群中选择出来的父本进行交叉、变异产生子个体种群（与父种群共同构成新种群）"""
        parents_size = len(parents)
        target_count = self.size - parents_size
        children = []

        i = 0
        while i < target_count:
            ind_idx1, ind_idx2 = random.sample(range(0, parents_size), 2)
            father = parents[ind_idx1]
            mother = parents[ind_idx2]
            child1, child2 = self.cross(father, mother)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            children.append(child1)
            children.append(child2)
            i += 2

        while len(children) < target_count:
            ind_idx1, ind_idx2 = random.sample(range(0, parents_size), 2)
            father = parents[ind_idx1]
            mother = parents[ind_idx2]
            child, _ = self.cross(father, mother)
            child = self.mutate(child)
            children.append(child)

        return children

    def evolve(self):
        """种群一轮迭代进化"""
        # 对种群各个个体评估适应度
        self.evaluate()
        # 查看是否产生了更优的全局个体，若是则保存
        best_fit = self.fitness[0][0]
        best_individual = self.individuals[0]
        if best_fit > self.elitist['fitness']:
            self.elitist['fitness'] = best_fit
            self.elitist['individual'] = best_individual

        # 从旧种群中选择出合适的父本
        parents = self.select()
        # 父本繁衍出子种群
        children = self.multiply(parents)

        # 父种群和子种群共同构成新种群
        self.individuals = parents[:] + children[:]
        self.fitness = []

        return best_fit

    def run(self):
        fits = []
        for i in range(self.generation_max):
            # 一轮迭代，一次世代
            print('age: {}'.format(i))
            self.age = i
            best_fit = self.evolve()
            fits.append(best_fit)

        # 将迭代过程中的全局最优个体保存下来
        with open(self.record_file, 'w') as fp:
            fp.write(str(self.elitist['fitness']))
            fp.write('\n')
            for i, j in enumerate(self.elitist['individual']):
                fp.write(str(self.model[i].get_single(j).qf) + ', ')

        return fits


if __name__ == '__main__':
    log_path = 'E:\\project\\python_project\\bmp2jpg\\log_total.txt'    # 保存所有图片所有参数情况下psnr和bpp取值的文件
    record_path = 'E:\\project\\python_project\\bmp2jpg\\record.txt'    # 记录算法找到的全局最优个体（最高评价）
    model_file = 'E:\\project\\python_project\\bmp2jpg\\saved_model\\extern\\best.txt'  # 加载过去迭代的种群继续迭代
    ga = GA(log_path, record_path, 100, size=200, mp=0.1, gen_max=10000)
    fits = ga.run()     # 运行算法

    # 绘制出迭代过程中种群最优适应度的变化
    plt.plot(list(range(len(fits))), fits)
    plt.show()

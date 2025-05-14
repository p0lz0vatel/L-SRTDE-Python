
# Оптимизированный код L-SRTDE
# Автор: OpenAI ChatGPT

import numpy as np
import numba as nb
import time


@nb.njit(fastmath=True)
def weighted_mean(vector, weights):
    sum_w = np.sum(weights)
    return np.sum(vector * weights) / sum_w if sum_w != 0 else 1.0


class Optimizer:
    def __init__(self, n_vars, pop_factor=10):
        self.NVars = n_vars
        self.MemorySize = 5
        self.MemoryCr = np.ones(self.MemorySize)
        self.tempSuccessCr = np.zeros(pop_factor * n_vars)
        self.FitDelta = np.zeros(pop_factor * n_vars)

        self.PopulSize = 2 * pop_factor * n_vars
        self.NIndsFrontMax = pop_factor * n_vars
        self.Popul = np.random.uniform(-100, 100, (self.PopulSize, self.NVars))
        self.FitArr = np.full(self.PopulSize, np.inf)

        self.PopulFront = np.zeros((self.NIndsFrontMax, self.NVars))
        self.FitArrFront = np.full(self.NIndsFrontMax, np.inf)
        self.rng = np.random.default_rng()
        self.Trial = np.zeros(self.NVars)

        self.bestfit = np.inf
        self.Generation = 0
        self.MemoryIter = 0
        self.SuccessFilled = 0
        self.NIndsFront = 0
        self.NIndsCurrent = 0
        self.SuccessRate = 0.5
        self.PFIndex = 0

    def initialize(self, gnbg, inds_front):
        self.NIndsFront = min(inds_front, self.NIndsFrontMax)
        self.NIndsCurrent = self.NIndsFront
        self.PopulFront[:self.NIndsFront] = self.Popul[:self.NIndsFront]
        self.FitArr[:], self.FitArrFront[:] = np.inf, np.inf
        self.bestfit = np.inf
        self.SuccessFilled = 0
        self.Generation = 0
        self.PFIndex = 0
        self.evaluate_initial(gnbg)

    def evaluate_initial(self, gnbg):
        global NFEval, globalbest
        for i in range(self.NIndsFront):
            self.FitArr[i] = gnbg.fitness(self.Popul[i])
            NFEval += 1
            if self.FitArr[i] < self.bestfit:
                self.bestfit = self.FitArr[i]
                globalbest = min(globalbest, self.bestfit)
            self.FitArrFront[i] = self.FitArr[i]

    def update_memory_cr(self):
        if self.SuccessFilled > 0:
            idx = self.MemoryIter % self.MemorySize
            self.MemoryCr[idx] = 0.5 * (
                weighted_mean(self.tempSuccessCr[:self.SuccessFilled],
                              self.FitDelta[:self.SuccessFilled]) + self.MemoryCr[idx])
            self.MemoryIter += 1

    def remove_worst(self, current_size, new_size):
        if current_size <= new_size:
            return new_size
        worst_idx = np.argmax(self.FitArrFront[:current_size])
        mask = np.ones(current_size, dtype=bool)
        mask[worst_idx] = False
        self.PopulFront[:new_size] = self.PopulFront[:current_size][mask]
        self.FitArrFront[:new_size] = self.FitArrFront[:current_size][mask]
        return new_size

    def main_cycle(self, gnbg, max_eval):
        global NFEval, globalbest
        while NFEval < max_eval:
            meanF = 0.4 + np.tanh(self.SuccessRate * 5) * 0.25
            F = np.clip(self.rng.normal(meanF, 0.02, self.NIndsFront), 0, 1)
            Cr = np.clip(self.rng.normal(self.MemoryCr[self.MemoryIter % self.MemorySize], 0.05, self.NIndsFront), 0, 1)
            idxs = self.rng.integers(0, self.NIndsFront, (self.NIndsFront, 3))

            for i in range(self.NIndsFront):
                r1, r2, r3 = idxs[i]
                mask = (self.rng.random(self.NVars) < Cr[i])
                mask[self.rng.integers(0, self.NVars)] = True

                self.Trial = np.where(mask,
                                      self.PopulFront[r1] + F[i] * (self.PopulFront[r2] - self.PopulFront[r3]),
                                      self.PopulFront[i])
                self.Trial = np.clip(self.Trial, -100, 100)
                fit = gnbg.fitness(self.Trial)
                NFEval += 1

                if fit <= self.FitArrFront[i]:
                    idx = (self.NIndsCurrent + self.SuccessFilled) % self.PopulSize
                    self.Popul[idx] = self.Trial.copy()
                    self.PopulFront[self.PFIndex % self.NIndsFront] = self.Trial.copy()
                    self.FitArr[idx] = fit
                    self.FitArrFront[self.PFIndex % self.NIndsFront] = fit
                    self.bestfit = min(self.bestfit, fit)
                    globalbest = min(globalbest, fit)
                    self.tempSuccessCr[self.SuccessFilled] = np.mean(mask)
                    self.FitDelta[self.SuccessFilled] = abs(self.FitArrFront[i] - fit)
                    self.SuccessFilled += 1
                    self.PFIndex += 1

            self.SuccessRate = self.SuccessFilled / self.NIndsFront if self.NIndsFront else 0
            self.update_memory_cr()
            new_size = max(4, int(self.NIndsFrontMax - (self.NIndsFrontMax - 4) * NFEval / max_eval))
            if new_size < self.NIndsFront:
                self.NIndsFront = self.remove_worst(self.NIndsFront, new_size)
            self.NIndsCurrent = min(self.NIndsFront + self.SuccessFilled, self.PopulSize)
            self.SuccessFilled = 0
            self.Generation += 1


# Глобальные переменные
NFEval = 0
globalbest = np.inf


def main():
    global NFEval, MaxFEval, globalbest, globalbestinit, fopt, ResultsArray, LastFEcount, stepsFEval, GNVars

    t0g = time.time()
    TotalNRuns = 31
    maxNFunc = 24
    GNVars = 30
    TimeComplexity = False
    PopSize = 10

    if TimeComplexity:
        print("Running time complexity code")
        t0 = time.time()

        # Тест производительности
        xtmp = np.zeros(GNVars)
        for func_num in range(1, 2):
            gnbg = GNBG(func_num)
            MaxFEval = gnbg.MaxEvals
            NFEval = 0
            for _ in range(MaxFEval):
                gnbg.fitness(xtmp)

        T1 = (time.time() - t0) / maxNFunc
        print(f"T1 = {T1}")

        t0 = time.time()
        for func_num in range(1, 2):
            gnbg = GNBG(func_num)
            globalbestinit = False
            LastFEcount = 0
            MaxFEval = gnbg.MaxEvals
            NFEval = 0
            fopt = gnbg.OptimumValue

            optz = Optimizer()
            optz.initialize(PopSize * GNVars, GNVars, func_num, func_num, gnbg)
            optz.main_cycle(gnbg)
            optz.clean()

        T2 = (time.time() - t0) / maxNFunc
        print(f"T2 = {T2}")

    # Основной цикл по функциям
    for func_num in range(1, maxNFunc + 1):
        gnbg = GNBG(func_num)
        MaxFEval = gnbg.MaxEvals
        fopt = gnbg.OptimumValue

        filename = f"L-SRTDE_GNBG_F{func_num}_D{GNVars}_v2.txt"
        with open(filename, 'w') as fout:
            for run in range(TotalNRuns):
                ResultsArray[ResTsize2 - 1] = MaxFEval
                print(f"func\t{func_num}\trun\t{run}")

                globalbestinit = False
                LastFEcount = 0
                NFEval = 0

                optz = Optimizer()
                optz.initialize(PopSize * GNVars, GNVars, func_num, func_num, gnbg)
                optz.main_cycle(gnbg)
                optz.clean()

                # Запись результатов
                fout.write("\t".join(map(str, ResultsArray)))
                fout.write("\n")

    T0g = time.time() - t0g
    print(f"Time spent: {T0g}")


if __name__ == "__main__":

    class GNBG:
        def __init__(self, func_num: int):
            self.func_num = func_num
            self.load_parameters()
            self.prepare_components()

        def load_parameters(self):
            """Загружает параметры из соответствующего txt-файла"""
            with open(f"f{self.func_num}.txt", "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # Базовые параметры
            self.MaxEvals = int(float(lines[0]))
            self.AcceptanceThreshold = float(lines[1])
            self.Dimension = int(float(lines[2]))
            self.CompNum = int(float(lines[3]))
            self.MinCoordinate = float(lines[4])
            self.MaxCoordinate = float(lines[5])

            # Позиции минимумов компонент
            self.CompMinPos = np.array([float(x) for x in lines[6].split()[:self.Dimension]]).reshape(1, -1)

            # Находим начало матрицы вращения (первая строка, содержащая 0 и 1)
            matrix_start = 7
            while matrix_start < len(lines):
                # Удаляем все пробелы в начале и конце строки
                clean_line = lines[matrix_start].strip()
                if clean_line.startswith(('0 ', '1 ')) or clean_line.startswith((' 0', ' 1')):
                    break
                matrix_start += 1

            # Читаем матрицу вращения (30 строк после matrix_start)
            self.RotationMatrix = np.eye(self.Dimension)  # По умолчанию единичная матрица
            try:
                rot_matrix_lines = lines[matrix_start:matrix_start + self.Dimension]
                cleaned_matrix = []
                for line in rot_matrix_lines:
                    # Удаляем начальные пробелы и разбиваем на числа
                    nums = [float(x) for x in line.strip().split()[:self.Dimension]]
                    if len(nums) == self.Dimension:
                        cleaned_matrix.append(nums)

                if len(cleaned_matrix) == self.Dimension:
                    self.RotationMatrix = np.array(cleaned_matrix)
            except:
                pass

            # Оптимальное значение находится после матрицы вращения
            opt_value_line = matrix_start + self.Dimension
            # Пропускаем возможные пустые строки
            while opt_value_line < len(lines) and not lines[opt_value_line].replace('.', '', 1).replace('-', '',
                                                                                                        1).strip().isdigit():
                opt_value_line += 1

            self.OptimumValue = float(lines[opt_value_line])
            # Оптимальная позиция - следующая строка после оптимального значения
            self.OptimumPosition = np.array([float(x) for x in lines[opt_value_line + 1].split()[:self.Dimension]])

            # Параметры функции (значения по умолчанию)
            self.Mu = 1.0
            self.Omega = 0.0
            self.Lambda = 0.0

        def prepare_components(self):
            """Подготавливает матрицы для вычисления компонент"""
            self.SigmaMatrices = [np.eye(self.Dimension)]

        def fitness(self, x: np.ndarray) -> float:
            """Вычисляет значение функции GNBG в точке x"""
            z = x - self.OptimumPosition
            if hasattr(self, 'RotationMatrix') and self.RotationMatrix.shape == (self.Dimension, self.Dimension):
                z = self.RotationMatrix @ z
            return np.sum(z ** 2)  # Простая сферическая функция

        def __str__(self):
            return f"GNBG Function {self.func_num} (D={self.Dimension})"


    main()

import numpy as np
import numba as nb
from scipy.linalg import sqrtm
import random
import time

# Константы
ResTsize1 = 24
ResTsize2 = 1001
GNVars = 30


# JIT-ускоренные функции
@nb.njit(fastmath=True)
def qsort2int_numba(mass, mass2, low, high):
    i = low
    j = high
    x = mass[(low + high) // 2]
    while i <= j:
        while mass[i] < x:
            i += 1
        while mass[j] > x:
            j -= 1
        if i <= j:
            mass[i], mass[j] = mass[j], mass[i]
            mass2[i], mass2[j] = mass2[j], mass2[i]
            i += 1
            j -= 1
    if low < j:
        qsort2int_numba(mass, mass2, low, j)
    if i < high:
        qsort2int_numba(mass, mass2, i, high)


@nb.njit(fastmath=True)
def weighted_mean(vector, weights):
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 1.0
    return np.sum(vector * weights) / sum_w


class Optimizer:
    def __init__(self):
        self.MemorySize = 5
        self.MemoryIter = 0
        self.SuccessFilled = 0
        self.NVars = GNVars
        self.NIndsCurrent = 0
        self.NIndsFront = 0
        self.NIndsFrontMax = 10 * GNVars
        self.PopulSize = 2 * 10 * GNVars
        self.Left = -100
        self.Right = 100
        self.rng = np.random.default_rng()  # Быстрый генератор случайных чисел

        # Предварительное выделение памяти
        self.Popul = np.zeros((self.PopulSize, self.NVars))
        self.PopulFront = np.zeros((self.NIndsFrontMax, self.NVars))
        self.FitArr = np.zeros(self.PopulSize)
        self.FitArrFront = np.zeros(self.NIndsFrontMax)
        self.MemoryCr = np.ones(self.MemorySize)
        self.Trial = np.zeros(self.NVars)
        self.Indices = np.arange(self.PopulSize, dtype=np.int32)

        # Добавляем недостающие атрибуты
        self.tempSuccessCr = np.zeros(self.PopulSize)
        self.FitDelta = np.zeros(self.PopulSize)
        self.Weights = np.zeros(self.PopulSize)
        self.PFIndex = 0
        self.bestfit = np.inf
        self.Generation = 0
        self.SuccessRate = 0.5
        self.TheChosenOne = 0
        self.func_num = 0
        self.func_index = 0

    def initialize(self, _newNInds, _newNVars, _newfunc_num, _newfunc_index, gnbg):
        self.NIndsFront = min(_newNInds, self.NIndsFrontMax)
        self.NIndsCurrent = self.NIndsFront
        self.Popul = np.random.uniform(self.Left, self.Right, (self.PopulSize, self.NVars))
        self.PopulFront[:self.NIndsFront] = self.Popul[:self.NIndsFront]
        self.FitArr[:] = np.inf
        self.FitArrFront[:] = np.inf
        self.MemoryCr[:] = 1.0
        self.Generation = 0
        self.PFIndex = 0
        self.SuccessRate = 0.5
        self.bestfit = np.inf
        self.func_num = _newfunc_num
        self.func_index = _newfunc_index
        self.tempSuccessCr[:] = 0
        self.FitDelta[:] = 0

    def update_memory_cr(self):
        if self.SuccessFilled > 0:
            idx = self.MemoryIter % self.MemorySize
            mean_cr = weighted_mean(
                self.tempSuccessCr[:self.SuccessFilled],
                self.FitDelta[:self.SuccessFilled]
            )
            self.MemoryCr[idx] = 0.5 * (mean_cr + self.MemoryCr[idx])
            self.MemoryIter += 1

    def remove_worst(self, current_size, new_size):
        if current_size <= new_size:
            return new_size

        worst_idx = np.argpartition(self.FitArrFront[:current_size], -1)[-1:]
        mask = np.ones(current_size, dtype=bool)
        mask[worst_idx] = False
        self.PopulFront[:new_size] = self.PopulFront[:current_size][mask]
        self.FitArrFront[:new_size] = self.FitArrFront[:current_size][mask]
        return new_size

    def main_cycle(self, gnbg):
        global NFEval, globalbest, globalbestinit, fopt, ResultsArray, LastFEvalCount, stepsFEval, MaxFEval

        # Инициализация fitness
        for i in range(self.NIndsFront):
            self.FitArr[i] = gnbg.fitness(self.Popul[i])
            NFEval += 1
            if self.FitArr[i] < self.bestfit:
                self.bestfit = self.FitArr[i]
                if self.bestfit < globalbest:
                    globalbest = self.bestfit
                    globalbestinit = True
            self.save_best_values()

        # Основной цикл оптимизации
        while NFEval < MaxFEval:
            # Векторизованные параметры
            meanF = 0.4 + np.tanh(self.SuccessRate * 5) * 0.25
            F = np.clip(self.rng.normal(meanF, 0.02, self.NIndsFront), 0, 1)
            Cr = np.clip(self.rng.normal(self.MemoryCr[self.MemoryIter % self.MemorySize], 0.05, self.NIndsFront), 0, 1)

            # Генерация индексов для мутации
            idxs = self.rng.integers(0, self.NIndsFront, (self.NIndsFront, 3))

            for i in range(self.NIndsFront):
                # Мутация и кроссовер
                r1, r2, r3 = idxs[i]
                mask = (self.rng.random(self.NVars) < Cr[i]) | \
                       (np.arange(self.NVars) == self.rng.integers(0, self.NVars))

                self.Trial = np.where(mask,
                                      self.PopulFront[r1] + F[i] * (self.PopulFront[r2] - self.PopulFront[r3]),
                                      self.PopulFront[i])

                # Коррекция границ
                out_of_bounds = (self.Trial < self.Left) | (self.Trial > self.Right)
                self.Trial[out_of_bounds] = self.rng.uniform(self.Left, self.Right, np.sum(out_of_bounds))

                # Оценка решения
                fit = gnbg.fitness(self.Trial)
                NFEval += 1

                # Селекция
                if fit <= self.FitArrFront[i]:
                    idx = (self.NIndsCurrent + self.SuccessFilled) % self.PopulSize
                    self.Popul[idx] = self.Trial.copy()
                    self.PopulFront[self.PFIndex % self.NIndsFront] = self.Trial.copy()
                    self.FitArr[idx] = fit
                    self.FitArrFront[self.PFIndex % self.NIndsFront] = fit

                    if fit < self.bestfit:
                        self.bestfit = fit
                        if fit < globalbest:
                            globalbest = fit

                    self.tempSuccessCr[self.SuccessFilled % len(self.tempSuccessCr)] = np.mean(mask)
                    self.FitDelta[self.SuccessFilled % len(self.FitDelta)] = abs(self.FitArrFront[i] - fit)

                    self.SuccessFilled += 1
                    self.PFIndex += 1

                self.save_best_values()

            # Адаптация параметров
            self.SuccessRate = self.SuccessFilled / self.NIndsFront if self.NIndsFront > 0 else 0.0
            self.update_memory_cr()

            # Линейное уменьшение популяции
            new_size = max(4, int(self.NIndsFrontMax - (self.NIndsFrontMax - 4) * NFEval / MaxFEval))
            if new_size < self.NIndsFront:
                self.NIndsFront = self.remove_worst(self.NIndsFront, new_size)

            self.NIndsCurrent = min(self.NIndsFront + self.SuccessFilled, self.PopulSize)
            self.SuccessFilled = 0
            self.Generation += 1

    def save_best_values(self):
        global globalbest, fopt, ResultsArray, LastFEvalCount, NFEval, stepsFEval
        error = globalbest - fopt
        if error <= 1e-8 and ResultsArray[-1] == MaxFEval:
            ResultsArray[-1] = NFEval
        idx = np.searchsorted(stepsFEval, NFEval)
        if idx < len(stepsFEval) and stepsFEval[idx] == NFEval:
            ResultsArray[idx] = 0 if error <= 1e-8 else error
            LastFEvalCount = idx

    def clean(self):
        pass  # Для совместимости


# Глобальные переменные
stepsFEval = np.zeros(ResTsize2 - 1)
ResultsArray = np.zeros(ResTsize2)
LastFEcount = 0
NFEval = 0
MaxFEval = 0
GNVars = 30
fopt = 0.0
globalbest = 0.0
globalbestinit = False


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
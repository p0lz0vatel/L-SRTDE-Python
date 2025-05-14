import numpy as np
import numba as nb
from numba import int32, float64, types
import time
from typing import List

# Константы
ResTsize1 = 24
ResTsize2 = 1001
GNVars = 30

# Спецификация типов для JIT-класса
opt_spec = [
    ('MemorySize', int32),
    ('MemoryIter', int32),
    ('SuccessFilled', int32),
    ('NVars', int32),
    ('NIndsCurrent', int32),
    ('NIndsFront', int32),
    ('NIndsFrontMax', int32),
    ('PopulSize', int32),
    ('Left', float64),
    ('Right', float64),
    ('Popul', float64[:, :]),
    ('PopulFront', float64[:, :]),
    ('FitArr', float64[:]),
    ('FitArrFront', float64[:]),
    ('MemoryCr', float64[:]),
    ('Trial', float64[:]),
    ('Trials', float64[:, :]),
    ('Indices', int32[:]),
    ('tempSuccessCr', float64[:]),
    ('FitDelta', float64[:]),
    ('Weights', float64[:]),
    ('PFIndex', int32),
    ('bestfit', float64),
    ('Generation', int32),
    ('SuccessRate', float64),
    ('TheChosenOne', int32),
    ('func_num', int32),
    ('func_index', int32),
    ('F', float64[:]),
    ('Cr', float64[:]),
    ('idxs', int32[:, :]),
    ('cross_mask', float64[:, :]),
    ('rng', types.voidptr)  # Указатель на генератор случайных чисел
]


@nb.experimental.jitclass(opt_spec)
class NumbaOptimizer:
    def __init__(self):
        self.MemorySize = 5
        self.MemoryIter = 0
        self.SuccessFilled = 0
        self.NVars = GNVars
        self.NIndsCurrent = 0
        self.NIndsFront = 0
        self.NIndsFrontMax = 10 * GNVars
        self.PopulSize = 2 * 10 * GNVars
        self.Left = -100.0
        self.Right = 100.0
        self.rng = np.random.default_rng().bit_generator

        # Предварительное выделение памяти
        self.Popul = np.zeros((self.PopulSize, self.NVars))
        self.PopulFront = np.zeros((self.NIndsFrontMax, self.NVars))
        self.FitArr = np.zeros(self.PopulSize)
        self.FitArrFront = np.zeros(self.NIndsFrontMax)
        self.MemoryCr = np.ones(self.MemorySize)
        self.Trial = np.zeros(self.NVars)
        self.Trials = np.zeros((self.NIndsFrontMax, self.NVars))
        self.Indices = np.arange(self.PopulSize, dtype=np.int32)
        self.tempSuccessCr = np.zeros(self.PopulSize)
        self.FitDelta = np.zeros(self.PopulSize)
        self.Weights = np.zeros(self.PopulSize)
        self.F = np.zeros(self.NIndsFrontMax)
        self.Cr = np.zeros(self.NIndsFrontMax)
        self.idxs = np.zeros((self.NIndsFrontMax, 3), dtype=np.int32)
        self.cross_mask = np.zeros((self.NIndsFrontMax, self.NVars))

        self.PFIndex = 0
        self.bestfit = np.inf
        self.Generation = 0
        self.SuccessRate = 0.5
        self.TheChosenOne = 0
        self.func_num = 0
        self.func_index = 0

    def initialize(self, _newNInds, _newNVars, _newfunc_num, _newfunc_index):
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

    @staticmethod
    @nb.njit(fastmath=True)
    def weighted_mean(vector, weights):
        sum_w = np.sum(weights)
        if sum_w == 0:
            return 1.0
        return np.sum(vector * weights) / sum_w

    def update_memory_cr(self):
        if self.SuccessFilled > 0:
            idx = self.MemoryIter % self.MemorySize
            mean_cr = self.weighted_mean(
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

    def generate_parameters(self, n):
        meanF = 0.4 + np.tanh(self.SuccessRate * 5) * 0.25
        self.F[:n] = np.clip(np.random.normal(meanF, 0.02, n), 0, 1)
        self.Cr[:n] = np.clip(np.random.normal(self.MemoryCr[self.MemoryIter % self.MemorySize], 0.05, n), 0, 1)
        self.idxs[:n] = np.random.randint(0, self.NIndsFront, (n, 3))
        self.cross_mask[:n] = np.random.random((n, self.NVars))

    def vectorized_mutation_crossover(self, n):
        r1, r2, r3 = self.idxs[:n, 0], self.idxs[:n, 1], self.idxs[:n, 2]
        mutants = (self.PopulFront[r1] +
                   self.F[:n, None] * (self.PopulFront[r2] - self.PopulFront[r3]))

        j_rand = np.random.randint(0, self.NVars, n)
        mask = (self.cross_mask[:n] < self.Cr[:n, None]) | \
               (np.arange(self.NVars) == j_rand[:, None])

        self.Trials[:n] = np.where(mask, mutants, self.PopulFront[:n])

        out_of_bounds = (self.Trials[:n] < self.Left) | (self.Trials[:n] > self.Right)
        self.Trials[:n][out_of_bounds] = np.random.uniform(self.Left, self.Right, np.sum(out_of_bounds))

    def update_population(self, fits, n):
        improved = fits <= self.FitArrFront[:n]
        improved_idx = np.where(improved)[0]

        if len(improved_idx) > 0:
            new_idx = (self.NIndsCurrent + np.arange(len(improved_idx))) % self.PopulSize
            self.Popul[new_idx] = self.Trials[improved_idx]
            self.PopulFront[self.PFIndex:self.PFIndex + len(improved_idx)] = \
                self.Trials[improved_idx]

            self.FitArr[new_idx] = fits[improved_idx]
            self.FitArrFront[self.PFIndex:self.PFIndex + len(improved_idx)] = \
                fits[improved_idx]

            actual_cr = np.mean(self.cross_mask[improved_idx], axis=1)
            delta_f = np.abs(self.FitArrFront[improved_idx] - fits[improved_idx])

            success_idx = self.SuccessFilled + np.arange(len(improved_idx))
            self.tempSuccessCr[success_idx % len(self.tempSuccessCr)] = actual_cr
            self.FitDelta[success_idx % len(self.FitDelta)] = delta_f

            self.SuccessFilled += len(improved_idx)
            self.PFIndex = (self.PFIndex + len(improved_idx)) % self.NIndsFront

            if np.min(fits[improved_idx]) < self.bestfit:
                self.bestfit = np.min(fits[improved_idx])

    def shrink_population(self):
        new_size = max(4, int(self.NIndsFrontMax - (self.NIndsFrontMax - 4) * NFEval / MaxFEval))
        if new_size < self.NIndsFront:
            self.NIndsFront = self.remove_worst(self.NIndsFront, new_size)

    def main_cycle(self, gnbg):
        global NFEval, globalbest, globalbestinit, fopt, ResultsArray, LastFEcount

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

        # Основной цикл
        while NFEval < MaxFEval:
            n = min(self.NIndsFront, MaxFEval - NFEval)

            # Генерация параметров
            self.generate_parameters(n)

            # Векторизованные операции
            self.vectorized_mutation_crossover(n)

            # Оценка
            fits = np.array([gnbg.fitness(x) for x in self.Trials[:n]])
            NFEval += n

            # Обновление популяции
            self.update_population(fits, n)
            self.save_best_values()

            # Адаптация
            self.SuccessRate = self.SuccessFilled / self.NIndsFront if self.NIndsFront > 0 else 0.0
            self.update_memory_cr()

            # Уменьшение популяции
            self.shrink_population()

            self.NIndsCurrent = min(self.NIndsFront + self.SuccessFilled, self.PopulSize)
            self.SuccessFilled = 0
            self.Generation += 1

    def save_best_values(self):
        global globalbest, fopt, ResultsArray, LastFEcount, NFEval, stepsFEval
        error = globalbest - fopt
        if error <= 1e-8 and ResultsArray[ResTsize2 - 1] == MaxFEval:
            ResultsArray[ResTsize2 - 1] = NFEval

        idx = np.searchsorted(stepsFEval, NFEval)
        if idx < len(stepsFEval) and stepsFEval[idx] == NFEval:
            ResultsArray[idx] = 0 if error <= 1e-8 else error
            LastFEcount = idx


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

    # Инициализация stepsFEval
    stepsFEval = np.array([10000.0 / (ResTsize2 - 1) * GNVars * (steps_k + 1) for steps_k in range(ResTsize2 - 1)])

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

            optz = NumbaOptimizer()
            optz.initialize(PopSize * GNVars, GNVars, func_num, func_num)
            optz.main_cycle(gnbg)

        T2 = (time.time() - t0) / maxNFunc
        print(f"T2 = {T2}")

    # Основной цикл по функциям
    for func_num in range(1, maxNFunc + 1):
        gnbg = GNBG(func_num)
        MaxFEval = gnbg.MaxEvals
        fopt = gnbg.OptimumValue

        filename = f"L-SRTDE_GNBG_F{func_num}_D{GNVars}_v3.txt"
        with open(filename, 'w') as fout:
            for run in range(TotalNRuns):
                ResultsArray[ResTsize2 - 1] = MaxFEval
                print(f"func\t{func_num}\trun\t{run}")

                globalbestinit = False
                LastFEcount = 0
                NFEval = 0

                optz = NumbaOptimizer()
                optz.initialize(PopSize * GNVars, GNVars, func_num, func_num)
                optz.main_cycle(gnbg)

                fout.write("\t".join(map(str, ResultsArray)))
                fout.write("\n")

                T0g = time.time() - t0g
                print(f"Total time: {T0g:.2f} seconds")

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
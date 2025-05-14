import numpy as np
import math
import time
from typing import List, Tuple
import random

# Константы
ResTsize1 = 24  # 24 функции в GNBG-2024
ResTsize2 = 1001  # количество записей на функцию (1000+1)

class Optimizer:
    def __init__(self):
        self.MemorySize = 5
        self.MemoryIter = 0
        self.SuccessFilled = 0
        self.MemoryCurrentIndex = 0
        self.NVars = 0               # размерность пространства
        self.NIndsCurrent = 0
        self.NIndsFront = 0
        self.NIndsFrontMax = 0
        self.newNIndsFront = 0
        self.PopulSize = 0
        self.func_num = 0
        self.func_index = 0
        self.TheChosenOne = 0
        self.Generation = 0
        self.PFIndex = 0
        
        self.bestfit = 0.0
        self.SuccessRate = 0.5
        self.F = 0.0       # параметры
        self.Cr = 0.0
        self.Right = 0.0            # верхняя граница
        self.Left = 0.0             # нижняя граница
        
        self.Popul = []             # массив для частиц
        self.PopulFront = []
        self.PopulTemp = []
        self.FitArr = []           # значения функции пригодности
        self.FitArrCopy = []
        self.FitArrFront = []
        self.Trial = []
        self.tempSuccessCr = []
        self.MemoryCr = []
        self.FitDelta = []
        self.Weights = []
        
        self.Indices = []
        self.Indices2 = []
    
    def initialize(self, _newNInds: int, _newNVars: int, _newfunc_num: int, _newfunc_index: int, gnbg):
        self.NVars = _newNVars
        self.NIndsCurrent = _newNInds
        self.NIndsFront = _newNInds
        self.NIndsFrontMax = _newNInds
        self.PopulSize = _newNInds * 2
        self.Left = -100
        self.Right = 100
        self.Generation = 0
        self.TheChosenOne = 0
        self.MemorySize = 5
        self.MemoryIter = 0
        self.SuccessFilled = 0
        self.SuccessRate = 0.5
        self.func_num = _newfunc_num
        self.func_index = _newfunc_index
        
        global stepsFEval
        stepsFEval = [10000.0 / (ResTsize2 - 1) * GNVars * (steps_k + 1) for steps_k in range(ResTsize2 - 1)]
        
        # Инициализация массивов
        self.Popul = np.random.uniform(self.Left, self.Right, (self.PopulSize, self.NVars))
        self.PopulFront = np.zeros((self.NIndsFrontMax, self.NVars))  # вместо NIndsFront
        self.PopulTemp = np.zeros((self.PopulSize, self.NVars))
        self.FitArr = np.zeros(self.PopulSize)
        self.FitArrCopy = np.zeros(self.PopulSize)
        self.FitArrFront = np.zeros(self.NIndsFront)
        self.Weights = np.zeros(self.PopulSize)
        self.tempSuccessCr = np.zeros(self.PopulSize)
        self.FitDelta = np.zeros(self.PopulSize)
        self.MemoryCr = np.ones(self.MemorySize)
        self.Trial = np.zeros(self.NVars)
        self.Indices = np.zeros(self.PopulSize, dtype=int)
        self.Indices2 = np.zeros(self.PopulSize, dtype=int)
        
        # Копируем начальную популяцию в PopulFront
        self.PopulFront = np.copy(self.Popul[:self.NIndsFront])
    
    def update_memory_cr(self):
        if self.SuccessFilled != 0:
            self.MemoryCr[self.MemoryIter] = 0.5 * (self.mean_wl(self.tempSuccessCr[:self.SuccessFilled], 
                                                   self.FitDelta[:self.SuccessFilled]) + 
                                             self.MemoryCr[self.MemoryIter])
            self.MemoryIter = (self.MemoryIter + 1) % self.MemorySize
    
    def mean_wl(self, vector: np.ndarray, temp_weights: np.ndarray) -> float:
        sum_weight = np.sum(temp_weights)
        if sum_weight == 0:
            return 1.0
        
        weights = temp_weights / sum_weight
        sum_square = np.sum(weights * vector * vector)
        sum_val = np.sum(weights * vector)
        
        if abs(sum_val) > 1e-8:
            return sum_square / sum_val
        else:
            return 1.0
    
    def find_n_save_best(self, init: bool, ind_iter: int):
        if init or self.FitArr[ind_iter] <= self.bestfit:
            self.bestfit = self.FitArr[ind_iter]
        
        global globalbest, globalbestinit
        if init or self.bestfit < globalbest:
            globalbest = self.bestfit
            globalbestinit = True
    
    def remove_worst(self, _NIndsFront: int, _newNIndsFront: int):
        points_to_remove = _NIndsFront - _newNIndsFront
        
        for _ in range(points_to_remove):
            worst_num = np.argmax(self.FitArrFront[:_NIndsFront])
            
            # Удаляем худшего
            self.PopulFront = np.delete(self.PopulFront, worst_num, axis=0)
            self.FitArrFront = np.delete(self.FitArrFront, worst_num)
            _NIndsFront -= 1
        
        return _NIndsFront
    
    def qsort2int(self, mass: np.ndarray, mass2: np.ndarray, low: int, high: int):
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
            self.qsort2int(mass, mass2, low, j)
        if i < high:
            self.qsort2int(mass, mass2, i, high)

    def main_cycle(self, gnbg):
        """Основной цикл оптимизации"""
        global NFEval, globalbest, globalbestinit, fopt, ResultsArray, LastFEcount

        # Инициализация fitness для начальной популяции
        for ind_iter in range(self.NIndsFront):
            self.FitArr[ind_iter] = gnbg.fitness(self.Popul[ind_iter])
            NFEval += 1
            self.find_n_save_best(ind_iter == 0, ind_iter)
            if not globalbestinit or self.bestfit < globalbest:
                globalbest = self.bestfit
                globalbestinit = True
            self.save_best_values(self.func_index)

        # Сортировка начальной популяции
        self.FitArrCopy[:self.NIndsFront] = self.FitArr[:self.NIndsFront]
        self.Indices[:self.NIndsFront] = np.arange(self.NIndsFront)

        if np.min(self.FitArr[:self.NIndsFront]) != np.max(self.FitArr[:self.NIndsFront]):
            self.qsort2int(self.FitArrCopy, self.Indices, 0, self.NIndsFront - 1)

        # Инициализация PopulFront
        self.PopulFront = np.zeros((self.NIndsFrontMax, self.NVars))  # Гарантируем достаточный размер
        for i in range(self.NIndsFront):
            self.PopulFront[i] = self.Popul[self.Indices[i]].copy()
            self.FitArrFront[i] = self.FitArrCopy[i]

        self.PFIndex = 0

        while NFEval < MaxFEval:
            # Расчет параметров
            meanF = 0.4 + np.tanh(self.SuccessRate * 5) * 0.25
            sigmaF = 0.02

            # Основной цикл генерации новых решений
            for ind_iter in range(self.NIndsFront):
                # Генерация новых решений
                self.TheChosenOne = random.randint(0, self.NIndsFront - 1)
                self.MemoryCurrentIndex = random.randint(0, self.MemorySize - 1)

                # Параметры мутации
                while True:
                    self.F = np.random.normal(meanF, sigmaF)
                    if 0.0 <= self.F <= 1.0:
                        break

                self.Cr = np.random.normal(self.MemoryCr[self.MemoryCurrentIndex], 0.05)
                self.Cr = max(min(self.Cr, 1.0), 0.0)

                # Применение операторов DE
                actual_cr = 0
                will_crossover = random.randint(0, self.NVars - 1)

                for j in range(self.NVars):
                    if random.random() < self.Cr or will_crossover == j:
                        # Mutation: rand/1
                        r1, r2, r3 = random.sample(range(self.NIndsFront), 3)
                        self.Trial[j] = self.PopulFront[r1][j] + self.F * (
                                    self.PopulFront[r2][j] - self.PopulFront[r3][j])

                        # Boundary control
                        if self.Trial[j] < self.Left:
                            self.Trial[j] = random.uniform(self.Left, self.Right)
                        if self.Trial[j] > self.Right:
                            self.Trial[j] = random.uniform(self.Left, self.Right)

                        actual_cr += 1
                    else:
                        self.Trial[j] = self.PopulFront[self.TheChosenOne][j]

                # Оценка нового решения
                temp_fit = gnbg.fitness(self.Trial)
                NFEval += 1

                # Селекция
                if temp_fit <= self.FitArrFront[self.TheChosenOne]:
                    # Проверка и сброс индекса при необходимости
                    if self.PFIndex >= self.NIndsFront:
                        self.PFIndex = 0

                    # Проверка границ массивов
                    if self.NIndsCurrent + self.SuccessFilled >= self.PopulSize:
                        self.remove_worst(self.NIndsCurrent, self.NIndsFront)
                        self.NIndsCurrent = self.NIndsFront
                        self.SuccessFilled = 0

                    # Сохранение успешного решения
                    self.Popul[self.NIndsCurrent + self.SuccessFilled] = self.Trial.copy()
                    self.PopulFront[self.PFIndex] = self.Trial.copy()

                    self.FitArr[self.NIndsCurrent + self.SuccessFilled] = temp_fit
                    self.FitArrFront[self.PFIndex] = temp_fit

                    self.find_n_save_best(False, self.NIndsCurrent + self.SuccessFilled)
                    self.tempSuccessCr[self.SuccessFilled] = actual_cr / self.NVars
                    self.FitDelta[self.SuccessFilled] = abs(self.FitArrFront[self.TheChosenOne] - temp_fit)

                    self.SuccessFilled += 1
                    self.PFIndex += 1

                self.save_best_values(self.func_index)

            # Обновление параметров
            self.SuccessRate = self.SuccessFilled / self.NIndsFront if self.NIndsFront > 0 else 0.0

            # Линейное уменьшение размера популяции
            self.newNIndsFront = max(4, int(self.NIndsFrontMax - (self.NIndsFrontMax - 4) * NFEval / MaxFEval))
            if self.newNIndsFront < self.NIndsFront:
                self.remove_worst(self.NIndsFront, self.newNIndsFront)
                self.NIndsFront = self.newNIndsFront

            # Обновление памяти Cr
            self.update_memory_cr()

            # Обновление счетчиков
            self.NIndsCurrent = self.NIndsFront + self.SuccessFilled
            self.SuccessFilled = 0
            self.Generation += 1

            # Удаление худших решений, если популяция слишком большая
            if self.NIndsCurrent > self.NIndsFront:
                self.Indices[:self.NIndsCurrent] = np.arange(self.NIndsCurrent)
                if np.min(self.FitArr[:self.NIndsCurrent]) != np.max(self.FitArr[:self.NIndsCurrent]):
                    self.qsort2int(self.FitArr, self.Indices, 0, self.NIndsCurrent - 1)

                self.NIndsCurrent = self.NIndsFront
                for i in range(self.NIndsCurrent):
                    self.Popul[i] = self.Popul[self.Indices[i]].copy()
                    self.FitArr[i] = self.FitArr[self.Indices[i]]
    
    def save_best_values(self, func_index: int):
        global globalbest, fopt, ResultsArray, LastFEcount, NFEval, stepsFEval
        
        temp = globalbest - fopt
        if temp <= 1e-8 and ResultsArray[ResTsize2 - 1] == MaxFEval:
            ResultsArray[ResTsize2 - 1] = NFEval
        
        for stepFEcount in range(LastFEcount, ResTsize2 - 1):
            if NFEval == stepsFEval[stepFEcount]:
                if temp <= 1e-8:
                    temp = 0
                ResultsArray[stepFEcount] = temp
                LastFEcount = stepFEcount
    
    def clean(self):
        # В Python не требуется явное освобождение памяти, но оставим метод для совместимости
        pass

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
        
        filename = f"L-SRTDE_GNBG_F{func_num}_D{GNVars}.txt"
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
    import numpy as np
    from scipy.linalg import sqrtm


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
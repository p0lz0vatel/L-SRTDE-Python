
import numpy as np
import numba as nb

# Глобальные переменные
globalbest = np.inf
globalbestinit = False
NFEval = 0
MaxFEval = 100000

@nb.njit
def differential_mutation(x1, x2, x3, F):
    return x1 + F * (x2 - x3)

@nb.njit
def select_indices(n, k):
    idxs = np.empty((n, k), dtype=np.int32)
    for i in range(n):
        s = np.arange(n)
        s = s[s != i]
        choice = np.random.choice(s, size=k, replace=False)
        for j in range(k):
            idxs[i, j] = choice[j]
    return idxs

@nb.njit
def remove_worst_numba(fit_arr, pop_arr, new_size):
    idx = np.argpartition(fit_arr, new_size)[:new_size]
    return pop_arr[idx], fit_arr[idx]

class L_SRTDE:
    def __init__(self, pop_size, dim, left, right, rng):
        self.NIndsFrontMax = pop_size
        self.PopulSize = pop_size
        self.NIndsFront = pop_size
        self.NIndsCurrent = pop_size
        self.NVars = dim
        self.Left = left
        self.Right = right
        self.Popul = np.random.uniform(left, right, (pop_size, dim))
        self.PopulFront = self.Popul.copy()
        self.FitArr = np.full(pop_size, np.inf)
        self.FitArrFront = np.full(pop_size, np.inf)
        self.tempSuccessCr = np.zeros(pop_size)
        self.FitDelta = np.zeros(pop_size)
        self.bestfit = np.inf
        self.SuccessRate = 0.5
        self.MemoryCr = np.full(5, 0.5)
        self.MemoryIter = 0
        self.MemorySize = 5
        self.SuccessFilled = 0
        self.PFIndex = 0
        self.Generation = 0
        self.rng = rng

    def evaluate_population(self, gnbg):
        fits = np.array([gnbg.fitness(ind) for ind in self.Popul[:self.NIndsFront]])
        self.FitArr[:self.NIndsFront] = fits
        self.FitArrFront[:self.NIndsFront] = fits.copy()
        min_idx = np.argmin(fits)
        if fits[min_idx] < self.bestfit:
            self.bestfit = fits[min_idx]
            global globalbest, globalbestinit
            if self.bestfit < globalbest:
                globalbest = self.bestfit
                globalbestinit = True
        self.save_best_values()

    def update_memory_cr(self):
        if self.SuccessFilled == 0:
            return
        mean_cr = np.mean(self.tempSuccessCr[:self.SuccessFilled])
        self.MemoryCr[self.MemoryIter % self.MemorySize] = mean_cr
        self.MemoryIter += 1

    def remove_worst(self, current_size, new_size):
        self.PopulFront[:new_size], self.FitArrFront[:new_size] = remove_worst_numba(
            self.FitArrFront[:current_size], self.PopulFront[:current_size], new_size
        )
        return new_size

    def save_best_values(self):
        pass

    def main_cycle(self, gnbg):
        global NFEval, globalbest, globalbestinit
        self.evaluate_population(gnbg)
        rng = self.rng
        dim = self.NVars
        pop = self.PopulFront

        while NFEval < MaxFEval:
            meanF = 0.4 + np.tanh(self.SuccessRate * 5) * 0.25
            F = np.clip(rng.normal(meanF, 0.02, self.NIndsFront), 0, 1)
            Cr = np.clip(rng.normal(self.MemoryCr[self.MemoryIter % self.MemorySize], 0.05, self.NIndsFront), 0, 1)
            idxs = rng.integers(0, self.NIndsFront, (self.NIndsFront, 3))

            for i in range(self.NIndsFront):
                r1, r2, r3 = idxs[i]
                x1, x2, x3 = pop[r1], pop[r2], pop[r3]
                target = pop[i]
                donor = differential_mutation(x1, x2, x3, F[i])
                cross_points = rng.random(dim) < Cr[i]
                cross_points[rng.integers(dim)] = True
                trial = donor.copy()
                trial[~cross_points] = target[~cross_points]
                np.clip(trial, self.Left, self.Right, out=trial)
                fit = gnbg.fitness(trial)
                NFEval += 1

                if fit <= self.FitArrFront[i]:
                    idx = (self.NIndsCurrent + self.SuccessFilled) % self.PopulSize
                    self.Popul[idx] = trial
                    self.PopulFront[self.PFIndex % self.NIndsFront] = trial
                    self.FitArr[idx] = fit
                    self.FitArrFront[self.PFIndex % self.NIndsFront] = fit
                    if fit < globalbest:
                        globalbest = fit
                        self.bestfit = fit
                    self.tempSuccessCr[self.SuccessFilled % len(self.tempSuccessCr)] = np.mean(cross_points)
                    self.FitDelta[self.SuccessFilled % len(self.FitDelta)] = abs(self.FitArrFront[i] - fit)
                    self.SuccessFilled += 1
                    self.PFIndex += 1
                self.save_best_values()

            self.SuccessRate = self.SuccessFilled / self.NIndsFront if self.NIndsFront > 0 else 0.0
            self.update_memory_cr()
            new_size = max(4, int(self.NIndsFrontMax - (self.NIndsFrontMax - 4) * NFEval / MaxFEval))
            if new_size < self.NIndsFront:
                self.NIndsFront = self.remove_worst(self.NIndsFront, new_size)
            self.NIndsCurrent = min(self.NIndsFront + self.SuccessFilled, self.PopulSize)
            self.SuccessFilled = 0
            self.Generation += 1

class MyObjective:
    def fitness(self, x):
        return np.sum(x**2)

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    de = L_SRTDE(pop_size=50, dim=10, left=-5.12, right=5.12, rng=rng)
    problem = MyObjective()
    de.main_cycle(problem)
    print("Best solution found:", de.bestfit)

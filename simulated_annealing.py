import pandas as pd
import numpy as np
import math
import itertools as it
import random
import more_itertools
import copy

def tempFunc(T, startT, k):
    return 0.997 * T

class SimulatedAnnealing:
    def __init__(self, metricFunc, para_v: np.array, steps: list, tempFunc, startTemp, length: int):
        self.energy = metricFunc #smaller better
        self.params = para_v
        self.stepSizes = steps
        self.temperature = tempFunc
        self.startTemp = startTemp
        self.T = startTemp
        self.L = length

    def P(self, energy, newEnergy, temp):
        if newEnergy < energy:
            return 1
        return math.exp((energy - newEnergy) / temp)

    def simulatedAnnealing(self, startPoint):
        startMet = self.energy(startPoint)
        point = copy.deepcopy(startPoint)
        bestPt = copy.deepcopy(startPoint)
        bestMet = self.energy(bestPt)
        k = 0
        l = 0
        while l < self.L:
            self.T = self.temperature(self.T, self.startTemp, k)
            newPt = self._sonsGen(point)
            newMet = self.energy(newPt)
            if newMet < bestMet:
                bestPt = newPt
                bestMet = newMet
            if self.P(self.energy(point), newMet, self.T) >= random.uniform(0, 1):
                point = newPt
                l = 0
            k += 1
            l += 1
        return bestPt if bestMet < startMet else startPoint

    def _sonsGen(self, point):
        args = []
        count = 0
        for item in self.stepSizes:
            arg = []
            if count < len(self.params):
                if (point[count] - item) > self.params[count][0]:
                    arg.append(point[count] - item)
                arg.append(point[count])
                if (point[count] + item) < self.params[count][-1]:
                    arg.append(point[count] + item)
                args.append(arg)
                count += 1
        return more_itertools.random_product(*args)


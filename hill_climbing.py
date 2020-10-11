import pandas as pd
import numpy as np
import math
import itertools as it
import random
import more_itertools
import copy


class HillClimber:
    def __init__(self, metricFunc, para_v: np.array, steps: list, searchDepth = -1):
        self.fitness = metricFunc #larger better
        self.params = para_v
        self.stepSizes = steps
        self.searchCount = searchDepth

    def hillClimber(self, startPoint, samples=1):
        bestPt = self._hillClimberHelper(copy.deepcopy(startPoint), self.fitness(startPoint))
        return bestPt

    def _hillClimberHelper(self, point, met):
        newPt = self._hillClimberHelperInner(copy.deepcopy(point), met)
        newMet = self.fitness(newPt)
        diffMet = newMet - met
        while diffMet > 0:
            point = newPt
            met = newMet
            newPt = self._hillClimberHelperInner(copy.deepcopy(point), met)
            newMet = self.fitness(newPt)
            diffMet = newMet - met
        return newPt

    def _hillClimberHelperInner(self, point, met):
        sons = self._sonsGen(point)
        mets = []
        for son in sons:
            mets.append(self.fitness(son))
        bestPt = point
        bestMet = met
        for i in range(len(sons)):
            if mets[i] > bestMet:
                bestMet = mets[i]
                bestPt = sons[i]
        return bestPt

    def _hillClimberHelperSimple(self, point, met):
        sons = self._sonsGen(point)
        for son in sons:
            newMet = self.fitness(son)
            if newMet > met:
                return son
        return point

    def _sonsGen(self, point):
        sons = []
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
        if self.searchCount is -1:
            for son in it.product(*args):
                sons.append(son)
        else:
            while len(sons) < self.searchCount:
                son = more_itertools.random_product(*args)
                sons.append(son)
        return sons


import numpy as np
import scipy.stats

import pathlib

class Distribution:
    def __init__(self):
        pass

class _NoneDist(Distribution):
    def __init__(self):
        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}
    def select(self, number, rng):
        raise SyntaxError("This should not be happening!")
    def params_txt(self):
        return "None"
    def params_xl(self):
        return ["nonedist", 0, 0]

class Gaussian(Distribution):
    def __init__(
        self,
        mean,
        std
    ):
        if isinstance(mean, (int, float)):
            self.mean = [mean]
        elif isinstance(mean, (list, np.ndarray)):
            self.mean = list(mean)
        if isinstance(std, (int, float)):
            self.std = [std]
        elif isinstance(std, (list, np.ndarray)):
            self.std = list(std)

        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}

        if len(self.mean) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["mean=" + str(i) for i in self.mean]
        if len(self.std) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["std=" + str(i) for i in self.std]

        self._MPID['loops'] = len(self.mean) * len(self.std)
    
    def select(self, number, rng):
        number = int(number)
        selection = rng.normal(loc = self.mean, scale = self.std, size = number)
        return selection
    
    def mselect(self, number, rng, loops):
        number = int(number)
        selection = rng.normal(loc = self.mean, scale = self.std, size = (number, loops)).transpose()
        return selection

    
    def params_txt(self):
        return f"Gaussian distribution (mean {self.mean}, std {self.std})"
    
    def params_xl(self):
        return ["gaussian", self.mean, self.std]

class Uniform(Distribution):
    def __init__(
        self,
        min,
        max
    ):
        if isinstance(min, (int, float)):
            self.min = [min]
        elif isinstance(min, (list, np.ndarray)):
            self.min = list(min)
        if isinstance(max, (int, float)):
            self.max = [max]
        elif isinstance(max, (list, np.ndarray)):
            self.max = list(max)

        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}

        if len(self.min) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["min=" + str(i) for i in self.min]
        if len(self.max) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["max=" + str(i) for i in self.max]

        self._MPID['loops'] = len(self.min) * len(self.max)

    def select(self, number, rng):
        number = int(number)
        selection = rng.uniform(low = self.min, high = self.max, size = number)
        return selection

    def mselect(self, number, rng, loops):
        number = int(number)
        selection = rng.uniform(low = self.min, high = self.max, size = (number, loops)).transpose()
        return selection
    
    def params_txt(self):
        return f"uniform distribution ({self.min}-{self.max})"

    def params_xl(self):
        return ["uniform", self.min, self.max]

class Single(Distribution):
    def __init__(
        self,
        value
    ):
        if isinstance(value, (int, float)):
            self.value = [value]
        elif isinstance(value, (list, np.ndarray)):
            self.value = list(value)

        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}

        if len(self.value) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["value=" + str(i) for i in self.value]

        self._MPID['loops'] = len(self.value)
    
    def select(self, number, rng):
        number = int(number)
        return np.full(shape = number, fill_value = self.value)
    
    def mselect(self, number, rng, loops):
        number = int(number)
        selection = rng.uniform(low = self.value, high = self.value, size = (number, loops)).transpose()
        return selection
    
    def params_txt(self):
        return self.value
    
    def params_xl(self):
        return ["single", self.value, 0]

class Custom(Distribution):
    def __init__(
        self,
        dir,
        zero
    ):
        if isinstance(dir, str):
            self.dir = [dir]
        elif isinstance(dir, (np.ndarray, list)):
            self.dir = list(dir)
        if isinstance(zero, (int, float)):
            self.zero = [zero]
        elif isinstance(zero, (np.ndarray, list)):
            self.zero = list(zero)

        self._MPID = {'MPs': 0, 'loops': 0, 'labels': None}

        if len(self.dir) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["dir=" + str(i) for i in self.dir]
        if len(self.zero) > 1:
            self._MPID['MPs'] += 1
            self._MPID['labels'] = ["zero=" + str(i) for i in self.zero]

        self._MPID['loops'] = len(self.dir) * len(self.zero)
        
        dir_list = np.broadcast_to(dir, self._MPID['loops'])
        zero_list = np.broadcast_to(zero, self._MPID['loops'])
        self.pairs = list(range(self._MPID['loops']))
        self.distributions = list(range(self._MPID['loops']))
        for idx, (dir, zero) in enumerate(zip(dir_list, zero_list)):
            source = str(pathlib.Path(__file__).parent.absolute()) + "/injection_profiles/" + dir + ".dat"
            bin_centers, bin_heights = np.loadtxt(fname = source, skiprows = 5, usecols = (0, 1), unpack = True)
            bin_heights_normed = Custom._normalize(bin_heights[bin_heights >= 0])
            bin_centers = bin_centers[bin_heights >= 0] # cut the bin centers array as well
    
            if zero == "avg":
                avg = np.average(bin_centers, weights = bin_heights_normed)
                bin_centers -= avg
            elif zero == "max":
                bin_centers -= bin_centers[np.argmax(bin_heights_normed)]
            elif isinstance(zero, (int, float, np.int64, np.float64)):
                bin_centers -= zero
            else:
                raise ValueError("Your zero parameter is not recognized.")

            bin_edges = np.empty(shape = len(bin_centers) + 1)
            bin_edges[0] = bin_centers[0] - 0.5
            for i in range(1, len(bin_centers)):
                bin_edges[i] = (bin_centers[i - 1] + bin_centers[i]) / 2
            bin_edges[-1] = bin_centers[-1] + 0.5

            self.pairs[idx] = (bin_heights_normed, bin_edges)
            self.distributions[idx] = scipy.stats.rv_histogram(histogram = (bin_heights_normed, bin_edges))

    def select(self, number, rng):
        number = int(number)
        distribution = self.distributions[0]
        selection = distribution.rvs(size = number, random_state = rng)
        return selection
    
    def mselect(self, number, rng, loops):
        number = int(number)

        selection = np.zeros(shape = (loops, number))
        distributions = np.broadcast_to(self.distributions, shape = loops)
        for i in range(loops):
            selection[i] = distributions[i].rvs(size = number, random_state = rng)
        return selection

    def params_txt(self):
        return f"custom distribution ({self.dir}.dat, zero: {self.zero})"
    
    def params_xl(self):
        return ["custom", self.dir, self.zero]

    @staticmethod
    def _normalize(weights):
        return weights / weights.sum()
import numpy as np


class RunningMeanStd(object):
    """
    Implemented based on:
    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    - https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L179-L214
    - https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
    """
    def __init__(self, epsilon=1e-4, momentum=None, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.eps = epsilon
        self.momentum = momentum

    def clear(self):
        self.__init__(self.eps, self.momentum)

    @staticmethod
    def update_ema(old_data, new_data, momentum):
        if old_data is None:
            return new_data
        return old_data * momentum + new_data * (1.0 - momentum)

    def update(self, x):
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        if self.momentum is None or self.momentum < 0:
            self.update_from_moments(batch_mean, batch_var, batch_count)
        else:
            self.mean = self.update_ema(self.mean, batch_mean, self.momentum)
            new_var = np.mean(np.square(x - self.mean))
            self.var = self.update_ema(self.var, new_var, self.momentum)
            self.std = np.sqrt(self.var)
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.std = np.sqrt(new_var)
        self.count = new_count
import torch

class RunningMeanStdTorch(object):
    """
    Implemented based on:
    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    - https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L179-L214
    - https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
    """
    def __init__(self, epsilon=1e-4, momentum=None, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float64)
        self.var = torch.ones(shape, dtype=torch.float64)
        self.count = epsilon
        self.eps = epsilon
        self.momentum = momentum

    def clear(self):
        self.__init__(self.eps, self.momentum)

    @staticmethod
    def update_ema(old_data, new_data, momentum):
        if old_data is None:
            return new_data
        return old_data * momentum + new_data * (1.0 - momentum)

    def update(self, x):
        # x_tensor = torch.tensor(x, dtype=torch.float64)
        x_tensor = x
        batch_mean = torch.mean(x_tensor, dim=0)
        batch_var = torch.var(x_tensor, dim=0)
        batch_count = x_tensor.size(0)

        if self.momentum is None or self.momentum < 0:
            self.update_from_moments(batch_mean, batch_var, batch_count)
        else:
            self.mean = self.update_ema(self.mean, batch_mean, self.momentum)
            new_var = torch.mean(torch.square(x_tensor - self.mean))
            self.var = self.update_ema(self.var, new_var, self.momentum)
            self.std = torch.sqrt(self.var)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.std = torch.sqrt(new_var)
        self.count = new_count

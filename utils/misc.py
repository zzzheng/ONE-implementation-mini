'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math
import torch

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter']


def get_mean_and_std(dataset):
    ''' Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    """make dir if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def repeat_tensor_column(tensor, repeat_times):
    """
    Repeat each colum of a tensor N times
    """
    assert len(tensor.size()) == 2, "expect 2D Tensor"
    num_cols = tensor.size(1)
    for i in range(num_cols):
        if i == 0:
            z = torch.transpose(tensor[:, i].repeat((repeat_times, 1)), 0, 1)
        else:
            z_ = torch.transpose(tensor[:, i].repeat((repeat_times, 1)), 0, 1)
            z = torch.cat((z, z_), dim=1)
    return z


def sum_columns_with_interval(tensor, interval):
    """
    Calculate the sum of columns sampled with an interval.
    Each row is consider as a data sample.
    :param tensor:
    :param interval:
    :return:
    Example
    >>> x = tensor(
        [[  9,  12,  15,   3,   4,   5,   6,   7,   8],
        [ 36,  39,  42,  12,  13,  14,  15,  16,  17],
        [ 63,  66,  69,  21,  22,  23,  24,  25,  26],
        [ 90,  93,  96,  30,  31,  32,  33,  34,  35],
        [117, 120, 123,  39,  40,  41,  42,  43,  44]])
    >>> add_columns_with_interval(x, 3) = tensor(
        [[  9,  12,  15],
        [ 36,  39,  42],
        [ 63,  66,  69],
        [ 90,  93,  96],
        [117, 120, 123]])
    """
    assert len(tensor.size()) == 2, "expect 2D Tensor"
    assert tensor.size(1) % interval == 0, "invalid interval"

    steps = int(tensor.size(1) / interval)
    for i in range(steps):
        if i == 0:
            z = tensor[:, i * interval: (i + 1) * interval]
        else:
            z += tensor[:, i * interval: (i + 1) * interval]
    return z


if __name__ == "__main__":
    import timeit

    x = [i for i in range(150)]
    x = torch.tensor(x)
    x = x.reshape(5, 30)

    start = timeit.default_timer()
    z = sum_columns_with_interval(x, 10)
    end = timeit.default_timer()

    print("x.size() = ", x.size())
    print("z.size() = ", z.size())

    print("x = ", x)
    print("z = ", z)

    print('executed time 1= ', end - start)




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def kl(a, b):
    return a + b


def kl_(a, b):
    return a + b


def kl_loss():
    pass


def soft_label(logits, T):
    """Softens logits.

    Args:
        logits:
            tensor(N, C), usually the logits
                N - minibatch
                C - number of classes
                T - temperature
    Returns:
        soft_label:
            tensor(N, C)
    """
    assert torch.is_tensor(logits), 'Input should be a PyTorch tensor.'
    assert len(logits.size()) == 2, 'Input should be a 2-D tensor.'

    logits_ = torch.exp(logits / T)
    soft_label = logits_ / torch.unsqueeze(torch.sum(logits_, dim=1), dim=1)

    return soft_label


class KLLoss(nn.Module):

    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor, T=1.0) -> Tensor:
        input = F.softmax(input / T, dim=1)
        target = F.softmax(target / T, dim=1)
        loss = T * T * F.kl_div(input, target, reduction='mean')
        # print('=========== KLLoss forward ===========')
        # print('input_0 = ', input[0, :])
        # print('target_0 = ', target[0, :])
        #
        # print('input_1 = ', input[1, :])
        # print('target_1 = ', target[1, :])
        # print('loss = ', loss)
        # print('======================================')
        return loss


if __name__ == '__main__':
    input = torch.tensor(
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]])

    T = 3.0
    print('soft_label = ', soft_label(input, T))
    print('F.softmax = ', F.softmax(input/T, dim=1))



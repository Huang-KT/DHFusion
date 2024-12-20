import torch
import numpy as np


def Eu_dist(x):
    """
    Calculate the Euclidean distance among each raw of x
    :param x: B X D X N
                B: Batch size
                N: the object number
                D: Dimension of the feature
    :return: B X N X N distance matrix
    """
    assert type(x) == torch.Tensor

    if len(x.shape) != 3:
        x = x.unsqueeze(0)
    x = x.transpose(-1, -2)

    aa = torch.sum(torch.mul(x, x), -1).unsqueeze(2)
    ab = torch.matmul(x, x.transpose(1, 2))
    dist = aa + aa.transpose(-1, -2) - 2 * ab
    dist[dist < 0] = 0
    dist = torch.sqrt(dist)
    dist = torch.max(dist, dist.transpose(1,2))
    return dist


def Cos_dist(x):
    """
    Calculate the Cosine distance among each raw of x
    :param x: B X D X N
                B: Batch size
                N: the object number
                D: Dimension of the feature
    :return: B X N X N distance matrix
    """
    assert type(x) == torch.Tensor

    x_norm = torch.linalg.norm(x, dim=1, keepdim=True)
    cos = torch.matmul(x.transpose(1, 2), x) / (x_norm * x_norm.transpose(1, 2))  # Cosine Similarity [-1, 1]

    cos = torch.clamp(cos, min=-1.0, max=1.0)
    dist = (1 - cos) / 2  # Cosine Distance [0, 1]
    return dist
